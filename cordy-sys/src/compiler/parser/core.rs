/// Core Parser Implementation
///
/// This implementation manages token advancing, error handling, and issues related to newline handling.
/// It also handles the core structures representing the code flow graph (`Code`, `Block`, and branch handling).

use std::collections::HashMap;
use fxhash::FxBuildHasher;
use indexmap::IndexSet;

use crate::compiler::parser::{Parser, ParserError};
use crate::compiler::parser::ParserErrorType;
use crate::compiler::parser::semantic::{LValueReference, ParserFunctionImpl, ReferenceType};
use crate::compiler::scanner::ScanToken;
use crate::reporting::Location;
use crate::trace;
use crate::vm::Opcode;
use crate::vm::operator::CompareOp;

use Opcode::{*};
use ParserErrorType::{*};
use ScanToken::{*};


/// A series of code blocks representing the control flow graph.
/// - Each `Block` is uniquely identifiable by its index within `blocks`, meaning unique and persistent IDs may be taken
/// - Blocks can form references to other blocks via these IDs in the form of `BlockEnd`
/// - A unique identifier for a single opcode location can take the form of a block ID plus a opcode ID
#[derive(Debug)]
pub struct Code {
    pub blocks: Vec<Block>,
    pub locals: HashMap<CodeId, String, FxBuildHasher>,
}


/// A block of code with no internal `Jump` (branching) opcodes, and strict unique ID tracking for each emitted opcode.
/// - No `Jump`s means this can be optimized as an atomic unit, with opcodes inserted or removed as necessary.
/// - Strict unique ID tracking means a ID can be held and new opcodes inserted after it, without distributing any other held IDs.
///
/// Note that some optimizations on a `Block` are destructive - i.e. they can delete Opcodes, and their corresponding IDs.
/// Since we don't track these deleted IDs, this means that all outstanding late bindings need to be resolved before optimizations.
#[derive(Debug, Eq, PartialEq)]
pub struct Block {
    /// The code within the block, with each location holding:
    /// - A `usize` ID -> this will never change, and is what uniquely identifies a given opcode within a block.
    /// - The `Location` and `Opcode` used to emit locations and code respectively upon teardown.
    code: Vec<(usize, Location, Opcode)>,
    /// The behavior upon reaching the end of the block
    end: BlockEnd
}

/// An enum representing the terminal behavior of a `Block`
#[derive(Debug, Eq, PartialEq)]
pub enum BlockEnd {
    /// The default value of a block, in that it does not connect to another block. `Return`, `Exit` or `Yield` opcodes will be inserted automatically.
    Exit,
    /// A block with an unconditional jump to the next one. This is used when we need to represent a converging branch:
    /// <pre>
    ///            +--------&lt;--------+
    ///            |                 |
    /// [ Next ] --+--&gt; [ Branch ] --+--&gt;
    /// </pre>
    Next(usize),
    /// A block with two possible branches.
    /// - `ty` determines the type of branch, which is used for the emitted `Opcode`. It **should not** be `BranchType::Jump`, as that should be represented by `Next`
    /// - `default` represents the next block if the branch is not taken
    /// - `branch` represents the next block if the branch is taken.
    ///
    /// Note that if `default != current + 1`, this will require two `Jump` opcodes to fully implement.
    Branch { ty: BranchType, default: usize, branch: usize },
}

impl BlockEnd {
    fn branch(ty: BranchType, default: usize, branch: usize ) -> BlockEnd {
        match ty {
            BranchType::Jump => BlockEnd::Next(branch),
            ty => BlockEnd::Branch { ty, default, branch }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum BranchType {
    Jump, JumpIfTrue, JumpIfFalse, JumpIfTruePop, JumpIfFalsePop, Compare(CompareOp), TestIterable, AssertTest, AssertCompare(CompareOp)
}

/// Enough information to uniquely identify any given opcode within a function:
///
/// - `block_id` : An index into the block of that function's code
/// - `opcode_id` : An index into the block's `code`
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CodeId {
    block_id: usize,
    opcode_id: usize,
}

/// An extension of `OpcodeId` to also identify an enclosing function.
///
/// N.B. This does not allow identifying global code
#[derive(Debug, Clone)]
pub struct OpcodeId {
    /// An index into the owning function, as part of the parser functions
    function_id: usize,
    code_id: CodeId,
}

/// An index into a given `Block`, used to identify **forward** branch targets.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ForwardBlockId(pub usize);

/// An index into a given `Block`, used to identify **reverse** branch targets.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ReverseBlockId(pub usize);



impl Code {

    pub fn new() -> Code {
        Code {
            blocks: vec![Block::new(BlockEnd::Exit)],
            locals: HashMap::with_hasher(FxBuildHasher::default()),
        }
    }

    /// Emits code within this code, joining all blocks together with standard jump instructions, and emitting `Location`s in parallel.
    ///
    /// Returns a vector of `block_id` -> `index` in the output.
    ///
    /// **Implementation Note:** This function does not take a `&mut Parser` for flexibility of partial borrows.
    pub fn emit(&mut self, code: &mut Vec<Opcode>, locations: &mut Vec<Location>, locals: &mut Vec<String>) -> Vec<usize> {
        let mut starts: Vec<usize> = Vec::new(); // Indices into the start location of each block. These are the target of all jump instructions
        let mut branches: Vec<(usize, usize, BranchType)> = Vec::new(); // List of un-filled-in branches by index, jump-to-block_id, and branch type

        for (block_id, block) in self.blocks.drain(..).enumerate() {
            // This block starts at the current index
            starts.push(code.len());

            // Emit all code within the block
            for (opcode_id, loc, op) in block.code {
                code.push(op);
                locations.push(loc);

                // Emit matching locals for each opcode
                if let Some(local) = self.locals.remove(&CodeId { block_id, opcode_id }) {
                    locals.push(local);
                }
            }

            // And handle the end of the block
            match block.end {
                BlockEnd::Exit => {}, // Don't emit anything, the top-level `exit` or `yield` will be emitted, or a `Return` will already be present
                BlockEnd::Next(next_id) => {
                    if block_id + 1 != next_id { // Only need a branch if we aren't branching to the next block
                        branches.push((code.len(), next_id, BranchType::Jump));
                        code.push(Opcode::placeholder());
                        locations.push(Location::empty());
                    }
                }
                BlockEnd::Branch { ty, default, branch } => {
                    branches.push((code.len(), branch, ty)); // Always emit the single branch instruction
                    code.push(Opcode::placeholder());
                    locations.push(Location::empty());
                    if block_id + 1 != default { // And emit another unconditional jump if necessary
                        branches.push((code.len(), default, BranchType::Jump));
                        code.push(Opcode::placeholder());
                        locations.push(Location::empty());
                    }
                }
            }
        }

        // Fix all branch locations
        for (index, block_id, ty) in branches {
            let offset = starts[block_id] as i32 - index as i32 - 1;
            code[index] = match ty {
                BranchType::Jump => Jump(offset),
                BranchType::JumpIfTrue => JumpIfTrue(offset),
                BranchType::JumpIfFalse => JumpIfFalse(offset),
                BranchType::JumpIfTruePop => JumpIfTruePop(offset),
                BranchType::JumpIfFalsePop => JumpIfFalsePop(offset),
                BranchType::Compare(op) => Compare(op, offset),
                BranchType::TestIterable => TestIterable(offset),
                BranchType::AssertTest => AssertTest(offset),
                BranchType::AssertCompare(op) => AssertCompare(op, offset),
            }
        }

        starts
    }


    fn push(&mut self, loc: Location, opcode: Opcode, local: Option<String>) {
        let block_id = self.blocks.len() - 1;
        let opcode_id = self.blocks.last_mut().unwrap().push(loc, opcode);

        if let Some(local) = local {
            let code_id = CodeId { block_id, opcode_id };
            self.locals.insert(code_id, local);
        }
    }

    /// Used to create a forward branch from this point, which the target will be filled in later. Usage:
    ///
    /// 1. Call `branch_forward()` to end the current block, and retrieve a `block_id` to fill in the branch later
    /// 2. Call `join_forward(block_id)` to fill in the branch on the given `block_id`
    ///
    /// N.B. This cannot be used to branch over functions.
    fn branch_forward(&mut self) -> ForwardBlockId {
        let block_id = self.blocks.len() - 1;
        self.blocks.push(Block::new(BlockEnd::Exit));
        ForwardBlockId(block_id)
    }

    /// Joins a forward branch to this point, by ending the current block with a `Next` and fixing the branch from `block_id` to the new block.
    fn join_forward(&mut self, block_id: ForwardBlockId, ty: BranchType) {
        let block_id = block_id.0;
        let branch_id = self.blocks.len();

        self.blocks[branch_id - 1].end(BlockEnd::Next(branch_id)); // End the current block with a `Next`
        self.blocks[block_id].end(BlockEnd::branch(ty, block_id + 1, branch_id)); // Fix the branch to the next block
        self.blocks.push(Block::new(BlockEnd::Exit)); // And push a new block
    }

    /// Used to create a reverse branch to this point, which the branch will be filled in later. Usage:
    ///
    /// 1. Call `branch_reverse()` to end the current block, and retrieve a `block_id` to be branched to later
    /// 2. Call `join_reverse(block_id)` to end the current block with a `Branch` back to the provided `block_id`
    ///
    /// N.B. This cannot be used to branch over functions.
    fn branch_reverse(&mut self) -> ReverseBlockId {
        let block_id = self.blocks.len();
        self.blocks[block_id - 1].end(BlockEnd::Next(block_id));
        self.blocks.push(Block::new(BlockEnd::Exit));
        ReverseBlockId(block_id)
    }

    /// Joins a reverse branch, by ending the current block with a `Branch` to the provided `block_id`
    fn join_reverse(&mut self, block_id: ReverseBlockId, ty: BranchType) {
        let block_id = block_id.0;
        let branch_id = self.blocks.len();

        self.blocks[branch_id - 1].end(BlockEnd::branch(ty, branch_id, block_id));
        self.blocks.push(Block::new(BlockEnd::Exit));
    }
}

/// Bouncers for `branch` and `join` methods on the current code block
impl<'a> Parser<'a> {
    pub fn branch_forward(&mut self) -> ForwardBlockId { self.code().branch_forward() }
    pub fn branch_reverse(&mut self) -> ReverseBlockId { self.code().branch_reverse() }
    pub fn join_forward(&mut self, block_id: ForwardBlockId, ty: BranchType) { self.code().join_forward(block_id, ty) }
    pub fn join_reverse(&mut self, block_id: ReverseBlockId, ty: BranchType) { self.code().join_reverse(block_id, ty) }

    fn code(&mut self) -> &mut Code {
        match self.current_locals().func {
            Some(func) => &mut self.functions[func].code,
            None => &mut self.output,
        }
    }
}


impl Block {
    fn new(end: BlockEnd) -> Block {
        Block { code: Vec::new(), end }
    }

    // ===== Optimization API ===== //

    pub fn set(&mut self, index: usize, op: Opcode) {
        self.code[index].2 = op;
    }

    pub fn remove(&mut self, index: usize) {
        self.code.remove(index);
    }

    pub fn get(&mut self, index: usize) -> Option<Opcode> {
        self.code.get(index).map(|(_, _, u)| *u)
    }

    // ===== Internals ===== //

    fn push(&mut self, loc: Location, opcode: Opcode) -> usize {
        let opcode_id = self.code.len();
        self.code.push((opcode_id, loc, opcode));
        opcode_id
    }

    fn insert(&mut self, id: usize, loc: Location, opcode: Opcode) {
        match self.code.iter().position(|(i, _, _)| i == &id) {
            // The target `OpcodeId` was found, so insert immediately after, and shift all opcodes to the right
            Some(code_id) => self.code.insert(code_id, (self.code.len(), loc, opcode)),
            // The target `OpcodeId` was not found, which means this was an empty block
            // Just push into the empty block
            None => {
                debug_assert!(self.code.is_empty()); // This should always be true, if we cannot find a matching `OpcodeId`
                self.code.push((0, loc, opcode))
            }
        }
    }

    fn end(&mut self, end: BlockEnd) {
        debug_assert!(self.end == BlockEnd::Exit, "Should not assign self.end twice, already was {:?}", self.end);
        self.end = end;
    }
}



/// A state to restore to while backtracking.
/// Only stores enough state to be necessary, as we don't need to backtrack through output tokens.
pub struct ParserState {
    input: Vec<(Location, ScanToken)>,
    errors: usize,
}

impl ParserState {
    fn store(parser: &Parser) -> ParserState {
        ParserState {
            input: Vec::new(),
            errors: parser.errors.len(),
        }
    }

    fn restore(self, parser: &mut Parser) {
        trace::trace_parser!("restoring input {:?} -> [{:?}, ...]", self.input, parser.input.front());

        for token in self.input.into_iter().rev() {
            parser.input.push_front(token);
        }

        parser.errors.truncate(self.errors);
        parser.error_recovery = false;
    }
}


impl<'a> Parser<'a> {

    pub fn begin(&mut self) -> bool {
        trace::trace_parser!("begin backtracking");
        assert!(self.restore_state.is_none(), "Recursive backtracking attempt");

        self.restore_state = Some(ParserState::store(self));
        !self.error_recovery
    }

    pub fn accept(&mut self) {
        trace::trace_parser!("accept backtracking");
        self.restore_state = None;
    }

    pub fn reject(&mut self) {
        trace::trace_parser!("reject backtracking");
        match self.restore_state.take() {
            Some(state) => state.restore(self),
            _ => panic!("reject() without begin()"),
        }
    }

    /// Returns a unique identifier for the next opcode in the current function.
    ///
    /// N.B. The parser **must** be within a function, this does not allow identifying targets within global code!
    pub fn next_opcode(&self) -> OpcodeId {
        let function_id = self.current_locals().func.expect("next_opcode() called in global scope!");
        let blocks = &self.functions[function_id].code.blocks;
        let block_id = blocks.len() - 1;
        let opcode_id = blocks[block_id].code.len();

        OpcodeId { function_id, code_id: CodeId { block_id, opcode_id } }
    }

    pub fn insert(&mut self, at: OpcodeId, loc: Location, opcode: Opcode) { Parser::do_insert(at, loc, opcode, &mut self.functions) }

    /// Inserts the provided instruction at the `OpcodeId`, and shifts all other code forward.
    ///
    /// **Implementation Note:** This does not take `&mut self` to allow the flexibility of partial borrows.
    pub fn do_insert(at: OpcodeId, loc: Location, opcode: Opcode, functions: &mut Vec<ParserFunctionImpl>) {
        functions[at.function_id].code.blocks[at.code_id.block_id].insert(at.code_id.opcode_id, loc, opcode);
    }

    /// In the parsing of `<term>` defined by the following grammar:
    /// ```grammar
    /// <term> := <open-token> <term-arguments> ? <close-token>
    ///
    /// <term-arguments>
    ///     | <term-argument> `,` <term-arguments>
    ///     | <term-argument> `,` ?
    /// ```
    ///
    /// This parses the suffix after a `<term-argument>`: A close token, or optional trailing comma, or comma followed by another token.
    ///
    /// Returns `true` if the close token was encountered, or an error (either case causing the loop to break).
    /// Does not consume the closing token.
    pub(super) fn parse_optional_trailing_comma<F : FnOnce(Option<ScanToken>) -> ParserErrorType>(&mut self, close_token: ScanToken, error: F) -> bool {
        trace::trace_parser!("rule <csv-term-suffix>");
        match self.peek() {
            Some(Comma) => {
                self.skip(); // Consume `,`
                if Some(&close_token) == self.peek() { // Check again, if this is a trailing comma and close token
                    return true; // Don't consume the close token, as it's used as a resync point
                }
            },
            Some(t) if t == &close_token => {
                return true; // Don't consume the close token, as it's used as a resync point
            },
            _ => {
                self.error_with(error);
                return true; // Since in this situation, we still want to break
            },
        }
        false
    }

    /// If the given token is present, accept it. Otherwise, flag an error and enter error recovery mode.
    pub fn expect(&mut self, token: ScanToken) {
        match self.peek() {
            Some(t) if t == &token => {
                trace::trace_parser!("expect {:?} -> pass", token);
                self.advance();
            },
            _ => self.error_with(move |t| ExpectedToken(token, t)),
        }
    }

    /// Acts as a resynchronization point for error mode.
    /// Accepts tokens from the input (ignoring the current state of error recovery mode), until we reach the expected token or an empty input.
    /// If we reach the expected token, it is consumed and error mode is unset.
    pub fn expect_resync(&mut self, token: ScanToken) {
        if let Some(t) = self.peek() { // First, check for expect() without raising an error
            if *t == token {
                trace::trace_parser!("expect_resync {:?} -> pass", token);
                self.advance();
                return;
            }
        }

        // Then if we fail, start resync. Initially set error recovery `false`, so we can peek ahead at the input.
        let error_recovery = self.error_recovery;
        loop {
            self.error_recovery = false;
            match self.peek() {
                Some(t) if *t == token => {
                    trace::trace_parser!("expect_resync {:?} -> synced", token);
                    self.advance();
                    break;
                },
                Some(_t) => {
                    trace::trace_parser!("expect_resync {:?} -> discarding {:?}", token, _t);
                    self.advance();
                },
                None => {
                    // Error recovery failed - we reached end of input
                    // If we were already in an error and trying to resync, then don't raise an additional error
                    if !error_recovery {
                        self.error(ExpectedToken(token, None));
                    }
                    break
                }
            }
        }
    }

    /// Like `advance()`, but returns the boxed `Identifier` token.
    /// **Important**: Must only be called once `peek()` has identified an `Identifier` token is present, as this will panic otherwise.
    pub fn advance_identifier(&mut self) -> String {
        match self.advance() {
            Some(Identifier(name)) => name,
            t => panic!("Token mismatch in advance_identifier() -> expected an Some(Identifier(String)), got a {:?} instead", t)
        }
    }

    /// Like `advance()`, but returns the boxed `String` literal token.
    /// **Important**: Must only be called once `peek()` has identified a `StringLiteral` token is present, as this will panic otherwise.
    pub fn advance_str(&mut self) -> String {
        match self.advance() {
            Some(StringLiteral(s)) => s,
            t => panic!("Token mismatch in advance_str() -> expected a Some(StringLiteral(String)), got a {:?} instead", t)
        }
    }

    /// Expects an identifier token, consumes it, and returns the name. If an identifier was not present, then raises the provided error, and returns a procedurally generated ID.
    pub fn expect_identifier<F : FnOnce(Option<ScanToken>) -> ParserErrorType>(&mut self, err: F) -> String {
        match self.peek() {
            Some(Identifier(_)) => match self.advance() {
                Some(Identifier(it)) => it,
                _ => panic!()
            },
            _ => {
                self.error_with(err);
                self.synthetic_name()
            }
        }
    }

    /// Specialization of `peek()`, does not consume newlines, for checking if a specific syntax token is on the same line as the preceding one.
    pub fn peek_no_newline(&self) -> Option<&ScanToken> {
        if self.error_recovery {
            return None
        }
        if let Some((_, token)) = self.input.iter().next() {
            return Some(token)
        }
        None
    }

    /// Peeks at the next incoming token.
    /// Note that this function only returns a read-only reference to the underlying token, suitable for matching
    /// If the token data needs to be unboxed, i.e. as with `Identifier` tokens, it must be extracted only via `advance()`
    /// This also does not consume newline tokens in the input, rather peeks _past_ them in order to find the next matching token.
    pub fn peek(&self) -> Option<&ScanToken> { self.peek_lookahead(0) }
    pub fn peek2(&self) -> Option<&ScanToken> { self.peek_lookahead(1) }
    pub fn peek3(&self) -> Option<&ScanToken> { self.peek_lookahead(2) }

    fn peek_lookahead(&self, mut lookahead: u8) -> Option<&ScanToken> {
        if !self.error_recovery {
            for (_, token) in &self.input {
                if token != &NewLine {
                    if lookahead == 0 {
                        return Some(token)
                    }
                    lookahead -= 1;
                } else {
                    //self.prevent_expression_statement = false;
                }
            }
        }
        None
    }

    /// Like `advance()` but discards the result.
    pub fn skip(&mut self) {
        self.advance_both();
    }

    /// Like `advance()` but returns the location of the advanced-by token.
    pub fn advance_with(&mut self) -> Location {
        self.advance_both().map(|(loc, _)| loc).unwrap_or_else(Location::empty)
    }

    pub fn advance(&mut self) -> Option<ScanToken> {
        self.advance_both().map(|(_, t)| t)
    }

    /// Advances and returns the next incoming token.
    /// Will also advance past any newline tokens, and so the advanced token will be the next token _after_ any newlines between the last token and the next.
    fn advance_both(&mut self) -> Option<(Location, ScanToken)> {
        if self.error_recovery {
            return None
        }
        while let Some((_, NewLine)) = self.input.front() {
            let token = self.input.pop_front().unwrap();
            if let Some(state) = &mut self.restore_state {
                state.input.push(token);
            }
        }
        trace::trace_parser!("advance {:?}", self.input.front());
        let ret = self.input.pop_front();
        if let Some(token) = &ret {
            if let Some(state) = &mut self.restore_state {
                state.input.push(token.clone());
            }
        }
        if let Some((loc, _)) = ret {
            self.last_location = Some(loc)
        }
        ret
    }

    /// If we previously delayed a `Pop` opcode from being omitted, push it now and reset the flag
    pub fn push_delayed_pop(&mut self) {
        if self.delay_pop_from_expression_statement {
            trace::trace_parser!("push Pop (delayed)");
            self.push_pop(1);
            self.delay_pop_from_expression_statement = false;
        }
    }

    /// Pushes a number of `Pop` opcodes
    pub fn push_pop(&mut self, pops: u32) {
        trace::trace_parser!("push Pop {}", pops);
        for _ in 0..pops {
            self.push(Pop);
        }
    }

    pub fn push_load_lvalue(&mut self, loc: Location, lvalue: LValueReference) {
        match lvalue {
            LValueReference::Local(index) |
            LValueReference::LocalThis(index) => self.push_at(PushLocal(index), loc),
            LValueReference::Global(index) => self.push_at(PushGlobal(index), loc),
            LValueReference::UpValue(index) |
            LValueReference::UpValueThis(index) => self.push_at(PushUpValue(index), loc),
            LValueReference::ThisField { upvalue, index, field_index } => {
                self.push_at(if upvalue { PushUpValue(index) } else { PushLocal(index) }, loc);
                self.push_at(GetField(field_index), loc);
            }

            LValueReference::Method(index) => self.push_at(Constant(index), loc),
            LValueReference::ThisMethod { upvalue, index, function_id } => {
                self.push_at(if upvalue { PushUpValue(index) } else { PushLocal(index) }, loc);
                self.push_at(GetMethod(function_id), loc);
            }
            LValueReference::NativeFunction(native) => self.push_at(NativeFunction(native), loc),

            LValueReference::LateBinding(mut binding) => {
                binding.update(ReferenceType::Load(self.next_opcode()));
                self.late_bindings.push(*binding);
            }

            LValueReference::Invalid => {},
            LValueReference::Named(_) | LValueReference::This => panic!("Invalid load"),
        }
    }

    pub fn push_store_lvalue_prefix(&mut self, lvalue: &LValueReference, loc: Location) {
        match lvalue {
            LValueReference::ThisField { upvalue, index, .. } => {
                self.push_at(if *upvalue { PushUpValue(*index) } else { PushLocal(*index) }, loc);
            },
            _ => {}, // No-op
        }
    }

    pub fn push_store_lvalue(&mut self, lvalue: LValueReference, loc: Location, prefix: bool) {
        match lvalue {
            LValueReference::Local(index) => self.push_at(StoreLocal(index, false), loc),
            LValueReference::Global(index) => self.push_at(StoreGlobal(index, false), loc),
            LValueReference::UpValue(index) => self.push_at(StoreUpValue(index), loc),

            ref lvalue @ LValueReference::ThisField { field_index, .. } => {
                // If the prefix was not pushed, we have to push `self` and then swap so `self` is below the target value
                if !prefix {
                    self.push_store_lvalue_prefix(lvalue, loc);
                    self.push_at(Swap, loc); // Need the `self` to be below the field to be assigned
                }
                self.push_at(SetField(field_index), loc);
            }

            LValueReference::LocalThis(_) |
            LValueReference::UpValueThis(_) |
            LValueReference::Method(_) |
            LValueReference::ThisMethod { .. } |
            LValueReference::NativeFunction(_) => self.error(InvalidAssignmentTarget),

            LValueReference::LateBinding(mut binding) => {
                binding.update(ReferenceType::Store(self.next_opcode()));
                self.late_bindings.push(*binding);
            }

            LValueReference::Invalid => {},
            LValueReference::Named(_) | LValueReference::This => panic!("Invalid store"),
        }
    }

    /// Pushes a new token into the output stream.
    /// Returns the index of the token pushed, which allows callers to later mutate that token if they need to.
    pub fn push(&mut self, opcode: Opcode) {
        self.push_at(opcode, self.prev_location());
    }

    pub fn push_at(&mut self, opcode: Opcode, loc: Location) {
        trace::trace_parser!("push {:?}", opcode);

        let local = match opcode {
            PushGlobal(index) | StoreGlobal(index, _) => Some(self.locals[0].get_name(index as usize)),
            PushLocal(index) | StoreLocal(index, _) => Some(self.locals[self.function_depth as usize].get_name(index as usize)),
            // todo: upvalues?
            _ => None
        };

        let code = match self.current_locals().func {
            Some(func) => &mut self.functions[func].code,
            None => &mut self.output
        };

        code.push(loc, opcode, local);
    }

    /// A specialization of `error()` which provides the last token (the result of `peek()`) to the provided error function
    /// This avoids ugly borrow checker issues where `match self.peek() { ... t => self.error(Error(t)) }` does not work, despite the semantics being identical.
    pub fn error_with<F : FnOnce(Option<ScanToken>) -> ParserErrorType>(&mut self, error: F) {
        Parser::do_error(self.next_location(), error(self.peek().cloned()), true, &mut self.errors, &mut self.error_recovery)
    }

    pub fn error(&mut self, error: ParserErrorType) {
        Parser::do_error(self.next_location(), error, true, &mut self.errors, &mut self.error_recovery)
    }

    pub fn error_at(&mut self, loc: Location, error: ParserErrorType) {
        Parser::do_error(loc, error, true, &mut self.errors, &mut self.error_recovery)
    }

    pub fn semantic_error(&mut self, error: ParserErrorType) {
        Parser::do_error(self.prev_location(), error, false, &mut self.errors, &mut self.error_recovery)
    }

    pub fn semantic_error_at(&mut self, loc: Location, error: ParserErrorType) {
        Parser::do_error(loc, error, false, &mut self.errors, &mut self.error_recovery)
    }

    /// Pushes an error into the output. If error recovery mode was active, no error is emitted.
    ///
    /// - `loc` is the code location associated with the error
    /// - `error` is the error type itself
    /// - `start_error_recovery` is `true` if this was a **lexical** error, and the following token stream is invalid. This will initiate error recovery mode until a resync.
    ///
    /// **Implementation Note:** This function does not take `&mut self` to allow flexibility of partial borrows.
    pub fn do_error(loc: Location, error: ParserErrorType, lexical: bool, errors: &mut IndexSet<ParserError, FxBuildHasher>, error_recovery: &mut bool) {
        trace::trace_parser!("push_err (error = {}) {:?}", error_recovery, error);
        if !*error_recovery {
            errors.insert(ParserError::new(error, loc));
        }
        if lexical {
            *error_recovery = true;
        }
    }

    /// Returns the source location of the previous token, aka the one just accepted.
    pub fn prev_location(&self) -> Location {
        self.last_location.unwrap_or_else(Location::empty)
    }

    /// Returns the source location of the next token, aka the one in `peek()`
    pub fn next_location(&self) -> Location {
        self.input.front().map(|&(loc, _)| loc).unwrap_or_else(Location::empty)
    }

    pub fn synthetic_name(&mut self) -> String {
        self.synthetic_id += 1;
        format!("${}", self.synthetic_id)
    }
}


#[cfg(test)]
mod tests {
    use crate::compiler::parser::core::{Block, BlockEnd, BranchType, Code};

    #[test]
    fn test_branch_forward() {
        let mut code = Code::new();
        // ... first block ...
        let branch = code.branch_forward();
        // ... second block ...
        code.join_forward(branch, BranchType::JumpIfTrue);
        // ... third block ...

        assert_eq!(code.blocks, vec![
            Block { code: vec![], end: BlockEnd::Branch { ty: BranchType::JumpIfTrue, default: 1, branch: 2 } },
            Block { code: vec![], end: BlockEnd::Next(2) },
            Block { code: vec![], end: BlockEnd::Exit }
        ])
    }

    #[test]
    fn test_branch_forward_twice() {
        let mut code = Code::new();
        // ... first block ...
        let branch1 = code.branch_forward();
        // ... block if true ...
        let branch2 = code.branch_forward();
        // ... empty block ...
        code.join_forward(branch1, BranchType::JumpIfFalse);
        // ... block if false ...
        code.join_forward(branch2, BranchType::Jump);
        // ... final block ...

        assert_eq!(code.blocks, vec![
            Block { code: vec![], end: BlockEnd::Branch { ty: BranchType::JumpIfFalse, default: 1, branch: 3 } },
            Block { code: vec![], end: BlockEnd::Next(4) },
            Block { code: vec![], end: BlockEnd::Next(3) },
            Block { code: vec![], end: BlockEnd::Next(4) },
            Block { code: vec![], end: BlockEnd::Exit },
        ])
    }

    #[test]
    fn test_branch_reverse() {
        let mut code = Code::new();
        // ... first block ...
        let branch = code.branch_reverse();
        // ... second block ...
        code.join_reverse(branch, BranchType::Jump);
        // ... third block ...

        assert_eq!(code.blocks, vec![
            Block { code: vec![], end: BlockEnd::Next(1) },
            Block { code: vec![], end: BlockEnd::Next(1) },
            Block { code: vec![], end: BlockEnd::Exit }
        ])
    }

    #[test]
    fn test_branch_forward_and_reverse() {
        let mut code = Code::new();
        // ... first block ...
        let branch1 = code.branch_reverse();
        // ... start of loop ...
        let branch2 = code.branch_forward();
         // ... body of loop if true ...
        code.join_reverse(branch1, BranchType::Jump);
        // ... empty block ...
        code.join_forward(branch2, BranchType::JumpIfFalse);
        // ... end of loop ...

        assert_eq!(code.blocks, vec![
            Block { code: vec![], end: BlockEnd::Next(1) },
            Block { code: vec![], end: BlockEnd::Branch { ty: BranchType::JumpIfFalse, default: 2, branch: 4 } },
            Block { code: vec![], end: BlockEnd::Next(1) },
            Block { code: vec![], end: BlockEnd::Next(4) },
            Block { code: vec![], end: BlockEnd::Exit },
        ])
    }
}