use std::collections::HashMap;
use fxhash::FxBuildHasher;

use crate::compiler::parser::Parser;
use crate::compiler::parser::semantic::ParserFunctionImpl;
use crate::Location;
use crate::vm::Opcode;
use crate::vm::operator::CompareOp;


/// A series of code blocks representing the control flow graph.
/// - Each `Block` is uniquely identifiable by its index within `blocks`, meaning unique and persistent IDs may be taken
/// - Blocks can form references to other blocks via these IDs in the form of `BlockEnd`
/// - A unique identifier for a single opcode location can take the form of a block ID plus a opcode ID
#[derive(Debug)]
pub struct Code {
    pub(super) blocks: Vec<Block>,
    locals: HashMap<CodeId, String, FxBuildHasher>,
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
    pub fn do_insert(at: OpcodeId, loc: Location, opcode: Opcode, functions: &mut [ParserFunctionImpl]) {
        functions[at.function_id].code.blocks[at.code_id.block_id].insert(at.code_id.opcode_id, loc, opcode);
    }
}


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
            use Opcode::{*};
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


    pub(super) fn push(&mut self, loc: Location, opcode: Opcode, local: Option<String>) {
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


impl BlockEnd {
    fn branch(ty: BranchType, default: usize, branch: usize ) -> BlockEnd {
        match ty {
            BranchType::Jump => BlockEnd::Next(branch),
            ty => BlockEnd::Branch { ty, default, branch }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::compiler::parser::core::graph::{Block, BlockEnd, BranchType, Code};

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