/// Core Parser Implementation
///
/// This implementation manages token advancing, error handling, and issues related to newline handling.
/// It also handles the core structures representing the code flow graph (`Code`, `Block`, and branch handling).

use fxhash::FxBuildHasher;
use indexmap::IndexSet;

use crate::compiler::parser::{Parser, ParserError};
use crate::compiler::parser::ParserErrorType;
use crate::compiler::parser::semantic::{LValueReference, ReferenceType};
use crate::compiler::scanner::ScanToken;
use crate::reporting::Location;
use crate::trace;
use crate::vm::Opcode;

pub use crate::compiler::parser::core::graph::{Code, Block, BranchType, ForwardBlockId, ReverseBlockId, OpcodeId};

use Opcode::{*};
use ParserErrorType::{*};
use ScanToken::{*};


mod graph;
mod optimizer;


impl<'a> Parser<'a> {

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
    pub fn advance_both(&mut self) -> Option<(Location, ScanToken)> {
        if self.error_recovery {
            return None
        }
        while let Some((_, NewLine)) = self.input.front() {
            self.input.pop_front().unwrap();
        }
        trace::trace_parser!("advance {:?}", self.input.front());
        let ret = self.input.pop_front();
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