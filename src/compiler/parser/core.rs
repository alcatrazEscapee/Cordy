/// Core Parser Implementation
///
/// This implementation manages token advancing, error handling, and issues related to newline handling.

use crate::compiler::parser::{Parser, ParserError};
use crate::compiler::parser::ParserErrorType;
use crate::compiler::scanner::ScanToken;
use crate::compiler::parser::semantic::{LValueReference, Reference};
use crate::reporting::Location;
use crate::trace;
use crate::vm::{Opcode, RuntimeError};

use ParserErrorType::{*};
use ScanToken::{*};
use Opcode::{*};

/// A state to restore to while backtracking.
/// Only stores enough state to be necessary, as we don't need to backtrack through output tokens.
pub struct ParserState {
    input: Vec<(Location, ScanToken)>,

    output_len: usize, // Validity check
    error_len: usize,

    prevent_expression_statement: bool,
}

impl ParserState {
    fn store(parser: &Parser) -> ParserState {
        ParserState {
            input: Vec::new(),
            output_len: parser.output.len(),
            error_len: parser.errors.len(),
            prevent_expression_statement: parser.prevent_expression_statement,
        }
    }

    fn restore(self: Self, parser: &mut Parser) {
        assert_eq!(self.output_len, parser.output.len(), "Output modified while backtracking!");
        trace::trace_parser!("restoring input {:?} -> [{:?}, ...]", self.input, parser.input.front());

        for token in self.input.into_iter().rev() {
            parser.input.push_front(token);
        }

        parser.errors.truncate(self.error_len);
        parser.error_recovery = false;
        parser.prevent_expression_statement = self.prevent_expression_statement;
    }
}


impl<'a> Parser<'a> {

    pub fn begin(self: &mut Self) -> bool {
        trace::trace_parser!("begin backtracking");
        assert!(self.restore_state.is_none(), "Recursive backtracking attempt");

        self.restore_state = Some(ParserState::store(self));
        !self.error_recovery
    }

    pub fn accept(self: &mut Self) {
        trace::trace_parser!("accept backtracking");
        self.restore_state = None;
    }

    pub fn reject(self: &mut Self) {
        trace::trace_parser!("reject backtracking");
        match (&mut self.restore_state).take() {
            Some(state) => state.restore(self),
            _ => panic!("reject() without begin()"),
        }
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
    pub(super) fn parse_optional_trailing_comma<F : FnOnce(Option<ScanToken>) -> ParserErrorType>(self: &mut Self, close_token: ScanToken, error: F) -> bool {
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
    pub fn expect(self: &mut Self, token: ScanToken) {
        match self.peek() {
            Some(t) if t == &token => {
                trace::trace_parser!("expect {:?} -> pass", token);
                self.advance();
            },
            _ => self.error_with(move |t| ExpectedToken(token, t)),
        }
    }

    /// Acts as a resynchronization point for error mode
    /// Accepts tokens from the input (ignoring the current state of error recovery mode), until we reach the expected token or an empty input.
    /// If we reach the expected token, it is consumed and error mode is unset.
    pub fn expect_resync(self: &mut Self, token: ScanToken) {
        if let Some(t) = self.peek() { // First, check for expect() without raising an error
            if *t == token {
                trace::trace_parser!("expect_resync {:?} -> pass", token);
                self.advance();
                return;
            }
        }
        loop {
            // Then if we fail, start resync. Initially set error recovery `false`, so we can peek ahead at the input.
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
                    self.error(ExpectedToken(token, None));
                    break
                }
            }
        }
    }

    /// Like `advance()`, but returns the boxed `Identifier` token.
    /// **Important**: Must only be called once `peek()` has identified an `Identifier` token is present, as this will panic otherwise.
    pub fn take_identifier(self: &mut Self) -> String {
        match self.advance() {
            Some(Identifier(name)) => name,
            t => panic!("Token mismatch in advance_identifier() -> expected an Some(Identifier(String)), got a {:?} instead", t)
        }
    }

    /// Like `advance()`, but returns the boxed `Int` token.
    /// **Important**: Must only be called once `peek()` has identified an `Int` token is present, as this will panic otherwise.
    pub fn take_int(self: &mut Self) -> i64 {
        match self.advance() {
            Some(ScanToken::IntLiteral(i)) => i,
            t => panic!("Token mismatch in advance_int() -> expected an Some(Int(i64)), got a {:?} instead", t)
        }
    }

    /// Like `advance()`, but returns the boxed `String` literal token.
    /// **Important**: Must only be called once `peek()` has identified a `StringLiteral` token is present, as this will panic otherwise.
    pub fn take_str(self: &mut Self) -> String {
        match self.advance() {
            Some(StringLiteral(s)) => s,
            t => panic!("Token mismatch in advance_str() -> expected a Some(StringLiteral(String)), got a {:?} instead", t)
        }
    }

    /// Specialization of `peek()`, does not consume newlines, for checking if a specific syntax token is on the same line as the preceding one.
    pub fn peek_no_newline(self: &Self) -> Option<&ScanToken> {
        if self.error_recovery {
            return None
        }
        for (_, token) in &self.input {
            return Some(token)
        }
        None
    }

    /// Peeks at the next incoming token.
    /// Note that this function only returns a read-only reference to the underlying token, suitable for matching
    /// If the token data needs to be unboxed, i.e. as with `Identifier` tokens, it must be extracted only via `advance()`
    /// This also does not consume newline tokens in the input, rather peeks _past_ them in order to find the next matching token.
    pub fn peek(self: &Self) -> Option<&ScanToken> { self.peek_lookahead(0) }
    pub fn peek2(self: &Self) -> Option<&ScanToken> { self.peek_lookahead(1) }
    pub fn peek3(self: &Self) -> Option<&ScanToken> { self.peek_lookahead(2) }

    fn peek_lookahead(self: &Self, mut lookahead: u8) -> Option<&ScanToken> {
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
    pub fn skip(self: &mut Self) {
        self.advance_both();
    }

    /// Like `advance()` but returns the location of the advanced-by token.
    pub fn advance_with(self: &mut Self) -> Location {
        self.advance_both().map(|(loc, _)| loc).unwrap_or_else(Location::empty)
    }

    pub fn advance(self: &mut Self) -> Option<ScanToken> {
        self.advance_both().map(|(_, t)| t)
    }

    /// Advances and returns the next incoming token.
    /// Will also advance past any newline tokens, and so the advanced token will be the next token _after_ any newlines between the last token and the next.
    pub fn advance_both(self: &mut Self) -> Option<(Location, ScanToken)> {
        if self.error_recovery {
            return None
        }
        while let Some((_, NewLine)) = self.input.front() {
            let token = self.input.pop_front().unwrap();
            if let Some(state) = &mut self.restore_state {
                state.input.push(token);
            }
            self.prevent_expression_statement = false;
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

    /// Reserves a space in the output code by inserting a `Noop` token.
    /// Returns an index to the token, which can later be used to set the correct value.
    ///
    /// N.B. This cannot reserve across functions - the reserved token must be set from within the current function's parse.
    pub fn reserve(self: &mut Self) -> usize {
        trace::trace_parser!("reserve at {}", self.current_function().len());
        let loc = self.prev_location();
        self.current_function_mut().push((loc, Noop));
        self.current_function().len() - 1
    }

    /// Returns an index to the next opcode to be emitted, to be used to emit a jump to this location.
    pub fn next_opcode(self: &Self) -> usize {
        self.current_function().len()
    }

    /// Given a `usize` index, pushes a jump instruction that jumps *back* to the target index.
    pub fn push_jump<F: FnOnce(i32) -> Opcode>(self: &mut Self, origin: usize, jump: F) {
        let jump_opcode: Opcode = jump(origin as i32 - self.next_opcode() as i32 - 1);
        trace::trace_parser!("push jump at {} -> {:?}", self.next_opcode(), jump_opcode);
        self.push(jump_opcode);
    }

    /// Given a `usize` index, which is obtained from `self.reserve()`, this fixes the jump instruction at that location to point to the next opcode.
    pub fn fix_jump<F : FnOnce(i32) -> Opcode>(self: &mut Self, reserved: usize, jump: F) {
        let jump_opcode: Opcode = jump(self.next_opcode() as i32 - reserved as i32 - 1);
        trace::trace_parser!("fixing jump at {} -> {:?}", reserved, jump_opcode);
        self.current_function_mut()[reserved].1 = jump_opcode;
    }

    /// If we previously delayed a `Pop` opcode from being omitted, push it now and reset the flag
    pub fn push_delayed_pop(self: &mut Self) {
        if self.delay_pop_from_expression_statement {
            trace::trace_parser!("push Pop (delayed)");
            self.push(Pop);
            self.delay_pop_from_expression_statement = false;
        }
    }

    /// Specialization of `push` which may push nothing, Pop, or PopN(n)
    pub fn push_pop(self: &mut Self, n: u32) {
        trace::trace_parser!("push Pop/PopN {}", n);
        match n {
            0 => {},
            1 => self.push(Pop),
            n => self.push(PopN(n))
        }
    }

    pub fn push_load_lvalue(self: &mut Self, loc: Location, lvalue: LValueReference) {
        match lvalue {
            LValueReference::Local(index) => self.push_with(PushLocal(index), loc),
            LValueReference::Global(index) => self.push_with(PushGlobal(index), loc),
            LValueReference::LateBoundGlobal(global) => {
                self.late_bound_globals.push(Reference::Load(global.update_opcode(self.functions.len() - 1, self.next_opcode() as usize)));
                self.push_with(Noop, loc); // Will be fixed when the global is declared, or caught at EoF as an error
            }
            LValueReference::UpValue(index) => self.push_with(PushUpValue(index), loc),
            LValueReference::Invalid => {},
            LValueReference::NativeFunction(native) => self.push_with(NativeFunction(native), loc),
            _ => panic!("Invalid load: {:?}", lvalue),
        }
    }

    pub fn push_store_lvalue(self: &mut Self, lvalue: LValueReference) {
        match lvalue {
            LValueReference::Local(index) => self.push(StoreLocal(index)),
            LValueReference::Global(index) => self.push(StoreGlobal(index)),
            LValueReference::LateBoundGlobal(global) => {
                self.late_bound_globals.push(Reference::Store(global.update_opcode(self.functions.len() - 1, self.next_opcode() as usize)));
                self.push(Noop); // Will be fixed when the global is declared, or caught at EoF as an error
            },
            LValueReference::UpValue(index) => self.push(StoreUpValue(index)),
            LValueReference::Invalid => {},
            _ => panic!("Invalid store: {:?}", lvalue),
        }
    }

    /// Pushes a new token into the output stream.
    /// Returns the index of the token pushed, which allows callers to later mutate that token if they need to.
    pub fn push(self: &mut Self, token: Opcode) {
        self.push_with(token, self.prev_location());
    }

    pub fn push_with(self: &mut Self, token: Opcode, location: Location) {
        trace::trace_parser!("push {:?}", token);
        match match &token {
            PushGlobal(id) | StoreGlobal(id) => Some((0, id)),
            PushLocal(id) | StoreLocal(id) => Some((self.function_depth as usize, id)),
            _ => None,
        } {
            Some((depth, id)) => {
                let local = self.locals[depth].get_name(*id as usize);
                self.current_locals_reference_mut().push(local);
            },
            _ => {}
        };
        self.current_function_mut().push((location, token));
    }

    /// A specialization of `error()` which provides the last token (the result of `peek()`) to the provided error function
    /// This avoids ugly borrow checker issues where `match self.peek() { ... t => self.error(Error(t)) }` does not work, despite the semantics being identical.
    pub fn error_with<F : FnOnce(Option<ScanToken>) -> ParserErrorType>(self: &mut Self, error: F) {
        let token = self.peek().cloned();
        self.error(error(token));
    }

    /// Pushes a new error token into the output error stream.
    pub fn error(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError::new(error, self.next_location()));
        }
        self.error_recovery = true;
    }

    /// Pushes a new error token into the output error stream, but does not initiate error recovery.
    /// This is useful for semantic errors which are valid lexically, but still need to report errors.
    pub fn semantic_error(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError::new(error, self.prev_location()));
        }
    }

    /// Pushes a new error token into the output error stream, based on the provided runtime error (produced by constant expressions at compile time)
    /// Does not initiate error recovery.
    pub fn runtime_error(self: &mut Self, loc: Location, error: Box<RuntimeError>) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError::new(Runtime(error), loc));
        }
    }

    /// Creates an optional error, which will be deferred until later to be emitted
    pub fn deferred_error(self: &Self, error: ParserErrorType) -> Option<ParserError> {
        if self.error_recovery {
            None
        } else {
            Some(ParserError::new(error, self.prev_location()))
        }
    }

    /// Returns the source location of the previous token, aka the one just accepted.
    pub fn prev_location(self: &Self) -> Location {
        self.last_location.unwrap_or_else(Location::empty)
    }

    /// Returns the source location of the next token, aka the one in `peek()`
    pub fn next_location(self: &Self) -> Location {
        self.input.front().map(|u| u.0).unwrap_or_else(Location::empty)
    }
}