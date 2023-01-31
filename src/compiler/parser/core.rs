/// Core Parser Implementation
///
/// This implementation manages token advancing, error handling, and issues related to newline handling.

use crate::compiler::parser::{Parser, ParserError};
use crate::compiler::parser::ParserErrorType;
use crate::compiler::scanner::ScanToken;
use crate::trace;
use crate::vm::opcode::Opcode;

use ParserErrorType::{*};
use ScanToken::{*};
use Opcode::{*};


impl<'a> Parser<'a> {

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

    /// Advances the token stream and pushes the provided token to the output stream.
    pub fn advance_push(self: &mut Self, token: Opcode) {
        self.advance();
        self.push(token);
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
            Some(ScanToken::Int(i)) => i,
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
    pub fn peek_no_newline(self: &mut Self) -> Option<&ScanToken> {
        if self.error_recovery {
            return None
        }
        for token in &self.input {
            return Some(token)
        }
        None
    }

    /// Peeks at the next incoming token.
    /// Note that this function only returns a read-only reference to the underlying token, suitable for matching
    /// If the token data needs to be unboxed, i.e. as with `Identifier` tokens, it must be extracted only via `advance()`
    /// This also does not consume newline tokens in the input, rather peeks _past_ them in order to find the next matching token.
    pub fn peek(self: &mut Self) -> Option<&ScanToken> {
        if self.error_recovery {
            return None
        }
        for token in &self.input {
            if token != &NewLine {
                return Some(token)
            } else {
                self.prevent_expression_statement = false;
            }
        }
        None
    }

    // Like `peek()` but peeks one ahead (making this technically a lookahead 2 parser)
    pub fn peek2(self: &mut Self) -> Option<&ScanToken> {
        if self.error_recovery {
            return None
        }
        let mut first: bool = false;
        for token in &self.input {
            if token != &NewLine {
                if !first {
                    first = true;
                } else {
                    return Some(token)
                }
            } else {
                self.prevent_expression_statement = false;
            }
        }
        None
    }

    /// Like `advance()` but discards the result.
    pub fn skip(self: &mut Self) {
        self.advance();
    }

    /// Advances and returns the next incoming token.
    /// Will also advance past any newline tokens, and so the advanced token will be the next token _after_ any newlines between the last token and the next.
    pub fn advance(self: &mut Self) -> Option<ScanToken> {
        if self.error_recovery {
            return None
        }
        while let Some(NewLine) = self.input.front() {
            trace::trace_parser!("newline {} at opcode {}, last = {:?}", self.lineno + 1, self.next_opcode(), self.line_numbers.last());
            self.input.pop_front();
            self.lineno += 1;
            self.prevent_expression_statement = false;
        }
        trace::trace_parser!("advance {:?}", self.input.front());
        self.input.pop_front()
    }

    /// Reserves a space in the output code by inserting a `Noop` token
    /// Returns an index to the token, which can later be used to set the correct value
    pub fn reserve(self: &mut Self) -> usize {
        trace::trace_parser!("reserve at {}", self.output.len());
        self.output.push(Noop);
        self.line_numbers.push(self.lineno);
        self.output.len() - 1
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

    /// Specialization of `push` which pushes either `StoreLocal` or `StoreGlobal` for a given local
    pub fn push_store_local(self: &mut Self, index: usize) {
        trace::trace_parser!("push StoreLocal/StoreGlobal {}", index);

        let local = &self.current_locals_mut().locals[index];
        if local.is_true_global() {
            self.push(StoreGlobal(index as u32, false))
        } else if local.is_global() {
            self.push(StoreGlobal(index as u32, true))
        } else {
            self.push(StoreLocal(index as u32))
        }
    }

    /// Specialization of `push` which pushes a `IncGlobalCount` if we need to, for the given local
    /// Called immediately after a local `initialized` is set to `true
    pub fn push_inc_global(self: &mut Self, local: usize) {
        if self.current_locals_mut().locals[local].is_true_global() {
            self.push(IncGlobalCount);
        }
    }

    /// Pushes a new token into the output stream.
    /// Returns the index of the token pushed, which allows callers to later mutate that token if they need to.
    pub fn push(self: &mut Self, token: Opcode) {
        trace::trace_parser!("push {:?} at L{:?}", token, self.lineno + 1);
        match &token {
            PushGlobal(id, _) | StoreGlobal(id, _) => self.locals_reference.push(self.locals[0].locals[*id as usize].name.clone()),
            PushLocal(id) | StoreLocal(id) => self.locals_reference.push(self.locals[self.function_depth as usize].locals[*id as usize].name.clone()),
            _ => {},
        }
        self.output.push(token);
        self.line_numbers.push(self.lineno);
    }

    /// Pops the last emitted token
    pub fn pop(self: &mut Self) {
        match self.output.pop().unwrap() {
            PushGlobal(_, _) | StoreGlobal(_, _) | PushLocal(_) | StoreLocal(_) => {
                self.locals_reference.pop();
            },
            _ => {},
        };
    }

    /// Returns the index of the last token that was just pushed.
    pub fn last(self: &Self) -> Option<Opcode> {
        self.output.last().copied()
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
            self.errors.push(ParserError::new(error, self.lineno as usize));
        }
        self.error_recovery = true;
    }

    /// Pushes a new error token into the output error stream, but does not initiate error recovery.
    /// This is useful for semantic errors which are valid lexically, but still need to report errors.
    pub fn semantic_error(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError::new(error, self.lineno as usize));
        }
    }

    /// Creates an optional error, which will be deferred until later to be emitted
    pub fn deferred_error(self: &Self, error: ParserErrorType) -> Option<ParserError> {
        if self.error_recovery {
            None
        } else {
            Some(ParserError::new(error, self.lineno as usize))
        }
    }
}