use std::iter::Peekable;
use std::vec::IntoIter;

use crate::compiler::scanner::{ScanResult, ScanToken};
use crate::stdlib::{StdBinding, StdBindingTree};
use crate::vm::Opcode;
use crate::trace;

use crate::compiler::scanner::ScanToken::{*};
use crate::compiler::parser::ParserErrorType::{*};
use crate::vm::Opcode::{*};


pub fn parse(bindings: StdBindingTree, scan_result: ScanResult) -> ParserResult {
    let mut parser: Parser = Parser::new(bindings, scan_result.tokens);
    parser.parse();
    ParserResult {
        code: parser.output,
        errors: parser.errors,
    }
}

pub struct ParserResult {
    pub code: Vec<Opcode>,
    pub errors: Vec<ParserError>,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ParserError {
    pub error: ParserErrorType,
    pub lineno: usize,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ParserErrorType {
    UnexpectedEoF,
    UnexpectedEoFExpecting(ScanToken),
    UnexpectedTokenAfterEoF(ScanToken),

    Expecting(ScanToken, ScanToken),

    ExpectedExpressionTerminal(ScanToken),
    ExpectedCommaOrEndOfArguments(ScanToken),
    ExpectedStatement(ScanToken),
}


struct Parser {
    bindings: StdBindingTree,
    input: Peekable<IntoIter<ScanToken>>,
    output: Vec<Opcode>,
    errors: Vec<ParserError>,

    lineno: usize,

    error: bool, // If we are in error recover mode, this flag is set
    lookahead: Option<ScanToken> // A token we pretend 'push back onto' the input iterator
}


impl Parser {

    fn new(bindings: StdBindingTree, tokens: Vec<ScanToken>) -> Parser {
        Parser {
            bindings,
            input: tokens.into_iter().peekable(),
            output: Vec::new(),
            errors: Vec::new(),

            lineno: 0,

            error: false,
            lookahead: None,
        }
    }

    fn parse(self: &mut Self) {
        trace::trace_parser!("rule <root>");
        self.parse_statements();
        if let Some(t) = self.peek() {
            let token: ScanToken = t.clone();
            self.push_err(UnexpectedTokenAfterEoF(token));
        }
        self.push(Exit);
    }

    fn parse_statements(self: &mut Self) {
        trace::trace_parser!("rule <statements>");
        loop {
            match self.peek() {
                Some(KeywordFn) => self.parse_function(),
                Some(KeywordLet) => self.parse_let(),
                Some(KeywordIf) => self.parse_if(),
                Some(KeywordLoop) => self.parse_loop(),
                Some(KeywordFor) => self.parse_for(),
                Some(ScanToken::Identifier(_)) => self.parse_assignment(),
                Some(OpenBrace) => self.parse_block_statement(),
                Some(CloseBrace) => break, // Don't consume, but break if we're in an error mode
                Some(KeywordExit) => {
                    self.advance();
                    self.push(Exit);
                }
                Some(t) => {
                    let token: ScanToken = t.clone();
                    self.push_err(ExpectedStatement(token));
                    self.advance();
                }
                None => break,
            }
        }
    }

    fn parse_block_statement(self: &mut Self) {
        trace::trace_parser!("rule <block-statement>");
        self.expect(OpenBrace);
        self.parse_statements();
        self.expect_resync(CloseBrace);
    }

    fn parse_function(self: &mut Self) {
        // todo
        //self.expect_identifier();
        if self.accept(OpenParen) {
            // todo:
            //self.parse_function_parameters();
        }
        self.parse_block_statement();
    }

    fn parse_let(self: &mut Self) {
        if self.accept(Equals) {
            self.parse_expression();
        }
    }

    fn parse_if(self: &mut Self) {
        self.parse_expression();
        self.parse_block_statement();
        while self.accept(KeywordElif) {
            self.parse_block_statement()
        }
        if self.accept(KeywordElse) {
            self.parse_block_statement()
        }
    }

    fn parse_loop(self: &mut Self) {
        self.parse_block_statement()
    }

    fn parse_for(self: &mut Self) {
        self.expect(KeywordIn);
        self.parse_expression();
        self.parse_block_statement();
    }

    fn parse_assignment(self: &mut Self) {
        let name: String = self.take_identifier();
        if let Some(Equals) = self.peek() { // Assignment Statement
            self.push(Opcode::Identifier(name));
            self.advance();
            self.parse_expression();
            self.push(StoreValue);
            return
        }
        let maybe_op: Option<Opcode> = match self.peek() {
            Some(PlusEquals) => Some(OpAdd),
            Some(MinusEquals) => Some(OpSub),
            Some(MulEquals) => Some(OpMul),
            Some(DivEquals) => Some(OpDiv),
            Some(AndEquals) => Some(OpBitwiseAnd),
            Some(OrEquals) => Some(OpBitwiseOr),
            Some(XorEquals) => Some(OpBitwiseXor),
            Some(LeftShiftEquals) => Some(OpLeftShift),
            Some(RightShiftEquals) => Some(OpRightShift),
            Some(ModEquals) => Some(OpMod),
            Some(PowEquals) => Some(OpPow),
            _ => None
        };
        if let Some(op) = maybe_op { // Assignment Expression i.e. a += 3
            self.push(Opcode::Identifier(name));
            self.push(Dupe);
            self.advance();
            self.parse_expression();
            self.push(op);
            self.push(StoreValue);
            return;
        }
        // In this case, we matched (and consumed) <name> without a following '=' or other assignment operator.
        // This must be part of an expression, but we need to first push the lookahead identifier.
        // A bare expression that starts with an identifier token also must end with a `Pop` opcode.
        self.push_lookahead(ScanToken::Identifier(name));
        self.parse_expression();
        self.push(Pop);
    }

    fn parse_expression(self: &mut Self) {
        trace::trace_parser!("rule <expression>");
        self.parse_expr_9();
    }


    // ===== Expression Parsing ===== //

    fn parse_expr_1_terminal(self: &mut Self) {
        trace::trace_parser!("rule <expr-1>");
        match self.peek() {
            Some(KeywordNil) => self.advance_push(Nil),
            Some(KeywordTrue) => self.advance_push(True),
            Some(KeywordFalse) => self.advance_push(False),
            Some(ScanToken::Int(_)) => {
                let int: i64 = self.take_int();
                self.push(Opcode::Int(int));
            },
            Some(ScanToken::Identifier(_)) => {
                let string: String = self.take_identifier();
                if let Some(binding) = self.locate_binding(&string) {
                    self.push(Bound(binding));
                } else {
                    self.push(Opcode::Identifier(string));
                }
            },
            Some(StringLiteral(_)) => {
                let string = self.take_str();
                self.push(Str(string));
            },
            Some(OpenParen) => {
                self.advance();
                self.parse_expr_9();
                self.expect(CloseParen);
            }
            Some(e) => {
                let token: ScanToken = e.clone();
                self.push(Nil);
                self.push_err(ExpectedExpressionTerminal(token));
            },
            _ => {
                self.push(Nil);
                self.push_err(UnexpectedEoF)
            }
        }
    }

    fn parse_expr_2_unary(self: &mut Self) {
        trace::trace_parser!("rule <expr-2>");
        // Prefix operators
        let mut stack: Vec<Opcode> = Vec::new();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(Minus) => Some(UnarySub),
                Some(BitwiseNot) => Some(UnaryBitwiseNot),
                Some(LogicalNot) => Some(UnaryLogicalNot),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    stack.push(op);
                },
                None => break
            }
        }

        self.parse_expr_1_terminal();

        // Suffix operators
        loop {
            match self.peek() {
                Some(OpenParen) => {
                    let mut count: u8 = 0;
                    self.advance();
                    match self.peek() {
                        Some(CloseParen) => { self.advance(); },
                        Some(_) => {
                            // First argument
                            self.parse_expression();
                            count += 1;
                            // Other arguments
                            while let Some(Comma) = self.peek() {
                                self.advance();
                                self.parse_expression();
                                count += 1;
                            }
                            match self.peek() {
                                Some(CloseParen) => self.skip(),
                                Some(c) => {
                                    let token: ScanToken = c.clone();
                                    self.push_err(ExpectedCommaOrEndOfArguments(token))
                                },
                                _ => self.push_err(UnexpectedEoF),
                            }
                        }
                        None => self.push_err(UnexpectedEoF),
                    }
                    self.push(OpFuncEval(count));
                },
                Some(OpenSquareBracket) => panic!("Unimplemented array or slice syntax"),
                _ => break
            }
        }

        // Prefix operators are lower precedence than suffix operators
        for op in stack.into_iter().rev() {
            self.push(op);
        }
    }

    fn parse_expr_3(self: &mut Self) {
        trace::trace_parser!("rule <expr-3>");
        self.parse_expr_2_unary();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(Mul) => Some(OpMul),
                Some(Div) => Some(OpDiv),
                Some(Mod) => Some(OpMod),
                Some(Pow) => Some(OpPow),
                Some(KeywordIs) => Some(OpIs),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    self.parse_expr_2_unary();
                    self.push(op);
                },
                None => break
            }
        }
    }

    fn parse_expr_4(self: &mut Self) {
        trace::trace_parser!("rule <expr-4>");
        self.parse_expr_3();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(Plus) => Some(OpAdd),
                Some(Minus) => Some(OpSub),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    self.parse_expr_3();
                    self.push(op);
                },
                None => break
            }
        }
    }

    fn parse_expr_5(self: &mut Self) {
        trace::trace_parser!("rule <expr-5>");
        self.parse_expr_4();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(LeftShift) => Some(OpLeftShift),
                Some(RightShift) => Some(OpRightShift),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    self.parse_expr_4();
                    self.push(op);
                },
                None => break
            }
        }
    }

    fn parse_expr_6(self: &mut Self) {
        trace::trace_parser!("rule <expr-6>");
        self.parse_expr_5();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(LessThan) => Some(OpLessThan),
                Some(LessThanEquals) => Some(OpLessThanEqual),
                Some(GreaterThan) => Some(OpGreaterThan),
                Some(GreaterThanEquals) => Some(OpGreaterThanEqual),
                Some(DoubleEquals) => Some(OpEqual),
                Some(NotEquals) => Some(OpNotEqual),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    self.parse_expr_5();
                    self.push(op);
                },
                None => break
            }
        }
    }

    fn parse_expr_7(self: &mut Self) {
        trace::trace_parser!("rule <expr-7>");
        self.parse_expr_6();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(BitwiseAnd) => Some(OpBitwiseAnd),
                Some(BitwiseOr) => Some(OpBitwiseOr),
                Some(BitwiseXor) => Some(OpBitwiseXor),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    self.parse_expr_6();
                    self.push(op);
                },
                None => break
            }
        }
    }

    fn parse_expr_8(self: &mut Self) {
        trace::trace_parser!("rule <expr-8>");
        self.parse_expr_7();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(Dot) => Some(OpFuncCompose),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    self.parse_expr_7();
                    self.push(op);
                },
                None => break
            }
        }
    }

    fn parse_expr_9(self: &mut Self) {
        trace::trace_parser!("rule <expr-9>");
        self.parse_expr_8();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(LogicalAnd) => Some(OpLogicalAnd),
                Some(LogicalOr) => Some(OpLogicalOr),
                _ => None
            };
            match maybe_op {
                Some(op) => {
                    self.advance();
                    self.parse_expr_8();
                    self.push(op);
                },
                None => break
            }
        }
    }


    // ===== Semantic Analysis ===== //


    fn locate_binding(self: &mut Self, name: &String) -> Option<StdBinding> {
        self.bindings.children.as_ref().unwrap().get(name.as_str()).map(|b| b.binding.unwrap())
    }


    // ===== Parser Core ===== //


    /// If the given token is present, accept it and return `true`. Otherwise, do nothing and return `false`.
    fn accept(self: &mut Self, token: ScanToken) -> bool {
        match self.peek() {
            Some(t) if *t == token => {
                self.advance();
                true
            }
            _ => false
        }
    }


    /// If the given token is present, accept it. Otherwise, flag an error and enter error recovery mode.
    fn expect(self: &mut Self, token: ScanToken) {
        self.skip_new_line();
        match self.peek() {
            Some(t) if t == &token => {
                trace::trace_parser!("expect {:?} -> pass", token);
                self.raw_next();
            }
            Some(t) => {
                let t0: ScanToken = t.clone();
                self.push_err(Expecting(token, t0))
            },
            None => self.push_err(UnexpectedEoFExpecting(token)),
        }
    }

    /// Acts as a resynchronization point for error mode
    /// Accepts tokens from the input (ignoring the current state of error recovery mode), until we reach the expected token or an empty input.
    /// If we reach the expected token, it is consumed and error mode is unset.
    fn expect_resync(self: &mut Self, token: ScanToken) {
        self.skip_new_line();

        if let Some(t) = self.peek() { // First, check for expect() without raising an error
            if *t == token {
                trace::trace_parser!("expect_resync {:?} -> pass", token);
                self.advance();
                return;
            }
        }
        loop { // Then if we fail, start resync
            self.skip_new_line();
            match self.raw_peek() {
                Some(t) if *t == token => {
                    trace::trace_parser!("expect_resync {:?} -> synced", token);
                    self.error = false;
                    self.raw_next();
                    break;
                },
                Some(_t) => {
                    trace::trace_parser!("expect_resync {:?} -> discarding {:?}", token, _t);
                    self.raw_next();
                },
                None => break
            }
        }
    }

    /// Advances the token stream and pushes the provided token to the output stream.
    fn advance_push(self: &mut Self, token: Opcode) {
        self.advance();
        self.push(token);
    }


    /// Like `advance()`, but returns the boxed `Identifier` token.
    /// **Important**: Must only be called once `peek()` has identified an `Identifier` token is present, as this will panic otherwise.
    fn take_identifier(self: &mut Self) -> String {
        match self.advance() {
            Some(ScanToken::Identifier(name)) => name,
            t => panic!("Token mismatch in advance_identifier() -> expected an Some(Identifier(String)), got a {:?} instead", t)
        }
    }

    /// Like `advance()`, but returns the boxed `Int` token.
    /// **Important**: Must only be called once `peek()` has identified an `Int` token is present, as this will panic otherwise.
    fn take_int(self: &mut Self) -> i64 {
        match self.advance() {
            Some(ScanToken::Int(i)) => i,
            t => panic!("Token mismatch in advance_int() -> expected an Some(Int(i64)), got a {:?} instead", t)
        }
    }

    /// Like `advance()`, but returns the boxed `String` literal token.
    /// **Important**: Must only be called once `peek()` has identified a `StringLiteral` token is present, as this will panic otherwise.
    fn take_str(self: &mut Self) -> String {
        match self.advance() {
            Some(StringLiteral(s)) => s,
            t => panic!("Token mismatch in advance_str() -> expected a Some(StringLiteral(String)), got a {:?} instead", t)
        }
    }

    /// Peeks at the next incoming token.
    /// Note that this function only returns a read-only reference to the underlying token, suitable for matching
    /// If the token data needs to be unboxed, i.e. as with `Identifier` tokens, it must be extracted only via `advance()`
    fn peek(self: &mut Self) -> Option<&ScanToken> {
        if self.error {
            return None
        }
        self.skip_new_line();
        self.raw_peek()
    }

    /// Like `advance()` but discards the result.
    fn skip(self: &mut Self) {
        self.advance();
    }

    /// Advances and returns the next incoming token.
    fn advance(self: &mut Self) -> Option<ScanToken> {
        if self.error {
            return None
        }
        self.skip_new_line();
        trace::trace_parser!("advance {:?}", self.input.peek());
        self.raw_next()
    }

    fn skip_new_line(self: &mut Self) {
        while let Some(NewLine) = self.raw_peek() {
            trace::trace_parser!("advance NewLine L{}", self.lineno);
            self.raw_next();
            self.lineno += 1;
            self.output.push(LineNumber(self.lineno));
        }
    }

    /// Pushes a new token into the output stream.
    fn push(self: &mut Self, token: Opcode) {
        trace::trace_parser!("push {:?}", token);
        self.output.push(token);
    }

    /// Pushes a new error token into the output error stream.
    fn push_err(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error, error);
        if !self.error {
            self.errors.push(ParserError {
                error,
                lineno: self.lineno,
            });
        }
        self.error = true;
    }

    /// Inserts a scan token back into the input stream.
    /// This is required as we technically have a lookahead 2 parser.
    fn push_lookahead(self: &mut Self, token: ScanToken) {
        assert!(self.lookahead.is_none());
        self.lookahead = Some(token);
    }

    // ===== Interactions with `input` ===== //

    fn raw_next(self: &mut Self) -> Option<ScanToken> {
        if self.lookahead.is_some() {
            trace::trace_parser!("take lookahead");
            return self.lookahead.take();
        }
        self.input.next()
    }

    fn raw_peek(self: &mut Self) -> Option<&ScanToken> {
        if self.lookahead.is_some() {
            return self.lookahead.as_ref();
        }
        self.input.peek()
    }
}


#[cfg(test)]
mod tests {
    use crate::compiler::{parser, scanner, test_common};
    use crate::compiler::scanner::ScanResult;
    use crate::compiler::parser::{Parser, ParserResult};
    use crate::{reporting, stdlib};
    use crate::vm::Opcode;

    use crate::vm::Opcode::{*};

    #[test] fn test_str_empty() { run_expr("", vec![Nil]); }
    #[test] fn test_str_int() { run_expr("123", vec![Int(123)]); }
    #[test] fn test_str_str() { run_expr("'abc'", vec![Str(String::from("abc"))]); }
    #[test] fn test_str_unary_minus() { run_expr("-3", vec![Int(3), UnarySub]); }
    #[test] fn test_str_binary_mul() { run_expr("3 * 6", vec![Int(3), Int(6), OpMul]); }
    #[test] fn test_str_binary_div() { run_expr("20 / 4 / 5", vec![Int(20), Int(4), OpDiv, Int(5), OpDiv]); }
    #[test] fn test_str_binary_pow() { run_expr("2 ** 10", vec![Int(2), Int(10), OpPow]); }
    #[test] fn test_str_binary_minus() { run_expr("6 - 7", vec![Int(6), Int(7), OpSub]); }
    #[test] fn test_str_binary_and_unary_minus() { run_expr("15 -- 7", vec![Int(15), Int(7), UnarySub, OpSub]); }
    #[test] fn test_str_binary_add_and_mod() { run_expr("1 + 2 % 3", vec![Int(1), Int(2), Int(3), OpMod, OpAdd]); }
    #[test] fn test_str_binary_add_and_mod_rev() { run_expr("1 % 2 + 3", vec![Int(1), Int(2), OpMod, Int(3), OpAdd]); }
    #[test] fn test_str_binary_shifts() { run_expr("1 << 2 >> 3", vec![Int(1), Int(2), OpLeftShift, Int(3), OpRightShift]); }
    #[test] fn test_str_binary_shifts_and_operators() { run_expr("1 & 2 << 3 | 5", vec![Int(1), Int(2), Int(3), OpLeftShift, OpBitwiseAnd, Int(5), OpBitwiseOr]); }
    #[test] fn test_str_function_composition() { run_expr("a . b", vec![Identifier(String::from("a")), Identifier(String::from("b")), OpFuncCompose]); }
    #[test] fn test_str_precedence_with_parens() { run_expr("(1 + 2) * 3", vec![Int(1), Int(2), OpAdd, Int(3), OpMul]); }
    #[test] fn test_str_precedence_with_parens_2() { run_expr("6 / (5 - 3)", vec![Int(6), Int(5), Int(3), OpSub, OpDiv]); }
    #[test] fn test_str_precedence_with_parens_3() { run_expr("-(1 - 3)", vec![Int(1), Int(3), OpSub, UnarySub]); }
    #[test] fn test_str_function_no_args() { run_expr("foo", vec![Identifier(String::from("foo"))]); }
    #[test] fn test_str_function_one_arg() { run_expr("foo(1)", vec![Identifier(String::from("foo")), Int(1), OpFuncEval(1)]); }
    #[test] fn test_str_function_many_args() { run_expr("foo(1,2,3)", vec![Identifier(String::from("foo")), Int(1), Int(2), Int(3), OpFuncEval(3)]); }
    #[test] fn test_str_multiple_unary_ops() { run_expr("- ~ ! 1", vec![Int(1), UnaryLogicalNot, UnaryBitwiseNot, UnarySub]); }
    #[test] fn test_str_multiple_function_calls() { run_expr("foo (1) (2) (3)", vec![Identifier(String::from("foo")), Int(1), OpFuncEval(1), Int(2), OpFuncEval(1), Int(3), OpFuncEval(1)]); }
    #[test] fn test_str_multiple_function_calls_some_args() { run_expr("foo () (1) (2, 3)", vec![Identifier(String::from("foo")), OpFuncEval(0), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(2)]); }
    #[test] fn test_str_multiple_function_calls_no_args() { run_expr("foo () () ()", vec![Identifier(String::from("foo")), OpFuncEval(0), OpFuncEval(0), OpFuncEval(0)]); }
    #[test] fn test_str_function_call_unary_op_precedence() { run_expr("- foo ()", vec![Identifier(String::from("foo")), OpFuncEval(0), UnarySub]); }
    #[test] fn test_str_function_call_unary_op_precedence_with_parens() { run_expr("(- foo) ()", vec![Identifier(String::from("foo")), UnarySub, OpFuncEval(0)]); }
    #[test] fn test_str_function_call_unary_op_precedence_with_parens_2() { run_expr("- (foo () )", vec![Identifier(String::from("foo")), OpFuncEval(0), UnarySub]); }
    #[test] fn test_str_function_call_binary_op_precedence() { run_expr("foo ( 1 ) + ( 2 ( 3 ) )", vec![Identifier(String::from("foo")), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(1), OpAdd]); }
    #[test] fn test_str_function_call_parens_1() { run_expr("foo . bar (1 + 3) (5)", vec![Identifier(String::from("foo")), Identifier(String::from("bar")), Int(1), Int(3), OpAdd, OpFuncEval(1), Int(5), OpFuncEval(1), OpFuncCompose]); }
    #[test] fn test_str_function_call_parens_2() { run_expr("( foo . bar (1 + 3) ) (5)", vec![Identifier(String::from("foo")), Identifier(String::from("bar")), Int(1), Int(3), OpAdd, OpFuncEval(1), OpFuncCompose, Int(5), OpFuncEval(1)]); }

    #[test] fn test_empty() { run("empty"); }
    #[test] fn test_expressions() { run("expressions"); }
    #[test] fn test_hello_world() { run("hello_world"); }
    #[test] fn test_invalid_expressions() { run("invalid_expressions"); }

    fn run_expr(text: &str, tokens: Vec<Opcode>) {
        let result: ScanResult = scanner::scan(&String::from(text));
        assert!(result.errors.is_empty());

        let mut parser: Parser = Parser::new(stdlib::bindings(), result.tokens);

        parser.parse_expression();

        assert_eq!(tokens, parser.output);
    }

    fn run(path: &'static str) {
        let root: String = test_common::get_test_resource_path("parser", path);
        let text: String = test_common::get_test_resource_src(&root);

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let parse_result: ParserResult = parser::parse(stdlib::bindings(), scan_result);

        let mut lines: Vec<String> = Vec::new();
        if !parse_result.code.is_empty() {
            lines.push(String::from("=== Parse Tokens ===\n"));
            for token in parse_result.code {
                lines.push(format!("{:?}", token));
            }
        }
        if !parse_result.errors.is_empty() {
            lines.push(String::from("\n=== Parse Errors ===\n"));
            for error in &parse_result.errors {
                lines.push(format!("{:?}", error));
            }
            lines.push(String::from("\n=== Formatted Parse Errors ===\n"));
            let mut source: String = String::from(path);
            source.push_str(".aocl");
            let src_lines: Vec<&str> = text.lines().collect();
            for error in &parse_result.errors {
                lines.push(reporting::format_parse_error(&src_lines, &source, error));
            }
        }

        test_common::compare_test_resource_content(&root, lines);
    }
}