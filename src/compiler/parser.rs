use std::iter::Peekable;
use std::vec::IntoIter;

use crate::compiler::scanner::{ScanResult, ScanToken};

use crate::compiler::scanner::ScanToken::{*};
use crate::compiler::parser::ParserToken::{*};



pub fn parse(scan_result: ScanResult) {
    let mut parser: Parser = Parser {
        input: scan_result.tokens.into_iter().peekable(),
        output: Vec::new(),

        lineno: 1,

        errors: Vec::new(),
        error: false,
    };
}


#[derive(Eq, PartialEq, Debug)]
pub enum ParserToken {

    BeginExpr,
    EndExpr,

    Nil,
    True,
    False,
    Int(i64),
    Str(String),
    Identifier(String),

    // Unary Operators
    UnarySub,
    UnaryLogicalNot,
    UnaryBitwiseNot,

    // Binary Operators
    // Ordered by precedence, highest to lowest
    OpFuncEval(u8),
    OpArrayEval(u8),

    OpMul,
    OpDiv,
    OpMod,

    OpAdd,
    OpSub,

    OpLeftShift,
    OpRightShift,

    OpLessThan,
    OpGreaterThan,
    OpLessThanEqual,
    OpGreaterThanEqual,
    OpEqual,
    OpNotEqual,

    OpBitwiseAnd,
    OpBitwiseOr,
    OpBitwiseXor,

    OpFuncCompose,

    OpLogicalAnd,
    OpLogicalOr,

    NewLine,
}

struct ParserError {
    error: ParserErrorType,
    index: usize,
}

enum ParserErrorType {
    UnexpectedToken(ScanToken, ScanToken), // actual, expected
    UnexpectedEoF(ScanToken) // expected
}


struct Parser {
    input: Peekable<IntoIter<ScanToken>>,
    output: Vec<ParserToken>,
    errors: Vec<ParserError>,

    lineno: usize,

    /// If we are in error recover mode, this flag is set
    error: bool,
}


impl Parser {

    fn parse(self: &mut Self) {
        self.parse_statements();
    }

    fn parse_statements(self: &mut Self) {
        loop {
            match self.advance() {
                Some(KeywordFn) => self.parse_function(),
                Some(KeywordLet) => self.parse_let(),
                Some(KeywordIf) => self.parse_if(),
                Some(KeywordLoop) => self.parse_loop(),
                Some(KeywordFor) => self.parse_for(),
                Some(_) => {}, // todo: 'unknown thing' error here
                None => break
            }
        }
    }

    fn parse_block_statement(self: &mut Self) {
        self.expect(OpenBrace);
        self.parse_statements();
        self.expect(CloseBrace);
    }

    fn parse_function(self: &mut Self) {
        // todo
        //self.expect_identifier();
        if self.accept(OpenParen) {
            // todo:
            //self.parse_function_parameters();
        }
        self.parse_optional_type();
        self.parse_block_statement();
    }

    fn parse_let(self: &mut Self) {
        self.parse_pattern();
        self.parse_optional_type();
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
        self.parse_pattern();
        self.expect(KeywordIn);
        self.parse_expression();
        self.parse_block_statement();
    }


    fn parse_pattern(self: &mut Self) {
        // todo
        //self.expect_identifier();
    }

    fn parse_optional_type(self: &mut Self) {
        // todo: future, implement type inference
        self.expect(Colon);
        self.parse_type();
    }

    fn parse_expression(self: &mut Self) {
        self.parse_expr_9()
    }


    // ===== Expression Parsing ===== //

    fn parse_expr_1_terminal(self: &mut Self) {
        match self.peek() {
            Some(KeywordNil) => self.advance_push(Nil),
            Some(KeywordTrue) => self.advance_push(True),
            Some(KeywordFalse) => self.advance_push(False),
            Some(ScanToken::Int(_)) => self.advance_int(),
            Some(ScanToken::Identifier(_)) => self.advance_identifier(),
            Some(StringLiteral(_)) => self.advance_str(),
            Some(OpenParen) => {
                self.advance();
                self.parse_expr_9();
                self.expect(CloseParen);
            }
            _ => {} // todo: every case in this style of match {} statement needs to set error recovery flags, throw an error, etc.
        }
    }

    fn parse_expr_2_unary(self: &mut Self) {
        // Prefix operators
        let mut stack: Vec<ParserToken> = Vec::new();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
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
                            loop {
                                // Other arguments
                                match self.peek() {
                                    Some(Comma) => {
                                        self.advance();
                                        self.parse_expression();
                                        count += 1;
                                    },
                                    Some(CloseParen) => {
                                        self.advance();
                                        break
                                    },
                                    _ => break // todo: report error
                                }
                            }
                        }
                        None => break // todo: report error
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
        self.parse_expr_2_unary();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
                Some(Mul) => Some(OpMul),
                Some(Div) => Some(OpDiv),
                Some(Mod) => Some(OpMod),
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
        self.parse_expr_3();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
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
        self.parse_expr_4();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
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
        self.parse_expr_5();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
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
        self.parse_expr_6();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
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
        self.parse_expr_7();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
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
        self.parse_expr_8();
        loop {
            let maybe_op: Option<ParserToken> = match self.peek() {
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


    // ===== Type (Expression) Parsing ===== //

    fn parse_type(self: &mut Self) {
        panic!("Unimplemented!")
    }

    /// If the given token is present, accept it and return `true`. Otherwise, do nothing and return `false`.
    fn accept(self: &mut Self, token: ScanToken) -> bool {
        // todo: handle if we're in error recovery mode
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
        match self.peek() {
            Some(t) if t == &token => {
                self.advance();
            }
            Some(_) => {}, // todo: enter error recovery mode with 'unexpected token' and 'unexpected eof'
            None => {}
        }
    }

    /// Advances the token stream and pushes the provided token to the output stream.
    fn advance_push(self: &mut Self, token: ParserToken) {
        self.advance();
        self.push(token);
    }


    /// Like `advance()`, but pushes the boxed `Identifier` token to the output stream.
    /// **Important**: Must only be called once `peek()` has identified an `Identifier` token is present, as this will panic otherwise.
    fn advance_identifier(self: &mut Self) {
        match self.advance() {
            Some(ScanToken::Identifier(name)) => self.push(ParserToken::Identifier(name)),
            t => panic!("Token mismatch in advance_identifier() -> expected an Some(Identifier(String)), got a {:?} instead", t)
        }
    }

    /// Like `advance()`, but pushes the boxed `Int` token to the output stream.
    /// **Important**: Must only be called once `peek()` has identified an `Int` token is present, as this will panic otherwise.
    fn advance_int(self: &mut Self) {
        match self.advance() {
            Some(ScanToken::Int(i)) => self.push(ParserToken::Int(i)),
            t => panic!("Token mismatch in advance_int() -> expected an Some(Int(i64)), got a {:?} instead", t)
        }
    }

    /// Like `advance()`, but pushes the boxed `String` literal token to the output stream.
    /// **Important**: Must only be called once `peek()` has identified a `StringLiteral` token is present, as this will panic otherwise.
    fn advance_str(self: &mut Self) {
        match self.advance() {
            Some(StringLiteral(s)) => self.push(Str(s)),
            t => panic!("Token mismatch in advance_str() -> expected a Some(StringLiteral(String)), got a {:?} instead", t)
        }
    }


    /// Peeks at the next incoming token.
    /// Note that this function only returns a read-only reference to the underlying token, suitable for matching
    /// If the token data needs to be unboxed, i.e. as with `Identifier` tokens, it must be extracted only via `advance()`
    fn peek(self: &mut Self) -> Option<&ScanToken> {
        while let Some(ScanToken::NewLine) = self.input.peek() {
            self.input.next();
            self.lineno += 1;
        }
        self.input.peek()
    }

    /// Advances and returns the next incoming token.
    fn advance(self: &mut Self) -> Option<ScanToken> {
        while let Some(ScanToken::NewLine) = self.input.peek() {
            self.input.next();
            self.lineno += 1;
        }
        self.input.next()
    }

    /// Pushes a new token into the output stream
    fn push(self: &mut Self, token: ParserToken) {
        self.output.push(token);
    }
}


#[cfg(test)]
mod tests {
    use crate::compiler::scanner;
    use crate::compiler::scanner::ScanResult;
    use crate::compiler::parser::{Parser, ParserToken};
    use crate::compiler::parser::ParserToken::{*};

    #[test] fn test_str_empty() { run_str("", vec![]); } // todo: change once error recovery is in
    #[test] fn test_str_int() { run_str("123", vec![Int(123)]); }
    #[test] fn test_str_str() { run_str("'abc'", vec![Str(String::from("abc"))]); }
    #[test] fn test_str_unary_minus() { run_str("-3", vec![Int(3), UnarySub]); }
    #[test] fn test_str_binary_mul() { run_str("3 * 6", vec![Int(3), Int(6), OpMul]); }
    #[test] fn test_str_binary_div() { run_str("20 / 4 / 5", vec![Int(20), Int(4), OpDiv, Int(5), OpDiv]); }
    #[test] fn test_str_binary_minus() { run_str("6 - 7", vec![Int(6), Int(7), OpSub]); }
    #[test] fn test_str_binary_and_unary_minus() { run_str("15 -- 7", vec![Int(15), Int(7), UnarySub, OpSub]); }
    #[test] fn test_str_binary_add_and_mod() { run_str("1 + 2 % 3", vec![Int(1), Int(2), Int(3), OpMod, OpAdd]); }
    #[test] fn test_str_binary_add_and_mod_rev() { run_str("1 % 2 + 3", vec![Int(1), Int(2), OpMod, Int(3), OpAdd]); }
    #[test] fn test_str_binary_shifts() { run_str("1 << 2 >> 3", vec![Int(1), Int(2), OpLeftShift, Int(3), OpRightShift]); }
    #[test] fn test_str_binary_shifts_and_operators() { run_str("1 & 2 << 3 | 5", vec![Int(1), Int(2), Int(3), OpLeftShift, OpBitwiseAnd, Int(5), OpBitwiseOr]); }
    #[test] fn test_str_function_composition() { run_str("a . b", vec![Identifier(String::from("a")), Identifier(String::from("b")), OpFuncCompose]); }
    #[test] fn test_str_precedence_with_parens() { run_str("(1 + 2) * 3", vec![Int(1), Int(2), OpAdd, Int(3), OpMul]); }
    #[test] fn test_str_precedence_with_parens_2() { run_str("6 / (5 - 3)", vec![Int(6), Int(5), Int(3), OpSub, OpDiv]); }
    #[test] fn test_str_precedence_with_parens_3() { run_str("-(1 - 3)", vec![Int(1), Int(3), OpSub, UnarySub]); }
    #[test] fn test_str_function_no_args() { run_str("foo", vec![Identifier(String::from("foo"))]); }
    #[test] fn test_str_function_one_arg() { run_str("foo(1)", vec![Identifier(String::from("foo")), Int(1), OpFuncEval(1)]); }
    #[test] fn test_str_function_many_args() { run_str("foo(1,2,3)", vec![Identifier(String::from("foo")), Int(1), Int(2), Int(3), OpFuncEval(3)]); }
    #[test] fn test_str_multiple_unary_ops() { run_str("- ~ ! 1", vec![Int(1), UnaryLogicalNot, UnaryBitwiseNot, UnarySub]); }
    #[test] fn test_str_multiple_function_calls() { run_str("foo (1) (2) (3)", vec![Identifier(String::from("foo")), Int(1), OpFuncEval(1), Int(2), OpFuncEval(1), Int(3), OpFuncEval(1)]); }
    #[test] fn test_str_multiple_function_calls_some_args() { run_str("foo () (1) (2, 3)", vec![Identifier(String::from("foo")), OpFuncEval(0), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(2)]); }
    #[test] fn test_str_multiple_function_calls_no_args() { run_str("foo () () ()", vec![Identifier(String::from("foo")), OpFuncEval(0), OpFuncEval(0), OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence() { run_str("- foo ()", vec![Identifier(String::from("foo")), OpFuncEval(0), UnarySub]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens() { run_str("(- foo) ()", vec![Identifier(String::from("foo")), UnarySub, OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens_2() { run_str("- (foo () )", vec![Identifier(String::from("foo")), OpFuncEval(0), UnarySub]); }
    #[test] fn test_function_call_binary_op_precedence() { run_str("foo ( 1 ) + ( 2 ( 3 ) )", vec![Identifier(String::from("foo")), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(1), OpAdd]); }

    fn run_str(text: &str, tokens: Vec<ParserToken>) {
        let result: ScanResult = scanner::scan(&String::from(text));
        assert!(result.errors.is_empty());

        let mut parser: Parser = Parser {
            input: result.tokens.into_iter().peekable(),
            output: Vec::new(),
            errors: Vec::new(),

            lineno: 0,

            error: false,
        };

        parser.parse_expression();

        assert_eq!(tokens, parser.output);
    }
}