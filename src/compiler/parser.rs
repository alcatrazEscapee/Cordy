use std::collections::HashMap;
use std::iter::Peekable;
use std::vec::IntoIter;

use crate::compiler::scanner::{ScanResult, ScanToken};
use crate::stdlib::StdBinding;
use crate::vm::opcode::Opcode;
use crate::trace;

use ScanToken::{*};
use ParserErrorType::{*};
use Opcode::{*};


pub fn parse(bindings: HashMap<&'static str, StdBinding>, scan_result: ScanResult) -> ParserResult {
    let mut parser: Parser = Parser::new(bindings, scan_result.tokens);
    parser.parse();
    ParserResult {
        code: parser.output,
        errors: parser.errors,
        globals: parser.globals,
        strings: parser.strings,
        constants: parser.constants,
    }
}

pub struct ParserResult {
    pub code: Vec<Opcode>,
    pub errors: Vec<ParserError>,

    pub globals: HashMap<String, u16>,
    pub strings: Vec<String>,
    pub constants: Vec<i64>,
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
    ExpectedCommaOrEndOfList(ScanToken),
    ExpectedColonOrEndOfSlice(ScanToken),
    ExpectedStatement(ScanToken),
    ExpectedVariableNameAfterLet(ScanToken),

    GlobalVariableConflictWithBinding(String),
    GlobalVariableConflictWithGlobal(String),

    AssignmentToNotVariable(String),
    BreakOutsideOfLoop,
    ContinueOutsideOfLoop,
}


struct Parser {
    bindings: HashMap<&'static str, StdBinding>,
    input: Peekable<IntoIter<ScanToken>>,
    output: Vec<Opcode>,
    errors: Vec<ParserError>,

    lineno: usize,

    error_recovery: bool, // If we are in error recover mode, this flag is set
    multiple_expression_statements_per_line: bool, // If a expression statement has already been parsed before a new line or ';', this flag is set
    lookahead: Option<ScanToken>, // A token we pretend 'push back onto' the input iterator

    globals: HashMap<String, u16>,
    strings: Vec<String>,
    constants: Vec<i64>,

    // Loop stack
    // Each frame represents a single loop, which `break` and `continue` statements refer to
    // `continue` jumps back to the beginning of the loop, aka the first `usize` (loop start)
    // `break` statements jump back to the end of the loop, which needs to be patched later. The values to be patched record themselves in the stack at the current loop level
    loops: Vec<(u16, Vec<u16>)>,
}

#[derive(Eq, PartialEq, Debug, Clone)]
enum VariableType {
    //StructOrDerivedBinding, // this one is complicated, since it is a type based dispatch against user defined types...
    //LocalVariable(usize), // Not implemented
    GlobalVariable(u16),
    TopLevelBinding(StdBinding),
    None,
}


impl Parser {

    fn new(bindings: HashMap<&'static str, StdBinding>, tokens: Vec<ScanToken>) -> Parser {
        Parser {
            bindings,
            input: tokens.into_iter().peekable(),
            output: Vec::new(),
            errors: Vec::new(),

            lineno: 0,

            error_recovery: false,
            multiple_expression_statements_per_line: false,
            lookahead: None,

            globals: HashMap::new(),
            strings: Vec::new(),
            constants: Vec::new(),

            loops: Vec::new(),
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
                Some(KeywordIf) => self.parse_if_statement(),
                Some(KeywordLoop) => self.parse_loop(),
                Some(KeywordFor) => self.parse_for(),
                Some(KeywordBreak) => self.parse_break_statement(),
                Some(KeywordContinue) => self.parse_continue_statement(),
                Some(ScanToken::Identifier(_)) => self.parse_assignment(),
                Some(OpenBrace) => self.parse_block_statement(),
                Some(CloseBrace) => break, // Don't consume, but break if we're in an error mode
                Some(KeywordExit) => {
                    self.advance();
                    self.push(Exit);
                },
                Some(Semicolon) => {
                    self.multiple_expression_statements_per_line = false;
                    self.advance();
                },
                Some(_) => self.parse_expression_statement(),
                None => break,
            }
        }
    }

    fn parse_block_statement(self: &mut Self) {
        trace::trace_parser!("rule <block-statement>");
        self.expect(OpenBrace);
        self.parse_statements();
        self.multiple_expression_statements_per_line = false;
        self.expect_resync(CloseBrace);
    }

    fn parse_function(self: &mut Self) {
        panic!("Functions not implemented in parser");
    }

    fn parse_let(self: &mut Self) {
        trace::trace_parser!("rule <let-statement>");
        self.advance();
        loop {
            // For now, assume that all variables are global
            let store_op: Opcode = match self.peek() {
                Some(ScanToken::Identifier(_)) => {
                    let name: String = self.take_identifier();
                    let gid: u16 = match self.declare_global(&name) {
                        Some(g) => g,
                        None => break
                    };
                    StoreGlobal(gid)
                },
                Some(t) => {
                    let token: ScanToken = t.clone();
                    self.push_err(ExpectedVariableNameAfterLet(token));
                    break
                },
                None => break
            };

            match self.peek() {
                Some(Equals) => { // x = <expr>, so parse the expression
                    self.advance();
                    self.multiple_expression_statements_per_line = true;
                    self.parse_expression();
                },
                _ => {
                    self.push(Nil); // Initialize to 'nil'
                },
            }

            self.push(store_op);

            match self.peek() {
                Some(Comma) => { // Multiple declarations
                    self.advance();
                    self.multiple_expression_statements_per_line = false;
                },
                _ => break,
            }
        }
    }

    fn parse_if_statement(self: &mut Self) {
        // Translation:
        // if <expr> {       | JumpIfFalsePop L1
        //     <statements>  | <statements>
        // }                 | L1:
        //                   |
        // if <expr> {       | JumpIfFalsePop L1
        //     <statements>  | <statements> ; Jump L2
        // } else <expr> {   | L1:
        //     <statements>  | <statements>
        // }                 | L2:

        trace::trace_parser!("rule <if-statement>");
        self.advance();
        self.parse_expression();
        let jump_if_false: u16 = self.reserve(); // placeholder for jump to the beginning of an if branch, if it exists
        self.parse_block_statement();

        // `elif` can be de-sugared to else { if <expr> { ... } else { ... } }
        // The additional scope around the else {} can be dropped as it doesn't contain anything already in a scope
        // So if we hit `elif`, we emit what we would for `else`, then recurse, then once we return, patch the final jump.
        match self.peek() {
            Some(KeywordElif) => {
                // Don't advance, as `parse_if_statement()` will advance the first token
                let jump: u16 = self.reserve();
                let after_if: u16 = self.next_opcode();
                self.output[jump_if_false as usize] = JumpIfFalsePop(after_if);
                self.parse_if_statement();
                let after_else: u16 = self.next_opcode();
                self.output[jump as usize] = Jump(after_else);
            },
            Some(KeywordElse) => {
                // `else` is present, so we first insert an unconditional jump, parse the next block, then fix the first jump
                self.advance();
                let jump: u16 = self.reserve();
                let after_if: u16 = self.next_opcode();
                self.output[jump_if_false as usize] = JumpIfFalsePop(after_if);
                self.parse_block_statement();
                let after_else: u16 = self.next_opcode();
                self.output[jump as usize] = Jump(after_else);
            },
            _ => {
                // No `else`, so we patch the original jump to jump to here
                let after: u16 = self.next_opcode();
                self.output[jump_if_false as usize] = JumpIfFalsePop(after);
            },
        }
    }

    fn parse_loop(self: &mut Self) {
        trace::trace_parser!("<loop-statement>");

        // Translation:
        // loop {            | L1:
        //     <statements>  | <statements>
        //     break         | Jump L2  -> Push onto loop stack, fix at end
        //     continue      | Jump L1  -> Set immediately, using the loop start value
        //     <statements>  | <statements>
        // }                 | Jump L1 ; L2:
        self.advance();

        let loop_start: u16 = self.next_opcode(); // Top of the loop, push onto the loop stack
        self.loops.push((loop_start, Vec::new()));

        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push(Jump(loop_start));

        let loop_end: u16 = self.next_opcode(); // After the jump, the next opcode is 'end of loop'. Repair all break statements
        let break_opcodes: Vec<u16> = self.loops.pop().unwrap().1;
        for break_opcode in break_opcodes {
            self.output[break_opcode as usize] = Jump(loop_end);
        }
    }

    fn parse_for(self: &mut Self) {
        panic!("for statements not implemented in parser");
    }

    fn parse_break_statement(self: &mut Self) {
        trace::trace_parser!("rule <break-statement>");
        self.advance();
        match self.loops.last() {
            Some(_) => {
                let jump: u16 = self.reserve();
                self.loops.last_mut().unwrap().1.push(jump);
            },
            None => self.push_err(BreakOutsideOfLoop),
        }
    }

    fn parse_continue_statement(self: &mut Self) {
        trace::trace_parser!("rule <continue-statement>");
        self.advance();
        match self.loops.last() {
            Some((loop_start, _)) => {
                let jump_to: u16 = *loop_start;
                self.push(Jump(jump_to));
            },
            None => self.push_err(ContinueOutsideOfLoop),
        }
    }

    fn parse_assignment(self: &mut Self) {
        trace::trace_parser!("rule <assignment-statement>");
        let name: String = self.take_identifier();
        let maybe_op: Option<Opcode> = match self.peek() {
            Some(Equals) => Some(OpEqual), // Fake operator
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
        if let Some(op) = maybe_op {

            // Resolve the variable name
            let (store_op, push_op) = match self.resolve_identifier(&name) {
                VariableType::GlobalVariable(gid) => (StoreGlobal(gid), PushGlobal(gid)),
                _ => {
                    let token: String = name.clone();
                    self.push_err(AssignmentToNotVariable(token));
                    return
                }
            };
            if op != OpEqual { // Only load the value in assignment expressions
                self.push(push_op);
            }
            self.advance();
            self.multiple_expression_statements_per_line = true;
            self.parse_expression();
            if op != OpEqual { // Only push the operator in assignment expressions
                self.push(op);
            }
            self.push(store_op);
        } else {
            // In this case, we matched (and consumed) <name> without a following '=' or other assignment operator.
            // This must be part of an expression, but we need to first push the lookahead identifier.
            // A bare expression that starts with an identifier token also must end with a `Pop` opcode.
            self.push_lookahead(ScanToken::Identifier(name));
            self.parse_expression_statement();
        }
    }

    fn parse_expression_statement(self: &mut Self) {
        trace::trace_parser!("rule <expression-statement>");
        match self.peek() {
            Some(t) => {
                let token: ScanToken = t.clone();
                if !self.multiple_expression_statements_per_line {
                    self.multiple_expression_statements_per_line = true;
                    self.parse_expression();
                    self.push(Pop);
                } else {
                    self.push_err(ExpectedStatement(token))
                }
            },
            None => {},
        }
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
                let cid: u16 = self.declare_constant(int);
                self.push(Opcode::Int(cid));
            },
            Some(ScanToken::Identifier(_)) => {
                let string: String = self.take_identifier();
                match self.resolve_identifier(&string) {
                    VariableType::GlobalVariable(gid) => self.push(PushGlobal(gid)),
                    VariableType::TopLevelBinding(b) => self.push(Bound(b)),
                    _ => self.push(Opcode::Identifier)
                };
            },
            Some(StringLiteral(_)) => {
                let string: String = self.take_str();
                let sid: u16 = self.declare_string(string);
                self.push(Str(sid));
            },
            Some(OpenParen) => {
                self.advance();
                self.parse_expression();
                self.expect(CloseParen);
            },
            Some(OpenSquareBracket) => {
                self.advance();
                let mut length: i64 = 0;
                loop {
                    match self.peek() {
                        Some(CloseSquareBracket) => break,
                        Some(_) => {
                            self.parse_expression();
                            length += 1;
                            match self.peek() {
                                Some(Comma) => self.skip(), // Allow trailing commas, as this loops to the top again
                                Some(CloseSquareBracket) => {}, // Skip
                                Some(t) => {
                                    let token: ScanToken = t.clone();
                                    self.push_err(ExpectedCommaOrEndOfList(token))
                                }
                                None => break
                            }
                        },
                        None => break,
                    }
                }
                let length: u16 = self.declare_constant(length);
                self.push(List(length));
                self.expect(CloseSquareBracket);
            },
            Some(e) => {
                let token: ScanToken = e.clone();
                self.push(Nil);
                self.push_err(ExpectedExpressionTerminal(token));
            },
            _ => {
                self.push(Nil);
                self.push_err(UnexpectedEoF)
            },
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
                Some(OpenSquareBracket) => {
                    self.advance(); // Consume the square bracket

                    // Consumed `[` so far
                    match self.peek() {
                        Some(Colon) => { // No first argument, so push zero. Don't consume the colon as it's the seperator
                            self.push(Nil);
                        },
                        _ => self.parse_expression(), // Otherwise we require a first expression
                    }

                    // Consumed `[` <expression> so far
                    match self.peek() {
                        Some(CloseSquareBracket) => { // One argument, so push OpIndex and exit the statement
                            self.advance();
                            self.push(OpIndex);
                            continue;
                        },
                        Some(Colon) => { // At least two arguments, so continue parsing
                            self.advance();
                        },
                        Some(t) => { // Anything else was a syntax error
                            let token: ScanToken = t.clone();
                            self.push_err(ExpectedColonOrEndOfSlice(token));
                            continue
                        },
                        _ => self.expect(CloseSquareBracket),
                    }

                    // Consumed `[` <expression> `:` so far
                    match self.peek() {
                        Some(Colon) => { // No second argument, but we have a third argument, so push -1. Don't consume the colon as it's the seperator
                            self.push(Nil);
                        },
                        Some(CloseSquareBracket) => { // No second argument, so push -1 and then exit
                            self.push(Nil);
                            self.advance();
                            self.push(OpSlice);
                            continue
                        }
                        _ => self.parse_expression(), // As we consumed `:`, we require a second expression
                    }

                    // Consumed `[` <expression> `:` <expression> so far
                    match self.peek() {
                        Some(CloseSquareBracket) => { // Two arguments, so push OpSlice and exit the statement
                            self.advance();
                            self.push(OpSlice);
                            continue;
                        },
                        Some(Colon) => {
                            // Three arguments, so continue parsing
                            self.advance();
                        },
                        Some(t) => { // Anything else was a syntax error
                            let token: ScanToken = t.clone();
                            self.push_err(ExpectedColonOrEndOfSlice(token));
                            continue;
                        },
                        _ => self.expect(CloseSquareBracket),
                    }

                    // Consumed `[` <expression> `:` <expression> `:` so far
                    match self.peek() {
                        Some(CloseSquareBracket) => { // Three arguments + default value for slice
                            self.push(Nil);
                        },
                        _ => self.parse_expression(),
                    }

                    self.expect(CloseSquareBracket);
                    self.push(OpSliceWithStep);
                },
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
                Some(BitwiseAnd) => Some(OpBitwiseAnd),
                Some(BitwiseOr) => Some(OpBitwiseOr),
                Some(BitwiseXor) => Some(OpBitwiseXor),
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
                Some(Dot) => Some(OpFuncCompose),
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
            match self.peek() {
                Some(LogicalAnd) => {
                    self.advance();
                    let jump_if_false: u16 = self.reserve();
                    self.push(Pop);
                    self.parse_expr_8();
                    let jump_to: u16 = self.next_opcode();
                    self.output[jump_if_false as usize] = JumpIfFalse(jump_to);
                },
                Some(LogicalOr) => {
                    self.advance();
                    let jump_if_true: u16 = self.reserve();
                    self.push(Pop);
                    self.parse_expr_8();
                    let jump_to: u16 = self.next_opcode();
                    self.output[jump_if_true as usize] = JumpIfTrue(jump_to);
                },
                _ => break
            }
        }
    }


    // ===== Semantic Analysis ===== //

    fn declare_string(self: &mut Self, str: String) -> u16 {
        self.strings.push(str);
        (self.strings.len() - 1) as u16
    }

    fn declare_constant(self: &mut Self, int: i64) -> u16 {
        self.constants.push(int);
        (self.constants.len() - 1) as u16
    }

    fn next_opcode(self: &Self) -> u16 {
        self.output.len() as u16
    }

    fn declare_global(self: &mut Self, name: &String) -> Option<u16> {
        // Global variables must not conflict with known top-level bindings, or other global variables
        // Otherwise they are allowed
        if self.bindings.contains_key(name.as_str()) {
            self.push_err(GlobalVariableConflictWithBinding(name.clone()));
            return None;
        }
        if self.globals.contains_key(name) {
            self.push_err(GlobalVariableConflictWithGlobal(name.clone()));
            return None;
        }

        // Declaration successful! Enter it in the globals table
        let gid: u16 = self.globals.len() as u16;
        self.globals.insert(name.clone(), gid);
        Some(gid)
    }

    fn resolve_identifier(self: &mut Self, name: &String) -> VariableType {
        // Resolution Strategy
        // 1. Struct Fields and Derived Bindings
        // 2. Local Variables
        // 3. Global Variables
        // 4. Top Level Bindings
        if let Some(gid) = self.globals.get(name) {
            return VariableType::GlobalVariable(*gid);
        }
        if let Some(b) = self.bindings.get(name.as_str()) {
            return VariableType::TopLevelBinding(*b);
        }
        VariableType::None
    }


    // ===== Parser Core ===== //


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
                    self.error_recovery = false;
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
        if self.error_recovery {
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
        if self.error_recovery {
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
            self.output.push(LineNumber(self.lineno as u16));
            self.multiple_expression_statements_per_line = false;
        }
    }

    /// Reserves a space in the output code by inserting a `Noop` token
    /// Returns an index to the token, which can later be used to set the correct value
    fn reserve(self: &mut Self) -> u16 {
        trace::trace_parser!("reserve at {}", self.output.len() - 1);
        self.output.push(Noop);
        (self.output.len() - 1) as u16
    }

    /// Pushes a new token into the output stream.
    /// Returns the index of the token pushed, which allows callers to later mutate that token if they need to.
    fn push(self: &mut Self, token: Opcode) {
        trace::trace_parser!("push {:?}", token);
        self.output.push(token);
    }

    /// Pushes a new error token into the output error stream.
    fn push_err(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError {
                error,
                lineno: self.lineno,
            });
        }
        self.error_recovery = true;
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
    use crate::compiler::{parser, scanner};
    use crate::compiler::scanner::ScanResult;
    use crate::compiler::parser::{Parser, ParserResult};
    use crate::{reporting, stdlib};
    use crate::stdlib::StdBinding;
    use crate::vm::opcode::Opcode;
    use crate::trace;

    use crate::vm::opcode::Opcode::{*};

    #[test] fn test_str_empty() { run_expr("", vec![Nil]); }
    #[test] fn test_str_int() { run_expr("123", vec![Int(123)]); }
    #[test] fn test_str_str() { run_expr("'abc'", vec![Str(0)]); }
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
    #[test] fn test_str_function_composition() { run_expr("a . b", vec![Identifier, Identifier, OpFuncCompose]); }
    #[test] fn test_str_precedence_with_parens() { run_expr("(1 + 2) * 3", vec![Int(1), Int(2), OpAdd, Int(3), OpMul]); }
    #[test] fn test_str_precedence_with_parens_2() { run_expr("6 / (5 - 3)", vec![Int(6), Int(5), Int(3), OpSub, OpDiv]); }
    #[test] fn test_str_precedence_with_parens_3() { run_expr("-(1 - 3)", vec![Int(1), Int(3), OpSub, UnarySub]); }
    #[test] fn test_str_function_no_args() { run_expr("foo", vec![Identifier]); }
    #[test] fn test_str_function_one_arg() { run_expr("foo(1)", vec![Identifier, Int(1), OpFuncEval(1)]); }
    #[test] fn test_str_function_many_args() { run_expr("foo(1,2,3)", vec![Identifier, Int(1), Int(2), Int(3), OpFuncEval(3)]); }
    #[test] fn test_str_multiple_unary_ops() { run_expr("- ~ ! 1", vec![Int(1), UnaryLogicalNot, UnaryBitwiseNot, UnarySub]); }
    #[test] fn test_str_multiple_function_calls() { run_expr("foo (1) (2) (3)", vec![Identifier, Int(1), OpFuncEval(1), Int(2), OpFuncEval(1), Int(3), OpFuncEval(1)]); }
    #[test] fn test_str_multiple_function_calls_some_args() { run_expr("foo () (1) (2, 3)", vec![Identifier, OpFuncEval(0), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(2)]); }
    #[test] fn test_str_multiple_function_calls_no_args() { run_expr("foo () () ()", vec![Identifier, OpFuncEval(0), OpFuncEval(0), OpFuncEval(0)]); }
    #[test] fn test_str_function_call_unary_op_precedence() { run_expr("- foo ()", vec![Identifier, OpFuncEval(0), UnarySub]); }
    #[test] fn test_str_function_call_unary_op_precedence_with_parens() { run_expr("(- foo) ()", vec![Identifier, UnarySub, OpFuncEval(0)]); }
    #[test] fn test_str_function_call_unary_op_precedence_with_parens_2() { run_expr("- (foo () )", vec![Identifier, OpFuncEval(0), UnarySub]); }
    #[test] fn test_str_function_call_binary_op_precedence() { run_expr("foo ( 1 ) + ( 2 ( 3 ) )", vec![Identifier, Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(1), OpAdd]); }
    #[test] fn test_str_function_call_parens_1() { run_expr("foo . bar (1 + 3) (5)", vec![Identifier, Identifier, Int(1), Int(3), OpAdd, OpFuncEval(1), Int(5), OpFuncEval(1), OpFuncCompose]); }
    #[test] fn test_str_function_call_parens_2() { run_expr("( foo . bar (1 + 3) ) (5)", vec![Identifier, Identifier, Int(1), Int(3), OpAdd, OpFuncEval(1), OpFuncCompose, Int(5), OpFuncEval(1)]); }
    #[test] fn test_str_function_composition_with_is() { run_expr("'123' . int is int . print", vec![Str(0), Bound(StdBinding::Int), Bound(StdBinding::Int), OpIs, OpFuncCompose, Bound(StdBinding::Print), OpFuncCompose]); }
    #[test] fn test_and() { run_expr("1 < 2 and 3 < 4", vec![Int(1), Int(2), OpLessThan, JumpIfFalse(8), Pop, Int(3), Int(4), OpLessThan]); }
    #[test] fn test_or() { run_expr("1 < 2 or 3 < 4", vec![Int(1), Int(2), OpLessThan, JumpIfTrue(8), Pop, Int(3), Int(4), OpLessThan]); }
    #[test] fn test_precedence_1() { run_expr("1 . 2 & 3 > 4", vec![Int(1), Int(2), Int(3), OpBitwiseAnd, OpFuncCompose, Int(4), OpGreaterThan]); }
    #[test] fn test_slice_01() { run_expr("1 [::]", vec![Int(1), Nil, Nil, Nil, OpSliceWithStep]); }
    #[test] fn test_slice_02() { run_expr("1 [2::]", vec![Int(1), Int(2), Nil, Nil, OpSliceWithStep]); }
    #[test] fn test_slice_03() { run_expr("1 [:3:]", vec![Int(1), Nil, Int(3), Nil, OpSliceWithStep]); }
    #[test] fn test_slice_04() { run_expr("1 [::4]", vec![Int(1), Nil, Nil, Int(4), OpSliceWithStep]); }
    #[test] fn test_slice_05() { run_expr("1 [2:3:]", vec![Int(1), Int(2), Int(3), Nil, OpSliceWithStep]); }
    #[test] fn test_slice_06() { run_expr("1 [2::3]", vec![Int(1), Int(2), Nil, Int(3), OpSliceWithStep]); }
    #[test] fn test_slice_07() { run_expr("1 [:3:4]", vec![Int(1), Nil, Int(3), Int(4), OpSliceWithStep]); }
    #[test] fn test_slice_08() { run_expr("1 [2:3:4]", vec![Int(1), Int(2), Int(3), Int(4), OpSliceWithStep]); }
    #[test] fn test_slice_09() { run_expr("1 [:]", vec![Int(1), Nil, Nil, OpSlice]); }
    #[test] fn test_slice_10() { run_expr("1 [2:]", vec![Int(1), Int(2), Nil, OpSlice]); }
    #[test] fn test_slice_11() { run_expr("1 [:3]", vec![Int(1), Nil, Int(3), OpSlice]); }
    #[test] fn test_slice_12() { run_expr("1 [2:3]", vec![Int(1), Int(2), Int(3), OpSlice]); }

    #[test] fn test_empty() { run("empty"); }
    #[test] fn test_expressions() { run("expressions"); }
    #[test] fn test_global_variables() { run("global_variables"); }
    #[test] fn test_global_assignments() { run("global_assignments"); }
    #[test] fn test_hello_world() { run("hello_world"); }
    #[test] fn test_if_statement_1() { run("if_statement_1"); }
    #[test] fn test_if_statement_2() { run("if_statement_2"); }
    #[test] fn test_if_statement_3() { run("if_statement_3"); }
    #[test] fn test_if_statement_4() { run("if_statement_4"); }
    #[test] fn test_invalid_expressions() { run("invalid_expressions"); }
    #[test] fn test_loop_1() { run("loop_1"); }
    #[test] fn test_loop_2() { run("loop_2"); }
    #[test] fn test_loop_3() { run("loop_3"); }
    #[test] fn test_loop_4() { run("loop_4"); }

    fn run_expr(text: &str, tokens: Vec<Opcode>) {
        let result: ScanResult = scanner::scan(&String::from(text));
        assert!(result.errors.is_empty());

        let mut parser: Parser = Parser::new(stdlib::bindings(), result.tokens);
        parser.parse_expression();

        // Tokens will contain int values as exact values, as it's easier to read as a test DSL
        // However, the parser will give us constant IDs
        // So, index each one to produce what it looks like
        let constants: Vec<i64> = parser.constants;
        let output: Vec<Opcode> = parser.output.into_iter()
            .map(|t| match t {
                Int(i) => Int(constants[i as usize] as u16),
                t => t
            })
            .collect::<Vec<Opcode>>();

        assert_eq!(tokens, output);
    }

    fn run(path: &'static str) {
        let root: String = trace::test::get_test_resource_path("parser", path);
        let text: String = trace::test::get_test_resource_src(&root);

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let parse_result: ParserResult = parser::parse(stdlib::bindings(), scan_result);
        let strings: Vec<String> = parse_result.strings;
        let constants: Vec<i64> = parse_result.constants;

        let mut lines: Vec<String> = Vec::new();

        if !parse_result.globals.is_empty() {
            lines.push(String::from("=== Globals ===\n"));
            let mut globals = parse_result.globals.iter().collect::<Vec<(&String, &u16)>>();
            globals.sort_by_key(|(_, u)| *u);
            for (global, gid) in globals {
                lines.push(format!("{:?} : {:?}", global, gid));
            }
            lines.push(String::new());
        }

        if !parse_result.code.is_empty() {
            lines.push(String::from("=== Parse Tokens ===\n"));
            for (ip, token) in parse_result.code.iter().enumerate() {
                lines.push(format!("{:0>4} {}", ip, match token {
                    Int(cid) => format!("Int({} -> {})", cid, constants[*cid as usize]),
                    Str(sid) => format!("Str({} -> {:?})", sid, strings[*sid as usize]),
                    t => format!("{:?}", t),
                }));
            }
        }
        if !parse_result.errors.is_empty() {
            lines.push(String::from("\n=== Parse Errors ===\n"));
            for error in &parse_result.errors {
                lines.push(format!("{:?}", error));
            }
            lines.push(String::from("\n=== Formatted Parse Errors ===\n"));
            let mut source: String = String::from(path);
            source.push_str(".cor");
            let src_lines: Vec<&str> = text.lines().collect();
            for error in &parse_result.errors {
                lines.push(reporting::format_parse_error(&src_lines, &source, error));
            }
        }

        trace::test::compare_test_resource_content(&root, lines);
    }
}