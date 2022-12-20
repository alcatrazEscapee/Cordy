use std::collections::{HashMap, VecDeque};

use crate::compiler::scanner::{ScanResult, ScanToken};
use crate::stdlib::StdBinding;
use crate::vm::opcode::Opcode;
use crate::trace;

use ScanToken::{*};
use ParserErrorType::{*};
use Opcode::{*};
use crate::vm::value::FunctionImpl;


pub fn parse(bindings: HashMap<&'static str, StdBinding>, scan_result: ScanResult) -> ParserResult {
    let mut parser: Parser = Parser::new(bindings, scan_result.tokens);
    parser.parse();
    ParserResult {
        code: parser.output,
        errors: parser.errors,

        strings: parser.strings,
        constants: parser.constants,
        functions: parser.functions,

        line_numbers: parser.line_numbers,
    }
}

pub struct ParserResult {
    pub code: Vec<Opcode>,
    pub errors: Vec<ParserError>,

    pub strings: Vec<String>,
    pub constants: Vec<i64>,
    pub functions: Vec<FunctionImpl>,

    pub line_numbers: Vec<u16>,
}


impl ParserResult {

    pub fn disassemble(self: &Self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut width: usize = 0;
        let mut longest: usize = (self.line_numbers.last().unwrap_or(&0) + 1) as usize;
        while longest > 0 {
            width += 1;
            longest /= 10;
        }

        let mut last_line_no: u16 = 0;
        for (ip, token) in self.code.iter().enumerate() {
            let line_no = self.line_numbers.get(ip).unwrap_or_else(|| self.line_numbers.last().unwrap());
            let label: String = if line_no + 1 != last_line_no {
                last_line_no = line_no + 1;
                format!("L{:0>width$}: ", line_no + 1, width = width)
            } else {
                " ".repeat(width + 3)
            };
            let asm: String = match token {
                Opcode::Int(cid) => format!("Int({} -> {})", cid, self.constants[*cid as usize]),
                Str(sid) => format!("Str({} -> {:?})", sid, self.strings[*sid as usize]),
                Function(fid) => format!("Function({} -> {:?})", fid, self.functions[*fid as usize]),
                t => format!("{:?}", t),
            };
            lines.push(format!("{}{:0>4} {}", label, ip % 10_000, asm));
        }
        lines
    }
}



#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ParserError {
    pub error: ParserErrorType,
    pub lineno: usize,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ParserErrorType {
    UnexpectedEoF,
    UnexpectedEofExpectingVariableNameAfterLet,
    UnexpectedEofExpectingFunctionNameAfterFn,
    UnexpectedEoFExpecting(ScanToken),
    UnexpectedTokenAfterEoF(ScanToken),

    Expecting(ScanToken, ScanToken),

    ExpectedExpressionTerminal(ScanToken),
    ExpectedCommaOrEndOfArguments(ScanToken),
    ExpectedCommaOrEndOfList(ScanToken),
    ExpectedColonOrEndOfSlice(ScanToken),
    ExpectedStatement(ScanToken),
    ExpectedVariableNameAfterLet(ScanToken),
    ExpectedFunctionNameAfterFn(ScanToken),
    ExpectedParameterOrEndOfList(ScanToken),
    ExpectedCommaOrEndOfParameters(ScanToken),

    LocalVariableConflict(String),
    UndeclaredIdentifier(String),

    AssignmentToNotVariable(String),
    BreakOutsideOfLoop,
    ContinueOutsideOfLoop,
}


struct Parser {
    bindings: HashMap<&'static str, StdBinding>,
    input: VecDeque<ScanToken>,
    output: Vec<Opcode>,
    errors: Vec<ParserError>,

    lineno: u16,
    line_numbers: Vec<u16>,

    error_recovery: bool, // If we are in error recover mode, this flag is set
    multiple_expression_statements_per_line: bool, // If a expression statement has already been parsed before a new line or ';', this flag is set

    locals: Vec<Local>, // Table of all locals, placed at the bottom of the stack
    scope_depth: u16, // Scope depth
    function_depth: u16,

    strings: Vec<String>,
    constants: Vec<i64>,
    functions: Vec<FunctionImpl>,

    // Loop stack
    // Each frame represents a single loop, which `break` and `continue` statements refer to
    // `continue` jumps back to the beginning of the loop, aka the first `usize` (loop start)
    // `break` statements jump back to the end of the loop, which needs to be patched later. The values to be patched record themselves in the stack at the current loop level
    loops: Vec<Loop>,
}

#[derive(Eq, PartialEq, Debug, Clone)]
struct Loop {
    start_index: u16,
    depth: u16,
    break_statements: Vec<u16>
}

impl Loop {
    fn new(start_index: u16, depth: u16) -> Loop {
        Loop { start_index, depth, break_statements: Vec::new() }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
struct Local {
    name: String,
    scope_depth: u16,
    function_depth: u16,
    index: u16,
    initialized: bool,
}

impl Local {
    fn new(name: String, scope_depth: u16, function_depth: u16, index: u16) -> Local {
        Local { name, scope_depth, function_depth, index, initialized: false }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
enum VariableType {
    //StructOrDerivedBinding, // this one is complicated, since it is a type based dispatch against user defined types...
    Local(u16),
    Global(u16),
    TopLevelBinding(StdBinding),
    None,
}


impl Parser {

    fn new(bindings: HashMap<&'static str, StdBinding>, tokens: Vec<ScanToken>) -> Parser {
        Parser {
            bindings,
            input: tokens.into_iter().collect::<VecDeque<ScanToken>>(),
            output: Vec::new(),
            errors: Vec::new(),

            lineno: 0,
            line_numbers: Vec::new(),

            error_recovery: false,
            multiple_expression_statements_per_line: false,

            locals: Vec::new(),
            scope_depth: 0,
            function_depth: 0,

            strings: Vec::new(),
            constants: Vec::new(),
            functions: Vec::new(),

            loops: Vec::new(),
        }
    }

    fn parse(self: &mut Self) {
        trace::trace_parser!("rule <root>");
        self.parse_statements();
        self.pop_locals_in_current_scope_depth(true); // Pop top level 'local' variables
        if let Some(t) = self.peek() {
            let token: ScanToken = t.clone();
            self.error(UnexpectedTokenAfterEoF(token));
        }
        self.push(Exit);
    }

    fn parse_statements(self: &mut Self) {
        trace::trace_parser!("rule <statements>");
        loop {
            match self.peek() {
                Some(KeywordFn) => self.parse_function_statement(),
                Some(KeywordReturn) => self.parse_return_statement(),
                Some(KeywordLet) => self.parse_let_statement(),
                Some(KeywordIf) => self.parse_if_statement(),
                Some(KeywordLoop) => self.parse_loop_statement(),
                Some(KeywordWhile) => self.parse_while_statement(),
                Some(KeywordFor) => self.parse_for(),
                Some(KeywordBreak) => self.parse_break_statement(),
                Some(KeywordContinue) => self.parse_continue_statement(),
                Some(Identifier(_)) => self.parse_assignment(),
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
        self.scope_depth += 1;
        self.multiple_expression_statements_per_line = false;
        self.parse_statements();
        self.multiple_expression_statements_per_line = false;
        self.pop_locals_in_current_scope_depth(true);
        self.scope_depth -= 1;
        self.expect_resync(CloseBrace);
    }

    fn parse_function_statement(self: &mut Self) {
        self.advance(); // Consume `fn`
        let maybe_name: Option<String> = match self.peek() {
            Some(Identifier(_)) => Some(self.take_identifier()),
            Some(t) => {
                let token: ScanToken = t.clone();
                self.error(ExpectedFunctionNameAfterFn(token));
                None // Continue parsing as we can resync on the ')' and '}'
            },
            None => {
                self.error(UnexpectedEofExpectingFunctionNameAfterFn);
                return
            }
        };

        self.expect(OpenParen);

        // Parameters
        let mut args: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                Some(Identifier(_)) => args.push(self.take_identifier()),
                Some(CloseParen) => break,
                Some(t) => {
                    let token = t.clone();
                    self.error(ExpectedParameterOrEndOfList(token));
                    break
                },
                None => break
            }

            match self.peek() {
                Some(Comma) => self.skip(),
                Some(CloseParen) => break,
                Some(t) => {
                    let token: ScanToken = t.clone();
                    self.error(ExpectedCommaOrEndOfParameters(token));
                    break
                },
                None => break
            }
        }

        self.expect_resync(CloseParen);

        if let Some(name) = maybe_name {
            // The function itself is a complicated local variable, and needs to be declared as such
            match self.declare_local(name.clone()) {
                Some(index) => {
                    self.locals[index].initialized = true;  // Functions are always initialized, as they can be recursive
                    let func_start: usize = self.next_opcode() as usize + 2; // Declare the function literal. + 2 to the head because of the leading Jump and function local
                    let func: u16 = self.declare_function(func_start, name, args.clone());
                    self.push(Function(func));  // And push the function object itself
                },
                None => {
                    // todo: error case
                }
            }
        }

        let jump: u16 = self.reserve(); // Jump over the function itself, the first time we encounter it

        // Functions have their own depth tracking in addition to scope
        // In addition, we let parameters have their own scope depth one outside locals to the function
        // This lets us 1) declare parameters here, in the right scope,
        // and 2) avoid popping parameters at the end of a function call (as they're handled by the `Return` opcode instead)
        self.function_depth += 1;
        self.scope_depth += 1;

        for arg in args {
            match self.declare_local(arg) {
                Some(index) => {
                    // They are automatically initialized, and we don't need to push `Nil` for them, since they're provided from the stack due to call semantics
                    self.locals[index].initialized = true;
                },
                None => {
                    // todo: error case, parameter conflict
                }
            }
        }

        // Slightly modified version of parse_block_statement() to handle return value injection
        trace::trace_parser!("rule <block-statement>");
        self.expect(OpenBrace);
        self.scope_depth += 1;
        self.multiple_expression_statements_per_line = false;
        self.parse_statements();

        // Handle implicit return values
        // The last expression in a function is interpreted as the return value, without an explicit `return` keyword
        // We just detect if an expression was just popped, and if so, return it
        let last_index: usize = self.output.len() - 1;
        let last: &Opcode = &self.output[last_index];
        if last == &Pop {
            // We identified a `Pop`, and as the next instruction will be `return`, we can replace it with `Noop`
            // Note that we can't replace it with `Return` directly, as there may be jumps pointing to the *next* token after this pop.
            self.output[last_index] = Noop;
        } else {
            // No final expression, so push and return `nil` instead
            self.push(Nil);
        }
        self.push(Return);

        self.multiple_expression_statements_per_line = false;
        self.pop_locals_in_current_scope_depth(true);
        self.scope_depth -= 1;

        self.pop_locals_in_current_scope_depth(false); // Pop the parameters from the parser's knowledge of locals, but don't emit Pop / PopN
        self.function_depth -= 1;
        self.scope_depth -= 1;

        self.expect_resync(CloseBrace);

        let end: u16 = self.next_opcode(); // Repair the jump
        self.output[jump as usize] = Jump(end);
    }

    fn parse_return_statement(self: &mut Self) {
        self.advance(); // Consume `return`
        match self.peek() {
            Some(CloseBrace) => { // Allow a bare return, but only when followed by a `}` or `;`, which we can recognize and discard properly.
                 self.push(Nil);
            },
            Some(Semicolon) => {
                self.push(Nil);
                self.advance();
            },
            _ => {
                // Otherwise we require an expression
                self.parse_expression();
            }
        }
        // As the VM cleans up it's own call stack properly, by discarding everything above the function's frame when exiting,
        // we don't need to clean up the call stack ourselves! And that's brilliant!
        self.push(Return);
    }

    // ===== Control Flow ===== //

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

    fn parse_while_statement(self: &mut Self) {
        trace::trace_parser!("<while-statement>");

        // Translation
        // while <expr> {    | L1: <expr> ; JumpIfFalsePop L2
        //     <statements>  | <statements>
        //     break         | Jump L2  -> Push onto loop stack, fix at end
        //     continue      | Jump L1  -> Set immediately, using the loop start value
        //     <statements>  | <statements>
        // }                 | Jump L1 ; L2:

        self.advance();

        let loop_start: u16 = self.next_opcode(); // Top of the loop, push onto the loop stack
        self.loops.push(Loop::new(loop_start, self.scope_depth));

        self.parse_expression(); // While condition
        self.push(Bound(StdBinding::Bool)); // Evaluate the condition with `<expr> . bool` automatically
        self.push(OpFuncCompose);
        let jump_if_false: u16 = self.reserve(); // Jump to the end
        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push(Jump(loop_start));

        let loop_end: u16 = self.next_opcode(); // After the jump, the next opcode is 'end of loop'. Repair all break statements
        let break_opcodes: Vec<u16> = self.loops.pop().unwrap().break_statements;
        for break_opcode in break_opcodes {
            self.output[break_opcode as usize] = Jump(loop_end);
        }

        self.output[jump_if_false as usize] = JumpIfFalsePop(loop_end); // Fix the initial conditional jump
    }

    fn parse_loop_statement(self: &mut Self) {
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
        self.loops.push(Loop::new(loop_start, self.scope_depth));

        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push(Jump(loop_start));

        let loop_end: u16 = self.next_opcode(); // After the jump, the next opcode is 'end of loop'. Repair all break statements
        let break_opcodes: Vec<u16> = self.loops.pop().unwrap().break_statements;
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
                self.pop_locals_in_current_loop();
                let jump: u16 = self.reserve();
                self.loops.last_mut().unwrap().break_statements.push(jump);
            },
            None => self.error(BreakOutsideOfLoop),
        }
    }

    fn parse_continue_statement(self: &mut Self) {
        trace::trace_parser!("rule <continue-statement>");
        self.advance();
        match self.loops.last() {
            Some(loop_stmt) => {
                let jump_to: u16 = loop_stmt.start_index;
                self.pop_locals_in_current_loop();
                self.push(Jump(jump_to));
            },
            None => self.error(ContinueOutsideOfLoop),
        }
    }

    // ===== Variables + Expressions ===== //

    fn parse_let_statement(self: &mut Self) {
        trace::trace_parser!("rule <let-statement>");
        self.advance();
        loop {
            let local: usize = match self.peek() {
                Some(Identifier(_)) => {
                    let name: String = self.take_identifier();
                    match self.declare_local(name) {
                        Some(l) => l,
                        None => break
                    }
                },
                Some(t) => {
                    let token: ScanToken = t.clone();
                    self.error(ExpectedVariableNameAfterLet(token));
                    break
                },
                None => {
                    self.error(UnexpectedEofExpectingVariableNameAfterLet);
                    break
                }
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

            // Local declarations don't have an explicit `store` opcode
            // They just push their value onto the stack, and we know the location will equal that of the Local's index
            // However, after we initialize a local we need to mark it initialized, so we can refer to it in expressions
            self.locals[local].initialized = true;

            match self.peek() {
                Some(Comma) => { // Multiple declarations
                    self.advance();
                    self.multiple_expression_statements_per_line = false;
                },
                _ => break,
            }
        }
    }

    fn parse_assignment(self: &mut Self) {
        trace::trace_parser!("rule <assignment-statement>");
        let maybe_op: Option<Opcode> = match self.peek2() {
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
            let name: String = self.take_identifier();
            let (store_op, push_op) = match self.resolve_identifier(&name) {
                VariableType::Local(local) => (StoreLocal(local), PushLocal(local)),
                VariableType::Global(local) => (StoreGlobal(local), PushGlobal(local)),
                _ => {
                    let token: String = name.clone();
                    self.error(AssignmentToNotVariable(token));
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
            // In this case, we matched <name> without a following '=' or other assignment operator.
            // This must be part of an expression statement.
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
                    self.error(ExpectedStatement(token))
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
            Some(Identifier(_)) => {
                let string: String = self.take_identifier();
                match self.resolve_identifier(&string) {
                    VariableType::Local(local) => self.push(PushLocal(local)),
                    VariableType::Global(local) => self.push(PushGlobal(local)),
                    VariableType::TopLevelBinding(b) => self.push(Bound(b)),
                    _ => self.error(UndeclaredIdentifier(string))
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
                                    self.error(ExpectedCommaOrEndOfList(token))
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
                self.error(ExpectedExpressionTerminal(token));
            },
            _ => {
                self.push(Nil);
                self.error(UnexpectedEoF)
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
                                    self.error(ExpectedCommaOrEndOfArguments(token))
                                },
                                _ => self.error(UnexpectedEoF),
                            }
                        }
                        None => self.error(UnexpectedEoF),
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
                            self.error(ExpectedColonOrEndOfSlice(token));
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
                            self.error(ExpectedColonOrEndOfSlice(token));
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

    fn declare_function(self: &mut Self, head: usize, name: String, args: Vec<String>) -> u16 {
        self.functions.push(FunctionImpl::new(head, name, args));
        (self.functions.len() - 1) as u16
    }


    fn next_opcode(self: &Self) -> u16 {
        self.output.len() as u16
    }

    /// After a `let <name>` or `fn <name>` declaration, tries to declare this as a local variable in the current scope
    /// Returns the index of the local variable in `locals` if the declaration was successful. Note that the index in `locals is **not** the same as the index used to reference the local (if it is a true local, it will be an offset due to the call stack)
    fn declare_local(self: &mut Self, name: String) -> Option<usize> {

        // Walk backwards down the locals stack, and check if there are any locals with the same name in the same scope
        for local in self.locals.iter().rev() {
            if local.scope_depth != self.scope_depth {
                break
            }
            if local.name == name {
                self.error(LocalVariableConflict(name.clone()));
                return None;
            }
        }

        // Count the current number of locals at the current function depth
        let mut local_index: u16 = 0;
        for local in &self.locals {
            if local.function_depth == self.function_depth {
                local_index += 1;
            }
        }


        // todo: do we handle binding conflicts at all?

        // Local variable is free of conflicts
        let local: Local = Local::new(name, self.scope_depth, self.function_depth, local_index);
        self.locals.push(local);
        Some(self.locals.len() - 1)
    }

    /// Pops all locals declared in the current scope, *before* decrementing the scope depth
    /// Also emits the relevant `OpPop` opcodes to the output token stream.
    fn pop_locals_in_current_scope_depth(self: &mut Self, emit_pop_token: bool) {
        let mut local_depth: u16 = 0;
        while let Some(local) = self.locals.last() {
            if local.scope_depth == self.scope_depth {
                self.locals.pop().unwrap();
                local_depth += 1;
            } else {
                break
            }
        }
        if emit_pop_token {
            self.push_pop(local_depth);
        }
    }

    /// Pops all locals that are within the most recent loop, and outputs the relevant `Pop` / `PopN` opcodes
    /// It **does not** modify the local variable stack, and only outputs these tokens for the purposes of a jump across scopes.
    fn pop_locals_in_current_loop(self: &mut Self) {
        let loop_stmt: &Loop = self.loops.last().unwrap();
        let mut local_depth: u16 = 0;

        for local in self.locals.iter().rev() {
            if local.scope_depth > loop_stmt.depth {
                local_depth += 1;
            } else {
                break
            }
        }
        self.push_pop(local_depth);
    }

    fn resolve_identifier(self: &mut Self, name: &String) -> VariableType {
        // Search for local variables in reverse order
        // Locals can be accessed in multiple ways:
        // 1. Locals (in the current call frame), with the same function depth
        // 2. Globals (relative to the stack base), with function depth == 0
        // 3. UpValues (todo)
        for local in self.locals.iter().rev() {
            if &local.name == name && local.initialized {
                if local.function_depth == 0 {
                    return VariableType::Global(local.index);
                } else if local.function_depth == self.function_depth {
                    return VariableType::Local(local.index);
                }
            }
        }

        if let Some(b) = self.bindings.get(name.as_str()) {
            return VariableType::TopLevelBinding(*b);
        }
        VariableType::None
    }


    // ===== Parser Core ===== //


    /// If the given token is present, accept it. Otherwise, flag an error and enter error recovery mode.
    fn expect(self: &mut Self, token: ScanToken) {
        match self.peek() {
            Some(t) if t == &token => {
                trace::trace_parser!("expect {:?} -> pass", token);
                self.advance();
            }
            Some(t) => {
                let t0: ScanToken = t.clone();
                self.error(Expecting(token, t0))
            },
            None => self.error(UnexpectedEoFExpecting(token)),
        }
    }

    /// Acts as a resynchronization point for error mode
    /// Accepts tokens from the input (ignoring the current state of error recovery mode), until we reach the expected token or an empty input.
    /// If we reach the expected token, it is consumed and error mode is unset.
    fn expect_resync(self: &mut Self, token: ScanToken) {
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
                    // Error recovery failed, so we need to reset it
                    self.error_recovery = true;
                    break
                }
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
            Some(Identifier(name)) => name,
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
    /// This also does not consume newline tokens in the input, rather peeks _past_ them in order to find the next matching token.
    fn peek(self: &Self) -> Option<&ScanToken> {
        if self.error_recovery {
            return None
        }
        for token in &self.input {
            if token != &NewLine {
                return Some(token)
            }
        }
        None
    }

    // Like `peek()` but peeks one ahead (making this technically a lookahead 2 parser)
    fn peek2(self: &mut Self) -> Option<&ScanToken> {
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
            }
        }
        None
    }

    /// Like `advance()` but discards the result.
    fn skip(self: &mut Self) {
        self.advance();
    }

    /// Advances and returns the next incoming token.
    /// Will also advance past any newline tokens, and so the advanced token will be the next token _after_ any newlines between the last token and the next.
    fn advance(self: &mut Self) -> Option<ScanToken> {
        if self.error_recovery {
            return None
        }
        while let Some(NewLine) = self.input.front() {
            trace::trace_parser!("newline {} at opcode {}, last = {:?}", self.lineno + 1, self.next_opcode(), self.line_numbers.last());
            self.input.pop_front();
            self.lineno += 1;
        }
        trace::trace_parser!("advance {:?}", self.input.front());
        self.input.pop_front()
    }

    /// Reserves a space in the output code by inserting a `Noop` token
    /// Returns an index to the token, which can later be used to set the correct value
    fn reserve(self: &mut Self) -> u16 {
        trace::trace_parser!("reserve at {}", self.output.len());
        self.output.push(Noop);
        (self.output.len() - 1) as u16
    }

    /// Specialization of `pop` which may push nothing, Pop, or PopN(n)
    fn push_pop(self: &mut Self, n: u16) {
        match n {
            0 => {},
            1 => self.push(Pop),
            n => self.push(PopN(n))
        }
    }

    /// Pushes a new token into the output stream.
    /// Returns the index of the token pushed, which allows callers to later mutate that token if they need to.
    fn push(self: &mut Self, token: Opcode) {
        trace::trace_parser!("push {:?}", token);
        self.output.push(token);
        self.line_numbers.push(self.lineno)
    }

    /// Pushes a new error token into the output error stream.
    fn error(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError {
                error,
                lineno: self.lineno as usize,
            });
        }
        self.error_recovery = true;
    }
}


#[cfg(test)]
mod tests {
    use crate::compiler::{parser, scanner};
    use crate::compiler::scanner::ScanResult;
    use crate::compiler::parser::{Parser, ParserResult};
    use crate::{reporting, stdlib};
    use crate::stdlib::StdBinding;
    use crate::stdlib::StdBinding::{Print, Read};
    use crate::vm::opcode::Opcode;
    use crate::trace;

    use crate::vm::opcode::Opcode::{*};


    #[test] fn test_empty_expr() { run_expr("", vec![Nil]); }
    #[test] fn test_int() { run_expr("123", vec![Int(123)]); }
    #[test] fn test_str() { run_expr("'abc'", vec![Str(0)]); }
    #[test] fn test_unary_minus() { run_expr("-3", vec![Int(3), UnarySub]); }
    #[test] fn test_binary_mul() { run_expr("3 * 6", vec![Int(3), Int(6), OpMul]); }
    #[test] fn test_binary_div() { run_expr("20 / 4 / 5", vec![Int(20), Int(4), OpDiv, Int(5), OpDiv]); }
    #[test] fn test_binary_pow() { run_expr("2 ** 10", vec![Int(2), Int(10), OpPow]); }
    #[test] fn test_binary_minus() { run_expr("6 - 7", vec![Int(6), Int(7), OpSub]); }
    #[test] fn test_binary_and_unary_minus() { run_expr("15 -- 7", vec![Int(15), Int(7), UnarySub, OpSub]); }
    #[test] fn test_binary_add_and_mod() { run_expr("1 + 2 % 3", vec![Int(1), Int(2), Int(3), OpMod, OpAdd]); }
    #[test] fn test_binary_add_and_mod_rev() { run_expr("1 % 2 + 3", vec![Int(1), Int(2), OpMod, Int(3), OpAdd]); }
    #[test] fn test_binary_shifts() { run_expr("1 << 2 >> 3", vec![Int(1), Int(2), OpLeftShift, Int(3), OpRightShift]); }
    #[test] fn test_binary_shifts_and_operators() { run_expr("1 & 2 << 3 | 5", vec![Int(1), Int(2), Int(3), OpLeftShift, OpBitwiseAnd, Int(5), OpBitwiseOr]); }
    #[test] fn test_function_composition() { run_expr("print . read", vec![Bound(Print), Bound(Read), OpFuncCompose]); }
    #[test] fn test_precedence_with_parens() { run_expr("(1 + 2) * 3", vec![Int(1), Int(2), OpAdd, Int(3), OpMul]); }
    #[test] fn test_precedence_with_parens_2() { run_expr("6 / (5 - 3)", vec![Int(6), Int(5), Int(3), OpSub, OpDiv]); }
    #[test] fn test_precedence_with_parens_3() { run_expr("-(1 - 3)", vec![Int(1), Int(3), OpSub, UnarySub]); }
    #[test] fn test_function_no_args() { run_expr("print", vec![Bound(Print)]); }
    #[test] fn test_function_one_arg() { run_expr("print(1)", vec![Bound(Print), Int(1), OpFuncEval(1)]); }
    #[test] fn test_function_many_args() { run_expr("print(1,2,3)", vec![Bound(Print), Int(1), Int(2), Int(3), OpFuncEval(3)]); }
    #[test] fn test_multiple_unary_ops() { run_expr("- ~ ! 1", vec![Int(1), UnaryLogicalNot, UnaryBitwiseNot, UnarySub]); }
    #[test] fn test_multiple_function_calls() { run_expr("print (1) (2) (3)", vec![Bound(Print), Int(1), OpFuncEval(1), Int(2), OpFuncEval(1), Int(3), OpFuncEval(1)]); }
    #[test] fn test_multiple_function_calls_some_args() { run_expr("print () (1) (2, 3)", vec![Bound(Print), OpFuncEval(0), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(2)]); }
    #[test] fn test_multiple_function_calls_no_args() { run_expr("print () () ()", vec![Bound(Print), OpFuncEval(0), OpFuncEval(0), OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence() { run_expr("- print ()", vec![Bound(Print), OpFuncEval(0), UnarySub]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens() { run_expr("(- print) ()", vec![Bound(Print), UnarySub, OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens_2() { run_expr("- (print () )", vec![Bound(Print), OpFuncEval(0), UnarySub]); }
    #[test] fn test_function_call_binary_op_precedence() { run_expr("print ( 1 ) + ( 2 ( 3 ) )", vec![Bound(Print), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(1), OpAdd]); }
    #[test] fn test_function_call_parens_1() { run_expr("print . read (1 + 3) (5)", vec![Bound(Print), Bound(Read), Int(1), Int(3), OpAdd, OpFuncEval(1), Int(5), OpFuncEval(1), OpFuncCompose]); }
    #[test] fn test_function_call_parens_2() { run_expr("( print . read (1 + 3) ) (5)", vec![Bound(Print), Bound(Read), Int(1), Int(3), OpAdd, OpFuncEval(1), OpFuncCompose, Int(5), OpFuncEval(1)]); }
    #[test] fn test_function_composition_with_is() { run_expr("'123' . int is int . print", vec![Str(0), Bound(StdBinding::Int), Bound(StdBinding::Int), OpIs, OpFuncCompose, Bound(StdBinding::Print), OpFuncCompose]); }
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

    #[test] fn test_let_eof() { run_err("let", "Unexpected end of file, was expecting variable name after 'let' keyword\n  at: line 1 (<test>)\n  at:\n\nlet\n"); }
    #[test] fn test_let_no_identifier() { run_err("let =", "Expecting a variable name after 'let' keyword, got '=' token instead\n  at: line 1 (<test>)\n  at:\n\nlet =\n"); }
    #[test] fn test_let_expression_eof() { run_err("let x =", "Unexpected end of file.\n  at: line 1 (<test>)\n  at:\n\nlet x =\n"); }
    #[test] fn test_let_no_expression() { run_err("let x = &", "Expected an expression terminal, got '&' token instead\n  at: line 1 (<test>)\n  at:\n\nlet x = &\n"); }

    #[test] fn test_break_past_locals() { run("break_past_locals"); }
    #[test] fn test_continue_past_locals() { run("continue_past_locals"); }
    #[test] fn test_empty() { run("empty"); }
    #[test] fn test_expressions() { run("expressions"); }
    #[test] fn test_function() { run("function"); }
    #[test] fn test_function_early_return() { run("function_early_return"); }
    #[test] fn test_function_early_return_nested_scope() { run("function_early_return_nested_scope"); }
    #[test] fn test_function_unspecified_return() { run("function_unspecified_return"); }
    #[test] fn test_function_with_parameters() { run("function_with_parameters"); }
    #[test] fn test_global_variables() { run("global_variables"); }
    #[test] fn test_global_assignments() { run("global_assignments"); }
    #[test] fn test_hello_world() { run("hello_world"); }
    #[test] fn test_if_statement_1() { run("if_statement_1"); }
    #[test] fn test_if_statement_2() { run("if_statement_2"); }
    #[test] fn test_if_statement_3() { run("if_statement_3"); }
    #[test] fn test_if_statement_4() { run("if_statement_4"); }
    #[test] fn test_invalid_expressions() { run("invalid_expressions"); }
    #[test] fn test_local_assignments() { run("local_assignments"); }
    #[test] fn test_local_variables() { run("local_variables"); }
    #[test] fn test_loop_1() { run("loop_1"); }
    #[test] fn test_loop_2() { run("loop_2"); }
    #[test] fn test_loop_3() { run("loop_3"); }
    #[test] fn test_loop_4() { run("loop_4"); }
    #[test] fn test_weird_locals() { run("weird_locals"); }
    #[test] fn test_while_1() { run("while_1"); }
    #[test] fn test_while_2() { run("while_2"); }
    #[test] fn test_while_3() { run("while_3"); }
    #[test] fn test_while_4() { run("while_4"); }


    fn run_expr(text: &'static str, tokens: Vec<Opcode>) {
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

    fn run_err(text: &'static str, output: &'static str) {
        let text: &String = &String::from(text);
        let text_lines: Vec<&str> = text.split("\n").collect();

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let parse_result: ParserResult = parser::parse(stdlib::bindings(), scan_result);
        let mut lines: Vec<String> = Vec::new();
        assert!(!parse_result.errors.is_empty());

        for error in &parse_result.errors {
            lines.push(reporting::format_parse_error(&text_lines, &String::from("<test>"), error));
        }

        assert_eq!(lines.join("\n"), output);
    }

    fn run(path: &'static str) {
        let root: String = trace::test::get_test_resource_path("parser", path);
        let text: String = trace::test::get_test_resource_src(&root);

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let parse_result: ParserResult = parser::parse(stdlib::bindings(), scan_result);
        let mut lines: Vec<String> = parse_result.disassemble();

        if !parse_result.errors.is_empty() {
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