use std::collections::VecDeque;

use crate::compiler::scanner::{ScanResult, ScanToken};
use crate::compiler::CompileResult;
use crate::stdlib::StdBinding;
use crate::vm::opcode::Opcode;
use crate::vm::value::FunctionImpl;
use crate::{stdlib, trace};

use ScanToken::{*};
use ParserErrorType::{*};
use Opcode::{*};
use StdBinding::{*};


pub fn parse(scan_result: ScanResult) -> CompileResult {
    let mut parser: Parser = Parser::new(scan_result.tokens);
    parser.parse();
    CompileResult {
        code: parser.output,
        errors: parser.errors,

        strings: parser.strings,
        constants: parser.constants,
        functions: parser.functions,

        line_numbers: parser.line_numbers,
        locals: parser.locals_reference,
        globals: parser.globals_reference,
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ParserError {
    pub error: ParserErrorType,
    pub lineno: usize,
}

impl ParserError {
    fn new(error: ParserErrorType, lineno: usize) -> ParserError {
        ParserError { error, lineno }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ParserErrorType {
    UnexpectedTokenAfterEoF(ScanToken),
    ExpectedToken(ScanToken, Option<ScanToken>),

    ExpectedExpressionTerminal(Option<ScanToken>),
    ExpectedCommaOrEndOfArguments(Option<ScanToken>),
    ExpectedCommaOrEndOfList(Option<ScanToken>),
    ExpectedColonOrEndOfSlice(Option<ScanToken>),
    ExpectedStatement(Option<ScanToken>),
    ExpectedVariableNameAfterLet(Option<ScanToken>),
    ExpectedVariableNameAfterFor(Option<ScanToken>),
    ExpectedFunctionNameAfterFn(Option<ScanToken>),
    ExpectedFunctionBlockOrArrowAfterFn(Option<ScanToken>),
    ExpectedParameterOrEndOfList(Option<ScanToken>),
    ExpectedCommaOrEndOfParameters(Option<ScanToken>),

    LocalVariableConflict(String),
    LocalVariableConflictWithNativeFunction(String),
    UndeclaredIdentifier(String),

    InvalidAssignmentTarget,

    BreakOutsideOfLoop,
    ContinueOutsideOfLoop,
}


struct Parser {
    input: VecDeque<ScanToken>,
    output: Vec<Opcode>,
    errors: Vec<ParserError>,

    lineno: u16,
    line_numbers: Vec<u16>,

    locals_reference: Vec<String>, // A reference for local names on a per-instruction basis, used for disassembly
    globals_reference: Vec<String>, // A reference for global names, in stack order, used for runtime errors due to invalid late bound globals

    // If we are in error recover mode, this flag is set
    error_recovery: bool,
    // If a expression statement has already been parsed before a new line or ';', this flag is set
    // This denies two unrelated expression statements on the same line, unless seperated by a token such as `;`, `{` or `}`
    prevent_expression_statement: bool,
    // We delay the last `Pop` emitted from an expression statement wherever possible
    // This allows more statement-like constructs to act like expression statements automatically
    // If this flag is `true`, then we need to emit a `Pop` or risk mangling the stack.
    delay_pop_from_expression_statement: bool,

    locals: Vec<Local>, // Table of all locals, placed at the bottom of the stack
    late_bound_globals: Vec<LateBoundGlobal>, // Table of all late bound globals, as they occur.
    synthetic_local_index: usize, // A counter for unique synthetic local variables (`$1`, `$2`, etc.)
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

    fn is_global(self: &Self) -> bool { self.function_depth == 0 }
    fn is_true_global(self: &Self) -> bool { self.function_depth == 0 && self.scope_depth == 0 }
}

#[derive(Eq, PartialEq, Debug, Clone)]
enum VariableType {
    //StructOrDerivedBinding, // this one is complicated, since it is a type based dispatch against user defined types...
    NativeFunction(StdBinding),
    Local(u16),
    Global(u16),
    TrueGlobal(u16),
    LateBoundGlobal(LateBoundGlobal),
    None,
}

#[derive(Eq, PartialEq, Debug, Clone)]
struct LateBoundGlobal {
    name: String,
    opcode: usize, // Index in output[] of the `load` opcode
    error: Option<ParserError>, // An error that would be thrown from here, if the variable does not end up bound
}

impl LateBoundGlobal {
    fn new(name: String, opcode: usize, error: Option<ParserError>) -> LateBoundGlobal {
        LateBoundGlobal { name, opcode, error }
    }
}


impl Parser {

    fn new(tokens: Vec<ScanToken>) -> Parser {
        Parser {
            input: tokens.into_iter().collect::<VecDeque<ScanToken>>(),
            output: Vec::new(),
            errors: Vec::new(),

            lineno: 0,
            line_numbers: Vec::new(),
            locals_reference: Vec::new(),
            globals_reference: Vec::new(),

            error_recovery: false,
            prevent_expression_statement: false,
            delay_pop_from_expression_statement: false,

            locals: Vec::new(),
            late_bound_globals: Vec::new(),

            synthetic_local_index: 0,
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
        self.push_delayed_pop();
        self.pop_locals_in_current_scope_depth(true); // Pop top level 'local' variables
        if let Some(t) = self.peek() {
            let token: ScanToken = t.clone();
            self.error(UnexpectedTokenAfterEoF(token));
        }
        for global in self.late_bound_globals.drain(..) {
            if let Some(error) = global.error {
                self.errors.push(error);
            }
        }
        self.push(Exit);
    }

    fn parse_statements(self: &mut Self) {
        trace::trace_parser!("rule <statements>");
        loop {
            trace::trace_parser!("rule <statement>");
            match self.peek() {
                Some(KeywordFn) => self.parse_named_function(),
                Some(KeywordReturn) => self.parse_return_statement(),
                Some(KeywordLet) => self.parse_let_statement(),
                Some(KeywordIf) => self.parse_if_statement(),
                Some(KeywordLoop) => self.parse_loop_statement(),
                Some(KeywordWhile) => self.parse_while_statement(),
                Some(KeywordFor) => self.parse_for_statement(),
                Some(KeywordBreak) => self.parse_break_statement(),
                Some(KeywordContinue) => self.parse_continue_statement(),
                Some(OpenBrace) => self.parse_block_statement(),
                Some(CloseBrace) => break, // Don't consume, but break if we're in an error mode
                Some(KeywordExit) => {
                    self.push_delayed_pop();
                    self.advance();
                    self.push(Exit);
                },
                Some(Semicolon) => {
                    self.prevent_expression_statement = false;
                    self.push_delayed_pop();
                    self.advance();
                },
                Some(_) => self.parse_expression_statement(),
                None => break,
            }
        }
    }

    fn parse_block_statement(self: &mut Self) {
        trace::trace_parser!("rule <block-statement>");
        self.push_delayed_pop();
        self.expect(OpenBrace);
        self.scope_depth += 1;
        self.prevent_expression_statement = false;
        self.parse_statements();
        self.prevent_expression_statement = false;
        self.pop_locals_in_current_scope_depth(true);
        self.scope_depth -= 1;
        self.expect_resync(CloseBrace);
    }

    fn parse_named_function(self: &mut Self) {
        trace::trace_parser!("rule <named-function>");

        // Function header - `fn <name> (<arg>, ...)
        self.push_delayed_pop();
        self.advance();
        let maybe_name: Option<String> = self.parse_function_name();
        self.expect(OpenParen);
        let args: Vec<String> = self.parse_function_parameters();
        self.expect_resync(CloseParen);

        // Named functions are a complicated local variable, and needs to be declared as such
        let mut func_id: Option<u16> = None;
        if let Some(name) = maybe_name {
            match self.declare_local(name.clone()) {
                Some(index) => {
                    self.locals[index].initialized = true;  // Functions are always initialized, as they can be recursive
                    self.push_inc_global(index);

                    let func_start: usize = self.next_opcode() as usize + 2; // Declare the function literal. + 2 to the head because of the leading Jump, Function()
                    let func: u16 = self.declare_function(func_start, name, args.clone());
                    func_id = Some(func);
                    self.push(Opcode::Function(func));  // And push the function object itself
                },
                None => {
                    // todo: error case
                }
            }
        }

        self.parse_function_body(args, func_id);
    }

    fn parse_expression_function(self: &mut Self) {
        trace::trace_parser!("rule <expression-function>");

        // Function header - `fn` (<arg>, ...)
        self.advance();
        self.expect(OpenParen);
        let args: Vec<String> = self.parse_function_parameters();
        self.expect_resync(CloseParen);

        // Expression functions don't declare themselves as a local variable that can be referenced.
        // Instead, as they're part of an expression, they just push a single function instance onto the stack
        let func_start: usize = self.next_opcode() as usize + 2; // Declare the function literal. + 2 to the head because of the leading Jump and function local
        let func: u16 = self.declare_function(func_start, String::from("_"), args.clone());
        self.push(Opcode::Function(func));  // And push the function object itself

        self.parse_function_body(args, Some(func));
    }

    fn parse_function_name(self: &mut Self) -> Option<String> {
        trace::trace_parser!("rule <function-name>");
        match self.peek() {
            Some(Identifier(_)) => Some(self.take_identifier()),
            _ => {
                self.error_with(|t| ExpectedFunctionNameAfterFn(t));
                None
            }
        }
    }

    fn parse_function_parameters(self: &mut Self) -> Vec<String> {
        trace::trace_parser!("rule <function-parameters>");
        // Parameters
        let mut args: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                Some(Identifier(_)) => args.push(self.take_identifier()),
                Some(CloseParen) => break,
                _ => {
                    self.error_with(|t| ExpectedParameterOrEndOfList(t));
                    break
                },
            }

            match self.peek() {
                Some(Comma) => self.skip(),
                Some(CloseParen) => break,
                _ => {
                    self.error_with(|t| ExpectedCommaOrEndOfParameters(t));
                    break
                },
            }
        }
        args
    }

    fn parse_function_body(self: &mut Self, args: Vec<String>, func_id: Option<u16>) {
        trace::trace_parser!("rule <function-body>");
        let jump = self.reserve(); // Jump over the function itself, the first time we encounter it
        let prev_pop_status: bool = self.delay_pop_from_expression_statement; // Stack semantics for the delayed pop

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
                    self.push_inc_global(index);
                },
                None => {
                    // todo: error case, parameter conflict
                }
            }
        }

        // Scope of the function itself
        self.scope_depth += 1;
        self.prevent_expression_statement = false;

        let is_block_function = match self.peek() {
            Some(OpenBrace) => {
                self.advance(); // Block-based function
                self.parse_statements(); // So parse statements
                if !self.delay_pop_from_expression_statement {
                    // If we haven't delayed a `Pop` (and there is a return value already on the stack), we need to push one
                    self.push(Opcode::Nil);
                }
                true
            },
            Some(Arrow) => {
                self.advance(); // Expression-based function
                self.parse_expression(); // So parse an expression
                false
            },
            _ => {
                self.error_with(|t| ExpectedFunctionBlockOrArrowAfterFn(t));
                true
            }
        };

        if is_block_function { // Expect the end of the function, so the Return opcode gets attributed on the same line as the '}'
            self.expect_resync(CloseBrace);
        }

        // Update the function, if present, with the tail of the function
        // This makes tracking ownership in containing functions easier during error reporting
        if let Some(func_id) = func_id {
            self.functions[func_id as usize].tail = self.next_opcode() as usize;
        }

        self.push(Return); // Returns the last expression in the function

        self.prevent_expression_statement = false;
        self.pop_locals_in_current_scope_depth(true);
        self.scope_depth -= 1;

        self.pop_locals_in_current_scope_depth(false); // Pop the parameters from the parser's knowledge of locals, but don't emit Pop / PopN
        self.function_depth -= 1;
        self.scope_depth -= 1;

        let end: u16 = self.next_opcode(); // Repair the jump
        self.output[jump] = Jump(end);
        self.delay_pop_from_expression_statement = prev_pop_status; // Exit the stack
    }

    fn parse_return_statement(self: &mut Self) {
        trace::trace_parser!("rule <return-statement>");
        self.push_delayed_pop();
        self.advance(); // Consume `return`
        match self.peek() {
            Some(CloseBrace) => { // Allow a bare return, but only when followed by a `}` or `;`, which we can recognize and discard properly.
                 self.push(Opcode::Nil);
            },
            Some(Semicolon) => {
                self.push(Opcode::Nil);
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
        self.push_delayed_pop();
        self.parse_expression();
        let jump_if_false = self.reserve(); // placeholder for jump to the beginning of an if branch, if it exists
        self.parse_block_statement();

        // We treat `if` statements as expressions - each branch will create an value on the stack, and we set this flag at the end of parsing the `if`
        // But, if we didn't delay a `Pop`, we need to push a fake `Nil` onto the stack to maintain the stack size
        if !self.delay_pop_from_expression_statement {
            self.delay_pop_from_expression_statement = true;
            self.push(Opcode::Nil);
        }

        // `elif` can be de-sugared to else { if <expr> { ... } else { ... } }
        // The additional scope around the else {} can be dropped as it doesn't contain anything already in a scope
        // So if we hit `elif`, we emit what we would for `else`, then recurse, then once we return, patch the final jump.
        match self.peek() {
            Some(KeywordElif) => {
                // Don't advance, as `parse_if_statement()` will advance the first token
                let jump = self.reserve();
                let after_if: u16 = self.next_opcode();
                self.output[jump_if_false] = JumpIfFalsePop(after_if);
                self.delay_pop_from_expression_statement = false;
                self.parse_if_statement();
                let after_else: u16 = self.next_opcode();
                self.output[jump] = Jump(after_else);
            },
            Some(KeywordElse) => {
                // `else` is present, so we first insert an unconditional jump, parse the next block, then fix the first jump
                self.advance();
                let jump = self.reserve();
                let after_if: u16 = self.next_opcode();
                self.output[jump_if_false as usize] = JumpIfFalsePop(after_if);
                self.delay_pop_from_expression_statement = false;
                self.parse_block_statement();
                if !self.delay_pop_from_expression_statement {
                    self.delay_pop_from_expression_statement = true;
                    self.push(Opcode::Nil);
                }
                let after_else: u16 = self.next_opcode();
                self.output[jump] = Jump(after_else);
            },
            _ => {
                // No `else`, but we need to wire in a fake `else` statement which just pushes `Nil` so each branch still pushes a value
                let jump = self.reserve();
                let after_if: u16 = self.next_opcode();
                self.output[jump_if_false as usize] = JumpIfFalsePop(after_if);
                self.push(Opcode::Nil);
                let after_else: u16 = self.next_opcode();
                self.output[jump] = Jump(after_else);
            },
        }
    }

    fn parse_while_statement(self: &mut Self) {
        trace::trace_parser!("rule <while-statement>");

        // Translation
        // while <expr> {    | L1: <expr> ; JumpIfFalsePop L2
        //     <statements>  | <statements>
        //     break         | Jump L2  -> Push onto loop stack, fix at end
        //     continue      | Jump L1  -> Set immediately, using the loop start value
        //     <statements>  | <statements>
        // }                 | Jump L1 ; L2:

        self.push_delayed_pop();
        self.advance();

        let loop_start: u16 = self.next_opcode(); // Top of the loop, push onto the loop stack
        self.loops.push(Loop::new(loop_start, self.scope_depth));

        self.parse_expression(); // While condition
        self.push(Bound(Bool)); // Evaluate the condition with `<expr> . bool` automatically
        self.push(OpFuncCompose);
        let jump_if_false = self.reserve(); // Jump to the end
        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push(Jump(loop_start));

        let loop_end: u16 = self.next_opcode(); // After the jump, the next opcode is 'end of loop'. Repair all break statements
        let break_opcodes: Vec<u16> = self.loops.pop().unwrap().break_statements;
        for break_opcode in break_opcodes {
            self.output[break_opcode as usize] = Jump(loop_end);
        }

        self.output[jump_if_false] = JumpIfFalsePop(loop_end); // Fix the initial conditional jump
    }

    fn parse_loop_statement(self: &mut Self) {
        trace::trace_parser!("rule <loop-statement>");

        // Translation:
        // loop {            | L1:
        //     <statements>  | <statements>
        //     break         | Jump L2  -> Push onto loop stack, fix at end
        //     continue      | Jump L1  -> Set immediately, using the loop start value
        //     <statements>  | <statements>
        // }                 | Jump L1 ; L2:

        self.push_delayed_pop();
        self.advance();

        let loop_start: u16 = self.next_opcode(); // Top of the loop, push onto the loop stack
        self.loops.push(Loop::new(loop_start, self.scope_depth));

        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push_delayed_pop(); // Loops can't return a value
        self.push(Jump(loop_start));

        let loop_end: u16 = self.next_opcode(); // After the jump, the next opcode is 'end of loop'. Repair all break statements
        let break_opcodes: Vec<u16> = self.loops.pop().unwrap().break_statements;
        for break_opcode in break_opcodes {
            self.output[break_opcode as usize] = Jump(loop_end);
        }
    }

    fn parse_for_statement(self: &mut Self) {
        trace::trace_parser!("rule <for-statement>");

        // `for` loops are handled in one of three ways:
        // If we detect a `for` <name> `in` `range`
        //   -> This uses direct iteration and bypasses creating a list from `range`
        //   -> In order to do this we create two synthetic variables for the `step` and `iteration_limit`
        // If we detect a `for` (<name>, <name>) `in` `enumerate`
        //   -> todo: Bypass creating synthetic variables and pattern destructuring (once we have it)
        // Otherwise, we destructure this as a C-style `for` loop.
        // In order to support iteration through arbitrary structures, but allow modification during iteration for plain lists, we emit some bridge bytecode to check the type of the iterable before converting it to a list.

        self.push_delayed_pop();
        self.advance(); // Consume `for`

        // `for` loops have declared (including synthetic) variables within their own scope
        self.scope_depth += 1;
        let local_x: Option<usize> = match self.peek() {
            Some(Identifier(_)) => {
                let name = self.take_identifier();
                self.declare_local(name)
            },
            _  => {
                self.error_with(|t| ExpectedVariableNameAfterFor(t));
                None
            }
        };

        self.expect(KeywordIn);

        match self.peek() {
            Some(Identifier(s)) if s == "range" => {
                let local_stop: usize = self.declare_synthetic_local();

                self.advance(); // Consume `range`
                self.expect(OpenParen);
                self.parse_expression(); // The first argument, places into `local_x`
                match self.peek() {
                    Some(CloseParen) => {
                        // Single argument `range()`, so we set push 0 for `local_x`, and then use simple increment (+1) and less than (< local_stop)
                        // We need to swap the `stop` value and initial value for `local_x`, which will be the result of the expression
                        self.advance(); // Consume `)`
                        let constant_0 = self.declare_constant(0);
                        self.push(Opcode::Int(constant_0)); // Initial value for `local_x`
                        self.push(Swap);

                        // Initialize the loop variable, as we're now in the main block and it can be referenced
                        if let Some(local_x) = local_x {
                            self.locals[local_x].initialized = true;
                        }

                        // Bounds check and branch to end
                        let jump = self.next_opcode();
                        self.push_load_local(local_x.unwrap_or(0));
                        self.push_load_local(local_stop);
                        self.push(OpLessThan);
                        let jump_if_false_pop = self.reserve();

                        // Parse the body of the loop and emit
                        self.parse_block_statement();
                        self.push_delayed_pop();

                        // Increment and jump to head
                        let constant_1 = self.declare_constant(1);
                        self.push_load_local(local_x.unwrap_or(0));
                        self.push(Opcode::Int(constant_1));
                        self.push(OpAdd);
                        self.push_store_local(local_x.unwrap_or(0));
                        self.push(Opcode::Pop);
                        self.push(Jump(jump));

                        // Fix the jump
                        self.output[jump_if_false_pop] = JumpIfFalsePop(self.next_opcode());

                        self.pop_locals_in_current_scope_depth(true);
                        self.scope_depth -= 1;
                        return
                    },
                    Some(Comma) => {
                        self.advance(); // Consume the `,` and continue
                    }
                    _ => self.error_with(|t| ExpectedCommaOrEndOfArguments(t)),
                }

                self.parse_expression();
                match self.peek() {
                    Some(CloseParen) => {
                        // Two argument `for x in range(start, stop)` intrinsic.
                        // Since the step is known we can use `<` as a comparison
                        // The two expressions (start, stop) have already been placed in the stack and we already declared a synthetic local for `stop`
                        self.advance();

                        // Initialize the loop variable, as we're now in the main block and it can be referenced
                        if let Some(local_x) = local_x {
                            self.locals[local_x].initialized = true;
                        }

                        // Bounds check and branch to end
                        let jump = self.next_opcode();
                        self.push_load_local(local_x.unwrap_or(0));
                        self.push_load_local(local_stop);
                        self.push(OpLessThan);
                        let jump_if_false_pop = self.reserve();

                        // Parse the body of the loop and emit
                        self.parse_block_statement();
                        self.push_delayed_pop();

                        // Increment and jump to head
                        let constant_1 = self.declare_constant(1);
                        self.push_load_local(local_x.unwrap_or(0));
                        self.push(Opcode::Int(constant_1));
                        self.push(OpAdd);
                        self.push_store_local(local_x.unwrap_or(0));
                        self.push(Opcode::Pop);
                        self.push(Jump(jump));

                        // Fix the jump
                        self.output[jump_if_false_pop] = JumpIfFalsePop(self.next_opcode());

                        self.pop_locals_in_current_scope_depth(true);
                        self.scope_depth -= 1;
                        return
                    },
                    Some(Comma) => {
                        self.advance(); // Consume `,`
                    },
                    _ => self.error_with(|t| ExpectedCommaOrEndOfArguments(t)),
                }
            },
            _ => {
                // Standard `for` loop destructuring.
                // We need synthetic variables for the iterable and the loop index. The length is checked each iteration (as it may change)
                let local_iter: usize = self.declare_synthetic_local();
                let local_i: usize = self.declare_synthetic_local();

                // Push `Nil` first, to initialize the loop variable's value
                self.push(Opcode::Nil);

                // Parse the expression, which will be the iterable (as it is declared first)
                self.parse_expression();

                // Initialize the loop variable, as we're now in the main block and it can be referenced
                // The loop variables cannot be referenced.
                if let Some(local_x) = local_x {
                    self.locals[local_x].initialized = true;
                }

                // Bytecode to check the type of the iterable - if it's a list, do nothing, otherwise, invoke `list` on it.
                self.push(Dup);
                self.push(Bound(StdBinding::List));
                self.push(OpIs);
                self.push(JumpIfTruePop(self.next_opcode() + 3));
                self.push(Bound(StdBinding::List));
                self.push(OpFuncCompose);

                // Push the loop index
                let constant_0 = self.declare_constant(0);
                self.push(Opcode::Int(constant_0));

                // Bounds check and branch
                let jump: u16 = self.next_opcode();
                self.push_load_local(local_i);
                self.push_load_local(local_iter);
                self.push(Bound(Len));
                self.push(OpFuncCompose);
                self.push(OpLessThan);
                let jump_if_false_pop = self.reserve();

                // Initialize the loop variable
                self.push_load_local(local_iter);
                self.push_load_local(local_i);
                self.push(OpIndex);
                self.push_store_local(local_x.unwrap_or(0));
                self.push(Opcode::Pop);

                // Parse the body of the loop and emit
                self.parse_block_statement();
                self.push_delayed_pop();

                // Increment and jump to head
                self.push_load_local(local_i);
                let constant_1 = self.declare_constant(1);
                self.push(Opcode::Int(constant_1));
                self.push(OpAdd);
                self.push_store_local(local_i);
                self.push(Opcode::Pop);
                self.push(Jump(jump));

                // Fix the jump
                self.output[jump_if_false_pop] = JumpIfFalsePop(self.next_opcode());

                self.pop_locals_in_current_scope_depth(true);
                self.scope_depth -= 1;
                return
            }
        }
    }

    fn parse_break_statement(self: &mut Self) {
        trace::trace_parser!("rule <break-statement>");
        self.push_delayed_pop();
        self.advance();
        match self.loops.last() {
            Some(_) => {
                self.pop_locals_in_current_loop();
                let jump = self.reserve();
                self.loops.last_mut().unwrap().break_statements.push(jump as u16);
            },
            None => self.error(BreakOutsideOfLoop),
        }
    }

    fn parse_continue_statement(self: &mut Self) {
        trace::trace_parser!("rule <continue-statement>");
        self.push_delayed_pop();
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
        self.push_delayed_pop();
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
                _ => {
                    self.error_with(|t| ExpectedVariableNameAfterLet(t));
                    break
                }
            };

            match self.peek() {
                Some(Equals) => { // x = <expr>, so parse the expression
                    self.advance();
                    self.prevent_expression_statement = true;
                    self.parse_expression();
                },
                _ => {
                    self.push(Opcode::Nil); // Initialize to 'nil'
                },
            }

            // Local declarations don't have an explicit `store` opcode
            // They just push their value onto the stack, and we know the location will equal that of the Local's index
            // However, after we initialize a local we need to mark it initialized, so we can refer to it in expressions
            self.locals[local].initialized = true;
            self.push_inc_global(local);

            match self.peek() {
                Some(Comma) => { // Multiple declarations
                    self.advance();
                    self.prevent_expression_statement = false;
                },
                _ => break,
            }
        }
    }

    fn parse_expression_statement(self: &mut Self) {
        trace::trace_parser!("rule <expression-statement>");
        if !self.prevent_expression_statement {
            self.prevent_expression_statement = true;
            self.push_delayed_pop();
            self.parse_expression();
            self.delay_pop_from_expression_statement = true;
        } else {
            self.error_with(|t| ExpectedStatement(t))
        }
    }

    fn parse_expression(self: &mut Self) {
        trace::trace_parser!("rule <expression>");
        self.parse_expr_10();
    }


    // ===== Expression Parsing ===== //

    fn parse_expr_1_terminal(self: &mut Self) {
        trace::trace_parser!("rule <expr-1>");
        match self.peek() {
            Some(KeywordNil) => self.advance_push(Opcode::Nil),
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
                    VariableType::NativeFunction(b) => self.push(Bound(b)),
                    VariableType::Local(local) => self.push(PushLocal(local)),
                    VariableType::Global(local) => self.push(PushGlobal(local, true)),
                    VariableType::TrueGlobal(local) => self.push(PushGlobal(local, false)),
                    VariableType::LateBoundGlobal(global) => {
                        self.late_bound_globals.push(global);
                        self.push(Noop) // Push `Noop` for now, fix it later
                    },
                    _ => self.error(UndeclaredIdentifier(string))
                };
            },
            Some(StringLiteral(_)) => {
                let string: String = self.take_str();
                let sid: u16 = self.declare_string(string);
                self.push(Opcode::Str(sid));
            },
            Some(OpenParen) => {
                self.advance(); // Consume the `(`
                if !self.parse_expr_1_partial_operator() {
                    self.parse_expression();
                    self.expect(CloseParen);
                }
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
                                _ => self.error_with(|t| ExpectedCommaOrEndOfList(t)),
                            }
                        },
                        None => break,
                    }
                }
                let length: u16 = self.declare_constant(length);
                self.push(Opcode::List(length));
                self.expect(CloseSquareBracket);
            },
            Some(KeywordFn) => {
                self.parse_expression_function();
            },
            _ => self.error_with(|t| ExpectedExpressionTerminal(t)),
        }
    }

    fn parse_expr_1_partial_operator(self: &mut Self) -> bool {
        trace::trace_parser!("rule <expr-1-partial-operator>");
        // Open `(` usually resolves precedence, so it begins parsing an expression from the top again
        // However it *also* can be used to wrap around a partial evaluation of a literal operator, for example
        // (-) => OperatorUnarySub
        // (+ 3) => Int(3) OperatorAdd OpFuncCompose
        // (*) => OperatorMul
        // So, if we peek ahead and see an operator, we know this is a expression of that sort and we need to handle accordingly
        // We *also* know that we will never see a binary operator begin an expression
        let mut unary: Option<StdBinding> = None;
        let mut binary: Option<StdBinding> = None;
        match self.peek() {
            // Unary operators *can* be present at the start of an expression, but we would see something after
            // So, we peek ahead again and see if the next token is a `)` - that's the only way you evaluate unary operators as functions
            Some(Minus) => unary = Some(OperatorUnarySub),
            Some(LogicalNot) => unary = Some(OperatorUnaryLogicalNot),
            Some(BitwiseNot) => unary = Some(OperatorUnaryBitwiseNot),

            Some(Mul) => binary = Some(OperatorMul),
            Some(Div) => binary = Some(OperatorDiv),
            Some(Pow) => binary = Some(OperatorPow),
            Some(Mod) => binary = Some(OperatorMod),
            Some(KeywordIs) => binary = Some(OperatorIs),
            Some(Plus) => binary = Some(OperatorAdd),
            // `-` cannot be a binary operator as it's ambiguous from a unary expression
            Some(LeftShift) => binary = Some(OperatorLeftShift),
            Some(RightShift) => binary = Some(OperatorRightShift),
            Some(BitwiseAnd) => binary = Some(OperatorBitwiseAnd),
            Some(BitwiseOr) => binary = Some(OperatorBitwiseOr),
            Some(BitwiseXor) => binary = Some(OperatorBitwiseXor),
            Some(LessThan) => binary = Some(OperatorLessThan),
            Some(LessThanEquals) => binary = Some(OperatorLessThanEqual),
            Some(GreaterThan) => binary = Some(OperatorGreaterThan),
            Some(GreaterThanEquals) => binary = Some(OperatorGreaterThanEqual),
            Some(DoubleEquals) => binary = Some(OperatorEqual),
            Some(NotEquals) => binary = Some(OperatorNotEqual),

            _ => {},
        }

        if let Some(op) = unary {
            match self.peek2() {
                Some(CloseParen) => {
                    self.advance(); // The unary operator
                    self.push(Bound(op)); // Push the binding - there is no partial evaluation so that's all we need
                    self.advance(); // The close paren
                    true
                },
                _ => false
            }
        } else if let Some(op) = binary {
            self.advance(); // The unary operator
            match self.peek() {
                Some(CloseParen) => {
                    // No expression follows this operator, so we have `(` <op> `)`, which just returns the operator itself
                    self.push(Bound(op));
                    self.advance();
                },
                _ => {
                    // Anything else, and we try and parse an expression and partially evaluate
                    self.parse_expression(); // Parse the expression following a binary prefix operator
                    self.push(Bound(op)); // Push the binding
                    self.push(OpFuncCompose); // And partially evaluate it
                    self.expect(CloseParen);
                }
            }
            true
        } else {
            false
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
                    self.advance();
                    match self.peek() {
                        Some(CloseParen) => {
                            self.advance();
                            self.push(OpFuncEval(0));
                        },
                        Some(_) => {
                            // Special case: if we can parse a partially-evaluated operator expression, we do so
                            // This means we can write `map (+3)` instead of `map ((+3))`
                            // If we parse something here, this will have consumed everything, otherwise we parse an expression as per usual
                            if self.parse_expr_1_partial_operator() {
                                // We still need to treat this as a function evaluation, however, because we pretended we didn't need to see the outer parenthesis
                                self.push(OpFuncEval(1));
                            } else {
                                // First argument
                                self.parse_expression();
                                let mut count: u8 = 1;

                                // Other arguments
                                while let Some(Comma) = self.peek() {
                                    self.advance();
                                    self.parse_expression();
                                    count += 1;
                                }
                                match self.peek() {
                                    Some(CloseParen) => self.skip(),
                                    _ => self.error_with(|t| ExpectedCommaOrEndOfArguments(t)),
                                }

                                // Evaluate the number of arguments we parsed
                                self.push(OpFuncEval(count));
                            }
                        }
                        _ => self.error_with(|t| ExpectedCommaOrEndOfArguments(t)),
                    }
                },
                Some(OpenSquareBracket) => {
                    self.advance(); // Consume the square bracket

                    // Consumed `[` so far
                    match self.peek() {
                        Some(Colon) => { // No first argument, so push zero. Don't consume the colon as it's the seperator
                            self.push(Opcode::Nil);
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
                        _ => {
                            self.error_with(|t| ExpectedColonOrEndOfSlice(t));
                            continue
                        },
                    }

                    // Consumed `[` <expression> `:` so far
                    match self.peek() {
                        Some(Colon) => { // No second argument, but we have a third argument, so push -1. Don't consume the colon as it's the seperator
                            self.push(Opcode::Nil);
                        },
                        Some(CloseSquareBracket) => { // No second argument, so push -1 and then exit
                            self.push(Opcode::Nil);
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
                        _ => {
                            self.error_with(|t| ExpectedColonOrEndOfSlice(t));
                            continue
                        },
                    }

                    // Consumed `[` <expression> `:` <expression> `:` so far
                    match self.peek() {
                        Some(CloseSquareBracket) => { // Three arguments + default value for slice
                            self.push(Opcode::Nil);
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
                    let jump_if_false = self.reserve();
                    self.push(Opcode::Pop);
                    self.parse_expr_8();
                    let jump_to: u16 = self.next_opcode();
                    self.output[jump_if_false] = JumpIfFalse(jump_to);
                },
                Some(LogicalOr) => {
                    self.advance();
                    let jump_if_true = self.reserve();
                    self.push(Opcode::Pop);
                    self.parse_expr_8();
                    let jump_to: u16 = self.next_opcode();
                    self.output[jump_if_true] = JumpIfTrue(jump_to);
                },
                _ => break
            }
        }
    }

    fn parse_expr_10(self: &mut Self) {
        trace::trace_parser!("rule <expr-10>");
        self.parse_expr_9();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(Equals) => Some(OpEqual), // Fake operator
                Some(PlusEquals) => Some(OpAdd), // Assignment Operators
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

            // We have to handle the left hand side, as we may need to rewrite the most recent tokens based on what we just parsed.
            // Valid LHS expressions for a direct assignment, and their translations are:
            // PushLocal(a)                => pop, and emit a StoreLocal(a) instead
            // <expr> <expr> OpIndex       => pop, and emit a StoreArray instead
            // <expr> <expr> OpPropertyGet => pop, and emit a StoreProperty instead
            // If we have a assignment-expression operator, like `+=`, then we need to do it slightly differently
            // PushLocal(a)                => parse <expr>, and emit <op>, StoreLocal(a)
            // <expr> <expr> OpIndex       => pop, emit OpIndexPeek, parse <expr>, then emit <op>, StoreArray
            // <expr> <expr> OpPropertyGet => pop, emit OpPropertyGetPeek, parse <expr>, then emit <op>, StoreProperty
            //
            // **Note**: Assignments are right-associative, so call <expr-10> recursively instead of <expr-9>
            if let Some(OpEqual) = maybe_op { // // Direct assignment statement
                self.advance();
                let last: usize = self.last();
                match self.output[last] {
                    PushLocal(id) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreLocal(id));
                    },
                    PushGlobal(id, local) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreGlobal(id, local));
                    },
                    OpIndex => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreArray);
                    },
                    // todo: property access
                    _ => self.error(InvalidAssignmentTarget),
                }
            } else if let Some(op) = maybe_op {  // Assignment Expression
                self.advance();
                let last: usize = self.last();
                match self.output[last] {
                    PushLocal(id) => {
                        self.parse_expr_10();
                        self.push(op);
                        self.push(StoreLocal(id));
                    },
                    PushGlobal(id, is_local) => {
                        self.parse_expr_10();
                        self.push(op);
                        self.push(StoreGlobal(id, is_local));
                    },
                    OpIndex => {
                        self.pop();
                        self.push(OpIndexPeek);
                        self.parse_expr_10();
                        self.push(op);
                        self.push(StoreArray);
                    },
                    // todo: property access
                    _ => self.error(InvalidAssignmentTarget),
                }
            } else {
                // Not any kind of assignment statement
                break
            }
        }
    }


    // ===== Semantic Analysis ===== //

    fn declare_string(self: &mut Self, str: String) -> u16 {
        if let Some(id) = self.strings.iter().position(|s| s == &str) {
            return id as u16
        }
        self.strings.push(str);
        (self.strings.len() - 1) as u16
    }

    fn declare_constant(self: &mut Self, int: i64) -> u16 {
        if let Some(id) = self.constants.iter().position(|i| *i == int) {
            return id as u16
        }
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

        // Lookup the name as a binding - if it is, it will be denied as we don't allow shadowing global native functions
        if let Some(_) = stdlib::lookup_named_binding(&name) {
            self.error(LocalVariableConflictWithNativeFunction(name.clone()));
            return None;
        }

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

        let local: usize = self.declare_local_internal(name);

        if self.function_depth == 0 && self.scope_depth == 0 {

            let name: String = self.locals[local].name.clone();
            let local_index: u16 = self.locals[local].index;

            // Fix references to this global
            for global in &self.late_bound_globals {
                if global.name == name {
                    self.output[global.opcode] = PushGlobal(local_index, false);
                }
            }

            // And remove them
            self.late_bound_globals.retain(|global| global.name != name);

            // Declare this global variable's name
            self.globals_reference.push(name);
        }

        Some(local)
    }

    /// Declares a synthetic local variable. Unlike `declare_local()`, this can never fail.
    /// Returns the index of the local variable in `locals`.
    fn declare_synthetic_local(self: &mut Self) -> usize {
        self.synthetic_local_index += 1;
        self.declare_local_internal(format!("${}", self.synthetic_local_index - 1))
    }

    fn declare_local_internal(self: &mut Self, name: String) -> usize {
        // Count the current number of locals at the current function depth
        let mut local_index: u16 = 0;
        for local in &self.locals {
            if local.function_depth == self.function_depth {
                local_index += 1;
            }
        }

        let local: Local = Local::new(name, self.scope_depth, self.function_depth, local_index);
        self.locals.push(local);
        self.locals.len() - 1
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

    fn resolve_local(self: &Self, local_id: u16, global: bool) -> &Local {
        let mut local_index: u16 = 0;
        let target_depth: u16 = if global { 0 } else { self.function_depth };
        for local in &self.locals {
            if local.function_depth == target_depth {
                if local_id == local_index {
                    return local
                }
                local_index += 1;
            }
        }
        panic!("Could not find a local {} in locals {:?}", local_id, self.locals);
    }

    /// Resolve an identifier, which can be one of many things, each of which are tried in-order
    ///
    /// 1. Native Functions. These cannot be shadowed (as it creates interesting conflict scenarios), and they are all technically global functions.
    /// 2. Locals (in the current call frame), with the same function depth. Locals at a higher scope are shadowed (hidden) by locals in a deeper scope.
    /// 3. Globals (relative to the origin of the stack, with function depth == 0 and stack depth == 0)
    /// 4. Late Bound Globals:
    ///     - If we are in a >0 function depth, we *do* allow late binding globals.
    ///     - Note: we have to use a special opcode which checks if the global actually exists first. *And*, we need to fix it later if the global does not end up being bound by the end of compilation.
    fn resolve_identifier(self: &mut Self, name: &String) -> VariableType {
        if let Some(b) = stdlib::lookup_named_binding(name) {
            return VariableType::NativeFunction(b);
        }

        for local in self.locals.iter().rev() {
            if &local.name == name && local.initialized {
                // Globals lie at function depth 0, and are referencable if the global is at scope 0 (and we're in a function), or the global is at any scope (and we're not in a function)
                if local.function_depth == 0 && (self.function_depth == 0 || local.scope_depth == 0) {
                    return if local.scope_depth == 0 {
                        VariableType::TrueGlobal(local.index)
                    } else {
                        VariableType::Global(local.index)
                    }
                }
                // Locals lie at any scope, as long as they're in the same function depth
                // The exception is that if the local and current function depth is zero, they are caught as global variables instead.
                if local.function_depth == self.function_depth {
                    return VariableType::Local(local.index);
                }
            }
        }

        if self.function_depth > 0 {
            // Assume a late bound global
            let error = self.deferred_error(UndeclaredIdentifier(name.clone()));
            let global = LateBoundGlobal::new(name.clone(), self.next_opcode() as usize, error);
            return VariableType::LateBoundGlobal(global);
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
            },
            _ => self.error_with(move |t| ExpectedToken(token, t)),
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
    fn peek(self: &mut Self) -> Option<&ScanToken> {
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
            } else {
                self.prevent_expression_statement = false;
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
            self.prevent_expression_statement = false;
        }
        trace::trace_parser!("advance {:?}", self.input.front());
        self.input.pop_front()
    }

    /// Reserves a space in the output code by inserting a `Noop` token
    /// Returns an index to the token, which can later be used to set the correct value
    fn reserve(self: &mut Self) -> usize {
        trace::trace_parser!("reserve at {}", self.output.len());
        self.output.push(Noop);
        self.line_numbers.push(self.lineno);
        self.output.len() - 1
    }

    /// If we previously delayed a `Pop` opcode from being omitted, push it now and reset the flag
    fn push_delayed_pop(self: &mut Self) {
        if self.delay_pop_from_expression_statement {
            trace::trace_parser!("push Pop (delayed)");
            self.push(Opcode::Pop);
            self.delay_pop_from_expression_statement = false;
        }
    }

    /// Specialization of `push` which may push nothing, Pop, or PopN(n)
    fn push_pop(self: &mut Self, n: u16) {
        trace::trace_parser!("push Pop/PopN {}", n);
        match n {
            0 => {},
            1 => self.push(Opcode::Pop),
            n => self.push(PopN(n))
        }
    }

    /// Specialization of `push` which pushes either `PushLocal` or `PushGlobal` for a given local
    fn push_load_local(self: &mut Self, local: usize) {
        trace::trace_parser!("push PushLocal/PushGlobal {}", local);
        let is_true_global: bool = self.locals[local].is_true_global();
        match self.locals[local].is_global() {
            true => self.push(PushGlobal(local as u16, !is_true_global)),
            _ => self.push(PushLocal(local as u16)),
        }
    }

    /// Specialization of `push` which pushes either `StoreLocal` or `StoreGlobal` for a given local
    fn push_store_local(self: &mut Self, local: usize) {
        trace::trace_parser!("push StoreLocal/StoreGlobal {}", local);
        let is_true_global: bool = self.locals[local].is_true_global();
        match self.locals[local].is_global() {
            true => self.push(StoreGlobal(local as u16, !is_true_global)),
            _ => self.push(StoreLocal(local as u16)),
        }
    }

    /// Specialization of `push` which pushes a `IncGlobalCount` if we need to, for the given local
    /// Called immediately after a local `initialized` is set to `true
    fn push_inc_global(self: &mut Self, local: usize) {
        if self.locals[local].is_true_global() {
            self.push(IncGlobalCount);
        }
    }

    /// Pushes a new token into the output stream.
    /// Returns the index of the token pushed, which allows callers to later mutate that token if they need to.
    fn push(self: &mut Self, token: Opcode) {
        trace::trace_parser!("push {:?} at L{:?}", token, self.lineno + 1);
        match &token {
            PushGlobal(id, _) | StoreGlobal(id, _) => {
                let local = self.resolve_local(*id, true);
                self.locals_reference.push(local.name.clone());
            },
            PushLocal(id) | StoreLocal(id) => {
                let local = self.resolve_local(*id, false);
                self.locals_reference.push(local.name.clone());
            },
            _ => {},
        }
        self.output.push(token);
        self.line_numbers.push(self.lineno);
    }

    /// Pops the last emitted token
    fn pop(self: &mut Self) {
        match self.output.pop().unwrap() {
            PushGlobal(_, _) | StoreGlobal(_, _) | PushLocal(_) | StoreLocal(_) => {
                self.locals_reference.pop();
            },
            _ => {},
        };
    }

    /// Returns the index of the last token that was just pushed.
    fn last(self: &Self) -> usize {
        self.output.len() - 1
    }

    /// A specialization of `error()` which provides the last token (the result of `peek()`) to the provided error function
    /// This avoids ugly borrow checker issues where `match self.peek() { ... t => self.error(Error(t)) }` does not work, despite the semantics being identical.
    fn error_with<F : FnOnce(Option<ScanToken>) -> ParserErrorType>(self: &mut Self, error: F) {
        let token = self.peek().cloned();
        self.error(error(token));
    }

    /// Pushes a new error token into the output error stream.
    fn error(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError::new(error, self.lineno as usize));
        }
        self.error_recovery = true;
    }

    /// Creates an optional error, which will be deferred until later to be emitted
    fn deferred_error(self: &Self, error: ParserErrorType) -> Option<ParserError> {
        if self.error_recovery {
            None
        } else {
            Some(ParserError::new(error, self.lineno as usize))
        }
    }
}


#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use crate::compiler::{parser, CompileResult, scanner};
    use crate::compiler::scanner::ScanResult;
    use crate::compiler::parser::Parser;
    use crate::reporting;
    use crate::stdlib::StdBinding;
    use crate::vm::opcode::Opcode;
    use crate::trace;

    use StdBinding::{Print, Read, OperatorAdd, OperatorDiv, OperatorMul};
    use Opcode::{*};


    #[test] fn test_empty_expr() { run_expr("", vec![]); }
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
    #[test] fn test_function_composition_with_is() { run_expr("'123' . int is int . print", vec![Str(0), Bound(StdBinding::Int), Bound(StdBinding::Int), OpIs, OpFuncCompose, Bound(Print), OpFuncCompose]); }
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
    #[test] fn test_binary_ops() { run_expr("(*) * (+) + (/)", vec![Bound(OperatorMul), Bound(OperatorAdd), OpMul, Bound(OperatorDiv), OpAdd]); }

    #[test] fn test_let_eof() { run_err("let", "Expecting a variable name after 'let' keyword, got end of input instead\n  at: line 1 (<test>)\n  at:\n\nlet\n"); }
    #[test] fn test_let_no_identifier() { run_err("let =", "Expecting a variable name after 'let' keyword, got '=' token instead\n  at: line 1 (<test>)\n  at:\n\nlet =\n"); }
    #[test] fn test_let_expression_eof() { run_err("let x =", "Expected an expression terminal, got end of input instead\n  at: line 1 (<test>)\n  at:\n\nlet x =\n"); }
    #[test] fn test_let_no_expression() { run_err("let x = &", "Expected an expression terminal, got '&' token instead\n  at: line 1 (<test>)\n  at:\n\nlet x = &\n"); }

    #[test] fn test_break_past_locals() { run("break_past_locals"); }
    #[test] fn test_constants() { run("constants"); }
    #[test] fn test_continue_past_locals() { run("continue_past_locals"); }
    #[test] fn test_empty() { run("empty"); }
    #[test] fn test_expressions() { run("expressions"); }
    #[test] fn test_for_intrinsic_range_start_stop() { run("for_intrinsic_range_start_stop"); }
    #[test] fn test_for_intrinsic_range_stop() { run("for_intrinsic_range_stop"); }
    #[test] fn test_for_no_intrinsic() { run("for_no_intrinsic"); }
    #[test] fn test_function() { run("function"); }
    #[test] fn test_function_early_return() { run("function_early_return"); }
    #[test] fn test_function_early_return_nested_scope() { run("function_early_return_nested_scope"); }
    #[test] fn test_function_implicit_return() { run("function_implicit_return"); }
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
    #[test] fn test_weird_expression_statements() { run("weird_expression_statements"); }
    #[test] fn test_weird_locals() { run("weird_locals"); }
    #[test] fn test_while_1() { run("while_1"); }
    #[test] fn test_while_2() { run("while_2"); }
    #[test] fn test_while_3() { run("while_3"); }
    #[test] fn test_while_4() { run("while_4"); }


    fn run_expr(text: &'static str, tokens: Vec<Opcode>) {
        let result: ScanResult = scanner::scan(&String::from(text));
        assert!(result.errors.is_empty());

        let mut parser: Parser = Parser::new(result.tokens);
        parser.parse_expression();
        assert_eq!(None, parser.peek());

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

        assert_eq!(output, tokens);
    }

    fn run_err(text: &'static str, output: &'static str) {
        let text: &String = &String::from(text);
        let text_lines: Vec<&str> = text.split("\n").collect();

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let parse_result: CompileResult = parser::parse(scan_result);
        let mut lines: Vec<String> = Vec::new();
        assert!(!parse_result.errors.is_empty());

        for error in &parse_result.errors {
            lines.push(reporting::format_parse_error(&text_lines, &String::from("<test>"), error));
        }

        assert_eq!(lines.join("\n"), output);
    }

    fn run(path: &'static str) {
        let root: PathBuf = trace::test::get_test_resource_path("parser", path);
        let text: String = trace::test::get_test_resource_src(&root);

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let parse_result: CompileResult = parser::parse(scan_result);
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