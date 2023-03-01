use std::collections::VecDeque;
use std::rc::Rc;

use crate::compiler::CompileResult;
use crate::compiler::parser::core::ParserState;
use crate::compiler::parser::semantic::{LateBoundGlobal, LValue, LValueReference, MaybeFunction, Reference};
use crate::compiler::scanner::{ScanResult, ScanToken};
use crate::stdlib::NativeFunction;
use crate::trace;
use crate::vm::{FunctionImpl, Opcode};

pub use crate::compiler::parser::errors::{ParserError, ParserErrorType};
pub use crate::compiler::parser::semantic::Locals;

use NativeFunction::{*};
use Opcode::{*};
use ParserErrorType::{*};
use ScanToken::{*};
use crate::reporting::{Location, Locations};

pub const RULE_REPL: ParseRule = |mut parser| parser.parse_incremental_repl();
pub const RULE_EVAL: ParseRule = |mut parser| parser.parse_incremental_eval();

pub type ParseRule = fn(Parser) -> ();
pub type Functions = Vec<MaybeFunction>;

mod core;
mod errors;
mod semantic;


/// Create a default empty `CompileResult`. This is semantically equivalent to parsing an empty program, but will output nothing.
pub fn default() -> CompileResult {
    parse_rule(vec![], |_| ())
}


/// Parse a complete `CompileResult` from the given `ScanResult`
pub fn parse(scan_result: ScanResult) -> CompileResult {
    parse_rule(scan_result.tokens, |mut parser| parser.parse())
}


pub fn parse_incremental(scan_result: ScanResult, code: &mut Vec<Opcode>, locals: &mut Vec<Locals>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, locations: &mut Locations, globals: &mut Vec<String>, rule: fn(Parser) -> ()) -> Vec<ParserError> {

    let mut errors: Vec<ParserError> = Vec::new();
    let mut maybe_functions: Functions = functions.drain(..).map(|u| MaybeFunction::wrap(&u)).collect();

    rule(Parser::new(scan_result.tokens, code, locals, &mut errors, strings, constants, &mut maybe_functions, locations, &mut Vec::new(), globals));

    for func in maybe_functions {
        functions.push(func.unwrap());
    }

    errors
}


fn parse_rule(tokens: Vec<(Location, ScanToken)>, rule: fn(Parser) -> ()) -> CompileResult {

    let mut code: Vec<Opcode> = Vec::new();
    let mut errors: Vec<ParserError> = Vec::new();

    let mut strings: Vec<String> = vec![String::new()];
    let mut constants: Vec<i64> = vec![0, 1];
    let mut functions: Functions = Vec::new();

    let mut locations: Locations = Vec::new();
    let mut globals: Vec<String> = Vec::new();
    let mut locals: Vec<String> = Vec::new();

    rule(Parser::new(tokens, &mut code, &mut Locals::empty(), &mut errors, &mut strings, &mut constants, &mut functions, &mut locations, &mut locals, &mut globals));

    CompileResult {
        code,
        errors,

        strings,
        constants,
        functions: functions.into_iter().map(MaybeFunction::unwrap).collect::<Vec<Rc<FunctionImpl>>>(),

        locations,
        locals,
        globals,
    }
}


pub struct Parser<'a> {
    input: VecDeque<(Location, ScanToken)>,

    /// Previous output, from invocations of the parser are taken as input here
    /// Output for this invocation of the parser is accumulated in `output`, and in the `code` field of `functions`.
    /// It is then baked, emitting into `raw_output` and `locations`
    raw_output: &'a mut Vec<Opcode>,
    output: Vec<(Location, Opcode)>,
    errors: &'a mut Vec<ParserError>,

    /// A 1-1 mapping of the output tokens to their location
    locations: &'a mut Locations,
    last_location: Option<Location>,

    locals_reference: &'a mut Vec<String>, // A reference for local names on a per-instruction basis, used for disassembly
    globals_reference: &'a mut Vec<String>, // A reference for global names, in stack order, used for runtime errors due to invalid late bound globals

    /// If we are in error recover mode, this flag is set
    error_recovery: bool,
    /// If a expression statement has already been parsed before a new line or ';', this flag is set
    /// This denies two unrelated expression statements on the same line, unless seperated by a token such as `;`, `{` or `}`
    prevent_expression_statement: bool,
    /// We delay the last `Pop` emitted from an expression statement wherever possible
    /// This allows more statement-like constructs to act like expression statements automatically
    /// If this flag is `true`, then we need to emit a `Pop` or risk mangling the stack.
    delay_pop_from_expression_statement: bool,

    /// A restore state for when backtracking rejects it's attempt.
    restore_state: Option<ParserState>,

    /// A stack of nested functions, each of which have their own table of locals.
    /// While this mirrors the call stack it may not be representative. The only thing we can assume is that when a function is declared, all locals in the enclosing function are accessible.
    locals: &'a mut Vec<Locals>,

    late_bound_globals: Vec<Reference<LateBoundGlobal>>, // Table of all late bound globals, as they occur.
    synthetic_local_index: usize, // A counter for unique synthetic local variables (`$1`, `$2`, etc.)
    scope_depth: u32, // Current scope depth
    function_depth: u32,

    strings: &'a mut Vec<String>,
    constants: &'a mut Vec<i64>,

    /// List of all functions known to the parser.
    /// Note that functions can be in two forms - a 'baked' representation which is held by the VM, and an 'unbaked' representation which is held by the parser.
    /// At the teardown stage, all functions are 'baked', by emitting their bytecode to the central output token stream, and fixing the reference to the function.
    functions: &'a mut Vec<MaybeFunction>,
}


impl Parser<'_> {

    fn new<'a, 'b : 'a>(tokens: Vec<(Location, ScanToken)>, output: &'b mut Vec<Opcode>, locals: &'b mut Vec<Locals>, errors: &'b mut Vec<ParserError>, strings: &'b mut Vec<String>, constants: &'b mut Vec<i64>, functions: &'b mut Functions, locations: &'b mut Locations, locals_reference: &'b mut Vec<String>, globals_reference: &'b mut Vec<String>) -> Parser<'a> {
        Parser {
            input: tokens.into_iter().collect::<VecDeque<(Location, ScanToken)>>(),
            raw_output: output,
            output: Vec::new(),
            errors,

            locations,
            last_location: None,

            locals_reference,
            globals_reference,

            error_recovery: false,
            prevent_expression_statement: false,
            delay_pop_from_expression_statement: false,
            restore_state: None,

            locals,
            late_bound_globals: Vec::new(),

            synthetic_local_index: 0,
            scope_depth: 0,
            function_depth: 0,

            strings,
            constants,
            functions
        }
    }

    fn parse(self: &mut Self) {
        trace::trace_parser!("rule <root>");
        self.parse_statements();
        self.push_delayed_pop();
        self.pop_locals(None, true, true, true); // Pop top level 'local' variables
        self.push(Exit);
        self.teardown();
    }

    fn parse_incremental_repl(self: &mut Self) {
        trace::trace_parser!("rule <root-incremental>");
        self.parse_statements();
        if self.delay_pop_from_expression_statement {
            self.push(NativeFunction(Print));
            self.push(Swap);
            self.push(OpFuncEval(1));
            self.push(Opcode::Pop);
        }
        // Don't pop locals
        self.push(Yield);
        self.teardown();
    }

    fn parse_incremental_eval(self: &mut Self) {
        self.parse_expression();
        self.push(Return); // Insert a `Return` at the end, to return out of `eval`'s frame
        self.teardown();
    }

    fn teardown(self: &mut Self) {
        // Emit code from output -> (raw_output, locations)
        for (loc, op) in self.output.drain(..) {
            self.raw_output.push(op);
            self.locations.push(loc);
        }

        // Emit functions
        for func in self.functions.iter_mut() {
            match func {
                MaybeFunction::Unbaked(parser_func) => {
                    let head = self.raw_output.len();
                    for (loc, op) in parser_func.code.drain(..) {
                        self.raw_output.push(op);
                        self.locations.push(loc);
                    }
                    let tail = self.raw_output.len() - 1;
                    func.bake(head, tail);
                },
                _ => {},
            }
        }

        if let Some(t) = self.peek() {
            let token: ScanToken = t.clone();
            self.error(UnexpectedTokenAfterEoF(token));
        }
        for global in self.late_bound_globals.drain(..) {
            if let Some(error) = global.into().error {
                self.errors.push(error);
            }
        }
    }

    fn parse_statements(self: &mut Self) {
        trace::trace_parser!("rule <statements>");
        loop {
            trace::trace_parser!("rule <statement>");
            match self.peek() {
                Some(At) => self.parse_annotated_named_function(),
                Some(KeywordFn) => self.parse_named_function(),
                Some(KeywordReturn) => self.parse_return_statement(),
                Some(KeywordLet) => self.parse_let_statement(),
                Some(KeywordIf) => self.parse_if_statement(),
                Some(KeywordLoop) => self.parse_loop_statement(),
                Some(KeywordWhile) => self.parse_while_statement(),
                Some(KeywordFor) => self.parse_for_statement(),
                Some(KeywordBreak) => self.parse_break_statement(),
                Some(KeywordContinue) => self.parse_continue_statement(),
                Some(KeywordDo) => {
                    // For now, this is just a bridge (almost undocumented) keyword, in order to trigger a scoped block
                    // Maybe later we can expand it to a `do {} while` loop, but probably not.
                    self.advance();
                    self.parse_block_statement();
                }
                Some(CloseBrace) => break,
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
        self.pop_locals(Some(self.scope_depth), true, true, true);
        self.scope_depth -= 1;
        self.expect_resync(CloseBrace);
    }

    fn parse_annotated_named_function(self: &mut Self) {
        trace::trace_parser!("rule <annotated-named-function");

        self.push_delayed_pop();
        self.advance(); // Consume `@`
        self.parse_expression(); // The annotation body
        match self.peek() {
            Some(At) => self.parse_annotated_named_function(),
            Some(KeywordFn) => self.parse_named_function(),
            _ => self.error_with(|t| ExpectedAnnotationOrNamedFunction(t)),
        }
        self.push(OpFuncEval(1)) // Evaluate the annotation
    }

    fn parse_named_function(self: &mut Self) {
        trace::trace_parser!("rule <named-function>");

        // Function header - `fn <name> (<arg>, ...)
        self.push_delayed_pop();
        self.advance();
        let maybe_name: Option<String> = self.parse_function_name();
        self.expect(OpenParen);
        let args: Vec<LValue> = self.parse_function_parameters();
        self.expect_resync(CloseParen);

        // Named functions are a complicated local variable, and needs to be declared as such
        if let Some(name) = maybe_name {
            if let Some(index) = self.declare_local(name.clone()) {
                self.init_local(index);
                let func: u32 = self.declare_function(name, &args);
                self.push(Opcode::Function(func));  // And push the function object itself
            }
        }

        self.parse_function_body(args);
    }

    fn parse_annotated_expression_function(self: &mut Self) {
        trace::trace_parser!("rule <annotated-expression-function");

        self.push_delayed_pop();
        self.advance(); // Consume `@`
        self.parse_expression(); // The annotation body
        match self.peek() {
            Some(At) => self.parse_annotated_expression_function(),
            Some(KeywordFn) => self.parse_expression_function(),
            _ => self.error_with(|t| ExpectedAnnotationOrAnonymousFunction(t)),
        }
        self.push(OpFuncEval(1)) // Evaluate the annotation
    }

    fn parse_expression_function(self: &mut Self) {
        trace::trace_parser!("rule <expression-function>");

        // Function header - `fn` (<arg>, ...)
        self.advance();
        self.expect(OpenParen);
        let args: Vec<LValue> = self.parse_function_parameters();
        self.expect_resync(CloseParen);

        // Expression functions don't declare themselves as a local variable that can be referenced.
        // Instead, as they're part of an expression, they just push a single function instance onto the stack
        let func: u32 = self.declare_function(String::from("_"), &args);
        self.push(Opcode::Function(func));  // And push the function object itself

        self.parse_function_body(args);
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

    fn parse_function_parameters(self: &mut Self) -> Vec<LValue> {
        trace::trace_parser!("rule <function-parameters>");
        let mut args: Vec<LValue> = Vec::new();
        if let Some(CloseParen) = self.peek() {
            return args; // Special case for no parameters, don't enter the loop
        }

        loop {
            match self.parse_lvalue() {
                Some(lvalue) => args.push(lvalue),
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

    fn parse_function_body(self: &mut Self, args: Vec<LValue>) {
        trace::trace_parser!("rule <function-body>");
        let prev_pop_status: bool = self.delay_pop_from_expression_statement; // Stack semantics for the delayed pop

        // Functions have their own depth tracking in addition to scope
        // In addition, we let parameters have their own scope depth one outside locals to the function
        // This lets us 1) declare parameters here, in the right scope,
        // and 2) avoid popping parameters at the end of a function call (as they're handled by the `Return` opcode instead)
        self.locals.push(Locals::new(Some(self.functions.len() - 1)));
        self.function_depth += 1;
        self.scope_depth += 1;

        // Collect arguments into pairs of the lvalue, and associated synthetic
        let mut args_with_synthetics: Vec<(LValue, Option<usize>)> = args.into_iter()
            .map(|mut arg| {
                let local = arg.declare_single_local(self);
                (arg, local)
            })
            .collect::<Vec<(LValue, Option<usize>)>>();

        // Declare pattern locals as locals immediately after arguments.
        // Once declared, initialize all (referencable) locals - so not synthetics.
        for (arg, _) in &mut args_with_synthetics {
            arg.declare_pattern_locals(self);
            arg.initialize_locals(self);
        }

        // Push initial values **only for pattern locals**
        for (arg, synthetic) in &mut args_with_synthetics {
            if synthetic.is_some() {
                arg.emit_default_values(self, false);
            }
        }

        // Destructure locals, if necessary
        for (arg, synthetic) in args_with_synthetics {
            if let Some(local) = synthetic {
                self.push(PushLocal(local as u32)); // Push the iterable to be destructured onto the stack
                arg.emit_destructuring(self, false, false); // Emit destructuring
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
                    self.push(Nil);
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

        // Since `Return` cleans up all function locals, we just discard them from the parser without emitting any `Pop` tokens.
        // But, we still need to do this first, as we need to ensure `LiftUpValue` opcodes are still emitted before the `Return`
        // We do this twice, once for function locals, and once for function parameters (since they live in their own scope)
        self.prevent_expression_statement = false;
        self.pop_locals(Some(self.scope_depth), true, false, true);
        self.scope_depth -= 1;

        self.pop_locals(Some(self.scope_depth), true, false, true);

        self.push(Return); // Must come before we pop locals

        self.locals.pop().unwrap();
        self.function_depth -= 1;
        self.scope_depth -= 1;

        // If this function has captured any upvalues, we need to emit the correct tokens for them now, including wrapping the function in a closure
        if !self.current_locals().upvalues.is_empty() {
            self.push(Closure);
            let it = self.current_locals().upvalues
                .iter()
                .map(|upvalue| if upvalue.is_local { CloseLocal(upvalue.index) } else { CloseUpValue(upvalue.index) })
                .collect::<Vec<Opcode>>();
            for op in it {
                self.push(op);
            }
        }

        self.delay_pop_from_expression_statement = prev_pop_status; // Exit the stack
    }

    fn parse_return_statement(self: &mut Self) {
        trace::trace_parser!("rule <return-statement>");
        self.push_delayed_pop();
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
        // the only thing we need to do here is make sure we emit `LiftUpValue` opcodes.
        self.pop_locals(None, false, false, true);
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

        if let Some(KeywordThen) = self.peek() {
            // If we see a top-level `if <expression> then`, we want to consider this an expression, with a top level `if-then-else` statement
            // So here, we shortcut into expression parsing.
            self.parse_expr_1_inline_if_then_else(false);
            return;
        }


        let jump_if_false = self.reserve(); // placeholder for jump to the beginning of an if branch, if it exists
        self.parse_block_statement();

        // We treat `if` statements as expressions - each branch will create an value on the stack, and we set this flag at the end of parsing the `if`
        // But, if we didn't delay a `Pop`, we need to push a fake `Nil` onto the stack to maintain the stack size
        if !self.delay_pop_from_expression_statement {
            self.delay_pop_from_expression_statement = true;
            self.push(Nil);
        }

        // `elif` can be de-sugared to else { if <expr> { ... } else { ... } }
        // The additional scope around the else {} can be dropped as it doesn't contain anything already in a scope
        // So if we hit `elif`, we emit what we would for `else`, then recurse, then once we return, patch the final jump.
        match self.peek() {
            Some(KeywordElif) => {
                // Don't advance, as `parse_if_statement()` will advance the first token
                let jump = self.reserve();
                self.fix_jump(jump_if_false, JumpIfFalsePop);
                self.delay_pop_from_expression_statement = false;
                self.parse_if_statement();
                self.fix_jump(jump, Jump);
            },
            Some(KeywordElse) => {
                // `else` is present, so we first insert an unconditional jump, parse the next block, then fix the first jump
                self.advance();
                let jump = self.reserve();
                self.fix_jump(jump_if_false, JumpIfFalsePop);
                self.delay_pop_from_expression_statement = false;
                self.parse_block_statement();
                if !self.delay_pop_from_expression_statement {
                    self.delay_pop_from_expression_statement = true;
                    self.push(Nil);
                }
                self.fix_jump(jump, Jump);
            },
            _ => {
                // No `else`, but we need to wire in a fake `else` statement which just pushes `Nil` so each branch still pushes a value
                let jump = self.reserve();
                self.fix_jump(jump_if_false, JumpIfFalsePop);
                self.push(Nil);
                self.fix_jump(jump, Jump);
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

        let jump: usize = self.begin_loop();

        self.parse_expression(); // While condition
        let jump_if_false = self.reserve(); // Jump to the end
        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push_delayed_pop(); // Inner loop expressions cannot yield out of the loop
        self.push_jump(jump, Jump);
        self.fix_jump(jump_if_false, JumpIfFalsePop); // Fix the initial conditional jump

        if let Some(KeywordElse) = self.peek() { // Parse `while {} else {}`
            self.advance();
            self.parse_block_statement();
            self.push_delayed_pop();
        }

        self.end_loop();
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

        let jump: usize = self.begin_loop(); // Top of the loop

        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push_delayed_pop(); // Loops can't return a value
        self.push_jump(jump, Jump);

        self.end_loop();
    }

    fn parse_for_statement(self: &mut Self) {
        trace::trace_parser!("rule <for-statement>");

        self.push_delayed_pop();
        self.advance(); // Consume `for`

        // `for` loops have declared (including synthetic) variables within their own scope
        // the variable binding of a `for` can also support pattern expressions.
        self.scope_depth += 1;
        let mut lvalue: LValue = self.parse_bare_lvalue().unwrap_or_default();

        self.expect(KeywordIn);

        lvalue.declare_locals(self);
        lvalue.emit_default_values(self, false);

        self.parse_expression();

        lvalue.initialize_locals(self);

        // At the beginning and end of the loop, this local sits on the top of the stack.
        // We still need to declare it's stack slot space, even though we don't reference it via load/store local opcodes
        self.declare_synthetic_local();

        // Applies to the top of the stack
        self.push(InitIterable);

        // Test
        let jump: usize = self.begin_loop();
        self.push(TestIterable);
        let jump_if_false_pop = self.reserve();

        // Initialize locals
        lvalue.emit_destructuring(self, false, false);

        // Parse the body of the loop, and emit the delayed pop - the stack is restored to the same state as the top of the loop.
        // So, we jump to the top of the loop, where we test/increment
        self.parse_block_statement();
        self.push_delayed_pop();

        // We want the variables declared in a `for` loop to be somewhat unique - if they get captured, we want them to be closed over each iteration of the loop
        // This effectively means there's a new heap-allocated variable for each iteration of the loop.
        // In order to do this, we just need to emit the proper `LiftUpValue` opcodes each iteration of the loop
        self.pop_locals(Some(self.scope_depth), false, false, true);

        self.push_jump(jump, Jump);

        // Fix the jump
        self.fix_jump(jump_if_false_pop, JumpIfFalsePop);

        // Cleanup the `for` loop locals, but don't emit lifts as we do them per-iteration.
        self.pop_locals(Some(self.scope_depth), true, true, false);
        self.scope_depth -= 1;

        if let Some(KeywordElse) = self.peek() { // Parse `for {} else {}`
            self.advance();
            self.parse_block_statement();
            self.push_delayed_pop();
        }

        self.end_loop();
    }

    fn parse_break_statement(self: &mut Self) {
        trace::trace_parser!("rule <break-statement>");
        self.push_delayed_pop();
        self.advance();
        match self.current_locals().loops.last() {
            Some(loop_stmt) => {
                self.pop_locals(Some(loop_stmt.scope_depth + 1), false, true, true);
                let jump = self.reserve();
                self.current_locals_mut().loops.last_mut().unwrap().break_statements.push(jump);
            },
            None => self.semantic_error(BreakOutsideOfLoop),
        }
    }

    fn parse_continue_statement(self: &mut Self) {
        trace::trace_parser!("rule <continue-statement>");
        self.push_delayed_pop();
        self.advance();
        match self.current_locals().loops.last() {
            Some(loop_stmt) => {
                let jump_to: usize = loop_stmt.start_index;
                self.pop_locals(Some(loop_stmt.scope_depth + 1), false, true, true);
                self.push_jump(jump_to, Jump);
            },
            None => self.semantic_error(ContinueOutsideOfLoop),
        }
    }

    // ===== Variables + Expressions ===== //

    fn parse_let_statement(self: &mut Self) {
        trace::trace_parser!("rule <let-statement>");
        self.push_delayed_pop();
        self.advance(); // Consume `let`

        loop {
            match self.parse_bare_lvalue() {
                Some(mut lvalue) => {
                    match self.peek() {
                        Some(Equals) => {
                            // `let` <lvalue> `=` <expression>
                            self.advance(); // Consume `=`
                            lvalue.declare_locals(self); // Declare local variables first, so we prepare local indices
                            lvalue.emit_default_values(self, true); // Then emit `Nil`
                            self.parse_expression(); // So the expression ends up on top of the stack
                            lvalue.initialize_locals(self); // Initialize them, before we emit store opcodes, but after the expression is parsed.
                            lvalue.emit_destructuring(self, true, false); // Emit destructuring to assign to all locals
                        },
                        _ => {
                            // `let` <lvalue>
                            lvalue.declare_locals(self);
                            lvalue.initialize_locals(self);
                            lvalue.emit_default_values(self, false); // `in_place = false` causes us to emit *all* `Nil` values, which is what we want.
                        },
                    }

                    match self.peek() {
                        Some(Comma) => {
                            self.advance(); // Consume `,`
                            self.prevent_expression_statement = false;
                        },
                        _ => break,
                    }
                },
                None => break,
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

    /// Returns an `LValue`, or `None`. If this returns `None`, an error will already have been raised.
    fn parse_lvalue(self: &mut Self) -> Option<LValue> {
        trace::trace_parser!("rule <lvalue>");

        match self.peek() {
            Some(Identifier(_)) => {
                let name = self.take_identifier();
                Some(LValue::Named(LValueReference::Named(name)))
            },
            Some(Underscore) => {
                self.advance();
                Some(LValue::Empty)
            },
            Some(Mul) => {
                self.advance();
                match self.peek() {
                    Some(Identifier(_)) => {
                        let name = self.take_identifier();
                        Some(LValue::VarNamed(LValueReference::Named(name)))
                    },
                    Some(Underscore) => {
                        self.advance();
                        Some(LValue::VarEmpty)
                    },
                    _ => {
                        self.error_with(|t| ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(t));
                        None
                    }
                }
            },
            Some(OpenParen) => {
                self.advance();
                let terms = self.parse_bare_lvalue();
                self.expect(CloseParen);
                terms
            },
            _ => {
                self.error_with(|t| ExpectedUnderscoreOrVariableNameOrPattern(t));
                None
            }
        }
    }

    /// Parses a 'bare' `LValue`, so a `LValue::Terms` without any surrounding `,`. Promotes the top level `LValue::Terms` into a single term, if possible.
    /// If `None` is returned, an error will have already been raised.
    fn parse_bare_lvalue(self: &mut Self) -> Option<LValue> {
        trace::trace_parser!("rule <bare-lvalue>");

        let mut terms: Vec<LValue> = Vec::new();
        let mut found_variadic_term: bool = false;
        loop {
            match self.parse_lvalue() {
                Some(lvalue) => {
                    if lvalue.is_variadic_term() {
                        if found_variadic_term {
                            self.semantic_error(MultipleVariadicTermsInPattern);
                        } else {
                            found_variadic_term = true;
                        }
                    }
                    terms.push(lvalue)
                },
                None => return None,
            }

            match self.peek() {
                Some(Comma) => self.skip(), // Expect more terms
                _ => return Some({
                    if terms.len() == 1 {
                        match &terms[0] {
                            LValue::Terms(_) => LValue::Terms(terms),
                            _ => terms.into_iter().next().unwrap()
                        }
                    } else {
                        LValue::Terms(terms)
                    }
                }),
            }
        }
    }


    // ===== Expression Parsing ===== //

    fn parse_expr_1_terminal(self: &mut Self) {
        trace::trace_parser!("rule <expr-1>");
        match self.peek() {
            Some(KeywordNil) => self.push_advance(Nil),
            Some(KeywordTrue) => self.push_advance(True),
            Some(KeywordFalse) => self.push_advance(False),
            Some(KeywordExit) => self.push_advance(Exit),
            Some(ScanToken::Int(_)) => {
                let int: i64 = self.take_int();
                let cid: u32 = self.declare_constant(int);
                self.push(Opcode::Int(cid));
            },
            Some(Identifier(_)) => {
                let name: String = self.take_identifier();
                let lvalue: LValueReference = self.resolve_identifier(name);
                self.push_load_lvalue(lvalue);
            },
            Some(StringLiteral(_)) => {
                let string: String = self.take_str();
                let sid: u32 = self.declare_string(string);
                self.push(Opcode::Str(sid));
            },
            Some(OpenParen) => {
                self.advance(); // Consume the `(`
                if self.parse_expr_1_partial_operator_left() {
                    return // Looks ahead, and parses <op> <expr> `)`
                }
                self.parse_expression(); // Parse <expr>
                if self.parse_expr_1_partial_operator_right() {
                    return // Looks ahead and parses <op> `)`
                }
                match self.peek() {
                    Some(Comma) => self.parse_expr_1_vector_literal(),
                    _ => self.expect(CloseParen), // Simple expression
                }
            },
            Some(OpenSquareBracket) => self.parse_expr_1_list_literal(),
            Some(OpenBrace) => self.parse_expr_1_dict_or_set_literal(),
            Some(At) => self.parse_annotated_expression_function(),
            Some(KeywordFn) => self.parse_expression_function(),
            Some(KeywordIf) => self.parse_expr_1_inline_if_then_else(true),
            _ => self.error_with(|t| ExpectedExpressionTerminal(t)),
        }
    }

    fn parse_expr_1_partial_operator_left(self: &mut Self) -> bool {
        trace::trace_parser!("rule <expr-1-partial-operator-left>");
        // Open `(` usually resolves precedence, so it begins parsing an expression from the top again
        // However it *also* can be used to wrap around a partial evaluation of a literal operator, for example
        // (-)   => OperatorUnarySub
        // (+ 3) => Int(3) OperatorAdd OpFuncEval
        // (*)   => OperatorMul
        // So, if we peek ahead and see an operator, we know this is a expression of that sort and we need to handle accordingly
        // We *also* know that we will never see a binary operator begin an expression
        let mut unary: Option<NativeFunction> = None;
        let mut binary: Option<NativeFunction> = None;
        match self.peek() {
            // Unary operators *can* be present at the start of an expression, but we would see something after
            // So, we peek ahead again and see if the next token is a `)` - that's the only way you evaluate unary operators as functions
            Some(Minus) => unary = Some(OperatorUnarySub),
            Some(Not) => unary = Some(OperatorUnaryNot),

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
                    self.push(NativeFunction(op)); // Push the binding - there is no partial evaluation so that's all we need
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
                    self.push(NativeFunction(op));
                    self.advance();
                },
                _ => {
                    // Anything else, and we try and parse an expression and partially evaluate.
                    // Note that this is *right* partial evaluation, i.e. `(< 3)` -> we evaluate the *second* argument of `<`
                    // This actually means we need to transform the operator if it is asymmetric, to one which looks identical, but is actually the operator in reverse.
                    let op = match op {
                        OperatorDiv => OperatorDivSwap,
                        OperatorPow => OperatorPowSwap,
                        OperatorMod => OperatorModSwap,
                        OperatorIs => OperatorIsSwap,
                        OperatorLeftShift => OperatorLeftShiftSwap,
                        OperatorRightShift => OperatorRightShiftSwap,
                        OperatorLessThan => OperatorLessThanSwap,
                        OperatorLessThanEqual => OperatorLessThanEqualSwap,
                        OperatorGreaterThan => OperatorGreaterThanSwap,
                        OperatorGreaterThanEqual => OperatorGreaterThanEqualSwap,
                        op => op,
                    };
                    self.push(NativeFunction(op)); // Push the operator
                    self.parse_expression(); // Parse the expression following a binary prefix operator
                    self.push(OpFuncEval(1)); // And partially evaluate it
                    self.expect(CloseParen);
                }
            }
            true
        } else {
            false
        }
    }

    fn parse_expr_1_partial_operator_right(self: &mut Self) -> bool {
        trace::trace_parser!("rule <expr-1-partial-operator-right>");
        // If we see the pattern of BinaryOp `)`, then we recognize this as a partial operator, but evaluated from the left hand side instead.
        // For non-symmetric operators, this means we use the normal operator as we partial evaluate the left hand argument (by having the operator on the right)
        let op = match self.peek() {
            Some(Mul) => Some(OperatorMul),
            Some(Div) => Some(OperatorDiv),
            Some(Pow) => Some(OperatorPow),
            Some(Mod) => Some(OperatorMod),
            Some(KeywordIs) => Some(OperatorIs),
            Some(Plus) => Some(OperatorAdd),
            // `-` cannot be a binary operator as it's ambiguous from a unary expression
            Some(LeftShift) => Some(OperatorLeftShift),
            Some(RightShift) => Some(OperatorRightShift),
            Some(BitwiseAnd) => Some(OperatorBitwiseAnd),
            Some(BitwiseOr) => Some(OperatorBitwiseOr),
            Some(BitwiseXor) => Some(OperatorBitwiseXor),
            Some(LessThan) => Some(OperatorLessThan),
            Some(LessThanEquals) => Some(OperatorLessThanEqual),
            Some(GreaterThan) => Some(OperatorGreaterThan),
            Some(GreaterThanEquals) => Some(OperatorGreaterThanEqual),
            Some(DoubleEquals) => Some(OperatorEqual),
            Some(NotEquals) => Some(OperatorNotEqual),

            _ => None,
        };

        match (op, self.peek2()) {
            (Some(op), Some(CloseParen)) => {
                self.advance(); // The operator
                self.push(NativeFunction(op)); // Push the operator
                self.push(Swap);
                self.push(OpFuncEval(1)); // And partially evaluate it
                self.expect(CloseParen);
                true
            },
            _ => false,
        }
    }

    fn parse_expr_1_vector_literal(self: &mut Self) {
        trace::trace_parser!("rule <expr-1-vector-literal>");

        // Vector literal `(` <expr> `,` [ <expr> [ `,` <expr> ] * ] ? `)`
        // Note that this might be a single element vector literal, which includes a trailing comma
        self.advance(); // Consume `,`
        match self.peek() {
            Some(CloseParen) => {
                self.advance(); // Consume `)`
                self.push(Opcode::Vector(1));
                return
            },
            _ => {},
        }

        // Otherwise, it must be > 1
        let mut length = 1;
        loop {
            self.parse_expression();
            length += 1;
            match self.peek() {
                Some(CloseParen) => {
                    // Exit
                    self.advance(); // Consume `)`
                    let constant_len = self.declare_constant(length);
                    self.push(Opcode::Vector(constant_len));
                    break
                }
                Some(Comma) => {
                    self.advance();
                    match self.peek() { // Check again, to allow trailing comma
                        Some(CloseParen) => {
                            self.advance();
                            let constant_len = self.declare_constant(length);
                            self.push(Opcode::Vector(constant_len));
                            break
                        },
                        _ => {},
                    }
                },
                _ => {
                    self.error_with(|t| ExpectedCommaOrEndOfVector(t));
                    break
                }
            }
        }
    }

    fn parse_expr_1_list_literal(self: &mut Self) {
        trace::trace_parser!("rule <expr-1-list-literal>");
        self.advance(); // Consume `[`
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
        let length: u32 = self.declare_constant(length);
        self.push(Opcode::List(length));
        self.expect(CloseSquareBracket);
    }

    fn parse_expr_1_dict_or_set_literal(self: &mut Self) {
        trace::trace_parser!("rule <expr-1-dict-or-set-literal>");
        self.advance(); // Consume `{`

        // As dict and set literals both start the same, we initially parse empty `{}` as a empty dict, and use the first argument to determine if it is a dict or set
        match self.peek() {
            Some(CloseBrace) => {
                // Just `{}` is a empty dict, not a set
                self.advance();
                self.push(Opcode::Dict(0));
                return
            },
            _ => {}, // Otherwise, continue parsing to determine if it is a dict or set
        }

        self.parse_expression();
        let is_dict: bool = match self.peek() {
            Some(Colon) => {
                // Found `{ <expr> `:`, so it must be a dict. Parse the first value, then continue as such
                self.advance();
                self.parse_expression();
                true
            },
            _ => false,
        };

        match self.peek() {
            Some(Comma) => self.skip(), // Skip a comma, so we're either at end or next value to get into the loop below
            _ => {},
        }

        let mut length: i64 = 1;
        loop {
            match self.peek() {
                Some(CloseBrace) => break,
                Some(_) => {
                    self.parse_expression();
                    if is_dict {
                        self.expect(Colon);
                        self.parse_expression();
                    }
                    length += 1;
                    match self.peek() {
                        Some(Comma) => self.skip(), // Allow trailing commas, as this loops to the top again
                        Some(CloseBrace) => {}, // Skip
                        _ => if is_dict {
                            self.error_with(|t| ExpectedCommaOrEndOfDict(t))
                        } else {
                            self.error_with(|t| ExpectedCommaOrEndOfSet(t))
                        },
                    }
                },
                None => break,
            }
        }
        let length: u32 = self.declare_constant(length);
        self.push(if is_dict { Opcode::Dict(length) } else { Opcode::Set(length) });
        self.expect(CloseBrace);
    }

    /// If `parse_prefix` is `false`, this will not parse the leading `if <condition>`
    fn parse_expr_1_inline_if_then_else(self: &mut Self, parse_prefix: bool) {
        trace::trace_parser!("rule <expr-1-inline-if-then-else>");

        if parse_prefix {
            self.advance(); // Consume `if`
            self.parse_expression(); // condition
        }
        let jump_if_false_pop = self.reserve();
        self.expect(KeywordThen);
        self.parse_expression(); // Value if true
        let jump = self.reserve();
        self.fix_jump(jump_if_false_pop, JumpIfFalsePop);
        self.expect(KeywordElse);
        self.parse_expression(); // Value if false
        self.fix_jump(jump, Jump);
    }

    fn parse_expr_2_unary(self: &mut Self) {
        trace::trace_parser!("rule <expr-2>");
        // Prefix operators
        let mut stack: Vec<Opcode> = Vec::new();
        loop {
            let maybe_op: Option<Opcode> = match self.peek() {
                Some(Minus) => Some(UnarySub),
                Some(Not) => Some(UnaryNot),
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
            // The opening token of a suffix operator must be on the same line
            match self.peek_no_newline() {
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
                            if self.parse_expr_1_partial_operator_left() {
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
                        _ => {
                            self.error_with(|t| ExpectedColonOrEndOfSlice(t));
                            continue
                        },
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
                        _ => {
                            self.error_with(|t| ExpectedColonOrEndOfSlice(t));
                            continue
                        },
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
                Some(KeywordIn) => Some(OpIn),
                Some(KeywordNot) => match self.peek2() {
                    Some(KeywordIn) => Some(UnaryNot), // Special flag for `not in`, which desugars to `!(x in y)
                    _ => None,
                }
                _ => None
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(UnaryNot) => {
                    let loc = self.advance_with() | self.advance_with();
                    self.parse_expr_2_unary();
                    self.push_with(OpIn, loc);
                    self.push(UnaryNot)
                }
                Some(op) => {
                    let loc = self.advance_with();
                    self.parse_expr_2_unary();
                    self.push_with(op, loc);
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
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    self.parse_expr_3();
                    self.push_with(op, loc);
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
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    self.parse_expr_4();
                    self.push_with(op, loc);
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
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    self.parse_expr_5();
                    self.push_with(op, loc);
                },
                None => break
            }
        }
    }

    fn parse_expr_7(self: &mut Self) {
        trace::trace_parser!("rule <expr-7>");
        self.parse_expr_6();
        loop {
            match self.peek() {
                Some(Dot) => {
                    let mut rhs = self.advance_with();
                    rhs |= self.with_location(|p| p.parse_expr_6());
                    self.push(Swap);
                    self.push_with(OpFuncEval(1), rhs);
                },
                _ => break
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
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    self.parse_expr_7();
                    self.push_with(op, loc);
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
                Some(LogicalAnd) => Some(OpBitwiseAnd), // Just markers
                Some(LogicalOr) => Some(OpBitwiseOr),
                _ => None,
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(OpBitwiseAnd) => {
                    self.advance();
                    let jump_if_false = self.reserve();
                    self.push(Opcode::Pop);
                    self.parse_expr_8();
                    self.fix_jump(jump_if_false, JumpIfFalse)
                },
                Some(OpBitwiseOr) => {
                    self.advance();
                    let jump_if_true = self.reserve();
                    self.push(Opcode::Pop);
                    self.parse_expr_8();
                    self.fix_jump(jump_if_true, JumpIfTrue);
                },
                _ => break
            }
        }
    }

    fn parse_expr_10(self: &mut Self) {
        trace::trace_parser!("rule <expr-10>");
        self.parse_expr_10_pattern_lvalue();
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

                // `.=` is special, as it needs to emit `Swap`, then `OpFuncEval(1)`
                Some(DotEquals) => Some(Swap),

                // Special assignment operators, use their own version of a binary operator
                // Also need to consume the extra token
                Some(Identifier(it)) if it == "max" => match self.peek2() {
                    Some(Equals) => {
                        self.advance();
                        Some(OpMax)
                    },
                    _ => None,
                },
                Some(Identifier(it)) if it == "min" => match self.peek2() {
                    Some(Equals) => {
                        self.advance();
                        Some(OpMin)
                    },
                    _ => None,
                },
                _ => None
            };

            // We have to handle the left hand side, as we may need to rewrite the most recent tokens based on what we just parsed.
            // Valid LHS expressions for a direct assignment, and their translations are:
            // PushLocal(a)                => pop, and emit a StoreLocal(a) instead
            // PushUpValue(a)              => pop, and emit a StoreUpValue(a) instead
            // <expr> <expr> OpIndex       => pop, and emit a StoreArray instead
            // <expr> <expr> OpPropertyGet => pop, and emit a StoreProperty instead
            // If we have a assignment-expression operator, like `+=`, then we need to do it slightly differently
            // PushLocal(a)                => parse <expr>, and emit <op>, StoreLocal(a)
            // PushUpValue(a)              => parse <expr>, and emit <op>, StoreUpValue(a)
            // <expr> <expr> OpIndex       => pop, emit OpIndexPeek, parse <expr>, then emit <op>, StoreArray
            // <expr> <expr> OpPropertyGet => pop, emit OpPropertyGetPeek, parse <expr>, then emit <op>, StoreProperty
            //
            // **Note**: Assignments are right-associative, so call <expr-10> recursively instead of <expr-9>
            if let Some(OpEqual) = maybe_op { // // Direct assignment statement
                self.advance();
                match self.last() {
                    Some(PushLocal(id)) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreLocal(id));
                    },
                    Some(PushUpValue(id)) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreUpValue(id));
                    },
                    Some(PushGlobal(id)) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreGlobal(id));
                    },
                    Some(OpIndex) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreArray);
                    },
                    // todo: property access
                    _ => self.error(InvalidAssignmentTarget),
                }
            } else if let Some(op) = maybe_op {

                self.advance();
                match self.last() {
                    Some(PushLocal(id)) => {
                        self.parse_expr_10();
                        match op {
                            Swap => {
                                self.push(op);
                                self.push(OpFuncEval(1));
                            }
                            op => self.push(op)
                        }
                        self.push(StoreLocal(id));
                    },
                    Some(PushUpValue(id)) => {
                        self.parse_expr_10();
                        match op {
                            Swap => {
                                self.push(op);
                                self.push(OpFuncEval(1));
                            }
                            op => self.push(op)
                        }
                        self.push(StoreUpValue(id));
                    },
                    Some(PushGlobal(id)) => {
                        self.parse_expr_10();
                        match op {
                            Swap => {
                                self.push(op);
                                self.push(OpFuncEval(1));
                            }
                            op => self.push(op)
                        }
                        self.push(StoreGlobal(id));
                    },
                    Some(OpIndex) => {
                        self.pop();
                        self.push(OpIndexPeek);
                        self.parse_expr_10();
                        match op {
                            Swap => {
                                self.push(op);
                                self.push(OpFuncEval(1));
                            }
                            op => self.push(op)
                        }
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

    fn parse_expr_10_pattern_lvalue(self: &mut Self) {
        // A subset of `<lvalue> = <rvalue>` get parsed as patterns. These are for non-trivial <lvalue>s, which cannot involve array or property access, and cannot involve operator-assignment statements.
        // We use backtracking as for these cases, we want to parse the `lvalue` separately.

        trace::trace_parser!("rule <expr-10-pattern-lvalue>");
        if self.begin() {
            if let Some(mut lvalue) = self.parse_bare_lvalue() {
                if !lvalue.is_named() {
                    if let Some(Equals) = self.peek() {
                        // At this point, we know enough that this is the only possible parse
                        // We have a nontrivial `<lvalue>`, followed by an assignment statement, so this must be a pattern assignment.
                        self.advance(); // Accept `=`
                        self.accept(); // First and foremost, accept the query.
                        lvalue.resolve_locals(self); // Resolve each local, raising an error if need be.
                        self.parse_expr_10(); // Recursively parse, since this is left associative, call <expr-10>
                        lvalue.emit_destructuring(self, false, true); // Emit the destructuring code, which cleans up everything
                        return; // And exit this rule
                    }
                }
            }
        }
        self.reject();
        self.parse_expr_9();
    }
}


#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::compiler::{CompileResult, parser, scanner};
    use crate::compiler::scanner::ScanResult;
    use crate::reporting::SourceView;
    use crate::stdlib::NativeFunction;
    use crate::trace;
    use crate::vm::Opcode;

    use NativeFunction::{OperatorAdd, OperatorDiv, OperatorMul, Print, ReadText};
    use Opcode::{*};

    #[test] fn test_int() { run_expr("123", vec![Int(123)]); }
    #[test] fn test_str() { run_expr("'abc'", vec![Str(1)]); }
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
    #[test] fn test_function_composition() { run_expr("print . read_text", vec![NativeFunction(Print), NativeFunction(ReadText), Swap, OpFuncEval(1)]); }
    #[test] fn test_precedence_with_parens() { run_expr("(1 + 2) * 3", vec![Int(1), Int(2), OpAdd, Int(3), OpMul]); }
    #[test] fn test_precedence_with_parens_2() { run_expr("6 / (5 - 3)", vec![Int(6), Int(5), Int(3), OpSub, OpDiv]); }
    #[test] fn test_precedence_with_parens_3() { run_expr("-(1 - 3)", vec![Int(1), Int(3), OpSub, UnarySub]); }
    #[test] fn test_function_no_args() { run_expr("print", vec![NativeFunction(Print)]); }
    #[test] fn test_function_one_arg() { run_expr("print(1)", vec![NativeFunction(Print), Int(1), OpFuncEval(1)]); }
    #[test] fn test_function_many_args() { run_expr("print(1,2,3)", vec![NativeFunction(Print), Int(1), Int(2), Int(3), OpFuncEval(3)]); }
    #[test] fn test_multiple_unary_ops() { run_expr("- ! 1", vec![Int(1), UnaryNot, UnarySub]); }
    #[test] fn test_multiple_function_calls() { run_expr("print (1) (2) (3)", vec![NativeFunction(Print), Int(1), OpFuncEval(1), Int(2), OpFuncEval(1), Int(3), OpFuncEval(1)]); }
    #[test] fn test_multiple_function_calls_some_args() { run_expr("print () (1) (2, 3)", vec![NativeFunction(Print), OpFuncEval(0), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(2)]); }
    #[test] fn test_multiple_function_calls_no_args() { run_expr("print () () ()", vec![NativeFunction(Print), OpFuncEval(0), OpFuncEval(0), OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence() { run_expr("- print ()", vec![NativeFunction(Print), OpFuncEval(0), UnarySub]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens() { run_expr("(- print) ()", vec![NativeFunction(Print), UnarySub, OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens_2() { run_expr("- (print () )", vec![NativeFunction(Print), OpFuncEval(0), UnarySub]); }
    #[test] fn test_function_call_binary_op_precedence() { run_expr("print ( 1 ) + ( 2 ( 3 ) )", vec![NativeFunction(Print), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(1), OpAdd]); }
    #[test] fn test_function_call_parens_1() { run_expr("print . read_text (1 + 3) (5)", vec![NativeFunction(Print), NativeFunction(ReadText), Int(1), Int(3), OpAdd, OpFuncEval(1), Int(5), OpFuncEval(1), Swap, OpFuncEval(1)]); }
    #[test] fn test_function_call_parens_2() { run_expr("( print . read_text (1 + 3) ) (5)", vec![NativeFunction(Print), NativeFunction(ReadText), Int(1), Int(3), OpAdd, OpFuncEval(1), Swap, OpFuncEval(1), Int(5), OpFuncEval(1)]); }
    #[test] fn test_function_composition_with_is() { run_expr("'123' . int is int . print", vec![Str(1), NativeFunction(NativeFunction::Int), NativeFunction(NativeFunction::Int), OpIs, Swap, OpFuncEval(1), NativeFunction(Print), Swap, OpFuncEval(1)]); }
    #[test] fn test_and() { run_expr("1 < 2 and 3 < 4", vec![Int(1), Int(2), OpLessThan, JumpIfFalse(8), Pop, Int(3), Int(4), OpLessThan]); }
    #[test] fn test_or() { run_expr("1 < 2 or 3 < 4", vec![Int(1), Int(2), OpLessThan, JumpIfTrue(8), Pop, Int(3), Int(4), OpLessThan]); }
    #[test] fn test_precedence_1() { run_expr("1 . 2 & 3 > 4", vec![Int(1), Int(2), Int(3), OpBitwiseAnd, Swap, OpFuncEval(1), Int(4), OpGreaterThan]); }
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
    #[test] fn test_binary_ops() { run_expr("(*) * (+) + (/)", vec![NativeFunction(OperatorMul), NativeFunction(OperatorAdd), OpMul, NativeFunction(OperatorDiv), OpAdd]); }
    #[test] fn test_if_then_else() { run_expr("(if true then 1 else 2)", vec![True, JumpIfFalsePop(4), Int(1), Jump(5), Int(2)]); }
    #[test] fn test_zero_equals_zero() { run_expr("0 == 0", vec![Int(0), Int(0), OpEqual]); }
    #[test] fn test_zero_equals_zero_no_spaces() { run_expr("0==0", vec![Int(0), Int(0), OpEqual]); }

    #[test] fn test_let_eof() { run_err("let", "Expected a variable binding, either a name, or '_', or pattern (i.e. 'x, (_, y), *z'), got end of input instead\n  at: line 1 (<test>)\n\n1 | let\n2 |     ^^^\n"); }
    #[test] fn test_let_no_identifier() { run_err("let =", "Expected a variable binding, either a name, or '_', or pattern (i.e. 'x, (_, y), *z'), got '=' token instead\n  at: line 1 (<test>)\n\n1 | let =\n2 |     ^\n"); }
    #[test] fn test_let_expression_eof() { run_err("let x =", "Expected an expression terminal, got end of input instead\n  at: line 1 (<test>)\n\n1 | let x =\n2 |         ^^^\n"); }
    #[test] fn test_let_no_expression() { run_err("let x = &", "Expected an expression terminal, got '&' token instead\n  at: line 1 (<test>)\n\n1 | let x = &\n2 |         ^\n"); }

    #[test] fn test_array_access_after_newline() { run("array_access_after_newline"); }
    #[test] fn test_array_access_no_newline() { run("array_access_no_newline"); }
    #[test] fn test_break_past_locals() { run("break_past_locals"); }
    #[test] fn test_constants() { run("constants"); }
    #[test] fn test_continue_past_locals() { run("continue_past_locals"); }
    #[test] fn test_empty() { run("empty"); }
    #[test] fn test_expressions() { run("expressions"); }
    #[test] fn test_for_else() { run("for_else"); }
    #[test] fn test_for_range_start_stop() { run("for_range_start_stop"); }
    #[test] fn test_for_range_start_stop_no_var() { run("for_range_start_stop_no_var"); }
    #[test] fn test_for_range_start_stop_step() { run("for_range_start_stop_step"); }
    #[test] fn test_for_range_start_stop_step_no_var() { run("for_range_start_stop_step_no_var"); }
    #[test] fn test_for_range_stop() { run("for_range_stop"); }
    #[test] fn test_for_range_stop_no_var() { run("for_range_stop_no_var"); }
    #[test] fn test_for_string() { run("for_string"); }
    #[test] fn test_function() { run("function"); }
    #[test] fn test_function_call_after_newline() { run("function_call_after_newline"); }
    #[test] fn test_function_call_no_newline() { run("function_call_no_newline"); }
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
    #[test] fn test_multiple_undeclared_variables() { run("multiple_undeclared_variables"); }
    #[test] fn test_pattern_expression() { run("pattern_expression"); }
    #[test] fn test_pattern_expression_nested() { run("pattern_expression_nested"); }
    #[test] fn test_weird_expression_statements() { run("weird_expression_statements"); }
    #[test] fn test_weird_closure_not_a_closure() { run("weird_closure_not_a_closure"); }
    #[test] fn test_weird_locals() { run("weird_locals"); }
    #[test] fn test_weird_loop_nesting_in_functions() { run("weird_loop_nesting_in_functions"); }
    #[test] fn test_weird_upvalue_index() { run("weird_upvalue_index"); }
    #[test] fn test_weird_upvalue_index_with_parameter() { run("weird_upvalue_index_with_parameter"); }
    #[test] fn test_while_1() { run("while_1"); }
    #[test] fn test_while_2() { run("while_2"); }
    #[test] fn test_while_3() { run("while_3"); }
    #[test] fn test_while_4() { run("while_4"); }
    #[test] fn test_while_else() { run("while_else"); }
    #[test] fn test_while_false_if_false() { run("while_false_if_false"); }


    fn run_expr(text: &'static str, expected: Vec<Opcode>) {
        let result: ScanResult = scanner::scan(&String::from(text));
        assert!(result.errors.is_empty());

        let compile = parser::parse(result);
        assert!(compile.errors.is_empty(), "Found parser errors: {:?}", compile.errors);

        // For the purposes of the test, we perform some transformations on the 'expected' opcodes
        // - Int tokens use the actual value, as opposed to a constant ID
        // - Jump opcodes replace relative jumps with absolute jumps
        // - The trailing `Pop`, `Exit` tokens are removed.
        let constants: Vec<i64> = compile.constants;
        let mut actual: Vec<Opcode> = compile.code.into_iter()
            .enumerate()
            .map(|(ip, t)| match t {
                Int(i) => Int(constants[i as usize] as u32),
                _ => t.to_absolute_jump(ip),
            })
            .collect::<Vec<Opcode>>();

        assert_eq!(actual.pop(), Some(Exit));
        assert_eq!(actual.pop(), Some(Pop));
        assert_eq!(actual, expected);
    }

    fn run_err(text: &'static str, expected: &'static str) {
        let text: String = String::from(text);
        let name: String = String::from("<test>");
        let view: SourceView = SourceView::new(&name, &text);

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let compile: CompileResult = parser::parse(scan_result);
        assert!(!compile.errors.is_empty());

        let mut actual: Vec<String> = Vec::new();
        for error in &compile.errors {
            actual.push(view.format(error));
        }

        assert_eq!(actual.join("\n"), expected);
    }

    fn run(path: &'static str) {
        let root: PathBuf = trace::get_test_resource_path("parser", path);
        let text: String = trace::get_test_resource_src(&root);
        let name: String = format!("{}.cor", path);
        let view: SourceView = SourceView::new(&name, &text);

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let parse_result: CompileResult = parser::parse(scan_result);
        let mut lines: Vec<String> = parse_result.disassemble(&view);
        if !parse_result.errors.is_empty() {
            for error in &parse_result.errors {
                lines.push(view.format(error));
            }
        }

        trace::compare_test_resource_content(&root, lines);
    }
}