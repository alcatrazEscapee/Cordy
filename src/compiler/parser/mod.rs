use std::collections::VecDeque;
use std::rc::Rc;

use crate::compiler::{CompileParameters, CompileResult};
use crate::compiler::parser::core::ParserState;
use crate::compiler::parser::expr::{Expr, ExprType};
use crate::compiler::parser::semantic::{LateBoundGlobal, LValue, LValueReference, ParserFunctionImpl, Reference};
use crate::compiler::parser::optimizer::Optimize;
use crate::compiler::scanner::{ScanResult, ScanToken};
use crate::stdlib::NativeFunction;
use crate::vm::{FunctionImpl, Opcode, StructTypeImpl};
use crate::vm::operator::{BinaryOp, UnaryOp};
use crate::trace;
use crate::reporting::{Location, Locations};

pub use crate::compiler::parser::errors::{ParserError, ParserErrorType};
pub use crate::compiler::parser::semantic::{Locals, Fields};

use NativeFunction::{*};
use Opcode::{*};
use ParserErrorType::{*};
use ScanToken::{*};


pub const RULE_REPL: ParseRule = |parser| parser.parse_incremental_repl();
pub const RULE_EVAL: ParseRule = |parser| parser.parse_incremental_eval();

pub type ParseRule = fn(&mut Parser) -> ();

mod core;
mod errors;
mod expr;

mod semantic;
mod codegen;
mod optimizer;


/// Create a default empty `CompileResult`. This is semantically equivalent to parsing an empty program, but will output nothing.
pub fn default() -> CompileResult {
    parse_rule(false, vec![], |_| ())
}


/// Parse a complete `CompileResult` from the given `ScanResult`
pub fn parse(enable_optimization: bool, scan_result: ScanResult) -> CompileResult {
    parse_rule(enable_optimization, scan_result.tokens, |parser| parser.parse())
}


pub fn parse_incremental(scan_result: ScanResult, params: &mut CompileParameters, rule: ParseRule) -> Vec<ParserError> {
    let mut errors: Vec<ParserError> = Vec::new();

    rule(&mut Parser::new(params.enable_optimization, scan_result.tokens, params.code, params.locals, params.fields, &mut errors, params.strings, params.constants, params.functions, params.structs, params.locations, &mut Vec::new(), params.globals));

    errors
}


fn parse_rule(enable_optimization: bool, tokens: Vec<(Location, ScanToken)>, rule: ParseRule) -> CompileResult {
    let mut result = CompileResult {
        code: Vec::new(),
        errors: Vec::new(),

        strings: vec![String::new()],
        constants: vec![0, 1],
        functions: Vec::new(),
        structs: Vec::new(),

        locations: Vec::new(),
        globals: Vec::new(),
        locals: Vec::new(),
        fields: Fields::new(),
    };

    rule(&mut Parser::new(enable_optimization, tokens, &mut result.code, &mut Locals::empty(), &mut result.fields, &mut result.errors, &mut result.strings, &mut result.constants, &mut result.functions, &mut result.structs, &mut result.locations, &mut result.locals, &mut result.globals));

    result
}


pub struct Parser<'a> {
    enable_optimization: bool,

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

    /// A table of all struct fields and types. This is used to resolve `-> <name>` references at compile time, to a `field index`. At runtime it is used as a lookup to resolve a `(type index, field index)` into a `field offset`, which is used to access the underlying field.
    fields: &'a mut Fields,
    /// A vector of all baked structs, known to the runtime. These are appended as a new struct is encountered.
    structs: &'a mut Vec<Rc<StructTypeImpl>>,

    late_bound_globals: Vec<Reference<LateBoundGlobal>>, // Table of all late bound globals, as they occur.
    synthetic_local_index: usize, // A counter for unique synthetic local variables (`$1`, `$2`, etc.)
    scope_depth: u32, // Current scope depth
    function_depth: u32,

    strings: &'a mut Vec<String>,
    constants: &'a mut Vec<i64>,

    /// List of all functions known to the parser, in their unbaked form.
    /// Note that this list is considered starting at the length of `baked_functions`
    functions: Vec<ParserFunctionImpl>,
    baked_functions: &'a mut Vec<Rc<FunctionImpl>>,
}


impl Parser<'_> {

    fn new<'a, 'b : 'a>(enable_optimization: bool, tokens: Vec<(Location, ScanToken)>, output: &'b mut Vec<Opcode>, locals: &'b mut Vec<Locals>, fields: &'b mut Fields, errors: &'b mut Vec<ParserError>, strings: &'b mut Vec<String>, constants: &'b mut Vec<i64>, baked_functions: &'b mut Vec<Rc<FunctionImpl>>, structs: &'b mut Vec<Rc<StructTypeImpl>>, locations: &'b mut Locations, locals_reference: &'b mut Vec<String>, globals_reference: &'b mut Vec<String>) -> Parser<'a> {
        Parser {
            enable_optimization,

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
            fields,
            structs,
            late_bound_globals: Vec::new(),

            synthetic_local_index: 0,
            scope_depth: 0,
            function_depth: 0,

            strings,
            constants,
            functions: Vec::new(),
            baked_functions,
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
        for mut func in self.functions.drain(..) {
            let head: usize = self.raw_output.len();
            for (loc, op) in func.code.drain(..) {
                self.raw_output.push(op);
                self.locations.push(loc);
            }
            for local in func.locals_reference.drain(..) {
                self.locals_reference.push(local);
            }

            let tail: usize = self.raw_output.len() - 1;
            let baked: Rc<FunctionImpl> = Rc::new(FunctionImpl::new(head, tail, func.name, func.args, func.default_args));

            self.baked_functions.push(baked);
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
                Some(KeywordDo) => self.parse_do_while_statement(),
                Some(KeywordFor) => self.parse_for_statement(),
                Some(KeywordBreak) => self.parse_break_statement(),
                Some(KeywordContinue) => self.parse_continue_statement(),
                Some(KeywordAssert) => self.parse_assert_statement(),
                Some(KeywordStruct) => self.parse_struct_statement(),
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

    fn parse_struct_statement(self: &mut Self) {
        self.push_delayed_pop();
        self.advance(); // Consume `struct`

        // Structs can only be declared in global scope
        // While technically we could support scoped structs - the struct constructor becomes a local, there, the use case is just not there
        // The struct object and fields would have to be globally known anyway.
        if self.function_depth != 0 || self.scope_depth != 0 {
            self.semantic_error(StructNotInGlobalScope);
            return;
        }

        let type_name: String = match self.peek() {
            Some(Identifier(_)) => self.take_identifier(),
            _ => return,
        };

        // Declare a local for the struct in the global scope
        match self.declare_local(type_name.clone()) {
            Some(local) => self.init_local(local),
            _ => return,
        }

        // Declare a type index, as at this point we know we're in totally global scope, and the type name must be unique
        let type_index: u32 = self.structs.len() as u32;
        let mut unique_fields: Vec<String> = Vec::new();

        self.expect(OpenParen);

        loop {
            match self.peek() {
                Some(Identifier(_)) => {
                    let name: String = self.take_identifier();

                    if unique_fields.contains(&name) {
                        self.semantic_error(DuplicateFieldName(name))
                    } else {
                        self.declare_field(type_index, unique_fields.len(), name.clone());
                        unique_fields.push(name);
                    }

                    // Consume `,` and allow trailing comma
                    match self.peek() {
                        Some(Comma) => self.skip(),
                        _ => {},
                    }
                },
                _ => break, // `CloseParen` also breaks the loop, and hits resync below
            }
        }

        self.push(Struct(type_index));
        self.structs.push(Rc::new(StructTypeImpl::new(type_name, unique_fields, type_index)));

        self.expect_resync(CloseParen);
    }

    fn parse_annotated_named_function(self: &mut Self) {
        trace::trace_parser!("rule <annotated-named-function");

        self.push_delayed_pop();
        self.advance(); // Consume `@`
        self.parse_expression(); // The annotation body
        match self.peek() {
            Some(At) => self.parse_annotated_named_function(),
            Some(KeywordFn) => self.parse_named_function(),
            _ => self.error_with(ExpectedAnnotationOrNamedFunction),
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
        let (args, default_args) = self.parse_function_parameters();
        self.expect_resync(CloseParen);

        // Named functions are a complicated local variable, and needs to be declared as such
        // Note that we always declare the function here, to preserve parser operation in the event of a parse error
        let name = maybe_name
            .map(|name| {
                if let Some(index) = self.declare_local(name.clone()) {
                    self.init_local(index);
                }
                name
            })
            .unwrap_or_else(|| String::from("<invalid>"));

        let func: u32 = self.declare_function(name, &args);
        self.push(Opcode::Function(func));

        // Emit the closed locals from the function body right away, because we are not in an expression context
        let closed_locals = self.parse_function_body(args, default_args);
        self.emit_closure_and_closed_locals(closed_locals);
    }

    fn parse_annotated_expression_function(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <annotated-expression-function");

        self.push_delayed_pop();
        let loc_start = self.advance_with(); // Consume `@`
        let func = self.parse_expr_top_level(); // The annotation body
        let loc_end = self.prev_location();
        let arg = match self.peek() {
            Some(At) => self.parse_annotated_expression_function(),
            Some(KeywordFn) => self.parse_expression_function(),
            _ => {
                self.error_with(ExpectedAnnotationOrAnonymousFunction);
                Expr::nil()
            },
        };
        func.eval(loc_start | loc_end, vec![arg]) // Evaluate the annotation
    }

    fn parse_expression_function(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expression-function>");

        // Function header - `fn` (<arg>, ...)
        self.advance();
        self.expect(OpenParen);
        let (args, default_args) = self.parse_function_parameters();
        self.expect_resync(CloseParen);

        // Expression functions don't declare themselves as a local variable that can be referenced.
        // Instead, as they're part of an expression, they just push a single function instance onto the stack
        let func: u32 = self.declare_function(String::from("_"), &args);
        let closed_locals = self.parse_function_body(args, default_args);
        Expr::function(func, closed_locals)
    }

    fn parse_function_name(self: &mut Self) -> Option<String> {
        trace::trace_parser!("rule <function-name>");
        match self.peek() {
            Some(Identifier(_)) => Some(self.take_identifier()),
            _ => {
                self.error_with(ExpectedFunctionNameAfterFn);
                None
            }
        }
    }

    /// Returns the pair of `lvalue` parameters, and `Expr` default values, if present.
    fn parse_function_parameters(self: &mut Self) -> (Vec<LValue>, Vec<Expr>) {
        trace::trace_parser!("rule <function-parameters>");

        let mut args: Vec<LValue> = Vec::new();
        let mut default_args: Vec<Expr> = Vec::new();
        if let Some(CloseParen) = self.peek() {
            return (args, default_args); // Special case for no parameters, don't enter the loop
        }

        loop {
            match self.parse_lvalue() {
                Some(lvalue) => args.push(lvalue),
                _ => {
                    self.error_with(ExpectedParameterOrEndOfList);
                    break
                },
            }

            // Default Arguments
            match self.peek() {
                // Sugar for `= nil`, so mark this as a default argument
                Some(QuestionMark) => {
                    self.skip(); // Consume `?`
                    default_args.push(Expr(Location::empty(), ExprType::Nil));
                },
                Some(Equals) => {
                    self.skip(); // Consume `=`
                    default_args.push(self.parse_expr_top_level()); // Parse an expression
                },
                _ => {
                    if !default_args.is_empty() {
                        // Cannot have non-default arguments after default arguments
                        self.semantic_error(NonDefaultParameterAfterDefaultParameter);
                    }
                },
            }

            match self.peek() {
                Some(Comma) => self.skip(), // Consume `,`
                Some(CloseParen) => break,
                _ => {
                    self.error_with(ExpectedCommaOrEndOfParameters);
                    break
                },
            }
        }
        (args, default_args)
    }

    fn parse_function_body(self: &mut Self, args: Vec<LValue>, default_args: Vec<Expr>) -> Vec<Opcode> {
        trace::trace_parser!("rule <function-body>");
        let prev_pop_status: bool = self.delay_pop_from_expression_statement; // Stack semantics for the delayed pop

        // Functions have their own depth tracking in addition to scope
        // In addition, we let parameters have their own scope depth one outside locals to the function
        // This lets us 1) declare parameters here, in the right scope,
        // and 2) avoid popping parameters at the end of a function call (as they're handled by the `Return` opcode instead)
        self.locals.push(Locals::new(Some(self.functions.len() - 1)));
        self.function_depth += 1;
        self.scope_depth += 1;

        // After the locals have been pushed, we now can push function code
        // Before the body of the function, we emit code for each default argument, and mark it as such.
        for arg in default_args {
            self.emit_expr(arg);
            self.current_function_impl().mark_default_arg();
        }

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
                self.error_with(ExpectedFunctionBlockOrArrowAfterFn);
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
        // We just collect and return the opcodes for it, as if this is part of an expression function, we need to hold them to be emitted later
        let closed_locals: Vec<Opcode> = self.current_locals().upvalues
            .iter()
            .map(|upvalue| if upvalue.is_local { CloseLocal(upvalue.index) } else { CloseUpValue(upvalue.index) })
            .collect::<Vec<Opcode>>();

        self.delay_pop_from_expression_statement = prev_pop_status; // Exit the stack

        closed_locals
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

        let jump_if_false = self.reserve(); // placeholder for jump to the beginning of an if branch, if it exists

        if let Some(KeywordThen) = self.peek() {
            // If we see a top-level `if <expression> then`, we want to consider this an expression, with a top level `if-then-else` statement
            // We duplicate the structure here, as there's not many optimizations we could've performed if we treat this as a top-level expression.
            // Note that unlike `if { }`, an `if then else` **does** count as an expression, and leaves a value on the stack, so we set the flag for delay pop = true
            self.advance();
            self.parse_expression(); // Value if true
            let jump = self.reserve();
            self.fix_jump(jump_if_false, JumpIfFalsePop);
            self.expect(KeywordElse);
            self.parse_expression(); // Value if false
            self.fix_jump(jump, Jump);
            self.delay_pop_from_expression_statement = true;
            return;
        }

        self.parse_block_statement();
        self.push_delayed_pop();

        // `elif` can be de-sugared to else { if <expr> { ... } else { ... } }
        // The additional scope around the else {} can be dropped as it doesn't contain anything already in a scope
        // So if we hit `elif`, we emit what we would for `else`, then recurse, then once we return, patch the final jump.
        match self.peek() {
            Some(KeywordElif) => {
                // Don't advance, as `parse_if_statement()` will advance the first token
                let jump = self.reserve();
                self.fix_jump(jump_if_false, JumpIfFalsePop);
                self.parse_if_statement();
                self.fix_jump(jump, Jump);
            },
            Some(KeywordElse) => {
                // `else` is present, so we first insert an unconditional jump, parse the next block, then fix the first jump
                self.advance();
                let jump = self.reserve();
                self.fix_jump(jump_if_false, JumpIfFalsePop);
                self.parse_block_statement();
                self.push_delayed_pop();
                self.fix_jump(jump, Jump);
            },
            _ => {
                // No `else`, but we still need to fix the initial jump
                self.fix_jump(jump_if_false, JumpIfFalsePop);
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

    /// Parses a `do { }`, which is a syntax for a scoped block, and optionally a `while` at the end
    fn parse_do_while_statement(self: &mut Self) {
        self.advance(); // Consume `do`

        let jump: usize = self.begin_loop();

        self.parse_block_statement(); // The statements
        self.push_delayed_pop(); // Inner loop expressions cannot yield out of the loop

        if let Some(KeywordWhile) = self.peek() {
            self.advance(); // Consume `while`
            self.parse_expression(); // While condition
            self.push_jump(jump, JumpIfTruePop); // Jump back to origin
        }

        if let Some(KeywordElse) = self.peek() { // Parse do { } while <expr> else { }
            self.advance(); // Consume `else`
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

    fn parse_assert_statement(self: &mut Self) {
        trace::trace_parser!("rule <assert-statement>");
        self.push_delayed_pop();
        self.advance(); // Consume `assert`

        // `assert x : y` is effectively
        // `if x {} else { throw an exception with description `y` }
        let mut loc = self.next_location();
        self.parse_expression();
        loc |= self.prev_location();
        let jump_if_true = self.reserve();
        match self.peek() {
            Some(Colon) => {
                // Optional error message, that is lazily evaluated
                self.advance(); // Consume `:`
                self.parse_expression();
            },
            _ => {
                // Push `Nil`
                self.push(Nil);
            }
        }

        self.push_with(AssertFailed, loc); // Make sure the `AssertFailed` token has the same location as the original expression that failed
        self.fix_jump(jump_if_true, JumpIfTruePop)
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
            self.error_with(ExpectedStatement)
        }
    }

    fn parse_expression(self: &mut Self) {
        trace::trace_parser!("rule <expression>");
        let mut expr: Expr = self.parse_expr_top_level();
        if self.enable_optimization {
            expr = expr.optimize();
        }
        self.emit_expr(expr);
    }

    #[must_use = "For parsing expressions from non-expressions, use parse_expression()"]
    fn parse_expr_top_level(self: &mut Self) -> Expr {
        self.parse_expr_10()
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
                        self.error_with(ExpectedUnderscoreOrVariableNameAfterVariadicInPattern);
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
                self.error_with(ExpectedUnderscoreOrVariableNameOrPattern);
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

    fn parse_expr_1_terminal(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-1>");
        match self.peek() {
            Some(KeywordNil) => {
                self.advance();
                Expr::nil()
            },
            Some(KeywordTrue) => {
                self.advance();
                Expr::bool(true)
            },
            Some(KeywordFalse) => {
                self.advance();
                Expr::bool(false)
            },
            Some(KeywordExit) => {
                self.advance();
                Expr::exit()
            },
            Some(ScanToken::Int(_)) => Expr::int(self.take_int()),
            Some(StringLiteral(_)) => Expr::str(self.take_str()),
            Some(Identifier(_)) => {
                let name: String = self.take_identifier();
                let loc: Location = self.prev_location();
                let lvalue: LValueReference = self.resolve_identifier(name);
                Expr::lvalue(loc, lvalue)
            },
            Some(OpenParen) => {
                let loc_start = self.advance_with(); // Consume the `(`
                match self.parse_expr_1_partial_operator_left() {
                    Some(expr) => return expr, // Looks ahead, and parses <op> <expr> `)`
                    _ => {},
                }
                let expr = self.parse_expr_top_level(); // Parse <expr>
                let expr = match self.parse_expr_1_partial_operator_right(expr) {
                    Ok(expr) => return expr, // Looks ahead and parses <op> `)`
                    Err(expr) => expr,
                };
                match self.peek() {
                    Some(Comma) => self.parse_expr_1_vector_literal(loc_start, expr),
                    _ => {
                        // Simple expression
                        self.expect(CloseParen);
                        expr
                    },
                }
            },
            Some(OpenSquareBracket) => self.parse_expr_1_list_or_slice_literal(),
            Some(OpenBrace) => self.parse_expr_1_dict_or_set_literal(),
            Some(At) => self.parse_annotated_expression_function(),
            Some(KeywordFn) => self.parse_expression_function(),
            Some(KeywordIf) => self.parse_expr_1_inline_if_then_else(),
            _ => {
                self.error_with(ExpectedExpressionTerminal);
                Expr::nil()
            },
        }
    }

    fn parse_expr_1_partial_operator_left(self: &mut Self) -> Option<Expr> {
        trace::trace_parser!("rule <expr-1-partial-operator-left>");
        // Open `(` usually resolves precedence, so it begins parsing an expression from the top again
        // However it *also* can be used to wrap around a partial evaluation of a literal operator, for example
        // (-)   => OperatorUnary(UnaryOp::Minus)
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

            // This is a unique case, as we can only partially evaluate this with an identifier, not an expression.
            // It also involves a unique operator, as it cannot be evaluated as a normal native operator (since again, it takes a field index, not a expression)
            // This operator also cannot stand alone: `(->)` is not valid, but `(->x)` is, presuming a field `x` exists.
            Some(Arrow) => {
                if let Some((loc, field_index)) = self.parse_expr_2_field_access() {
                    self.expect(CloseParen);
                    return Some(Expr::get_field_function(loc, field_index))
                }
            },

            _ => {},
        }

        if let Some(op) = unary {
            match self.peek2() {
                Some(CloseParen) => {
                    // A bare `( <unary op> )`, which requires no partial evaluation
                    let loc = self.advance_with(); // The unary operator
                    self.advance(); // The close paren
                    Some(Expr::native(loc, op))
                },
                _ => None
            }
        } else if let Some(op) = binary {
            let loc = self.advance_with(); // The unary operator
            match self.peek() {
                Some(CloseParen) => {
                    // No expression follows this operator, so we have `(` <op> `)`, which just returns the operator itself
                    self.advance();
                    Some(Expr::native(loc, op))
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
                        OperatorAdd => OperatorAddSwap,
                        OperatorLeftShift => OperatorLeftShiftSwap,
                        OperatorRightShift => OperatorRightShiftSwap,
                        OperatorLessThan => OperatorLessThanSwap,
                        OperatorLessThanEqual => OperatorLessThanEqualSwap,
                        OperatorGreaterThan => OperatorGreaterThanSwap,
                        OperatorGreaterThanEqual => OperatorGreaterThanEqualSwap,
                        op => op,
                    };
                    let arg = self.parse_expr_top_level(); // Parse the expression following a binary prefix operator
                    self.expect(CloseParen);
                    Some(Expr::native(loc, op).eval(loc, vec![arg]))
                }
            }
        } else {
            None
        }
    }

    fn parse_expr_1_partial_operator_right(self: &mut Self, expr: Expr) -> Result<Expr, Expr> {
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
                let loc = self.advance_with(); // The operator
                self.expect(CloseParen);
                Ok(Expr::native(loc, op).eval(loc, vec![expr])) // Return Ok(), with our new expression
            },
            _ => Err(expr), // Return the expression as an error, indicating we did not parse anything
        }
    }

    fn parse_expr_1_vector_literal(self: &mut Self, mut loc: Location, arg1: Expr) -> Expr {
        trace::trace_parser!("rule <expr-1-vector-literal>");

        let mut args: Vec<Expr> = vec![arg1];

        // Vector literal `(` <expr> `,` [ <expr> [ `,` <expr> ] * ] ? `)`
        // Note that this might be a single element vector literal, which includes a trailing comma
        self.advance(); // Consume `,`
        match self.peek() {
            Some(CloseParen) => {
                loc |= self.advance_with(); // Consume `)`
                return Expr::vector(loc, args)
            },
            _ => {},
        }

        // Otherwise, it must be > 1
        loop {
            args.push(self.parse_expr_top_level());
            match self.peek() {
                Some(CloseParen) => {
                    self.advance(); // Consume `)`, then exit
                    break
                }
                Some(Comma) => {
                    self.advance();
                    match self.peek() { // Check again, to allow trailing comma
                        Some(CloseParen) => {
                            self.advance();
                            break
                        },
                        _ => {},
                    }
                },
                _ => {
                    self.error_with(ExpectedCommaOrEndOfVector);
                    break
                }
            }
        }
        Expr::vector(loc | self.prev_location(), args)
    }

    fn parse_expr_1_list_or_slice_literal(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-1-list-or-slice-literal>");
        let loc_start = self.advance_with(); // Consume `[`

        match self.peek() {
            Some(CloseSquareBracket) => { // Empty list
                self.advance(); // Consumes `]`
                return Expr::list(loc_start | self.prev_location(), Vec::new());
            },
            Some(Colon) => { // Slice with no first argument
                return self.parse_expr_1_slice_literal(loc_start, Expr::nil());
            },
            _ => {}
        }

        // Unsure if a slice or a list so far, so we parse the first expression and check for a colon, square bracket, or comma
        let arg = self.parse_expr_top_level();
        match self.peek() {
            Some(CloseSquareBracket) => { // Single element list
                let loc_end = self.advance_with(); // Consume `]`
                return Expr::list(loc_start | loc_end, vec![arg]);
            },
            Some(Colon) => { // Slice with a first argument
                return self.parse_expr_1_slice_literal(loc_start, arg);
            },
            Some(Comma) => self.skip(), // Consume `,`
            _ => self.error_with(ExpectedCommaOrEndOfList), // todo: new error type for comma, list, or slice
        }

        // Parse the next argument, then break into the loop until we exit the list literal
        let mut args: Vec<Expr> = vec![arg];
        loop {
            match self.peek() {
                Some(CloseSquareBracket) => break,
                Some(_) => {
                    args.push(self.parse_expr_top_level());
                    match self.peek() {
                        Some(Comma) => self.skip(),
                        Some(CloseSquareBracket) => {}, // Skip
                        _ => self.error_with(ExpectedCommaOrEndOfList),
                    }
                },
                None => break,
            }
        }
        self.expect(CloseSquareBracket);
        Expr::list(loc_start | self.prev_location(), args)
    }

    fn parse_expr_1_slice_literal(self: &mut Self, loc_start: Location, arg1: Expr) -> Expr {
        self.advance(); // Consume `:`

        let arg2: Expr = match self.peek() {
            Some(Colon) => Expr::nil(), // No second argument, but we have a third argument. Don't consume the colon as it's the seperator
            Some(CloseSquareBracket) => { // No second argument, so a unbounded slice
                let loc_end = self.advance_with();
                return Expr::slice_literal(loc_start | loc_end, arg1, Expr::nil(), None);
            }
            _ => self.parse_expr_top_level(), // As we consumed `:`, we require a second expression
        };

        // Consumed `[` <expression> `:` <expression> so far
        match self.peek() {
            Some(CloseSquareBracket) => { // Two argument slice
                let loc_end = self.advance_with();
                return Expr::slice_literal(loc_start | loc_end, arg1, arg2, None);
            },
            Some(Colon) => {
                // Three arguments, so continue parsing
                self.advance();
            },
            _ => {
                self.error_with(ExpectedColonOrEndOfSlice);
                return Expr::nil();
            },
        }

        // Consumed `[` <expression> `:` <expression> `:` so far
        let arg3: Expr = match self.peek() {
            Some(CloseSquareBracket) => Expr::nil(),
            _ => self.parse_expr_top_level(),
        };

        self.expect(CloseSquareBracket);
        Expr::slice_literal(loc_start | self.prev_location(), arg1, arg2, Some(arg3))
    }

    fn parse_expr_1_dict_or_set_literal(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-1-dict-or-set-literal>");
        let mut loc = self.advance_with(); // Consume `{`

        // As dict and set literals both start the same, we initially parse empty `{}` as a empty dict, and use the first argument to determine if it is a dict or set
        match self.peek() {
            Some(CloseBrace) => {
                // Just `{}` is a empty dict, not a set
                loc |= self.advance_with();
                return Expr::dict(loc, vec![])
            },
            _ => {}, // Otherwise, continue parsing to determine if it is a dict or set
        }

        let mut args: Vec<Expr> = Vec::new();

        args.push(self.parse_expr_top_level());
        let is_dict: bool = match self.peek() {
            Some(Colon) => {
                // Found `{ <expr> `:`, so it must be a dict. Parse the first value, then continue as such
                self.advance();
                args.push(self.parse_expr_top_level());
                true
            },
            _ => false,
        };

        match self.peek() {
            Some(Comma) => self.skip(), // Skip a comma, so we're either at end or next value to get into the loop below
            _ => {},
        }

        loop {
            match self.peek() {
                Some(CloseBrace) => break,
                Some(_) => {
                    args.push(self.parse_expr_top_level());
                    if is_dict {
                        self.expect(Colon);
                        args.push(self.parse_expr_top_level());
                    }
                    match self.peek() {
                        Some(Comma) => self.skip(), // Allow trailing commas, as this loops to the top again
                        Some(CloseBrace) => {}, // Skip
                        _ => if is_dict {
                            self.error_with(ExpectedCommaOrEndOfDict)
                        } else {
                            self.error_with(ExpectedCommaOrEndOfSet)
                        },
                    }
                },
                None => break,
            }
        }
        self.expect(CloseBrace);
        loc |= self.prev_location();
        if is_dict { Expr::dict(loc, args) } else { Expr::set(loc, args) }
    }

    fn parse_expr_1_inline_if_then_else(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-1-inline-if-then-else>");

        self.advance(); // Consume `if`
        let condition = self.parse_expr_top_level(); // condition
        self.expect(KeywordThen);
        let if_true = self.parse_expr_top_level(); // Value if true
        self.expect(KeywordElse);
        let if_false = self.parse_expr_top_level(); // Value if false

        condition.if_then_else(if_true, if_false)
    }

    fn parse_expr_2_unary(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-2>");

        // Prefix operators
        let mut stack: Vec<(Location, UnaryOp)> = Vec::new();
        loop {
            match self.peek() {
                Some(Minus) => {
                    let loc = self.advance_with();
                    stack.push((loc, UnaryOp::Minus));
                },
                Some(Not) => {
                    let loc = self.advance_with();
                    stack.push((loc, UnaryOp::Not));
                },
                _ => break
            }
        }

        let mut expr: Expr = self.parse_expr_1_terminal();

        // Suffix operators
        loop {
            // The opening token of a suffix operator must be on the same line
            match self.peek_no_newline() {
                Some(OpenParen) => {
                    let mut loc_start = self.advance_with();
                    match self.peek() {
                        Some(CloseParen) => {
                            loc_start |= self.advance_with();
                            expr = expr.eval(loc_start, vec![]);
                        },
                        Some(_) => {
                            // Special case: if we can parse a partially-evaluated operator expression, we do so
                            // This means we can write `map (+3)` instead of `map ((+3))`
                            // If we parse something here, this will have consumed everything, otherwise we parse an expression as per usual
                            if let Some(partial_expr) = self.parse_expr_1_partial_operator_left() {
                                // We still need to treat this as a function evaluation, however, because we pretended we didn't need to see the outer parenthesis
                                expr = expr.eval(loc_start | self.prev_location(), vec![partial_expr]);
                                continue;
                            } else {
                                // First argument
                                let mut args: Vec<Expr> = vec![self.parse_expr_top_level()];

                                // Other arguments
                                while let Some(Comma) = self.peek() {
                                    self.advance();
                                    args.push(self.parse_expr_top_level());
                                }
                                match self.peek() {
                                    Some(CloseParen) => self.skip(),
                                    _ => self.error_with(ExpectedCommaOrEndOfArguments),
                                }

                                expr = expr.eval(loc_start | self.prev_location(), args)
                            }
                        }
                        _ => self.error_with(ExpectedCommaOrEndOfArguments),
                    }
                },
                Some(OpenSquareBracket) => {
                    let loc_start = self.advance_with(); // Consume the square bracket

                    // Consumed `[` so far
                    let arg1: Expr = match self.peek() {
                        Some(Colon) => Expr::nil(), // No first argument. Don't consume the colon as it's the seperator
                        _ => self.parse_expr_top_level(), // Otherwise we require a first expression
                    };

                    // Consumed `[` <expression> so far
                    match self.peek() {
                        Some(CloseSquareBracket) => { // One argument, so an index expression
                            let loc_end = self.advance_with();
                            expr = expr.index(loc_start | loc_end, arg1);
                            continue;
                        },
                        Some(Colon) => { // At least two arguments, so continue parsing
                            self.advance();
                        },
                        _ => {
                            self.error_with(ExpectedColonOrEndOfSlice);
                            continue
                        },
                    }

                    // Consumed `[` <expression> `:` so far
                    let arg2: Expr = match self.peek() {
                        Some(Colon) => Expr::nil(), // No second argument, but we have a third argument. Don't consume the colon as it's the seperator
                        Some(CloseSquareBracket) => { // No second argument, so a unbounded slice
                            let loc_end = self.advance_with();
                            expr = expr.slice(loc_start | loc_end, arg1, Expr::nil());
                            continue
                        }
                        _ => self.parse_expr_top_level(), // As we consumed `:`, we require a second expression
                    };

                    // Consumed `[` <expression> `:` <expression> so far
                    match self.peek() {
                        Some(CloseSquareBracket) => { // Two argument slice
                            let loc_end = self.advance_with();
                            expr = expr.slice(loc_start | loc_end, arg1, arg2);
                            continue;
                        },
                        Some(Colon) => {
                            // Three arguments, so continue parsing
                            self.advance();
                        },
                        _ => {
                            self.error_with(ExpectedColonOrEndOfSlice);
                            continue
                        },
                    }

                    // Consumed `[` <expression> `:` <expression> `:` so far
                    let arg3: Expr = match self.peek() {
                        Some(CloseSquareBracket) => Expr::nil(),
                        _ => self.parse_expr_top_level(),
                    };

                    self.expect(CloseSquareBracket);
                    expr = expr.slice_step(loc_start | self.prev_location(), arg1, arg2, arg3);
                },
                _ => match self.peek() { // Re-match, since this is allowed to break over newlines
                    Some(Arrow) => {
                        if let Some((loc, field_index)) = self.parse_expr_2_field_access() {
                            expr = expr.get_field(loc, field_index);
                        }
                    },
                    _ => break
                }
            }
        }

        // Prefix operators are lower precedence than suffix operators
        for (loc, op) in stack.into_iter().rev() {
            expr = expr.unary(loc, op)
        }

        expr
    }

    /// Parses a `-> <field>` - either returns a `(Location, field_index)` pairing, or `None` and raises a parse error.
    fn parse_expr_2_field_access(self: &mut Self) -> Option<(Location, u32)> {
        let loc_start = self.advance_with(); // Consume `->`
        match self.peek() {
            Some(Identifier(_)) => {
                let field: String = self.take_identifier();
                match self.resolve_field(&field) {
                    Some(field_index) => return Some((loc_start | self.prev_location(), field_index)),
                    None => self.semantic_error(InvalidFieldName(field))
                }
            },
            _ => self.error_with(ExpectedFieldNameAfterArrow)
        }
        None
    }

    fn parse_expr_3(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-3>");
        let mut expr: Expr = self.parse_expr_2_unary();
        loop {
            let maybe_op: Option<BinaryOp> = match self.peek() {
                Some(Mul) => Some(BinaryOp::Mul),
                Some(Div) => Some(BinaryOp::Div),
                Some(Mod) => Some(BinaryOp::Mod),
                Some(Pow) => Some(BinaryOp::Pow),
                Some(KeywordIs) => Some(BinaryOp::Is),
                Some(KeywordIn) => Some(BinaryOp::In),
                Some(KeywordNot) => match self.peek2() {
                    Some(KeywordIn) => Some(BinaryOp::NotEqual), // Special flag for `not in`, which is implemented as `!(x in y)
                    _ => None,
                }
                _ => None
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(BinaryOp::NotEqual) => {
                    let loc = self.advance_with() | self.advance_with();
                    expr = expr.binary(loc, BinaryOp::In, self.parse_expr_2_unary()).not(loc);
                }
                Some(op) => {
                    let loc = self.advance_with();
                    expr = expr.binary(loc, op, self.parse_expr_2_unary());
                },
                None => break
            }
        }
        expr
    }

    fn parse_expr_4(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-4>");
        let mut expr: Expr = self.parse_expr_3();
        loop {
            let maybe_op: Option<BinaryOp> = match self.peek() {
                Some(Plus) => Some(BinaryOp::Add),
                Some(Minus) => Some(BinaryOp::Sub),
                _ => None
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    expr = expr.binary(loc, op, self.parse_expr_3());
                },
                None => break
            }
        }
        expr
    }

    fn parse_expr_5(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-5>");
        let mut expr: Expr = self.parse_expr_4();
        loop {
            let maybe_op: Option<BinaryOp> = match self.peek() {
                Some(LeftShift) => Some(BinaryOp::LeftShift),
                Some(RightShift) => Some(BinaryOp::RightShift),
                _ => None
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    expr = expr.binary(loc, op, self.parse_expr_4());
                },
                None => break
            }
        }
        expr
    }

    fn parse_expr_6(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-6>");
        let mut expr: Expr = self.parse_expr_5();
        loop {
            let maybe_op: Option<BinaryOp> = match self.peek() {
                Some(BitwiseAnd) => Some(BinaryOp::And),
                Some(BitwiseOr) => Some(BinaryOp::Or),
                Some(BitwiseXor) => Some(BinaryOp::Xor),
                _ => None
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => expr = expr.binary(self.advance_with(), op, self.parse_expr_5()),
                None => break
            }
        }
        expr
    }

    fn parse_expr_7(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-7>");
        let mut expr: Expr = self.parse_expr_6();
        loop {
            match self.peek() {
                Some(Dot) => {
                    let mut loc = self.advance_with();
                    let rhs = self.parse_expr_6();
                    loc |= self.prev_location();
                    expr = expr.compose(loc, rhs);
                },
                _ => break
            }
        }
        expr
    }

    fn parse_expr_8(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-8>");
        let mut expr: Expr = self.parse_expr_7();
        loop {
            let maybe_op: Option<BinaryOp> = match self.peek() {
                Some(LessThan) => Some(BinaryOp::LessThan),
                Some(LessThanEquals) => Some(BinaryOp::LessThanEqual),
                Some(GreaterThan) => Some(BinaryOp::GreaterThan),
                Some(GreaterThanEquals) => Some(BinaryOp::GreaterThanEqual),
                Some(DoubleEquals) => Some(BinaryOp::Equal),
                Some(NotEquals) => Some(BinaryOp::NotEqual),
                _ => None
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    expr = expr.binary(loc, op, self.parse_expr_7());
                },
                None => break
            }
        }
        expr
    }

    fn parse_expr_9(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-9>");
        let mut expr: Expr = self.parse_expr_8();
        loop {
            let maybe_op: Option<BinaryOp> = match self.peek() {
                Some(LogicalAnd) => Some(BinaryOp::And), // Just markers
                Some(LogicalOr) => Some(BinaryOp::Or),
                _ => None,
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(op) => {
                    let loc = self.advance_with();
                    expr = expr.logical(loc, op, self.parse_expr_8());
                },
                _ => break
            }
        }
        expr
    }

    fn parse_expr_10(self: &mut Self) -> Expr {
        trace::trace_parser!("rule <expr-10>");
        let mut expr: Expr = self.parse_expr_10_pattern_lvalue();
        loop {
            let mut loc = self.next_location();
            let maybe_op: Option<BinaryOp> = match self.peek() {
                Some(Equals) => Some(BinaryOp::Equal), // Fake operator
                Some(PlusEquals) => Some(BinaryOp::Add), // Assignment Operators
                Some(MinusEquals) => Some(BinaryOp::Sub),
                Some(MulEquals) => Some(BinaryOp::Mul),
                Some(DivEquals) => Some(BinaryOp::Div),
                Some(AndEquals) => Some(BinaryOp::And),
                Some(OrEquals) => Some(BinaryOp::Or),
                Some(XorEquals) => Some(BinaryOp::Xor),
                Some(LeftShiftEquals) => Some(BinaryOp::LeftShift),
                Some(RightShiftEquals) => Some(BinaryOp::RightShift),
                Some(ModEquals) => Some(BinaryOp::Mod),
                Some(PowEquals) => Some(BinaryOp::Pow),

                // `.=` is special, as it needs to emit `Swap`, then `OpFuncEval(1)`
                Some(DotEquals) => Some(BinaryOp::NotEqual), // Marker

                // Special assignment operators, use their own version of a binary operator
                // Also need to consume the extra token
                Some(Identifier(it)) if it == "max" => match self.peek2() {
                    Some(Equals) => {
                        self.advance();
                        loc |= self.next_location();
                        Some(BinaryOp::Max)
                    },
                    _ => None,
                },
                Some(Identifier(it)) if it == "min" => match self.peek2() {
                    Some(Equals) => {
                        self.advance();
                        loc |= self.next_location();
                        Some(BinaryOp::Min)
                    },
                    _ => None,
                },
                _ => None
            };

            // We have to handle the left hand side, as we may need to rewrite the left hand side based on what we just parsed.
            // Valid LHS expressions for a direct assignment, and their translations are:
            //
            // PushLocal(a)    => StoreLocal(a, <expr>) -> also works the same for globals, upvalues, and late bound globals
            // Index(a, b)     => StoreArray(a, b, <expr>)
            // GetField(a, b)  => SetField(a, b, <expr>)
            //
            // If we have a assignment-expression operator, like `+=`, then we need to do it slightly differently:
            //
            // PushLocal(a)    => StoreLocal(Op(PushLocal(a), <expr>)) -> also works the same for globals, upvalues, and late bound globals
            // Index(a, b)     => SwapArray(a, b, Op, <expr>) -> this is special as it needs to use `OpIndexPeek` for the read, then `StoreArray` for the write
            // GetField(a, b)  => SwapField(a, b, Op, <expr>) -> this is special as it needs to use `GetFieldPeek` for the read, then `SetField` for the write
            //
            // **Note**: Assignments are right-associative, so call <expr-10> recursively instead of <expr-9>
            if let Some(BinaryOp::Equal) = maybe_op { // // Direct assignment statement
                self.advance();
                expr = match expr {
                    Expr(_, ExprType::LValue(lvalue @ (LValueReference::Local(_) | LValueReference::UpValue(_) | LValueReference::Global(_) | LValueReference::LateBoundGlobal(_)))) => {
                        let rhs = self.parse_expr_10();
                        Expr::assign_lvalue(loc, lvalue, rhs)
                    },
                    Expr(_, ExprType::Index(array, index)) => {
                        let rhs = self.parse_expr_10();
                        Expr::assign_array(loc, *array, *index, rhs)
                    },
                    Expr(_, ExprType::GetField(lhs, field_index)) => {
                        let rhs = self.parse_expr_10();
                        lhs.set_field(loc, field_index, rhs)
                    },
                    _ => {
                        self.error(InvalidAssignmentTarget);
                        Expr::nil()
                    },
                }
            } else if let Some(op) = maybe_op {
                self.advance();
                expr = match expr {
                    Expr(lvalue_loc, ExprType::LValue(lvalue @ (LValueReference::Local(_) | LValueReference::UpValue(_) | LValueReference::Global(_) | LValueReference::LateBoundGlobal(_)))) => {
                        let lhs = Expr::lvalue(lvalue_loc, lvalue.clone());
                        let rhs = self.parse_expr_10();
                        Expr::assign_lvalue(loc, lvalue, match op {
                            BinaryOp::NotEqual => lhs.compose(loc, rhs),
                            op => lhs.binary(loc, op, rhs),
                        })
                    },
                    Expr(_, ExprType::Index(array, index)) => {
                        let rhs = self.parse_expr_10();
                        Expr::assign_op_array(loc, *array, *index, op, rhs)
                    },
                    Expr(_, ExprType::GetField(lhs, field_index)) => {
                        let rhs = self.parse_expr_10();
                        lhs.swap_field(loc, field_index, rhs, op)
                    },
                    _ => {
                        self.error(InvalidAssignmentTarget);
                        Expr::nil()
                    },
                }
            } else {
                // Not any kind of assignment statement
                break
            }
        }
        expr
    }

    fn parse_expr_10_pattern_lvalue(self: &mut Self) -> Expr {
        // A subset of `<lvalue> = <rvalue>` get parsed as patterns. These are for non-trivial <lvalue>s, which cannot involve array or property access, and cannot involve operator-assignment statements.
        // We use backtracking as for these cases, we want to parse the `lvalue` separately.

        trace::trace_parser!("rule <expr-10-pattern-lvalue>");
        if self.begin() {
            if let Some(mut lvalue) = self.parse_bare_lvalue() {
                if !lvalue.is_named() {
                    if let Some(Equals) = self.peek() {
                        // At this point, we know enough that this is the only possible parse
                        // We have a nontrivial `<lvalue>`, followed by an assignment statement, so this must be a pattern assignment.
                        let loc = self.advance_with(); // Accept `=`
                        self.accept(); // First and foremost, accept the query.
                        lvalue.resolve_locals(self); // Resolve each local, raising an error if need be.
                        let expr: Expr = self.parse_expr_10(); // Recursively parse, since this is left associative, call <expr-10>
                        return Expr::assign_pattern(loc, lvalue, expr); // And exit this rule
                    }
                }
            }
        }
        self.reject();
        self.parse_expr_9()
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
    use crate::vm::operator::{BinaryOp, UnaryOp};

    use NativeFunction::{OperatorAdd, OperatorDiv, OperatorMul, Print, ReadText};
    use Opcode::{*};

    #[test] fn test_int() { run_expr("123", vec![Int(123)]); }
    #[test] fn test_str() { run_expr("'abc'", vec![Str(1)]); }
    #[test] fn test_unary_minus() { run_expr("-3", vec![Int(3), Unary(UnaryOp::Minus)]); }
    #[test] fn test_binary_mul() { run_expr("3 * 6", vec![Int(3), Int(6), Binary(BinaryOp::Mul)]); }
    #[test] fn test_binary_div() { run_expr("20 / 4 / 5", vec![Int(20), Int(4), Binary(BinaryOp::Div), Int(5), Binary(BinaryOp::Div)]); }
    #[test] fn test_binary_pow() { run_expr("2 ** 10", vec![Int(2), Int(10), Binary(BinaryOp::Pow)]); }
    #[test] fn test_binary_minus() { run_expr("6 - 7", vec![Int(6), Int(7), Binary(BinaryOp::Sub)]); }
    #[test] fn test_binary_and_unary_minus() { run_expr("15 -- 7", vec![Int(15), Int(7), Unary(UnaryOp::Minus), Binary(BinaryOp::Sub)]); }
    #[test] fn test_binary_add_and_mod() { run_expr("1 + 2 % 3", vec![Int(1), Int(2), Int(3), Binary(BinaryOp::Mod), Binary(BinaryOp::Add)]); }
    #[test] fn test_binary_add_and_mod_rev() { run_expr("1 % 2 + 3", vec![Int(1), Int(2), Binary(BinaryOp::Mod), Int(3), Binary(BinaryOp::Add)]); }
    #[test] fn test_binary_shifts() { run_expr("1 << 2 >> 3", vec![Int(1), Int(2), Binary(BinaryOp::LeftShift), Int(3), Binary(BinaryOp::RightShift)]); }
    #[test] fn test_binary_shifts_and_operators() { run_expr("1 & 2 << 3 | 5", vec![Int(1), Int(2), Int(3), Binary(BinaryOp::LeftShift), Binary(BinaryOp::And), Int(5), Binary(BinaryOp::Or)]); }
    #[test] fn test_function_composition() { run_expr("print . read_text", vec![NativeFunction(Print), NativeFunction(ReadText), Swap, OpFuncEval(1)]); }
    #[test] fn test_precedence_with_parens() { run_expr("(1 + 2) * 3", vec![Int(1), Int(2), Binary(BinaryOp::Add), Int(3), Binary(BinaryOp::Mul)]); }
    #[test] fn test_precedence_with_parens_2() { run_expr("6 / (5 - 3)", vec![Int(6), Int(5), Int(3), Binary(BinaryOp::Sub), Binary(BinaryOp::Div)]); }
    #[test] fn test_precedence_with_parens_3() { run_expr("-(1 - 3)", vec![Int(1), Int(3), Binary(BinaryOp::Sub), Unary(UnaryOp::Minus)]); }
    #[test] fn test_function_no_args() { run_expr("print", vec![NativeFunction(Print)]); }
    #[test] fn test_function_one_arg() { run_expr("print(1)", vec![NativeFunction(Print), Int(1), OpFuncEval(1)]); }
    #[test] fn test_function_many_args() { run_expr("print(1,2,3)", vec![NativeFunction(Print), Int(1), Int(2), Int(3), OpFuncEval(3)]); }
    #[test] fn test_multiple_unary_ops() { run_expr("- ! 1", vec![Int(1), Unary(UnaryOp::Not), Unary(UnaryOp::Minus)]); }
    #[test] fn test_multiple_function_calls() { run_expr("print (1) (2) (3)", vec![NativeFunction(Print), Int(1), OpFuncEval(1), Int(2), OpFuncEval(1), Int(3), OpFuncEval(1)]); }
    #[test] fn test_multiple_function_calls_some_args() { run_expr("print () (1) (2, 3)", vec![NativeFunction(Print), OpFuncEval(0), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(2)]); }
    #[test] fn test_multiple_function_calls_no_args() { run_expr("print () () ()", vec![NativeFunction(Print), OpFuncEval(0), OpFuncEval(0), OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence() { run_expr("- print ()", vec![NativeFunction(Print), OpFuncEval(0), Unary(UnaryOp::Minus)]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens() { run_expr("(- print) ()", vec![NativeFunction(Print), Unary(UnaryOp::Minus), OpFuncEval(0)]); }
    #[test] fn test_function_call_unary_op_precedence_with_parens_2() { run_expr("- (print () )", vec![NativeFunction(Print), OpFuncEval(0), Unary(UnaryOp::Minus)]); }
    #[test] fn test_function_call_binary_op_precedence() { run_expr("print ( 1 ) + ( 2 ( 3 ) )", vec![NativeFunction(Print), Int(1), OpFuncEval(1), Int(2), Int(3), OpFuncEval(1), Binary(BinaryOp::Add)]); }
    #[test] fn test_function_call_parens_1() { run_expr("print . read_text (1 + 3) (5)", vec![NativeFunction(Print), NativeFunction(ReadText), Int(1), Int(3), Binary(BinaryOp::Add), OpFuncEval(1), Int(5), OpFuncEval(1), Swap, OpFuncEval(1)]); }
    #[test] fn test_function_call_parens_2() { run_expr("( print . read_text (1 + 3) ) (5)", vec![NativeFunction(Print), NativeFunction(ReadText), Int(1), Int(3), Binary(BinaryOp::Add), OpFuncEval(1), Swap, OpFuncEval(1), Int(5), OpFuncEval(1)]); }
    #[test] fn test_function_composition_with_is() { run_expr("'123' . int is int . print", vec![Str(1), NativeFunction(NativeFunction::Int), NativeFunction(NativeFunction::Int), Binary(BinaryOp::Is), Swap, OpFuncEval(1), NativeFunction(Print), Swap, OpFuncEval(1)]); }
    #[test] fn test_and() { run_expr("1 < 2 and 3 < 4", vec![Int(1), Int(2), Binary(BinaryOp::LessThan), JumpIfFalse(8), Pop, Int(3), Int(4), Binary(BinaryOp::LessThan)]); }
    #[test] fn test_or() { run_expr("1 < 2 or 3 < 4", vec![Int(1), Int(2), Binary(BinaryOp::LessThan), JumpIfTrue(8), Pop, Int(3), Int(4), Binary(BinaryOp::LessThan)]); }
    #[test] fn test_precedence_1() { run_expr("1 . 2 & 3 > 4", vec![Int(1), Int(2), Int(3), Binary(BinaryOp::And), Swap, OpFuncEval(1), Int(4), Binary(BinaryOp::GreaterThan)]); }
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
    #[test] fn test_binary_ops() { run_expr("(*) * (+) + (/)", vec![NativeFunction(OperatorMul), NativeFunction(OperatorAdd), Binary(BinaryOp::Mul), NativeFunction(OperatorDiv), Binary(BinaryOp::Add)]); }
    #[test] fn test_if_then_else() { run_expr("(if true then 1 else 2)", vec![True, JumpIfFalsePop(4), Int(1), Jump(5), Int(2)]); }
    #[test] fn test_zero_equals_zero() { run_expr("0 == 0", vec![Int(0), Int(0), Binary(BinaryOp::Equal)]); }
    #[test] fn test_zero_equals_zero_no_spaces() { run_expr("0==0", vec![Int(0), Int(0), Binary(BinaryOp::Equal)]); }

    #[test] fn test_let_eof() { run_err("let", "Expected a variable binding, either a name, or '_', or pattern (i.e. 'x, (_, y), *z'), got end of input instead\n  at: line 1 (<test>)\n\n1 | let\n2 |     ^^^\n"); }
    #[test] fn test_let_no_identifier() { run_err("let =", "Expected a variable binding, either a name, or '_', or pattern (i.e. 'x, (_, y), *z'), got '=' token instead\n  at: line 1 (<test>)\n\n1 | let =\n2 |     ^\n"); }
    #[test] fn test_let_expression_eof() { run_err("let x =", "Expected an expression terminal, got end of input instead\n  at: line 1 (<test>)\n\n1 | let x =\n2 |         ^^^\n"); }
    #[test] fn test_let_no_expression() { run_err("let x = &", "Expected an expression terminal, got '&' token instead\n  at: line 1 (<test>)\n\n1 | let x = &\n2 |         ^\n"); }
    #[test] fn test_top_level_fn_no_name() { run_err("fn () {}", "Expecting a function name after 'fn' keyword, got '(' token instead\n  at: line 1 (<test>)\n\n1 | fn () {}\n2 |    ^\n"); }
    #[test] fn test_expression_function_with_name() { run_err("(fn hello() {})", "Expected a '(' token, got identifier 'hello' instead\n  at: line 1 (<test>)\n\n1 | (fn hello() {})\n2 |     ^^^^^\n"); }
    #[test] fn test_top_level_function_in_error_recovery_mode() { run_err("+ fn hello() {}", "Expected an expression terminal, got '+' token instead\n  at: line 1 (<test>)\n\n1 | + fn hello() {}\n2 | ^\n"); }

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
    #[test] fn test_local_variable_reference() { run("local_variable_reference"); }
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

        let compile = parser::parse(false, result);
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

        let compile: CompileResult = parser::parse(false, scan_result);
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

        let parse_result: CompileResult = parser::parse(false, scan_result);
        let mut lines: Vec<String> = parse_result.disassemble(&view);
        if !parse_result.errors.is_empty() {
            for error in &parse_result.errors {
                lines.push(view.format(error));
            }
        }

        trace::compare_test_resource_content(&root, lines);
    }
}