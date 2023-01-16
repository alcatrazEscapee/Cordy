use std::collections::VecDeque;
use std::rc::Rc;

use crate::compiler::scanner::{ScanResult, ScanToken};
use crate::compiler::CompileResult;
use crate::stdlib::NativeFunction;
use crate::vm::opcode::Opcode;
use crate::vm::value::FunctionImpl;
use crate::{stdlib, trace};
use crate::misc::MaybeRc;

use ScanToken::{*};
use ParserErrorType::{*};
use Opcode::{*};
use NativeFunction::{*};

pub const RULE_INCREMENTAL: ParseRule = |mut parser| parser.parse_incremental();
pub const RULE_EXPRESSION: ParseRule = |mut parser| parser.parse_expression();

pub type ParseRule = fn(Parser) -> ();


/// Create a default empty `CompileResult`. This is semantically equivalent to parsing an empty program, but will output nothing.
pub fn default() -> CompileResult {
    parse_rule(vec![], |_| ())
}


/// Parse a complete `CompileResult` from the given `ScanResult`
pub fn parse(scan_result: ScanResult) -> CompileResult {
    parse_rule(scan_result.tokens, |mut parser| parser.parse())
}


pub fn parse_incremental(scan_result: ScanResult, code: &mut Vec<Opcode>, locals: &mut Vec<Locals>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, line_numbers: &mut Vec<u16>, globals: &mut Vec<String>, rule: fn(Parser) -> ()) -> Vec<ParserError> {

    let mut errors: Vec<ParserError> = Vec::new();
    let mut maybe_functions: Vec<MaybeRc<FunctionImpl>> = functions.iter().map(|u| MaybeRc::Rc(u.clone())).collect();

    rule(Parser::new(scan_result.tokens, code, locals, &mut errors, strings, constants, &mut maybe_functions, line_numbers, &mut Vec::new(), globals));

    for maybe in maybe_functions {
        match maybe {
            MaybeRc::Raw(func) => functions.push(Rc::new(func)),
            _ => {}
        }
    }

    errors
}


fn parse_rule(tokens: Vec<ScanToken>, rule: fn(Parser) -> ()) -> CompileResult {

    let mut code: Vec<Opcode> = Vec::new();
    let mut errors: Vec<ParserError> = Vec::new();

    let mut strings: Vec<String> = vec![String::new()];
    let mut constants: Vec<i64> = vec![0, 1];
    let mut functions: Vec<MaybeRc<FunctionImpl>> = Vec::new();

    let mut line_numbers: Vec<u16> = Vec::new();
    let mut globals: Vec<String> = Vec::new();
    let mut locals: Vec<String> = Vec::new();

    rule(Parser::new(tokens, &mut code, &mut vec![Locals::new()], &mut errors, &mut strings, &mut constants, &mut functions, &mut line_numbers, &mut locals, &mut globals));

    CompileResult {
        code,
        errors,

        strings,
        constants,
        functions: functions.into_iter().map(|u| u.into_rc()).collect::<Vec<Rc<FunctionImpl>>>(),

        line_numbers,
        locals,
        globals,
    }
}


// ===== Parser Data Structures ===== //


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ParserError {
    pub error: ParserErrorType,
    pub lineno: usize,
}

impl ParserError {
    fn new(error: ParserErrorType, lineno: usize) -> ParserError {
        ParserError { error, lineno }
    }

    pub fn is_eof(self: &Self) -> bool {
        match &self.error {
            ExpectedToken(_, it) => it.is_none(),
            ExpectedExpressionTerminal(it) | ExpectedCommaOrEndOfArguments(it) | ExpectedCommaOrEndOfList(it)  | ExpectedCommaOrEndOfVector(it) | ExpectedCommaOrEndOfDict(it) | ExpectedCommaOrEndOfSet(it) | ExpectedColonOrEndOfSlice(it) | ExpectedStatement(it) | ExpectedVariableNameAfterLet(it) | ExpectedVariableNameAfterFor(it) | ExpectedFunctionNameAfterFn(it) | ExpectedFunctionBlockOrArrowAfterFn(it) | ExpectedParameterOrEndOfList(it) | ExpectedCommaOrEndOfParameters(it) | ExpectedPatternTerm(it) | ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(it) | ExpectedUnderscoreOrVariableNameOrPattern(it) => it.is_none(),
            UnexpectedTokenAfterEoF(_) | LocalVariableConflict(_) | LocalVariableConflictWithNativeFunction(_) | UndeclaredIdentifier(_) | InvalidAssignmentTarget | MultipleVariadicTermsInPattern | LetWithPatternBindingNoExpression | BreakOutsideOfLoop | ContinueOutsideOfLoop => false,
        }
    }
}


#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ParserErrorType {
    UnexpectedTokenAfterEoF(ScanToken),

    ExpectedToken(ScanToken, Option<ScanToken>),
    ExpectedExpressionTerminal(Option<ScanToken>),
    ExpectedCommaOrEndOfArguments(Option<ScanToken>),
    ExpectedCommaOrEndOfList(Option<ScanToken>),
    ExpectedCommaOrEndOfVector(Option<ScanToken>),
    ExpectedCommaOrEndOfDict(Option<ScanToken>),
    ExpectedCommaOrEndOfSet(Option<ScanToken>),
    ExpectedColonOrEndOfSlice(Option<ScanToken>),
    ExpectedStatement(Option<ScanToken>),
    ExpectedVariableNameAfterLet(Option<ScanToken>),
    ExpectedVariableNameAfterFor(Option<ScanToken>),
    ExpectedFunctionNameAfterFn(Option<ScanToken>),
    ExpectedFunctionBlockOrArrowAfterFn(Option<ScanToken>),
    ExpectedParameterOrEndOfList(Option<ScanToken>),
    ExpectedCommaOrEndOfParameters(Option<ScanToken>),
    ExpectedPatternTerm(Option<ScanToken>),
    ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(Option<ScanToken>),
    ExpectedUnderscoreOrVariableNameOrPattern(Option<ScanToken>),

    LocalVariableConflict(String),
    LocalVariableConflictWithNativeFunction(String),
    UndeclaredIdentifier(String),

    InvalidAssignmentTarget,
    MultipleVariadicTermsInPattern,
    LetWithPatternBindingNoExpression,
    BreakOutsideOfLoop,
    ContinueOutsideOfLoop,
}


pub struct Parser<'a> {
    input: VecDeque<ScanToken>,
    output: &'a mut Vec<Opcode>,
    errors: &'a mut Vec<ParserError>,

    lineno: u16,
    line_numbers: &'a mut Vec<u16>,

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

    /// A stack of nested functions, each of which have their own table of locals.
    /// While this mirrors the call stack it may not be representative. The only thing we can assume is that when a function is declared, all locals in the enclosing function are accessible.
    locals: &'a mut Vec<Locals>,

    late_bound_globals: Vec<LateBoundGlobal>, // Table of all late bound globals, as they occur.
    synthetic_local_index: usize, // A counter for unique synthetic local variables (`$1`, `$2`, etc.)
    scope_depth: u16, // Current scope depth
    function_depth: u16,

    strings: &'a mut Vec<String>,
    constants: &'a mut Vec<i64>,
    functions: &'a mut Vec<MaybeRc<FunctionImpl>>,
}

#[derive(Debug)]
pub struct Locals {
    /// The local variables in the current function.
    /// Top level local variables are considered global, even though they still might be block scoped ('true global' are top level function in no block scope, as they can never go out of scope).
    locals: Vec<Local>,
    /// An array of captured upvalues for this function, either due to an inner function requiring them, or this function needing to capture locals from it's enclosing function
    upvalues: Vec<UpValue>,
    /// Loop stack
    /// Each frame represents a single loop, which `break` and `continue` statements refer to
    /// `continue` jumps back to the beginning of the loop, aka the first `usize` (loop start)
    /// `break` statements jump back to the end of the loop, which needs to be patched later. The values to be patched record themselves in the stack at the current loop level
    loops: Vec<Loop>,
}

impl Locals {
    pub fn empty() -> Vec<Locals> {
        vec![Locals::new()]
    }

    pub fn len(self: &Self) -> usize {
        self.locals.len()
    }

    fn new() -> Locals {
        Locals { locals: Vec::new(), upvalues: Vec::new(), loops: Vec::new() }
    }
}

#[derive(Debug)]
struct UpValue {
    /// `true` = local variable in enclosing function, `false` = upvalue in enclosing function
    is_local: bool,

    /// Either a reference to an index in the enclosing function's `locals` (which are stack offset),
    /// or a reference to the enclosing function's `upvalues` (which can be accessed via stack offset 0 -> upvalues, if it is a closure
    index: u16,
}

impl UpValue {
    fn new(is_local: bool, index: u16) -> UpValue {
        UpValue { is_local, index }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
struct Loop {
    start_index: u16,
    scope_depth: u16,
    break_statements: Vec<u16>
}

impl Loop {
    fn new(start_index: u16, depth: u16) -> Loop {
        Loop { start_index, scope_depth: depth, break_statements: Vec::new() }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
struct Local {
    name: String,
    index: u16,
    scope_depth: u16,
    function_depth: u16,
    initialized: bool,
}

impl Local {
    fn new(name: String, index: usize, scope_depth: u16, function_depth: u16) -> Local {
        Local { name, index: index as u16, scope_depth, function_depth, initialized: false }
    }

    /// Sets this local as `initialized`, meaning it is pushed onto the stack and can be referred to in expressions.
    fn initialize(self: &mut Self) { self.initialized = true; }

    fn is_global(self: &Self) -> bool { self.function_depth == 0 }
    fn is_true_global(self: &Self) -> bool { self.function_depth == 0 && self.scope_depth == 0 }

    /// Resolves the `local` as a `VariableType`, based on the function and scope depth of the local.
    fn resolve_type(self: &Self) -> VariableType {
        if self.is_true_global() {
            VariableType::TrueGlobal(self.index)
        } else if self.is_global() {
            VariableType::Global(self.index)
        } else {
            VariableType::Local(self.index)
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
enum VariableType {
    NativeFunction(NativeFunction),
    Local(u16),
    Global(u16),
    TrueGlobal(u16),
    LateBoundGlobal(LateBoundGlobal),
    UpValue(u16, bool), // index, is_local
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

#[derive(Debug, Clone)]
enum VariableBinding {
    Pattern(Pattern),
    Named(usize),
    Empty
}

impl VariableBinding {

    /// Initializes all local variables used in this pattern, or named variable binding
    fn init_all_locals(self: &Self, parser: &mut Parser) {
        match self {
            VariableBinding::Pattern(it) => it.init_all_locals(parser),
            VariableBinding::Named(local) => parser.current_locals_mut().locals[*local].initialize(),
            VariableBinding::Empty => {},
        }
    }

    /// Pushes default values for all local variables used in this pattern or named variable binding
    fn push_local_default_values(self: &Self, parser: &mut Parser) {
        match self {
            VariableBinding::Pattern(it) => it.emit_local_default_values(parser),
            VariableBinding::Named(_) => parser.push(Nil),
            VariableBinding::Empty => {},
        }
    }

    /// Emits code for store local variables associated to this pattern or named variable binding, and then pops the value being stored.
    fn push_store_locals_and_pop(self: &Self, parser: &mut Parser) {
        match self {
            VariableBinding::Pattern(it) => it.emit_destructuring(parser),
            VariableBinding::Named(local) => {
                parser.push_store_local(*local);
                parser.push(Opcode::Pop);
            },
            VariableBinding::Empty => parser.push(Opcode::Pop),
        }
    }
}

#[derive(Debug, Clone)]
enum Pattern {
    TermEmpty,
    TermVarEmpty,
    Term(usize),
    TermVar(usize),
    Terms(Vec<Pattern>)
}

impl Pattern {

    /// Emits code for initializing all local variables (to `Nil`) as part of this pattern.
    fn emit_local_default_values(self: &Self, parser: &mut Parser) {
        match self {
            Pattern::TermEmpty | Pattern::TermVarEmpty => {},
            Pattern::Term(_) | Pattern::TermVar(_) => parser.push(Nil),
            Pattern::Terms(terms) => {
                for term in terms {
                    term.emit_local_default_values(parser);
                }
            }
        }
    }

    /// Initializes all local variables used in this pattern
    fn init_all_locals(self: &Self, parser: &mut Parser) {
        match self {
            Pattern::TermEmpty | Pattern::TermVarEmpty => {},
            Pattern::Term(local) | Pattern::TermVar(local) => {
                parser.current_locals_mut().locals[*local].initialize();
                parser.push_inc_global(*local);
            },
            Pattern::Terms(terms) => {
                for term in terms {
                    term.init_all_locals(parser);
                }
            }
        }
    }

    /// Emits code for destructuring this pattern
    /// Assumes the variable containing the to-be-destructured iterable sits atop the stack, and that all local variables are already present (and initialized to `Nil`) in their respective stack slots.
    /// This will ultimately pop the iterable on top of the stack, once all variables have been assigned from this pattern.
    fn emit_destructuring(self: &Self, parser: &mut Parser) {
        let terms = match self {
            Pattern::Terms(v) => v,
            _ => panic!("Top level pattern must be a `Pattern::Terms`")
        };

        let is_variadic = terms.iter()
            .any(|t| t.is_variadic_term());
        let len: i64 = if is_variadic { terms.len() - 1 } else { terms.len() } as i64;
        let constant_len = parser.declare_constant(len);

        parser.push(if is_variadic { CheckLengthGreaterThan(constant_len) } else { CheckLengthEqualTo(constant_len) });

        let mut index: i64 = 0;
        for term in terms {
            match term {
                Pattern::TermEmpty => {
                    // Just advance the index
                    index += 1;
                },
                Pattern::TermVarEmpty => {
                    // Advance the index by the missing elements (start indexing in reverse)
                    index = -(len - index);
                },
                Pattern::Term(local) => {
                    let constant_index = parser.declare_constant(index);

                    parser.push(Opcode::Int(constant_index));
                    parser.push(OpIndexPeek); // [ it[index], index, it, ...]
                    parser.push_store_local(*local); // stores it[index]
                    parser.push(PopN(2)); // [it, ...]

                    index += 1;
                },
                Pattern::TermVar(local) => {
                    // index = the next index in the iterable to take = the number of elements already accessed
                    // therefor, len - index = the number of elements left we need to access exactly, which must be indices -1, -2, ... -(len - index)
                    // so our slice excludes these, so the 'high' value must be -(len - index)
                    // this is then also exactly what we set our index to next
                    let constant_low = parser.declare_constant(index);
                    index = -(len - index);
                    let constant_high = parser.declare_constant(index);

                    parser.push(Dup); // [it, it, ...]
                    parser.push(Opcode::Int(constant_low)); // [low, it, it, ...]
                    parser.push(if index == 0 { Nil } else { Opcode::Int(constant_high) }); // [high, low, it, it, ...]
                    parser.push(OpSlice); // [it[low:high], it, ...]
                    parser.push_store_local(*local); // stores it[low:high]
                    parser.push(Opcode::Pop); // [it, ...]
                },
                terms @ Pattern::Terms(_) => {
                    // Index as if this was a `Term`, but then invoke the emit recursively, with the value still on the stack, treating it as the iterable.
                    let constant_index = parser.declare_constant(index);

                    parser.push(Opcode::Int(constant_index));
                    parser.push(OpIndexPeek); // [ it[index], index, it, ...]
                    terms.emit_destructuring(parser); // [ index, it, ...]
                    parser.push(Opcode::Pop); // [it, ...]

                    index += 1;
                }
            }
        }

        parser.push(Opcode::Pop); // Pop the iterable
    }

    fn is_simple(self: &Self) -> Option<usize> {
        let terms = match self {
            Pattern::Terms(v) => v,
            _ => panic!("Top level pattern must be a `Pattern::Terms`")
        };

        if terms.iter().all(|t| t.is_simple_term()) {
            Some(terms.len())
        } else {
            None
        }
    }

    fn is_variadic_term(self: &Self) -> bool {
        match self {
            Pattern::TermVarEmpty | Pattern::TermVar(_) => true,
            _ => false
        }
    }

    fn is_simple_term(self: &Self) -> bool {
        match self {
            Pattern::Term(_) => true,
            _ => false
        }
    }
}


// ===== Main Parser Implementation ===== //



impl Parser<'_> {

    fn new<'a, 'b : 'a>(tokens: Vec<ScanToken>, output: &'b mut Vec<Opcode>, locals: &'b mut Vec<Locals>, errors: &'b mut Vec<ParserError>, strings: &'b mut Vec<String>, constants: &'b mut Vec<i64>, functions: &'b mut Vec<MaybeRc<FunctionImpl>>, line_numbers: &'b mut Vec<u16>, locals_reference: &'b mut Vec<String>, globals_reference: &'b mut Vec<String>) -> Parser<'a> {
        Parser {
            input: tokens.into_iter().collect::<VecDeque<ScanToken>>(),
            output,
            errors,

            lineno: 0,
            line_numbers,
            locals_reference,
            globals_reference,

            error_recovery: false,
            prevent_expression_statement: false,
            delay_pop_from_expression_statement: false,

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
        self.pop_locals_in_current_scope_depth(true); // Pop top level 'local' variables
        self.teardown();
    }

    fn parse_incremental(self: &mut Self) {
        trace::trace_parser!("rule <root-incremental>");
        self.parse_statements();
        if self.delay_pop_from_expression_statement {
            self.push(NativeFunction(Print));
            self.push(Swap);
            self.push(OpFuncEval(1));
            self.push(Opcode::Pop);
        }
        // Don't pop locals
        self.teardown();
    }

    fn teardown(self: &mut Self) {
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
                    self.current_locals_mut().locals[index].initialize();  // Functions are always initialized, as they can be recursive
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
        self.locals.push(Locals::new());
        self.function_depth += 1;
        self.scope_depth += 1;

        for arg in args {
            match self.declare_local(arg) {
                Some(index) => {
                    // They are automatically initialized, and we don't need to push `Nil` for them, since they're provided from the stack due to call semantics
                    self.current_locals_mut().locals[index].initialize();
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

        // Update the function, if present, with the tail of the function
        // This makes tracking ownership in containing functions easier during error reporting
        if let Some(func_id) = func_id {
            self.functions[func_id as usize].unbox_mut().tail = self.next_opcode() as usize;
        }

        self.push(Return); // Returns the last expression in the function

        let end: u16 = self.next_opcode(); // Repair the jump, to skip the function body itself
        self.output[jump] = Jump(end);

        // Since `Return` cleans up all function locals, we just discard them from the parser without emitting any `Pop` tokens.
        // We do this twice, once for function locals, and once for function parameters (since they live in their own scope)
        self.prevent_expression_statement = false;
        self.pop_locals_in_current_scope_depth(false);
        self.scope_depth -= 1;

        self.pop_locals_in_current_scope_depth(false);
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
            self.push(Nil);
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
                    self.push(Nil);
                }
                let after_else: u16 = self.next_opcode();
                self.output[jump] = Jump(after_else);
            },
            _ => {
                // No `else`, but we need to wire in a fake `else` statement which just pushes `Nil` so each branch still pushes a value
                let jump = self.reserve();
                let after_if: u16 = self.next_opcode();
                self.output[jump_if_false as usize] = JumpIfFalsePop(after_if);
                self.push(Nil);
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
        let loop_depth: u16 = self.scope_depth;
        self.current_locals_mut().loops.push(Loop::new(loop_start, loop_depth));

        self.parse_expression(); // While condition
        let jump_if_false = self.reserve(); // Jump to the end
        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push_delayed_pop(); // Inner loop expressions cannot yield out of the loop
        self.push(Jump(loop_start));

        let loop_end: u16 = self.next_opcode(); // After the jump, the next opcode is 'end of loop'. Repair all break statements
        let break_opcodes: Vec<u16> = self.current_locals_mut().loops.pop().unwrap().break_statements;
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
        let loop_depth: u16 = self.scope_depth;
        self.current_locals_mut().loops.push(Loop::new(loop_start, loop_depth));

        self.parse_block_statement(); // Inner loop statements, and jump back to front
        self.push_delayed_pop(); // Loops can't return a value
        self.push(Jump(loop_start));

        let loop_end: u16 = self.next_opcode(); // After the jump, the next opcode is 'end of loop'. Repair all break statements
        let break_opcodes: Vec<u16> = self.current_locals_mut().loops.pop().unwrap().break_statements;
        for break_opcode in break_opcodes {
            self.output[break_opcode as usize] = Jump(loop_end);
        }
    }

    fn parse_for_statement(self: &mut Self) {
        trace::trace_parser!("rule <for-statement>");

        self.push_delayed_pop();
        self.advance(); // Consume `for`

        // `for` loops have declared (including synthetic) variables within their own scope
        // the variable binding of a `for` can also support pattern expressions.
        self.scope_depth += 1;
        let local_x: VariableBinding = self.parse_variable_binding().unwrap_or(VariableBinding::Empty);

        self.expect(KeywordIn);

        local_x.push_local_default_values(self);

        self.parse_expression();

        local_x.init_all_locals(self);

        // At the beginning and end of the loop, this local sits on the top of the stack.
        // We still need to declare it's stack slot space, even though we don't reference it via load/store local opcodes
        self.declare_synthetic_local();

        // Applies to the top of the stack
        self.push(InitIterable);

        // Test
        let jump: u16 = self.next_opcode();
        self.push(TestIterable);
        let jump_if_false_pop = self.reserve();

        // Initialize locals
        local_x.push_store_locals_and_pop(self);

        // Parse the body of the loop, and emit the delayed pop - the stack is restored to the same state as the top of the loop.
        // So, we jump to the top of the loop, where we test/increment
        self.parse_block_statement();
        self.push_delayed_pop();
        self.push(Jump(jump));

        // Fix the jump
        self.output[jump_if_false_pop] = JumpIfFalsePop(self.next_opcode());

        // Cleanup the `for` loop locals
        self.pop_locals_in_current_scope_depth(true);
        self.scope_depth -= 1;
    }

    fn parse_break_statement(self: &mut Self) {
        trace::trace_parser!("rule <break-statement>");
        self.push_delayed_pop();
        self.advance();
        match self.current_locals().loops.last() {
            Some(_) => {
                self.pop_locals_in_current_loop();
                let jump = self.reserve();
                self.current_locals_mut().loops.last_mut().unwrap().break_statements.push(jump as u16);
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
                let jump_to: u16 = loop_stmt.start_index;
                self.pop_locals_in_current_loop();
                self.push(Jump(jump_to));
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
            let variable: VariableBinding = match self.parse_variable_binding() {
                // Note that `let _` is rather silly but we can emit perfectly legal code for it, so we allow it
                Some(v) => v,
                None => break
            };

            let expression_present = match self.peek() {
                Some(Equals) => { // x = <expr>, so parse the expression
                    self.advance();
                    self.prevent_expression_statement = true;

                    // Before we parse an expression, if we're declaring a pattern variable, then emit the pattern's locals
                    // We need the expression to be on the top of the stack to do destructuring
                    if let VariableBinding::Pattern(pattern) = &variable {
                        pattern.emit_local_default_values(self)
                    }

                    self.parse_expression();
                    true
                },
                _ => {
                    match &variable {
                        VariableBinding::Pattern(pattern) => {
                            // If the pattern is a simple pattern (just `x, y, z, ...`), then this is actually legal - and we need to push the respective amount of `Nil`, which we assumed was a pattern instead of a sequence of `let` declarations.
                            if let Some(n) = pattern.is_simple() {
                                for _ in 0..n {
                                    self.push(Nil)
                                }
                            } else {
                                self.semantic_error(LetWithPatternBindingNoExpression);
                            }
                        },
                        _ => {
                            self.push(Nil); // Initialize to 'nil'
                        }
                    }
                    false
                },
            };

            match &variable {
                VariableBinding::Pattern(pattern) => {
                    // A pattern binding needs to emit a sequence of code that takes the top operand on the stack, and initializes all local variables
                    // This sequence requires the variables to already have stack slots, which we setup before parsing the expression
                    // So, we just need to emit code for the destructuring, and initialize all local variables
                    pattern.init_all_locals(self);
                    if pattern.is_simple().is_none() || expression_present {
                        pattern.emit_destructuring(self);
                    }
                },
                VariableBinding::Named(local) => {
                    // Local declarations don't have an explicit `store` opcode
                    // They just push their value onto the stack, and we know the location will equal that of the Local's index
                    // However, after we initialize a local we need to mark it initialized, so we can refer to it in expressions
                    self.current_locals_mut().locals[*local].initialize();
                    self.push_inc_global(*local);
                },
                VariableBinding::Empty => {
                    self.push(Opcode::Pop); // No variable to bind to, so just pop the expression (or `Nil`, if we emitted it)
                },
            }

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


    // ===== Pattern Parsing ===== //

    /// Parses a variable binding.
    /// A variable binding may be a named variable (like in `let x`), an empty binding (`let _`, although in this context this is useless), or a pattern (`let x, (_, y), *z = `
    /// All local variables named in either the named variable or pattern will be defined as locals by this method, and the variable binding is returned, if present. Otherwise, `None` is returned and an error will have been raised already.
    fn parse_variable_binding(self: &mut Self) -> Option<VariableBinding> {
        trace::trace_parser!("rule <variable-binding>");

        if let Some(p) = self.parse_pattern() {

            // Recognize special patterns, which we do not want to emit pattern-like declarations for
            // These are single non-variadic terms, which can either be a local variable or empty binding
            let terms = match &p {
                Pattern::Terms(terms) => terms,
                _ => panic!("Should always be top-level Pattern::Terms")
            };

            trace::trace_parser!("pattern {:?}", p);

            if terms.len() == 1 {
                match &terms[0] {
                    Pattern::Term(local) => return Some(VariableBinding::Named(*local)),
                    Pattern::TermEmpty => return Some(VariableBinding::Empty),
                    _ => {} // Single, non-simple term
                }
            }

            Some(VariableBinding::Pattern(p))
        } else {
            match self.peek() {
                Some(Identifier(_)) => {
                    let name: String = self.take_identifier();
                    match self.declare_local(name) {
                        Some(local) => Some(VariableBinding::Named(local)),
                        None => None
                    }
                }
                Some(Underscore) => {
                    self.advance();
                    Some(VariableBinding::Empty)
                },
                _ => {
                    self.error_with(|t| ExpectedUnderscoreOrVariableNameOrPattern(t));
                    None
                }
            }
        }
    }

    /// Optionally parses a `Pattern`, which will always be a top level `Pattern::Terms`
    /// If this does not parse a pattern (i.e. returns `None`), the parser state will not be modified.
    fn parse_pattern(self: &mut Self) -> Option<Pattern> {
        if self.detect_pattern() {
            trace::trace_parser!("rule <pattern>");
            let mut terms: Vec<Pattern> = Vec::new();
            let mut found_variadic_term: bool = false;

            terms.push(self.parse_pattern_term());
            loop {
                match self.peek() {
                    Some(Comma) => {
                        self.advance();
                        let term = self.parse_pattern_term();
                        if term.is_variadic_term() {
                            if found_variadic_term {
                                self.error(MultipleVariadicTermsInPattern);
                            } else {
                                found_variadic_term = true;
                            }
                        }
                        trace::trace_parser!("pattern term {:?}", term);
                        terms.push(term)
                    },
                    _ => break
                }
            }

            Some(Pattern::Terms(terms))
        } else {
            None
        }
    }

    fn parse_pattern_term(self: &mut Self) -> Pattern {
        trace::trace_parser!("rule <pattern-term>");
        match self.peek() {
            Some(Identifier(_)) => {
                let name: String = self.take_identifier();
                match self.declare_local(name) {
                    Some(local) => Pattern::Term(local),
                    _ => Pattern::TermEmpty,
                }
            },
            Some(Underscore) => {
                self.advance();
                Pattern::TermEmpty
            },
            Some(Mul) => {
                self.advance();
                match self.peek() {
                    Some(Identifier(_)) => {
                        let name: String = self.take_identifier();
                        match self.declare_local(name) {
                            Some(local) => Pattern::TermVar(local),
                            _ => Pattern::TermEmpty,
                        }
                    },
                    Some(Underscore) => {
                        self.advance();
                        Pattern::TermVarEmpty
                    },
                    _ => {
                        self.error_with(|t| ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(t));
                        Pattern::TermEmpty
                    }
                }
            },
            Some(OpenParen) => {
                self.advance();
                let term = match self.parse_pattern() {
                    Some(t) => t,
                    None => {
                        self.error_with(|t| ExpectedPatternTerm(t));
                        Pattern::TermEmpty
                    }
                };
                self.expect(CloseParen);
                term
            },
            _ => {
                self.error_with(|t| ExpectedPatternTerm(t));
                Pattern::TermEmpty
            },
        }
    }

    fn detect_pattern(self: &mut Self) -> bool {
        match self.peek() {
            Some(Identifier(_)) => match self.peek2() {
                Some(Comma) => true,
                _ => false
            },
            Some(Underscore) | Some(OpenParen) | Some(Mul) => true,
            _ => false
        }
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
                    VariableType::NativeFunction(b) => self.push(NativeFunction(b)),
                    VariableType::Local(local) => self.push(PushLocal(local)),
                    VariableType::Global(local) => self.push(PushGlobal(local, true)),
                    VariableType::TrueGlobal(local) => self.push(PushGlobal(local, false)),
                    VariableType::LateBoundGlobal(global) => {
                        self.late_bound_globals.push(global);
                        self.push(Noop) // Push `Noop` for now, fix it later
                    },
                    VariableType::UpValue(index, is_local) => self.push(PushUpValue(index, is_local)),
                    _ => self.semantic_error(UndeclaredIdentifier(string))
                };
            },
            Some(StringLiteral(_)) => {
                let string: String = self.take_str();
                let sid: u16 = self.declare_string(string);
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
            Some(KeywordFn) => self.parse_expression_function(),
            Some(KeywordIf) => self.parse_expr_1_inline_if_then_else(),
            _ => self.error_with(|t| ExpectedExpressionTerminal(t)),
        }
    }

    fn parse_expr_1_partial_operator_left(self: &mut Self) -> bool {
        trace::trace_parser!("rule <expr-1-partial-operator-left>");
        // Open `(` usually resolves precedence, so it begins parsing an expression from the top again
        // However it *also* can be used to wrap around a partial evaluation of a literal operator, for example
        // (-) => OperatorUnarySub
        // (+ 3) => Int(3) OperatorAdd OpFuncCompose
        // (*) => OperatorMul
        // So, if we peek ahead and see an operator, we know this is a expression of that sort and we need to handle accordingly
        // We *also* know that we will never see a binary operator begin an expression
        let mut unary: Option<NativeFunction> = None;
        let mut binary: Option<NativeFunction> = None;
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
        let length: u16 = self.declare_constant(length);
        self.push(Opcode::List(length));
        self.expect(CloseSquareBracket);
    }

    fn parse_expr_1_dict_or_set_literal(self: &mut Self) {
        trace::trace_interpreter!("rule <expr-1-dict-or-set-literal>");
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
        let length: u16 = self.declare_constant(length);
        self.push(if is_dict { Opcode::Dict(length) } else { Opcode::Set(length) });
        self.expect(CloseBrace);
    }

    fn parse_expr_1_inline_if_then_else(self: &mut Self) {
        trace::trace_parser!("rule <expr-1-inline-if-then-else>");

        self.advance(); // Consume `if`
        self.parse_expression(); // condition
        let jump_if_false_pop = self.reserve();
        self.expect(KeywordThen);
        self.parse_expression(); // Value if true
        let jump = self.reserve();
        self.output[jump_if_false_pop] = JumpIfFalsePop(self.next_opcode());
        self.expect(KeywordElse);
        self.parse_expression(); // Value if false
        self.output[jump] = Jump(self.next_opcode());

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
                    Some(KeywordIn) => Some(UnaryLogicalNot), // Special flag for `not in`, which desugars to `!(x in y)
                    _ => None,
                }
                _ => None
            };
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
            match maybe_op {
                Some(UnaryLogicalNot) => {
                    self.advance();
                    self.advance();
                    self.parse_expr_2_unary();
                    self.push(OpIn);
                    self.push(UnaryLogicalNot)
                }
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
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
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
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
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
            if maybe_op.is_some() && self.peek2() == Some(&CloseParen) {
                break
            }
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
            match self.peek() {
                Some(Dot) => {
                    self.advance();
                    self.parse_expr_6();
                    self.push(Swap);
                    self.push(OpFuncEval(1));
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
                    let jump_to: u16 = self.next_opcode();
                    self.output[jump_if_false] = JumpIfFalse(jump_to);
                },
                Some(OpBitwiseOr) => {
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
                match self.last() {
                    Some(PushLocal(id)) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreLocal(id));
                    },
                    Some(PushGlobal(id, local)) => {
                        self.pop();
                        self.parse_expr_10();
                        self.push(StoreGlobal(id, local));
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
                        self.push(op);
                        self.push(StoreLocal(id));
                    },
                    Some(PushGlobal(id, is_local)) => {
                        self.parse_expr_10();
                        self.push(op);
                        self.push(StoreGlobal(id, is_local));
                    },
                    Some(OpIndex) => {
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
        self.functions.push(MaybeRc::new(FunctionImpl::new(head, name, args)));
        (self.functions.len() - 1) as u16
    }


    fn next_opcode(self: &Self) -> u16 {
        self.output.len() as u16
    }

    /// After a `let <name>` or `fn <name>` declaration, tries to declare this as a local variable in the current scope
    /// Returns the index of the local variable in `self.current_locals().locals`
    fn declare_local(self: &mut Self, name: String) -> Option<usize> {

        // Lookup the name as a binding - if it is, it will be denied as we don't allow shadowing global native functions
        if let Some(_) = stdlib::find_native_function(&name) {
            self.semantic_error(LocalVariableConflictWithNativeFunction(name.clone()));
            return None
        }

        // Ensure there are no conflicts within the current scope, as we don't allow shadowing in the same scope.
        for local in &self.locals.last().unwrap().locals {
            if local.scope_depth == self.scope_depth && local.name == name {
                self.semantic_error(LocalVariableConflict(name.clone()));
                return None
            }
        }

        let index = self.declare_local_internal(name);
        let local = &self.locals.last().unwrap().locals[index];

        if local.is_true_global() {

            // Fix references to this global
            for global in &self.late_bound_globals {
                if global.name == local.name {
                    self.output[global.opcode] = PushGlobal(local.index, false);
                }
            }

            // And remove them
            self.late_bound_globals.retain(|global| global.name != local.name);

            // Declare this global variable's name
            self.globals_reference.push(local.name.clone());
        }

        Some(index)
    }

    /// Declares a synthetic local variable. Unlike `declare_local()`, this can never fail.
    /// Returns the index of the local variable in `locals`.
    fn declare_synthetic_local(self: &mut Self) -> usize {
        self.synthetic_local_index += 1;
        self.declare_local_internal(format!("${}", self.synthetic_local_index - 1))
    }

    /// Declares a local variable by the name `name` in the current scope.
    fn declare_local_internal(self: &mut Self, name: String) -> usize {
        let local: Local = Local::new(name, self.locals.last().unwrap().locals.len(), self.scope_depth, self.function_depth);
        self.locals.last_mut().unwrap().locals.push(local);
        self.locals.last().unwrap().locals.len() - 1
    }

    /// Pops all locals declared in the current scope, *before* decrementing the scope depth
    /// Also emits the relevant `OpPop` opcodes to the output token stream.
    fn pop_locals_in_current_scope_depth(self: &mut Self, emit_pop_token: bool) {
        let local_count = self.current_locals().locals.len();
        let scope_depth = self.scope_depth;
        self.current_locals_mut().locals.retain(|e| e.scope_depth != scope_depth);
        if emit_pop_token {
            self.push_pop((local_count - self.current_locals().locals.len()) as u16);
        }
    }

    /// Pops all locals that are within the most recent loop, and outputs the relevant `Pop` / `PopN` opcodes
    /// It **does not** modify the local variable stack, and only outputs these tokens for the purposes of a jump across scopes.
    fn pop_locals_in_current_loop(self: &mut Self) {
        let loop_stmt: &Loop = self.current_locals().loops.last().unwrap();
        let mut count: u16 = 0;

        for local in self.current_locals().locals.iter().rev() {
            if local.scope_depth > loop_stmt.scope_depth {
                count += 1;
            } else {
                break
            }
        }
        self.push_pop(count);
    }

    /// Returns the current function's `locals`
    fn current_locals(self: &Self) -> &Locals {
        self.locals.last().unwrap()
    }

    /// Returns the current function's `locals`
    fn current_locals_mut(self: &mut Self) -> &mut Locals {
        self.locals.last_mut().unwrap()
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
        if let Some(b) = stdlib::find_native_function(name) {
            return VariableType::NativeFunction(b);
        }

        // 1. Search for locals in the current function. This may return `Local`, `Global`, or `TrueGlobal` based on the scope of the variable.
        for local in self.current_locals().locals.iter().rev() {
            if &local.name == name && local.initialized {
                return local.resolve_type()
            }
        }

        // 2. If we are in function depth > 0, we search in enclosing functions (and global scope), for values that can be captured by this function.
        //   - Globals that are not true globals can be captured in the same manner as upvalues (these are fairly uncommon in practice)
        //   - Locals in an enclosing function can be captured.
        if self.function_depth > 0 {
            for depth in (0..self.function_depth).rev() { // Iterate through the range of [function_depth - 1, ... 0]
                for local in self.locals[depth as usize].locals.iter().rev() { // In reverse, as we go inner -> outer scopes
                    if &local.name == name && local.initialized && (local.function_depth > 0 || local.scope_depth > 0) { // Note that it must **not** be a true global, anything else can be captured as an upvalue
                        return self.resolve_upvalue(depth, local.index);
                    }
                }
            }
        }

        // 3. If we are in a function depth > 0, then we can also resolve true globals
        // If we are in function depth == 0, any true globals will be caught and resolved as locals by (1.) (but the opcodes for global load/store will still be emitted)
        if self.function_depth > 0 {
            for local in self.locals[0].locals.iter().rev() {
                if &local.name == name && local.initialized {
                    if local.is_true_global() {
                        return VariableType::TrueGlobal(local.index)
                    }
                }
            }
        }

        // 4. If we are in function depth > 0, we can assume that the variable must be a late bound global.
        // We cannot late bind globals in function depth == 0, as all code there is still procedural.
        if self.function_depth > 0 {
            // Assume a late bound global
            let error = self.deferred_error(UndeclaredIdentifier(name.clone()));
            let global = LateBoundGlobal::new(name.clone(), self.next_opcode() as usize, error);
            return VariableType::LateBoundGlobal(global);
        }

        // In global scope if we still could not resolve a variable, we return `None`
        VariableType::None
    }

    /// Resolves an `UpValue` reference.
    /// For a given reference to a local, defined at a function depth `local_depth` at index `local_index`, this will
    /// bubble up the upvalue through each of the enclosing functions between here and `self.function_depth`, and ensure the variable is added as an `UpValue`.
    fn resolve_upvalue(self: &mut Self, local_depth: u16, local_index: u16) -> VariableType {

        // Capture the local at index `local_index` in the function at `local_depth`
        // If it already exists (is `is_local` and has the same `index` as the target), just grab the upvalue index, otherwise add it and bubble up
        let mut maybe_index: Option<usize> = None;
        for (i, upvalue) in self.locals[local_depth as usize].upvalues.iter().enumerate() {
            if upvalue.index == local_index && upvalue.is_local {
                maybe_index = Some(i);
                break
            }
        }

        // If we did not find it, then capture the local - add this as an upvalue to the function at this depth
        let mut is_local = true;
        let mut index = if let Some(index) = maybe_index {
            index as u16
        } else {
            self.locals[local_depth as usize].upvalues.push(UpValue::new(true, local_index));
            (self.locals[local_depth as usize].upvalues.len() - 1) as u16
        };

        // For any function depths between the enclosing function (self.function_depth - 1), and the function above where we referenced the local (local_depth + 1),
        // we need to add this as an upvalue, referencing an upvalue one level down, to each depth.
        for depth in local_depth + 1..self.function_depth {

            // Only add it if we haven't found an upvalue with the same index and `!is_local` (which is unique).
            let mut found: bool = false;
            for upvalue in &self.locals[depth as usize].upvalues {
                if upvalue.index == index && !upvalue.is_local {
                    index = upvalue.index; // Update the index
                    found = true; // And mark that we found an existing one
                    break
                }
            }

            // If we did not find an upvalue, then we must add one, referencing the upvalue from the outer function
            if !found {
                self.locals[depth as usize].upvalues.push(UpValue::new(false, index));
                index = (self.locals[depth as usize].upvalues.len() - 1) as u16
            }

            // This is no longer a local upvalue reference, the final upvalue will be a reference to a enclosing upvalue
            is_local = false;
        }

        // Finally, the last value of `index` will be one set from the directly enclosing function
        // We can thus return a `UpValue` reference, which contains the index of the upvalue in the enclosing function
        // We also return if the upvalue was local or not, as that's also needed to emit the correct bytecode
        return VariableType::UpValue(index, is_local)
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
                    // Error recovery failed - we reached end of input
                    self.error(ExpectedToken(token, None));
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

    /// Specialization of `peek()`, does not consume newlines, for checking if a specific syntax token is on the same line as the preceeding one.
    fn peek_no_newline(self: &mut Self) -> Option<&ScanToken> {
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

    /// Specialization of `push` which pushes either `StoreLocal` or `StoreGlobal` for a given local
    fn push_store_local(self: &mut Self, index: usize) {
        trace::trace_parser!("push StoreLocal/StoreGlobal {}", index);

        let local = &self.current_locals_mut().locals[index];
        if local.is_true_global() {
            self.push(StoreGlobal(index as u16, false))
        } else if local.is_global() {
            self.push(StoreGlobal(index as u16, true))
        } else {
            self.push(StoreLocal(index as u16))
        }
    }

    /// Specialization of `push` which pushes a `IncGlobalCount` if we need to, for the given local
    /// Called immediately after a local `initialized` is set to `true
    fn push_inc_global(self: &mut Self, local: usize) {
        if self.current_locals_mut().locals[local].is_true_global() {
            self.push(IncGlobalCount);
        }
    }

    /// Pushes a new token into the output stream.
    /// Returns the index of the token pushed, which allows callers to later mutate that token if they need to.
    fn push(self: &mut Self, token: Opcode) {
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
    fn pop(self: &mut Self) {
        match self.output.pop().unwrap() {
            PushGlobal(_, _) | StoreGlobal(_, _) | PushLocal(_) | StoreLocal(_) => {
                self.locals_reference.pop();
            },
            _ => {},
        };
    }

    /// Returns the index of the last token that was just pushed.
    fn last(self: &Self) -> Option<Opcode> {
        self.output.last().copied()
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

    /// Pushes a new error token into the output error stream, but does not initiate error recovery.
    /// This is useful for semantic errors which are valid lexically, but still need to report errors.
    fn semantic_error(self: &mut Self, error: ParserErrorType) {
        trace::trace_parser!("push_err (error = {}) {:?}", self.error_recovery, error);
        if !self.error_recovery {
            self.errors.push(ParserError::new(error, self.lineno as usize));
        }
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
    use crate::reporting;
    use crate::stdlib::NativeFunction;
    use crate::vm::opcode::Opcode;
    use crate::trace;

    use NativeFunction::{Print, ReadText, OperatorAdd, OperatorDiv, OperatorMul};
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
    #[test] fn test_multiple_unary_ops() { run_expr("- ~ ! 1", vec![Int(1), UnaryLogicalNot, UnaryBitwiseNot, UnarySub]); }
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

    #[test] fn test_let_eof() { run_err("let", "Expected a variable binding, either a name, or '_', or pattern (i.e. 'x, (_, y), *z'), got end of input instead\n  at: line 1 (<test>)\n  at:\n\nlet\n"); }
    #[test] fn test_let_no_identifier() { run_err("let =", "Expected a variable binding, either a name, or '_', or pattern (i.e. 'x, (_, y), *z'), got '=' token instead\n  at: line 1 (<test>)\n  at:\n\nlet =\n"); }
    #[test] fn test_let_expression_eof() { run_err("let x =", "Expected an expression terminal, got end of input instead\n  at: line 1 (<test>)\n  at:\n\nlet x =\n"); }
    #[test] fn test_let_no_expression() { run_err("let x = &", "Expected an expression terminal, got '&' token instead\n  at: line 1 (<test>)\n  at:\n\nlet x = &\n"); }

    #[test] fn test_array_access_after_newline() { run("array_access_after_newline"); }
    #[test] fn test_array_access_no_newline() { run("array_access_no_newline"); }
    #[test] fn test_break_past_locals() { run("break_past_locals"); }
    #[test] fn test_constants() { run("constants"); }
    #[test] fn test_continue_past_locals() { run("continue_past_locals"); }
    #[test] fn test_empty() { run("empty"); }
    #[test] fn test_expressions() { run("expressions"); }
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
    #[test] fn test_weird_expression_statements() { run("weird_expression_statements"); }
    #[test] fn test_weird_locals() { run("weird_locals"); }
    #[test] fn test_weird_loop_nesting_in_functions() { run("weird_loop_nesting_in_functions"); }
    #[test] fn test_weird_upvalue_index() { run("weird_upvalue_index"); }
    #[test] fn test_weird_upvalue_index_with_parameter() { run("weird_upvalue_index_with_parameter"); }
    #[test] fn test_while_1() { run("while_1"); }
    #[test] fn test_while_2() { run("while_2"); }
    #[test] fn test_while_3() { run("while_3"); }
    #[test] fn test_while_4() { run("while_4"); }
    #[test] fn test_while_false_if_false() { run("while_false_if_false"); }


    fn run_expr(text: &'static str, expected: Vec<Opcode>) {
        let result: ScanResult = scanner::scan(&String::from(text));
        assert!(result.errors.is_empty());

        let compile = parser::parse(result);
        assert!(compile.errors.is_empty(), "Found parser errors: {:?}", compile.errors);

        // Tokens will contain int values as exact values, as it's easier to read as a test DSL
        // However, the parser will give us constant IDs
        // So, index each one to produce what it looks like
        // We also want to also trim the trailing `Pop`, `Exit`, as the parser will insert that
        let constants: Vec<i64> = compile.constants;
        let mut actual: Vec<Opcode> = compile.code.into_iter()
            .map(|t| match t {
                Int(i) => Int(constants[i as usize] as u16),
                t => t
            })
            .collect::<Vec<Opcode>>();

        assert_eq!(actual.pop(), Some(Exit));
        assert_eq!(actual.pop(), Some(Pop));
        assert_eq!(actual, expected);
    }

    fn run_err(text: &'static str, expected: &'static str) {
        let text: &String = &String::from(text);
        let text_lines: Vec<&str> = text.split("\n").collect();

        let scan_result: ScanResult = scanner::scan(&text);
        assert!(scan_result.errors.is_empty());

        let compile: CompileResult = parser::parse(scan_result);
        let mut actual: Vec<String> = Vec::new();
        assert!(!compile.errors.is_empty());

        for error in &compile.errors {
            actual.push(reporting::format_parse_error(&text_lines, &String::from("<test>"), error));
        }

        assert_eq!(actual.join("\n"), expected);
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