use itertools::Itertools;
/// This module contains core semantic analysis related functions in the parser, as we merge the parse and semantic phases of the compiler into a single pass.
/// This also contains core structures that are used by the parser for semantic analysis.
///
/// The functions declared in this module are public to be used by `parser/mod.rs`, but the module `semantic` is not exported itself.

use crate::compiler::parser::{Parser, ParserError, ParserErrorType};
use crate::misc::MaybeRc;
use crate::stdlib;
use crate::vm::value::FunctionImpl;
use crate::vm::opcode::Opcode;

use ParserErrorType::{*};
use Opcode::{*};



#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Loop {
    pub(super) start_index: u32,
    pub(super) scope_depth: u32,
    pub(super) break_statements: Vec<u32>
}

impl Loop {
    pub fn new(start_index: u32, depth: u32) -> Loop {
        Loop { start_index, scope_depth: depth, break_statements: Vec::new() }
    }
}



#[derive(Debug)]
pub struct Locals {
    /// The local variables in the current function.
    /// Top level local variables are considered global, even though they still might be block scoped ('true global' are top level function in no block scope, as they can never go out of scope).
    pub(super) locals: Vec<Local>,
    /// An array of captured upvalues for this function, either due to an inner function requiring them, or this function needing to capture locals from it's enclosing function
    pub(super) upvalues: Vec<UpValue>,
    /// Loop stack
    /// Each frame represents a single loop, which `break` and `continue` statements refer to
    /// `continue` jumps back to the beginning of the loop, aka the first `usize` (loop start)
    /// `break` statements jump back to the end of the loop, which needs to be patched later. The values to be patched record themselves in the stack at the current loop level
    pub(super) loops: Vec<Loop>,
}

impl Locals {
    pub fn empty() -> Vec<Locals> {
        vec![Locals::new()]
    }

    pub fn len(self: &Self) -> usize {
        self.locals.len()
    }

    pub fn new() -> Locals {
        Locals { locals: Vec::new(), upvalues: Vec::new(), loops: Vec::new() }
    }
}

#[derive(Debug)]
pub struct UpValue {
    /// `true` = local variable in enclosing function, `false` = upvalue in enclosing function
    pub(super) is_local: bool,

    /// Either a reference to an index in the enclosing function's `locals` (which are stack offset),
    /// or a reference to the enclosing function's `upvalues` (which can be accessed via stack offset 0 -> upvalues, if it is a closure
    pub(super) index: u32,
}

impl UpValue {
    pub fn new(is_local: bool, index: u32) -> UpValue {
        UpValue { is_local, index }
    }
}



#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Local {
    /// The local variable name
    pub(super) name: String,
    /// The index of the local variable within it's `Locals` array.
    /// - At runtime, this matches the stack offset of this variable.
    /// - At compile time, this is used to index into `Parser.locals`
    index: u32,
    scope_depth: u32,
    function_depth: u32,
    initialized: bool,
    /// `true` if this local variable has been captured as an `UpValue`. This means when it is popped, the corresponding `UpValue` must also be popped.
    captured: bool,
}

impl Local {
    pub fn new(name: String, index: usize, scope_depth: u32, function_depth: u32) -> Local {
        Local { name, index: index as u32, scope_depth, function_depth, initialized: false, captured: false }
    }

    /// Sets this local as `initialized`, meaning it is pushed onto the stack and can be referred to in expressions.
    pub fn initialize(self: &mut Self) { self.initialized = true; }

    pub fn is_global(self: &Self) -> bool { self.function_depth == 0 }
    pub fn is_true_global(self: &Self) -> bool { self.function_depth == 0 && self.scope_depth == 0 }

    /// Resolves the `local` as a `VariableType`, based on the function and scope depth of the local.
    pub fn resolve_type(self: &Self) -> VariableType {
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
pub enum VariableType {
    NativeFunction(stdlib::NativeFunction),
    Local(u32),
    Global(u32),
    TrueGlobal(u32),
    LateBoundGlobal(LateBoundGlobal),
    UpValue(u32),
    None,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct LateBoundGlobal {
    name: String,
    opcode: usize, // Index in output[] of the `load` opcode
    pub(super) error: Option<ParserError>, // An error that would be thrown from here, if the variable does not end up bound
}

impl LateBoundGlobal {
    pub fn new(name: String, opcode: usize, error: Option<ParserError>) -> LateBoundGlobal {
        LateBoundGlobal { name, opcode, error }
    }
}


/// An `LValue` is a value used on the left hand side (hence `L`) of an expression. It may be declared, or undeclared.
///
/// Examples of possible `<lvalue>`s:
///
/// - `_` : An empty `LValue`, which simply discards it's associated `RValue`
/// - `x` : A named `LValue`, which assigns it's associated `RValue` to the variable `x` (which may be any type of assignable variable, including a yet-undeclared variable)
/// - `(x, y)` : A pattern `LValue`, which destructures it's input and assigns them to `x` and `y`.
/// - `x, *y, _` : A complex pattern `LValue`.
///
/// An `LValue` can be used in a number of different situations:
///
/// - In a `let <lvalue>` statement, the `LValue` immediately follows the `let`, and declares all it's contents. `LValue`s used here must be non-variadic, non-nested. Variables must be undeclared, and will be declared by the statement.
/// - In a `let <lvalue> = <expression>` statement, the `LValue` must be undeclared and will be declared by the statement.
/// - In a `fn <name> ? ( <lvalue> )` statement, the single `LValue` represents all arguments to the function. Variables are declared as new locals to the function, and relevant destructuring code is emitted if needed.
#[derive(Debug, Clone)]
pub enum LValue {
    /// A `_` term.
    Empty,
    /// A `*_` term.
    VarEmpty,
    /// A `<name>` term.
    Named(LValueReference),
    /// A `* <name>` term.
    VarNamed(LValueReference),
    /// A pattern `LValue` with at least one top-level `( <lvalue> )`
    Terms(Vec<LValue>),
}

#[derive(Debug, Clone)]
pub enum LValueReference {
    Named(String),
    Local(u32),
    Invalid,
}



impl LValue {

    /// Returns `true` if the `LValue` is a top-level variadic, such as `* <name>` or `*_`
    pub fn is_variadic_term(self: &Self) -> bool { match self { LValue::VarEmpty | LValue::VarNamed(_) => true, _ => false } }

    fn as_terms(self: &Self) -> &Vec<LValue> { match self { LValue::Terms(it) => it, _ => panic!("Expected LValue::Terms") } }

    /// Converts this `LValue` into a code-representation string.
    pub fn to_code_str(self: &Self) -> String {
        match self {
            LValue::Empty => String::from("_"),
            LValue::VarEmpty => String::from("*_"),
            LValue::Named(LValueReference::Named(it)) => it.clone(),
            LValue::VarNamed(LValueReference::Named(it)) => format!("*{}", it),
            LValue::Terms(it) => format!("({})", it.iter().map(|u| u.to_code_str()).join(", ")),
            _ => panic!("Cannot convert a {:?} to a code string", self),
        }
    }

    /// Declares all variables associated with this `LValue` as locals in the current scope.
    /// Will panic if the `LValue` has terms which are not `LValueReference::Named`.
    pub fn declare_locals(self: &mut Self, parser: &mut Parser) {
        match self {
            LValue::Named(it) | LValue::VarNamed(it) => {
                let name = std::mem::take(it).as_named();
                if let Some(local) = parser.declare_local(name) {
                    *it = LValueReference::Local(local as u32);
                }
            },
            LValue::Terms(lvalue) => {
                for term in lvalue {
                    term.declare_locals(parser);
                }
            },
            _ => {},
        }
    }

    /// This is meant to be paired with `declare_pattern_locals()`, together which will declare all local variables used by this `LValue`.
    /// This method will **always** declare a single local variable in the current scope, which is meant to be the input to the pattern.
    /// If the pattern needs more or less, which would normally be handled via destructuring, this will declare a synthetic variable instead.
    ///
    /// If a synthetic local was needed, returns the index of the local as `Some(local)`. Otherwise returns `None`
    pub fn declare_single_local(self: &mut Self, parser: &mut Parser) -> Option<usize> {
        match self {
            LValue::Named(_) | LValue::VarNamed(_) => {
                self.declare_locals(parser);
                return None;
            },
            LValue::Terms(_) | LValue::Empty | LValue::VarEmpty => Some(parser.declare_synthetic_local()),
        }
    }

    pub fn declare_pattern_locals(self: &mut Self, parser: &mut Parser) {
        match self {
            LValue::Terms(lvalue) => {
                for term in lvalue {
                    term.declare_locals(parser);
                }
            },
            _ => {},
        }
    }

    /// Initializes all local variables associated with this `LValue`. This allows them to be referenced in expressions.
    /// Also emits `IncGlobalCount` opcodes for globally declared variables.
    pub fn initialize_locals(self: &mut Self, parser: &mut Parser) {
        match self {
            LValue::Named(LValueReference::Local(local)) | LValue::VarNamed(LValueReference::Local(local)) => {
                parser.current_locals_mut().locals[*local as usize].initialize();
                parser.push_inc_global(*local as usize);
            },
            LValue::Terms(lvalue) => {
                for term in lvalue {
                    term.initialize_locals(parser);
                }
            },
            _ => {},
        }
    }

    /// Emits default values (`Nil`) for all newly declared local variables for this `LValue`.
    ///
    /// If `in_place` is `true`, this (and the associated function `emit_destructuring()`, will assume the value is constructed **in-place**, i.e. the result
    /// variables are to be placed on top of the stack once finished. This enables a few minor bytecode optimizations for specific `LValue`s:
    ///
    /// - Instead of emitting `Nil`, we don't emit anything as part of default value initialization.
    /// - Instead of destructuring, we either leave the value in-place (for a `<name>` `LValue`), or immediately pop (for a `_` `LValue`)
    pub fn emit_default_values(self: &Self, parser: &mut Parser, in_place: bool) {
        match self {
            LValue::Terms(terms) => {
                for term in terms {
                    term.emit_default_values(parser, false);
                }
            },
            LValue::Named(_) | LValue::VarNamed(_) if !in_place => parser.push(Nil),
            _ => {},
        }
    }

    /// Emits destructuring code for this `LValue`. Assumes the `RValue` is present on top of the stack, and all variables
    ///
    /// If `in_place` is `true`, this will use an optimized destructuring for specific `LValue`s:
    ///
    /// - `_` `LValue`s will emit a single `Pop`, rather than destructuring.
    /// - `<name>` `LValue`s will assume their value is constructed in place, and not emit any destructuring code.
    pub fn emit_destructuring(self: &Self, parser: &mut Parser, in_place: bool) {
        match self {
            LValue::Empty | LValue::VarEmpty => parser.push(Pop),
            LValue::Named(LValueReference::Local(index)) | LValue::VarNamed(LValueReference::Local(index)) if !in_place => {
                parser.push_store_local(*index as usize);
                parser.push(Pop);
            }
            LValue::Terms(_) => self.emit_terms_destructuring(parser),
            _ => {},
        }
    }

    fn emit_terms_destructuring(self: &Self, parser: &mut Parser) {
        let terms = self.as_terms();

        let is_variadic = terms.iter().any(|t| t.is_variadic_term());
        let len: i64 = if is_variadic { terms.len() - 1 } else { terms.len() } as i64;
        let constant_len = parser.declare_constant(len);

        parser.push(if is_variadic { CheckLengthGreaterThan(constant_len) } else { CheckLengthEqualTo(constant_len) });

        let mut index: i64 = 0;
        for term in terms {
            match term {
                LValue::Empty => {
                    // Just advance the index
                    index += 1;
                },
                LValue::VarEmpty => {
                    // Advance the index by the missing elements (start indexing in reverse)
                    index = -(len - index);
                },
                LValue::Named(LValueReference::Local(local)) => {
                    let constant_index = parser.declare_constant(index);

                    parser.push(Int(constant_index));
                    parser.push(OpIndexPeek); // [ it[index], index, it, ...]
                    parser.push_store_local(*local as usize); // stores it[index]
                    parser.push(PopN(2)); // [it, ...]

                    index += 1;
                },
                LValue::VarNamed(LValueReference::Local(local)) => {
                    // index = the next index in the iterable to take = the number of elements already accessed
                    // therefor, len - index = the number of elements left we need to access exactly, which must be indices -1, -2, ... -(len - index)
                    // so our slice excludes these, so the 'high' value must be -(len - index)
                    // this is then also exactly what we set our index to next
                    let constant_low = parser.declare_constant(index);
                    index = -(len - index);
                    let constant_high = parser.declare_constant(index);

                    parser.push(Dup); // [it, it, ...]
                    parser.push(Int(constant_low)); // [low, it, it, ...]
                    parser.push(if index == 0 { Nil } else { Int(constant_high) }); // [high, low, it, it, ...]
                    parser.push(OpSlice); // [it[low:high], it, ...]
                    parser.push_store_local(*local as usize); // stores it[low:high]
                    parser.push(Pop); // [it, ...]
                },
                terms @ LValue::Terms(_) => {
                    // Index as if this was a `Term`, but then invoke the emit recursively, with the value still on the stack, treating it as the iterable.
                    let constant_index = parser.declare_constant(index);

                    parser.push(Int(constant_index));
                    parser.push(OpIndexPeek); // [ it[index], index, it, ...]
                    terms.emit_destructuring(parser, false); // [ index, it, ...]
                    parser.push(Pop); // [it, ...]

                    index += 1;
                },
                _ => {},
            }
        }

        parser.push(Pop); // Pop the iterable
    }
}

impl Default for LValue {
    fn default() -> Self { LValue::Empty }
}

impl LValueReference {
    fn as_named(self) -> String {
        match self { LValueReference::Named(it) => it, _ => panic!("Expected LValueReference::Named") }
    }
}

impl Default for LValueReference {
    fn default() -> Self {
        LValueReference::Invalid
    }
}



impl<'a> Parser<'a> {
    pub fn declare_string(self: &mut Self, str: String) -> u32 {
        if let Some(id) = self.strings.iter().position(|s| s == &str) {
            return id as u32
        }
        self.strings.push(str);
        (self.strings.len() - 1) as u32
    }

    pub fn declare_constant(self: &mut Self, int: i64) -> u32 {
        if let Some(id) = self.constants.iter().position(|i| *i == int) {
            return id as u32
        }
        self.constants.push(int);
        (self.constants.len() - 1) as u32
    }

    pub fn declare_function(self: &mut Self, head: usize, name: String, args: &Vec<LValue>) -> u32 {
        self.functions.push(MaybeRc::new(FunctionImpl::new(head, name, args.iter().map(|u| u.to_code_str()).collect())));
        (self.functions.len() - 1) as u32
    }


    pub fn next_opcode(self: &Self) -> u32 {
        self.output.len() as u32
    }

    /// After a `let <name>` or `fn <name>` declaration, tries to declare this as a local variable in the current scope.
    /// Returns the index of the local variable in `self.current_locals().locals`, or `None` if the variable could not be declared.
    /// Note that if `None` is returned, a semantic error will already have been raised.
    pub fn declare_local(self: &mut Self, name: String) -> Option<usize> {

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
    pub fn declare_synthetic_local(self: &mut Self) -> usize {
        self.synthetic_local_index += 1;
        self.declare_local_internal(format!("${}", self.synthetic_local_index - 1))
    }

    /// Declares a local variable by the name `name` in the current scope.
    fn declare_local_internal(self: &mut Self, name: String) -> usize {
        let local: Local = Local::new(name, self.locals.last().unwrap().locals.len(), self.scope_depth, self.function_depth);
        self.locals.last_mut().unwrap().locals.push(local);
        self.locals.last().unwrap().locals.len() - 1
    }

    /// Manages popping local variables and upvalues. Does a number of tasks, based on what is requested.
    ///
    /// 1. Pops local variables from the parser's local variable stack.
    ///    This is required when the variable goes out of scope, as we can no longer reference it.
    ///    Note however, if we're using this to emit pop tokens for non-sequential code (i.e. a `break` or `return`), we want to keep these variables around.
    ///
    ///    This behavior is thus controlled by `modify_lvt` parameter.
    ///
    /// 2. Emit `Pop` tokens for the local variables that would be caught by (1.). This is required without (1.) when we are in non-sequential code, (i.e. a `break` or `return`), as we need to make sure the runtime stack is correct.
    ///    Note that this is also not required after a `return`, as the `Return` opcode will clear everything above the current frame pointer, hence not needing explicit `Pop` opcodes.
    ///
    ///    This behavior is controlled by the `emit_pop` parameter.
    ///
    /// 3. Emit `LiftUpValue` opcodes for any captured local variables. This ensures that these variables live on the heap as long as required, by any closures that have captured them.
    ///    This can be used without (1.) or (2.) to create the illusion of having local variables local to a loop structure, i.e. `for` loop - each iteration of the loop, we just emit `LiftUpValue` tokens, and only do a `Pop` at the final iteration of the loop.
    ///
    ///    This behavior is controlled by the `emit_lifts` parameter.
    ///
    /// ---
    ///
    /// This function also can pop different local variables. If `scope` is `Some(scope)`, then it will pop all local variables **equal to and including** a certain scope depth. If `scope` is `None`, it will pop all locals in the current function. This can be used in different situations:
    ///
    /// - At the closure of a block scope, only a single scope needs to be popped, so `scope` is `Some(self.current_scope_depth)`
    /// - At a `break` or `continue` statement, this jumps multiple block scopes, so any scope above `loop.scope_depth` needs to be popped.
    /// - At a `return` statement, this jumps out of all scopes in the current function, so any scopes above that need to be popped.
    pub fn pop_locals(self: &mut Self, scope: Option<u32>, modify_lvt: bool, emit_pop: bool, emit_lifts: bool) {

        let len = self.current_locals().locals.len() as u32;
        let mut pop_count = 0;
        for local_index in (0..len).rev() {
            let local = &self.current_locals().locals[local_index as usize];

            if match scope {
                Some(min_scope) => local.scope_depth >= min_scope,
                None => true,
            } {
                pop_count += 1;

                if local.captured && emit_lifts {
                    self.push(LiftUpValue(local_index));
                }

                if modify_lvt {
                    // Pop the local
                    self.current_locals_mut().locals.pop().unwrap();

                    // And pop any matching upvalues
                    if let Some(upvalue) = self.current_locals_mut().upvalues.last() {
                        if upvalue.index == local_index && upvalue.is_local {
                            self.current_locals_mut().upvalues.pop().unwrap();
                        }
                    }
                }
            }
        }

        // Emit pop tokens for the locals that were popped
        if emit_pop && pop_count > 0 {
            self.push_pop(pop_count);
        }
    }

    /// Returns the current function's `locals`
    pub fn current_locals(self: &Self) -> &Locals {
        self.locals.last().unwrap()
    }

    /// Returns the current function's `locals`
    pub fn current_locals_mut(self: &mut Self) -> &mut Locals {
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
    pub fn resolve_identifier(self: &mut Self, name: &String) -> VariableType {
        if let Some(b) = stdlib::find_native_function(name) {
            return VariableType::NativeFunction(b);
        }

        // 1. Search for locals in the current function. This may return `Local`, `Global`, or `TrueGlobal` based on the scope of the variable.
        //   - Locals that are captured as upvalues, but are now being referenced as locals again, emit upvalue references, as the stack stops getting updated after a value is lifted into an upvalue.
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
                    if &local.name == name && local.initialized && !local.is_true_global() { // Note that it must **not** be a true global, anything else can be captured as an upvalue
                        let index = local.index;
                        self.locals[depth as usize].locals[index as usize].captured = true;
                        return self.resolve_upvalue(depth, index);
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
    pub fn resolve_upvalue(self: &mut Self, local_depth: u32, local_index: u32) -> VariableType {

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
        let mut index = if let Some(index) = maybe_index {
            index as u32
        } else {
            self.locals[local_depth as usize].upvalues.push(UpValue::new(true, local_index));
            (self.locals[local_depth as usize].upvalues.len() - 1) as u32
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
                index = (self.locals[depth as usize].upvalues.len() - 1) as u32
            }
        }

        // Finally, the last value of `index` will be one set from the directly enclosing function
        // We can thus return a `UpValue` reference, which contains the index of the upvalue in the enclosing function
        return VariableType::UpValue(index)
    }
}