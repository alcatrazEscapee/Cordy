/// This module contains core semantic analysis related functions in the parser, as we merge the parse and semantic phases of the compiler into a single pass.
/// This also contains core structures that are used by the parser for semantic analysis.
///
/// The functions declared in this module are public to be used by `parser/mod.rs`, but the module `semantic` is not exported itself.

use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::Debug;
use fxhash::FxBuildHasher;
use indexmap::IndexMap;
use itertools::Itertools;

use crate::compiler::parser::{Parser, ParserError, ParserErrorType};
use crate::core;
use crate::core::Pattern;
use crate::reporting::Location;
use crate::vm::{FunctionImpl, IntoValue, Opcode, StoreOp, StructTypeImpl, ValuePtr};

use Opcode::{*};
use ParserErrorType::{*};


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Loop {
    pub(super) start_index: usize,
    pub(super) scope_depth: u32,
    pub(super) break_statements: Vec<usize>
}

impl Loop {
    fn new(start_index: usize, depth: u32) -> Loop {
        Loop { start_index, scope_depth: depth, break_statements: Vec::new() }
    }
}



#[derive(Debug, Clone)]
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

    /// Ordinal into `self.functions` to access `self.functions[func].code`
    /// If not present, it is assumed to be global code.
    /// Note this is not quite the same as the function ID, as if there are baked functions present, this will differ by the amount of baked functions
    func: Option<usize>,
}

impl Locals {
    /// Returns a new empty `Locals` stack.
    pub fn empty() -> Vec<Locals> {
        vec![Locals::new(None)]
    }

    /// Returns a new empty `Locals` instance, corresponding to the given function, if present.
    pub(super) fn new(func: Option<usize>) -> Locals {
        Locals { locals: Vec::new(), upvalues: Vec::new(), loops: Vec::new(), func }
    }

    /// Returns the length of the locals, effectively the number of variables declared in this frame.
    /// `locals[0].len()` will be the number of global variables, for a given `Locals` stack.
    pub fn len(&self) -> usize {
        self.locals.len()
    }

    /// Returns the name of a local with the given `index`.
    pub(super) fn get_name(&self, index: usize) -> String {
        self.locals[index].name.clone()
    }

    /// Returns the topmost `Loop` statement on the stack, or `None` if the stack is empty.
    pub(super) fn top_loop(&mut self) -> Option<&mut Loop> {
        self.loops.last_mut()
    }

    /// Enumerates the current locals' `upvalues`, and emits the correct `CloseLocal` or `CloseUpValue` tokens for each.
    pub(super) fn closed_locals(&self) -> Vec<Opcode> {
        self.upvalues.iter()
            .map(|upvalue| if upvalue.is_local { CloseLocal(upvalue.index) } else { CloseUpValue(upvalue.index) })
            .collect::<Vec<Opcode>>()
    }
}

#[derive(Debug, Clone)]
pub struct Fields {
    /// A mapping of `field name` to `field index`. This is used to record unique fields.
    /// For example, `struct Foo(a, b, c)` would generate the fields `"a"`, `"b"`, and `"c"` at index `0`, `1`, and `2`, respectively.
    fields: HashMap<String, u32, FxBuildHasher>,

    /// A table which maps pairs of `(type index, field index)` to a `field offset`
    /// The `type index` is known at runtime, based on the runtime type of the struct in use.
    /// The `field index` is known at compile time, based on the identifier that it resolves to.
    /// The resultant `field offset` is a index into a specific struct object's `Vec<Value>` of fields.
    lookup: HashMap<(u32, u32), usize, FxBuildHasher>,

    /// The next available `type_index`
    types: u32,
}

impl Fields {
    pub(super) fn new() -> Fields {
        Fields {
            fields: HashMap::with_hasher(FxBuildHasher::default()),
            lookup: HashMap::with_hasher(FxBuildHasher::default()),
            types: 0,
        }
    }

    pub fn get_field_offset(&self, type_index: u32, field_index: u32) -> Option<usize> {
        self.lookup.get(&(type_index, field_index)).copied()
    }

    pub fn get_field_name(&self, field_index: u32) -> String {
        self.fields.iter()
            .find(|(_, v)| field_index == **v)
            .unwrap()
            .0
            .clone()
    }
}

#[derive(Debug, Clone)]
struct UpValue {
    /// `true` = local variable in enclosing function, `false` = upvalue in enclosing function
    is_local: bool,

    /// Either a reference to an index in the enclosing function's `locals` (which are stack offset),
    /// or a reference to the enclosing function's `upvalues` (which can be accessed via stack offset 0 -> upvalues, if it is a closure
    index: u32,
}

impl UpValue {
    fn new(is_local: bool, index: u32) -> UpValue {
        UpValue { is_local, index }
    }
}



#[derive(Eq, PartialEq, Debug, Clone)]
struct Local {
    /// The local variable name
    name: String,
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
    fn new(name: String, index: usize, scope_depth: u32, function_depth: u32) -> Local {
        Local { name, index: index as u32, scope_depth, function_depth, initialized: false, captured: false }
    }

    fn is_global(&self) -> bool {
        self.function_depth == 0 && self.scope_depth == 0
    }
}


#[derive(Debug, Clone)]
pub struct LateBinding {
    /// A unique index for every late bound global.
    index: usize,
    name: String,
    /// This is a pair of (function ordinal, code ordinal)
    /// Note that the function ordinal is an index into `self.functions`, not an absolute function ID (and is thus a `usize` not a `u32`)
    opcode: (usize, usize),
    /// The reference type of this late bound global, for fixing later
    ty: ReferenceType,
    loc: Location,
    /// If true, an error should be raised. Represents the state of the error recovery mode when hit
    error: bool,
}

impl LateBinding {
    pub fn new(index: usize, name: String, ordinal: usize, opcode: usize, loc: Location, error: bool) -> LateBinding {
        LateBinding {
            index,
            name,
            opcode: (ordinal, opcode),
            ty: ReferenceType::Invalid,
            loc,
            error,
        }
    }

    pub(super) fn update(&mut self, ty: ReferenceType, parser: &Parser) {
        self.opcode = (parser.functions.len() - 1, parser.next_opcode());
        self.ty = ty;
    }

    pub fn to_error(self) -> Option<ParserError> {
        match self.error {
            true => Some(ParserError::new(UndeclaredIdentifier(self.name), self.loc)),
            false => None
        }
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
#[derive(Debug, Clone, Default)]
pub enum LValue {
    /// A `_` term.
    #[default]
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

#[derive(Debug, Clone, Default)]
pub enum LValueReference {
    #[default]
    Invalid,
    Named(String),
    Local(u32),
    Global(u32),
    LateBinding(LateBinding),
    UpValue(u32),
    Method(u32),
    NativeFunction(core::NativeFunction),
}


#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ReferenceType {
    Invalid, Load, Store, StorePattern
}



impl LValue {

    /// Returns `true` if the `LValue` is a top-level variadic, such as `* <name>` or `*_`
    pub fn is_variadic_term(&self) -> bool { matches!(self, LValue::VarEmpty | LValue::VarNamed(_)) }

    /// Returns `true` if the `LValue` is a top-level `LValue::Named`, i.e. `<name>`.
    pub fn is_named(&self) -> bool { matches!(self, LValue::Named(_)) }

    /// Returns `true` if the `LValue` is non-trivial, i.e. it contains any of `_`, `*`, or nested elements.
    pub fn is_non_trivial(&self) -> bool {
        match self {
            LValue::Named(_) => false,
            LValue::Empty | LValue::VarEmpty | LValue::VarNamed(_) => true,
            LValue::Terms(terms) => terms.iter().any(|term| term.is_non_trivial() || matches!(term, LValue::Terms(_)))
        }
    }

    fn into_terms(self) -> Vec<LValue> { match self { LValue::Terms(it) => it, _ => panic!("Expected LValue::Terms") } }

    /// Converts this `LValue` into a code-representation string.
    pub fn to_code_str(&self) -> String {
        match self {
            LValue::Empty => String::from("_"),
            LValue::VarEmpty => String::from("*_"),
            LValue::Named(LValueReference::Named(it)) => it.clone(),
            LValue::VarNamed(LValueReference::Named(it)) => format!("*{}", it),
            LValue::Terms(it) => format!("({})", it.iter().map(|u| u.to_code_str()).join(", ")),
            _ => panic!("Cannot convert a {:?} to a code string", self),
        }
    }

    /// Resolves each identifier as a local variable that is currently declared.
    /// This will raise semantic errors for undeclared variables.
    pub(super) fn resolve_locals(&mut self, loc: Location, parser: &mut Parser) {
        match self {
            LValue::Named(it) | LValue::VarNamed(it) => {
                let name: String = it.as_named();
                *it = parser.resolve_mutable_reference(loc, name);
            },
            LValue::Terms(lvalue) => {
                for term in lvalue {
                    term.resolve_locals(loc, parser);
                }
            },
            _ => {},
        };
    }

    /// Declares all variables associated with this `LValue` as locals in the current scope.
    /// Will panic if the `LValue` has terms which are not `LValueReference::Named`.
    pub(super) fn declare_locals(&mut self, parser: &mut Parser) {
        match self {
            LValue::Named(it) | LValue::VarNamed(it) => {
                let name: String = it.as_named();
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
    pub(super) fn declare_single_local(&mut self, parser: &mut Parser) -> Option<usize> {
        match self {
            LValue::Named(_) | LValue::VarNamed(_) => {
                self.declare_locals(parser);
                None
            },
            LValue::Terms(_) | LValue::Empty | LValue::VarEmpty => Some(parser.declare_synthetic_local()),
        }
    }

    pub(super) fn declare_pattern_locals(&mut self, parser: &mut Parser) {
        if let LValue::Terms(lvalue) = self {
            for term in lvalue {
                term.declare_locals(parser);
            }
        }
    }

    /// Initializes all local variables associated with this `LValue`. This allows them to be referenced in expressions.
    /// Also emits `IncGlobalCount` opcodes for globally declared variables.
    pub(super) fn initialize_locals(&mut self, parser: &mut Parser) {
        match self {
            LValue::Named(LValueReference::Local(local)) |
            LValue::VarNamed(LValueReference::Local(local)) => parser.init_local(*local as usize),
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
    pub(super) fn emit_default_values(&self, parser: &mut Parser, in_place: bool) {
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
    ///   - `<name>` `LValue`s will assume their value is constructed in place, and not emit any destructuring code.
    ///
    /// If `in_expr` is `true`, this will assume the top level destructuring is part of an expression, and the last `Pop` token won't be emitted.
    /// This leaves the stack untouched after destructuring, with the iterable still on top.
    pub(super) fn emit_destructuring(self, parser: &mut Parser, in_place: bool, in_expression: bool) {
        match self {
            LValue::Empty | LValue::VarEmpty => {
                if !in_expression {
                    parser.push(Pop)
                }
            },
            LValue::Named(local) | LValue::VarNamed(local) => {
                if !in_place {
                    parser.push_store_lvalue(local);
                    if !in_expression {
                        parser.push(Pop);
                    }
                }
            }
            LValue::Terms(_) => {
                let pattern = self.build_pattern(parser);
                parser.declare_pattern(pattern);
                if !in_expression {
                    parser.push(Pop); // Push the final pop
                }
            },
        }
    }

    fn build_pattern(self, parser: &mut Parser) -> Pattern<ParserStoreOp> {
        let terms = self.into_terms();
        let is_variadic = terms.iter().any(|t| t.is_variadic_term());
        let len = if is_variadic { terms.len() - 1 } else { terms.len() };

        let mut pattern: Pattern<ParserStoreOp> = Pattern::new(len, is_variadic);

        let mut index: i64 = 0;
        for term in terms {
            match term {
                LValue::Empty => {
                    // Just advance the index
                    index += 1;
                },
                LValue::VarEmpty => {
                    // Advance the index by the missing elements (start indexing in reverse)
                    index = -(len as i64 - index);
                },
                LValue::Named(lvalue) => {
                    pattern.push_index(index, lvalue.as_store_op(parser));
                    index += 1;
                },
                LValue::VarNamed(lvalue) => {
                    let low = index;
                    index = -(len as i64 - index);
                    let high = index;

                    pattern.push_slice(low, high, lvalue.as_store_op(parser));
                },
                terms @ LValue::Terms(_) => {
                    pattern.push_pattern(index, terms.build_pattern(parser));
                    index += 1;
                },
            }
        }
        pattern
    }
}

impl LValueReference {

    fn as_named(&mut self) -> String {
        match std::mem::take(self) {
            LValueReference::Named(it) => it,
            _ => panic!("Expected LValueReference::Named"),
        }
    }

    fn as_store_op(self, parser: &mut Parser) -> ParserStoreOp {
        match self {
            LValueReference::Local(index) => ParserStoreOp::Bound(StoreOp::Local(index)),
            LValueReference::Global(index) => ParserStoreOp::Bound(StoreOp::Global(index)),
            LValueReference::LateBinding(mut global) => {
                let index = global.index;

                global.update(ReferenceType::StorePattern, parser);
                parser.late_bindings.push(global);

                ParserStoreOp::LateBoundGlobal(index)
            },
            LValueReference::UpValue(index) => ParserStoreOp::Bound(StoreOp::UpValue(index)),
            LValueReference::Invalid => ParserStoreOp::Invalid, // Error has already been raised
            _ => panic!("Invalid store: {:?}", self),
        }
    }
}

#[derive(Debug)]
pub enum ParserStoreOp {
    Bound(StoreOp),
    LateBoundGlobal(usize),
    Invalid
}


#[derive(Debug)]
pub struct ParserFunctionImpl {
    /// Function name and argument names
    name: String,
    args: Vec<String>,

    /// These are indexes into the function code, where the function call position should jump to.
    /// They are indexed from the first function call (zero default arguments), increasing.
    /// So `[Nil, Int(1), Int(2), Plus, ...]` would have entries `[1, 4]` as it's argument set, and length is the number of default arguments.
    default_args: Vec<usize>,

    /// If the last argument in this function is a variadic argument, meaning it needs special behavior when invoked with >= `max_args()`
    var_arg: bool,

    /// If the function is a FFI native function
    native: bool,

    /// Bytecode for the function body itself
    code: Vec<(Location, Opcode)>,

    /// Entries for `locals_reference`, that need to be held until the function code is emitted
    locals_reference: Vec<String>,

    /// Constant index for this function, which is used to fix the function later
    constant_id: u32,
}

impl ParserFunctionImpl {
    /// Empties and returns the source code for this parser function.
    pub(super) fn emit_code(&mut self) -> std::vec::Drain<'_, (Location, Opcode)> {
        self.code.drain(..)
    }

    /// Empties and returns the local references for this parser function.
    pub(super) fn emit_locals(&mut self) -> std::vec::Drain<'_, String> {
        self.locals_reference.drain(..)
    }

    /// Bakes this parser function into an immutable `FunctionImpl`.
    /// The `head` and `tail` pointers are computed based on the surrounding code.
    pub(super) fn bake(self, constants: &mut [ValuePtr], head: usize, tail: usize) {
        constants[self.constant_id as usize] = FunctionImpl::new(head, tail, self.name, self.args, self.default_args, self.var_arg, self.native).to_value();
    }

    /// Marks a default argument as finished.
    pub(super) fn mark_default_arg(&mut self) {
        self.default_args.push(self.code.len());
    }

    pub(super) fn set_native(&mut self) {
        self.native = true;
    }
}


/// A representation of a module (or struct) which may expose methods and fields that can be bound.
pub struct Module {
    name: String,
    is_module: bool,

    /// A map of method names to function indices. When resolving module methods, they don't do a `GetField` and rather bind directly to the method at compile-time.
    methods: IndexMap<String, u32>,
    fields: Vec<String>,
}

impl Module {
    pub fn new(name: String, is_module: bool) -> Module {
        Module { name, is_module, methods: IndexMap::new(), fields: Vec::new() }
    }

    pub fn bake(self, instance_type: u32, constructor_type: u32) -> StructTypeImpl {
        let methods: Vec<u32> = self.methods.into_values().collect();
        StructTypeImpl::new(self.name, self.fields, methods, instance_type, constructor_type, self.is_module)
    }
}

enum LateBound {
    Local(u32),
    Method(u32)
}

#[derive(Debug, Clone)]
pub struct FunctionLibrary {
    modules: Vec<String>,
    methods: Vec<FunctionLibraryEntry>,
}

impl FunctionLibrary {
    pub fn new() -> FunctionLibrary {
        FunctionLibrary { modules: Vec::new(), methods: Vec::new() }
    }

    /// Returns, for a given known `handle_id`, the pair of `(module_name, method_name)`
    pub fn lookup(&self, handle_id: u32) -> &FunctionLibraryEntry {
        &self.methods[handle_id as usize]
    }

    pub fn len(&self) -> (usize, usize) {
        (self.modules.len(), self.methods.len())
    }

    pub fn truncate(&mut self, len: (usize, usize)) {
        self.modules.truncate(len.0);
        self.methods.truncate(len.1);
    }
}


#[derive(Debug, Clone)]
pub struct FunctionLibraryEntry {
    pub module_id: u32,
    pub module_name: String,
    pub nargs: usize,

    pub method_name: String,
    method_cname: CString,
}

impl FunctionLibraryEntry {
    fn new(module_id: u32, module_name: String, method_name: String, nargs: usize) -> FunctionLibraryEntry {
        let method_cname = CString::new(method_name.as_bytes()).unwrap(); // Safe because we know method names cannot contain '\0'
        FunctionLibraryEntry { module_id, module_name, nargs, method_name, method_cname }
    }
    pub fn symbol(&self) -> &[u8] {
        self.method_cname.as_bytes_with_nul()
    }
}


impl<'a> Parser<'a> {

    // ===== Loops ===== //

    /// Marks the beginning of a loop type statement, for the purposes of tracking `break` and `continue` statements.
    pub fn begin_loop(&mut self) -> usize {
        let loop_start: usize = self.next_opcode(); // Top of the loop, push onto the loop stack
        let loop_depth: u32 = self.scope_depth;
        self.current_locals_mut().loops.push(Loop::new(loop_start, loop_depth));
        loop_start
    }

    /// Marks the end of a loop, at the point where `break` statements should jump to (so after any `else` statements attached to the loop)
    pub fn end_loop(&mut self) {
        let break_opcodes: Vec<usize> = self.current_locals_mut().loops.pop().unwrap().break_statements;
        for break_opcode in break_opcodes {
            self.fix_jump(break_opcode, Jump);
        }
    }

    pub fn declare_const<T : IntoValue>(&mut self, value: T) -> u32 {
        let value = value.to_value();
        if let Some(id) = self.constants.iter().position(|i| i == &value) {
            return id as u32
        }
        self.constants.push(value);
        (self.constants.len() - 1) as u32
    }

    /// Declares a function with a given name and arguments.
    /// Returns the constant identifier for this function, however the function itself is currently located in `self.functions`, not `self.constants`.
    /// Instead, we push a dummy `Nil` into the constants array, and store the constant index on our parser function. During teardown, we inject these into the right spots.
    pub fn declare_function(&mut self, name: String, args: &[LValue], var_arg: bool) -> u32 {
        let constant_id: u32 = self.constants.len() as u32;

        self.constants.push(ValuePtr::nil());
        self.functions.push(ParserFunctionImpl {
            name,
            args: args.iter().map(|u| u.to_code_str()).collect(),
            default_args: Vec::new(),
            var_arg,
            native: false,
            code: Vec::new(),
            locals_reference: Vec::new(),
            constant_id,
        });
        constant_id
    }

    /// Declares a `Pattern`, stores the pattern, and then emits a `ExecPattern` opcode for it
    fn declare_pattern(&mut self, pattern: Pattern<ParserStoreOp>) {
        let pattern_id: u32 = (self.patterns.len() + self.baked_patterns.len()) as u32;

        self.patterns.push(pattern);
        self.push(ExecPattern(pattern_id));
    }

    /// After a `let <name>`, `fn <name>`, or `struct <name>` declaration, tries to declare this as a local variable in the current scope.
    /// Returns the index of the local variable in `self.current_locals().locals`, or `None` if the variable could not be declared.
    /// Note that if `None` is returned, a semantic error will already have been raised.
    ///
    /// - If `is_const` is true, this variable will not be able to be reassigned. This is the case for modules and structs.
    pub fn declare_local(&mut self, name: String) -> Option<usize> {

        // Lookup the name as a binding - if it is, it will be denied as we don't allow shadowing global native functions
        if core::NativeFunction::find(&name).is_some() {
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

        let index = self.declare_local_internal(name.clone());
        let local = &self.locals.last().unwrap().locals[index];

        if local.is_global() {
            // Resolve any late bound references to this global
            self.resolve_late_bindings(name.clone(), LateBound::Local(local.index));

            // Declare this global variable's name
            self.globals_reference.push(name);
        }

        Some(index)
    }

    /// Upon declaring a global variable, struct field or method, this will resolve any references to this variable if they exist.
    /// It may raise errors due to the variable not being assignable (i.e. a module method).
    fn resolve_late_bindings(&mut self, name: String, result: LateBound) {
        let mut i = 0;
        while i < self.late_bindings.len() {
            let binding = &self.late_bindings[i];

            // If the binding matches, it will be declared here
            // The first declaration is always the matching one, as the precedence for methods/fields -> globals is always maintained
            if binding.name == name {
                let binding = self.late_bindings.swap_remove(i); // Remove this binding - this drops the borrow on self.bindings
                let (func_index, code_index) = binding.opcode;
                let binding_index = binding.index;
                let (_, opcode) = &mut self.functions[func_index].code[code_index];

                // Update the binding if possible
                match (binding.ty, &result) {
                    (ReferenceType::Load, LateBound::Local(index)) => *opcode = PushGlobal(*index),
                    (ReferenceType::Load, LateBound::Method(index)) => *opcode = Constant(*index),
                    (ReferenceType::Store, LateBound::Local(index)) => *opcode = StoreGlobal(*index, false),
                    (ReferenceType::StorePattern, LateBound::Local(local)) => {
                        // Need to update all the matching late bound globals in the pattern, which should be referred to by `ExecPattern`
                        let pattern_index: usize = match opcode {
                            ExecPattern(index) => *index as usize - self.baked_patterns.len(),
                            _ => panic!("LoadPattern should find a ExecPattern to update"),
                        };

                        let pattern = &mut self.patterns[pattern_index];
                        pattern.visit(&mut |op| {
                            if let ParserStoreOp::LateBoundGlobal(index) = op {
                                if *index == binding_index {
                                    *op = ParserStoreOp::Bound(StoreOp::Global(*local))
                                }
                            }
                        });
                    },
                    (ReferenceType::Invalid, _) => panic!("Invalid reference type set!"),
                    (_, LateBound::Method(_)) => self.error_at(binding.loc, InvalidAssignmentTarget),
                }
                continue; // Skip the below i += 1, since this binding got swap-removed
            }

            i += 1;
        }
    }

    /// Declares a synthetic local variable. Unlike `declare_local()`, this can never fail.
    /// Returns the index of the local variable in `locals`.
    pub fn declare_synthetic_local(&mut self) -> usize {
        self.synthetic_local_index += 1;
        self.declare_local_internal(format!("${}", self.synthetic_local_index - 1))
    }

    /// Declares a local variable by the name `name` in the current scope.
    fn declare_local_internal(&mut self, name: String) -> usize {
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
    ///    This behavior is controlled by `modify_lvt` parameter.
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
    pub fn pop_locals(&mut self, scope: Option<u32>, modify_lvt: bool, emit_pop: bool, emit_lifts: bool) {

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
    pub fn current_locals(&self) -> &Locals {
        self.locals.last().unwrap()
    }

    /// Returns the current function's `locals`
    pub fn current_locals_mut(&mut self) -> &mut Locals {
        self.locals.last_mut().unwrap()
    }

    /// Returns the output code of the current function
    pub fn current_function(&self) -> &Vec<(Location, Opcode)> {
        match &self.current_locals().func {
            Some(func) => &self.functions[*func].code,
            None => &self.output
        }
    }

    /// Returns the output code of the current function
    pub fn current_function_mut(&mut self) -> &mut Vec<(Location, Opcode)> {
        match self.current_locals().func {
            Some(func) => &mut self.functions[func].code,
            None => &mut self.output
        }
    }

    /// Returns a mutable reference to the current `ParserFunctionImpl`. Will panic if a function is currently not being parsed.
    pub fn current_function_impl(&mut self) -> &mut ParserFunctionImpl {
        let func: usize = self.current_locals().func.unwrap();
        &mut self.functions[func]
    }

    /// Returns the locals reference of the current function, like `current_function_mut()`
    pub fn current_locals_reference_mut(&mut self) -> &mut Vec<String> {
        match self.current_locals().func {
            Some(func) => &mut self.functions[func].locals_reference,
            None => self.locals_reference,
        }
    }

    /// Resolves a reference like `resolve_reference()`, but requires that the reference returned be mutable (i.e. can be assigned to).
    /// If the bound reference is not mutable, it will return `Invalid` instead.
    ///
    /// - Native Functions cannot be assigned to as they cannot be shadowed
    /// - Methods in enclosing modules are not assignable
    pub fn resolve_mutable_reference(&mut self, loc: Location, name: String) -> LValueReference {
        match self.resolve_reference(name) {
            LValueReference::NativeFunction(_) |
            LValueReference::Method(_) => {
                self.error_at(loc, InvalidAssignmentTarget);
                LValueReference::Invalid
            }
            lvalue => lvalue
        }
    }

    /// Resolve an identifier, which can be one of many things, each of which are tried in-order:
    ///
    /// 1. Native Functions. These cannot be shadowed (as it creates interesting conflict scenarios), and they are all technically global functions.
    /// 2. Locals in the current function (the same function depth).
    ///    - Higher (nested) scope locals shadow lower scoped locals.
    /// 3. Locals in enclosing functions.
    ///    - These are bound as _upvalues_, as they may be referenced once the function's scope has exited.
    ///    - Upvalues are also shadowed by locals in higher (nested) scopes, and by ones in higher (nested) enclosing functions.
    /// 4. Functions defined in enclosing modules.
    ///    - Functions can be referenced directly (without need to emit a `Module->method` `GetField` opcode) by name, while in a function defined on that module.
    ///    - todo: with `self` methods on modules, this also introduces more complexity here with binding to `self->` methods
    /// 5. Global Variables (with both function depth == 0 and scope depth == 0)
    ///
    /// ### Late Binding
    /// We also allow late binding for two explicit scenarios:
    ///
    /// - Functions defined in enclosing modules may be late bound, as neither may be invoked before the definition of the module is completed (as modules have no directly executable code)
    /// - Global variables may be late bound (for semantic convenience), but this binding needs to be checked at runtime (via the `InitGlobal` opcode). Note this can only occur within a function scope (as otherwise code is purely procedural).
    ///
    /// In the event that late binding may be possible, we return a special 'late bound' `LValueReference`, which holds an error or information to be replaced when the variable is later bound.
    ///
    /// Finally, if no valid identifier can be found (i.e. in global scope, with no matching native function, local, or global), this will return `LValueReference::Invalid`. In this case, a semantic error will have already been raised.
    pub fn resolve_reference(&mut self, name: String) -> LValueReference {
        if let Some(b) = core::NativeFunction::find(&name) {
            return LValueReference::NativeFunction(b);
        }

        // 1. Search for locals in the current function. This may return `Local`, or `Global` based on the scope of the variable.
        //   - Locals that are captured as upvalues, but are now being referenced as locals again, emit upvalue references, as the stack stops getting updated after a value is lifted into an upvalue.
        for local in self.current_locals().locals.iter().rev() {
            if local.name == name && local.initialized {
                return if local.is_global() {
                    LValueReference::Global(local.index)
                } else {
                    LValueReference::Local(local.index)
                }
            }
        }

        // 2. If we are in function depth > 0, we search in enclosing functions (and global scope), for values that can be captured by this function.
        //   - Globals that are not true globals can be captured in the same manner as upvalues (these are fairly uncommon in practice)
        //   - Locals in an enclosing function can be captured.
        if self.function_depth > 0 {
            for depth in (0..self.function_depth).rev() { // Iterate through the range of [function_depth - 1, ... 0]
                for local in self.locals[depth as usize].locals.iter().rev() { // In reverse, as we go inner -> outer scopes
                    if local.name == name && local.initialized && !local.is_global() { // Note that it must **not** be a true global, anything else can be captured as an upvalue
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
                if local.name == name && local.initialized && local.is_global() {
                    return LValueReference::Global(local.index)
                }
            }
        }

        // 4. If we are within a function and module, we may bind to functions defined on this, or enclosing modules
        if self.function_depth > 0 && self.module_depth > 0 {

            // Search modules from top-down to find matching methods, then fields
            // Field and method names are unique on a single module so we don't need to worry about conflicts
            for module in self.modules.iter().rev() {
                match module.methods.get(&name) {
                    Some(function_id) => {
                        return LValueReference::Method(*function_id)
                    }
                    None => {}
                }
            }

            // todo: bind to fields, which requires a `self->` reference
            // todo: consider case of binding to parent module with self:
            // module Outer {
            //     fn a() {}
            //     struct Inner(a) {
            //         fn b() { a }  <-- does this raise an error because `self->a` is not accessible? or bind to `Outer->a` ?
        }

        // 4. / 5. Late Bindings
        // Either within a function or within a module
        if self.function_depth > 0 || self.module_depth > 0 {
            let binding = LateBinding::new(self.late_binding_next_index, name, self.functions.len() - 1, self.next_opcode(), self.prev_location(), !self.error_recovery);

            self.late_binding_next_index += 1;
            return LValueReference::LateBinding(binding);
        }

        // In global scope if we still could not resolve a variable, we return `None`
        self.semantic_error(UndeclaredIdentifier(name));
        LValueReference::Invalid
    }

    /// Resolves an `UpValue` reference.
    /// For a given reference to a local, defined at a function depth `local_depth` at index `local_index`, this will
    /// bubble up the upvalue through each of the enclosing functions between here and `self.function_depth`, and ensure the variable is added as an `UpValue`.
    fn resolve_upvalue(&mut self, local_depth: u32, local_index: u32) -> LValueReference {

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
        LValueReference::UpValue(index)
    }

    /// Initializes a local, so it can be referenced.
    /// Marks the corresponding `Local` as initialized, and also (if necessary), pushes a `IncGlobalCount` opcode.
    pub fn init_local(&mut self, index: usize) {
        let local = &mut self.current_locals_mut().locals[index];
        local.initialized = true;
        if local.is_global() {
            self.push(InitGlobal);
        }
    }

    /// Declares a field as part of the current module. If the field name is already used, raises a semantic error.
    pub fn declare_field(&mut self, type_index: u32, name: String) {
        let fields: &mut Vec<String> = &mut self.modules.last_mut().unwrap().fields;
        if fields.contains(&name) {
            self.semantic_error(DuplicateFieldName(name))
        } else {
            fields.push(name.clone());
            let offset = fields.len() - 1;
            self.declare_field_offset(name, type_index, offset);
        }
    }

    /// Declares a method as part of the current module. If the method name is already in use (either via a field or method), raises a semantic error.
    ///
    /// Returns `true` if the declaration was successful.
    pub fn declare_method(&mut self, type_index: u32, name: String, loc: Location, args: &[LValue], var_arg: bool) {
        let module: &mut Module = self.modules.last_mut().unwrap();

        // Need to also check conflicts with fields
        // Technically we can have fields that conflict with methods, so long as the methods in question aren't `self` methods:
        // ```
        // struct Foo(bar) {
        //     fn bar() {}
        // }
        // ```
        // This references a `Foo->bar` and a `Foo(nil)->bar` which are distinct. But it introduces ambiguity if `bar` was a instance method:
        // ```
        // Foo(nil)->bar // is this a field or method?
        // ```
        // In the interest of preventing confusion, we deny this behavior completely.
        //
        // Some prior art:
        // - Java allows conflicts, but (1) access levels, and (2) you can't reference functions without invoking them, and (3) you can't invoke fields -> no ambiguity
        // - Python fields and methods are identical, meaning conflicts can occur and they must be unique (same situation here)
        if module.fields.contains(&name) || module.methods.contains_key(&name) {
            self.semantic_error_at(loc, DuplicateFieldName(name.clone()));
        }

        // Always declare the function, it will be cleaned up later
        let function_id: u32 = self.declare_function(name.clone(), &args, var_arg);
        let methods = &mut self.modules.last_mut().unwrap().methods;
        let offset = methods.len();

        methods.insert(name.clone(), function_id);
        self.declare_field_offset(name.clone(), type_index, offset);
        self.resolve_late_bindings(name, LateBound::Method(function_id));
    }

    fn declare_field_offset(&mut self, name: String, type_index: u32, field_offset: usize) {
        let next_field_index: u32 = self.fields.fields.len() as u32;
        let field_index: u32 = *self.fields.fields
            .entry(name)
            .or_insert(next_field_index);

        self.fields.lookup.insert((type_index, field_index), field_offset);
    }

    /// Declares a new type, and returns the corresponding `type index`.
    pub fn declare_type(&mut self) -> u32 {
        self.fields.types += 1;
        self.fields.types - 1
    }

    pub fn declare_native(&mut self, module_name: String, method_name: String, nargs: usize) -> u32 {
        let module_index = self.library.modules.iter()
            .position(|v| v == &module_name)
            .unwrap_or_else(|| {
                self.library.modules.push(module_name.clone());
                self.library.modules.len() - 1
            });

        self.library.methods.push(FunctionLibraryEntry::new(module_index as u32, module_name, method_name, nargs));
        (self.library.methods.len() - 1) as u32
    }

    /// Resolves a field name to a specific field index. If the field is not present, raises a parse error.
    /// Returns the `field_index`, if one was found, or `None` if not.
    pub fn resolve_field(&self, name: &String) -> Option<u32> {
        self.fields.fields.get(name).copied()
    }
}