/// This module contains core semantic analysis related functions in the parser, as we merge the parse and semantic phases of the compiler into a single pass.
/// This also contains core structures that are used by the parser for semantic analysis.
///
/// The functions declared in this module are public to be used by `parser/mod.rs`, but the module `semantic` is not exported itself.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::CString;
use std::fmt::Debug;
use fxhash::FxBuildHasher;
use indexmap::IndexMap;
use itertools::Itertools;

use crate::compiler::parser::{Parser, ParserError, ParserErrorType};
use crate::core;
use crate::core::Pattern;
use crate::reporting::Location;
use crate::vm::{FunctionImpl, IntoValue, Method, Opcode, StoreOp, StructTypeImpl, ValuePtr};
use crate::compiler::parser::core::{BranchType, Code, ForwardBlockId, OpcodeId, ReverseBlockId};
use crate::compiler::parser::expr::Expr;

use Opcode::{*};
use ParserErrorType::{*};


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Loop {
    pub(super) start_id: ReverseBlockId,
    pub(super) break_ids: Vec<ForwardBlockId>,

    /// The scope immediately enclosing the loop. This scope is used for `break` statements to jump to, which needs to pop everything inside the loop scope
    enclosing_scope: u32,

    /// The scope within the loop, containing per-loop-iteration variables.
    /// This is the scope that a `continue` statement needs to jump to, which is the minimal scope still inside the loop.
    inner_scope: u32,
}

impl Loop {
    fn new(start_id: ReverseBlockId, enclosing_scope: u32, inner_scope: u32) -> Loop {
        Loop { start_id, break_ids: Vec::new(), enclosing_scope, inner_scope }
    }

    pub fn enclosing_scope(&self) -> u32 { self.enclosing_scope }
    pub fn inner_scope(&self) -> u32 { self.inner_scope }
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
    pub(super) func: Option<usize>,
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
        self.locals[index].name.clone().to_named()
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
    fields: HashMap<String, FieldReference, FxBuildHasher>,

    /// A table which maps pairs of `(type index, field index)` to a `field offset`
    /// The `type index` is known at runtime, based on the runtime type of the struct in use.
    /// The `field index` is known at compile time, based on the identifier that it resolves to.
    /// The resultant `field offset` is a index into a specific struct object's `Vec<ValuePtr>` of fields.
    lookup: HashMap<(u32, u32), usize, FxBuildHasher>,

    /// The next available `type_index`
    types: u32,
}

#[derive(Debug, Clone)]
struct FieldReference {
    field_index: u32,
    /// If `None`, this field has been declared.
    /// If `Some(loc)`, this is the location of the first place where this field was referenced, but it has not been declared yet.
    loc: Option<Location>,
}

impl Fields {
    pub(super) fn new() -> Fields {
        Fields {
            fields: HashMap::with_hasher(FxBuildHasher::default()),
            lookup: HashMap::with_hasher(FxBuildHasher::default()),
            types: 0,
        }
    }

    pub fn get_offset(&self, type_index: u32, field_index: u32) -> Option<usize> {
        self.lookup.get(&(type_index, field_index)).copied()
    }

    pub fn get_name(&self, field_index: u32) -> String {
        self.fields.iter()
            .find(|(_, v)| field_index == v.field_index)
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



#[derive(Debug, Clone)]
struct Local {
    /// The local variable name - either a string name or the `self` keyword
    name: Reference,
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
    fn new(name: Reference, index: usize, scope_depth: u32, function_depth: u32) -> Local {
        Local { name, index: index as u32, scope_depth, function_depth, initialized: false, captured: false }
    }

    fn is_global(&self) -> bool {
        self.function_depth == 0 && self.scope_depth == 0
    }

    fn as_reference(&self) -> LValueReference {
        if self.is_global() {
            return LValueReference::Global(self.index)
        }
        match self.name {
            Reference::Named(_) => LValueReference::Local(self.index),
            Reference::This | Reference::LateThis => LValueReference::LocalThis(self.index),
        }
    }
}


#[derive(Debug, Clone)]
pub struct LateBinding {
    /// A unique index for every late binding, along with the name (not unique).
    index: usize,
    name: String,
    /// This is a late-bound reference to `self`, which is required when the late binding is later declared as a `self` method.
    /// If the binding didn't have a `self` available, this will be `Invalid`, which needs to raise an error upon trying to bind to a `self` method.
    this: LValueReference,
    /// The type of the reference. This indicates what needs to be updated later (either an opcode location, or a pattern).
    ty: ReferenceType,
    loc: Location,
    /// A fallback binding. If this is present, an error should never be raised, and the fallback should be used instead.
    /// The fallback here will always be a global, and this will be the index
    fallback: Option<u32>,
    /// If true, an error should be raised. Represents the state of the error recovery mode when hit
    error: bool,
}

impl LateBinding {
    fn with_fallback(name: String, fallback: u32, parser: &mut Parser) -> Box<LateBinding> {
        LateBinding::new(name, Some(fallback), parser)
    }

    fn with_error(name: String, parser: &mut Parser) -> Box<LateBinding> {
        LateBinding::new(name, None, parser)
    }

    fn new(name: String, fallback: Option<u32>, parser: &mut Parser) -> Box<LateBinding> {
        let index = parser.late_binding_next_index;
        let loc = parser.prev_location();
        let error = !parser.error_recovery;
        let this = parser.resolve_reference(Reference::LateThis);

        parser.late_binding_next_index += 1;

        Box::new(LateBinding { index, name, this, ty: ReferenceType::Invalid, loc, fallback, error })
    }

    pub fn update(&mut self, ty: ReferenceType) {
        self.ty = ty;
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
    /// `Invalid` is used as a the default type of an invalid reference. Either initializing default references, or when continuing in the site of an error, this is used.
    #[default]
    Invalid,

    /// `This` and `Named` are used when parsing `LValue`s that are resolved later due to backtracking.
    /// This is used, for example in function parameters, or in pattern assignment statements.
    This, // The `self` type in instance methods
    Named(String),

    /// The below cases are _assignable_ `LValueReference`s. These are used when an `LValue` is directly constructed, i.e. within an expression.
    /// They can be converted from a load to a store reference without mutating the underlying functionality.
    Local(u32),
    Global(u32),
    UpValue(u32),
    /// `ThisField` represents a bound self field.
    /// - `self` is loaded either by a `Local(index)` or a `UpValue(index)`
    /// - The field is accessed via a `GetField(field_index)` or `SetField(field_index)`
    ThisField { upvalue: bool, index: u32, field_index: u32 },

    /// The below cases are _not assignable_ `LValueReference`s. These are used similarly in expressions, but forbid themselves from being used as an assignment target.
    /// `LocalThis` and `UpValueThis` are variants of `Local` and `UpValue` respectively, which are not assignable, as they represent `self`
    LocalThis(u32),
    UpValueThis(u32),
    Method(u32),
    /// `ThisMethod` represents a bound self method.
    /// - `self` is loaded either by a `Local(index)` or a `UpValue(index)`
    /// - The method is accessed via a `GetMethod(function_id)`
    ThisMethod { upvalue: bool, index: u32, function_id: u32 },
    NativeFunction(core::NativeFunction),

    /// Late Bindings can be either methods or globals - their assignability is unknown. This is resolved when the binding is resolved by checking the `ReferenceType` field on the binding.
    LateBinding(Box<LateBinding>),
}


#[derive(Debug, Clone)]
pub enum ReferenceType {
    Invalid,
    Load(OpcodeId),
    Store(OpcodeId),
    /// The index is an index into `Parser.patterns`, not the actual pattern ID.
    StorePattern(usize)
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
            LValue::Named(LValueReference::This) => String::from("self"),
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
                *it = parser.resolve_mutable_reference(loc, it.into_reference());
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
            LValue::Named(LValueReference::This) => {
                parser.declare_self_local();
            },
            LValue::Named(it) | LValue::VarNamed(it) => {
                let name: String = it.as_named();
                if let Some(local) = parser.declare_local(name, false) {
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
                    parser.push_store_lvalue(local, parser.prev_location(), false);
                    if !in_expression {
                        parser.push(Pop);
                    }
                }
            }
            LValue::Terms(_) => {
                let index = parser.patterns.len();
                let pattern = self.build_pattern(parser, index);
                parser.declare_pattern(pattern);
                if !in_expression {
                    parser.push(Pop); // Push the final pop
                }
            },
        }
    }

    fn build_pattern(self, parser: &mut Parser, pattern_index: usize) -> Pattern<ParserStoreOp> {
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
                    pattern.push_index(index, lvalue.as_store_op(parser, pattern_index));
                    index += 1;
                },
                LValue::VarNamed(lvalue) => {
                    let low = index;
                    index = -(len as i64 - index);
                    let high = index;

                    pattern.push_slice(low, high, lvalue.as_store_op(parser, pattern_index));
                },
                terms @ LValue::Terms(_) => {
                    pattern.push_pattern(index, terms.build_pattern(parser, pattern_index));
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
            _ => panic!("at LValueReference::as_named()"),
        }
    }

    /// Unwraps this `LValueReference` into a pure `Reference` which can then be resolved as a local variable.
    fn into_reference(&mut self) -> Reference {
        match std::mem::take(self) {
            LValueReference::Named(it) => Reference::Named(it),
            LValueReference::This => Reference::This,
            _ => panic!("at LValueReference::into_reference()"),
        }
    }

    fn as_store_op(self, parser: &mut Parser, pattern_index: usize) -> ParserStoreOp {
        match self {
            LValueReference::Local(index) => ParserStoreOp::Bound(StoreOp::Local(index)),
            LValueReference::Global(index) => ParserStoreOp::Bound(StoreOp::Global(index)),
            LValueReference::UpValue(index) => ParserStoreOp::Bound(StoreOp::UpValue(index)),

            LValueReference::LateBinding(mut binding) => {
                let index = binding.index;

                binding.update(ReferenceType::StorePattern(pattern_index));
                parser.late_bindings.push(*binding);

                ParserStoreOp::LateBoundGlobal(index)
            }

            LValueReference::Invalid => ParserStoreOp::Invalid, // Error has already been raised
            _ => panic!("Invalid store from within pattern: {:?}", self),

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
    ///
    /// These are the block IDs corresponding to the initial default arguments. Each entry matches where the function call position should start from:
    ///
    /// <pre>
    /// default_args[0] --+                     default_args[1] --+
    ///                   |                                       |
    /// [ Nil : Next ] ---+---&gt; [ Int(1) Int(2) Plus : Next ] ----+---&gt; [ ... function body ... ]
    /// </pre>
    pub(super) default_args: Vec<ReverseBlockId>,

    /// If the last argument in this function is a variadic argument, meaning it needs special behavior when invoked with >= `max_args()`
    variadic: bool,
    /// If the function is a FFI native function
    native: bool,
    /// If the function is a instance method on a struct
    instance: bool,

    /// Bytecode for the function body itself
    pub(super) code: Code,

    /// Constant index for this function, which is used to fix the function later
    constant_id: u32,
}

impl ParserFunctionImpl {
    /// Bakes this parser function into an immutable `FunctionImpl`.
    /// The `head` and `tail` pointers are computed based on the surrounding code.
    pub(super) fn bake(self, constants: &mut [ValuePtr], default_args: Vec<usize>, head: usize, tail: usize) {
        constants[self.constant_id as usize] = FunctionImpl::new(head, tail, self.name, self.args, default_args, self.variadic, self.native, self.instance).to_value();
    }

    pub(super) fn set_native(&mut self) {
        self.native = true;
    }
}

pub struct ParserFunctionParameters {
    /// The vector of parsed (but not resolved) `LValue` parameters.
    pub args: Vec<LValue>,
    /// Any default argument values, if present
    pub default_args: Vec<Expr>,
    /// If `true`, this function has a variadic `*x` argument as the last argument
    pub variadic: bool,
    /// If `true`, this function has an explicit `self` parameter as it's first argument
    pub instance: bool,
}


/// A representation of a module (or struct) which may expose methods and fields that can be bound.
pub struct Module {
    name: String,
    is_module: bool,

    methods: IndexMap<String, Method>,
    fields: Vec<String>,
}

impl Module {
    pub fn new(name: String, is_module: bool) -> Module {
        Module { name, is_module, methods: IndexMap::new(), fields: Vec::new() }
    }

    pub fn bake(self, instance_type: u32, constructor_type: u32) -> StructTypeImpl {
        StructTypeImpl::new(self.name, self.fields, self.methods.into_values().collect(), instance_type, constructor_type, self.is_module)
    }
}

enum LateBound {
    Local(u32),
    Method { function_id: u32, instance: bool }
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


/// A type to query `resolve_reference()` with, which includes both named variables and keyword `self`.
///
/// N.B. Intentionally does not implement `Eq` as we want `This` and `LateThis` to be equivalent (see `is_equivalent`) in almost all scenarios.
#[derive(Debug, Clone)]
pub enum Reference {
    Named(String),
    This,
    /// This is identical to `This` in all respects, except in the case of an error, querying `LateThis` will return `Invalid` **without** raising a semantic error.
    /// It is necessary to check if we can bind to `self` for the purposes of late-bound methods.
    LateThis,
}

impl Reference {
    fn is_self(&self) -> bool {
        matches!(self, Reference::This | Reference::LateThis)
    }

    fn is_equivalent(&self, other: &Reference) -> bool {
        match other {
            Reference::Named(name) => self.is_same(name),
            Reference::This | Reference::LateThis => self.is_self(),
        }
    }

    fn is_same(&self, other: &String) -> bool {
        match self {
            Reference::Named(name) => name == other,
            Reference::This | Reference::LateThis => false,
        }
    }

    fn to_named(self) -> String {
        match self {
            Reference::Named(name) => name,
            Reference::This | Reference::LateThis => String::from("self")
        }
    }
}

impl<'a> Parser<'a> {

    // ===== Loops ===== //

    /// Marks the beginning of a loop type statement, for the purposes of tracking `break` and `continue` statements.
    pub fn begin_loop(&mut self) -> ReverseBlockId {
        let loop_start = self.branch_reverse(); // Top of the loop, push onto the loop stack
        let depth: u32 = self.scope_depth;
        self.current_locals_mut().loops.push(Loop::new(loop_start, depth + 1, depth + 1));
        loop_start
    }

    /// Marks the beginning of a `for` loop type statement, for the purposes of tracking `break` and `continue` statements.
    ///
    /// The primary difference between this and `begin_loop()` is the scope depth is tracked differently.
    /// - In a typical loop, the enclosing and inner scopes are the same.
    /// - `for` loops have two separate scopes, with the enclosing being one higher than the inner scope
    pub fn begin_for_loop(&mut self) -> ReverseBlockId {
        let loop_start = self.branch_reverse(); // Top of the loop, push onto the loop stack
        let depth: u32 = self.scope_depth;
        self.current_locals_mut().loops.push(Loop::new(loop_start, depth, depth + 1));
        loop_start
    }

    /// Marks the end of a loop, at the point where `break` statements should jump to (so after any `else` statements attached to the loop)
    pub fn end_loop(&mut self) {
        for jump in self.current_locals_mut().loops.pop().unwrap().break_ids {
            self.join_forward(jump, BranchType::Jump);
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

    // ===== Structs / Modules ===== //

    pub fn begin_module(&mut self, is_module: bool) -> String {
        // Structs can only be declared in global scope (function and scope depth)
        // Scoped structs or modules introduce a couple issues and have better existing solutions:
        // 1. Scoped structs -> struct and module methods are allowed to be closures.
        //    This is just weird, and better served by a struct declared globally with fields that are referenced (via `self` methods) in the struct
        // 2. Nested structs + modules are still possible, but only within module scope depth:
        // ```
        // module outer {
        //     module inner { // we can support this, and both are still ""global""
        // ```
        // 3. Instances of structs, and the module itself, can always escape an inner scope, so the object must be known globally anyway.
        //    In these cases, it makes more sense to restrict them to only being available in global scope.
        if self.function_depth != 0 || self.scope_depth != 0 {
            self.semantic_error(StructNotInGlobalScope);
        }

        let module_name: String = self.expect_identifier(|t| ExpectedStructNameAfterStruct(t, is_module));

        // Declare a local for the struct in the global scope
        self.declare_local(module_name.clone(), true);

        // Increment module depth, after initial checks, now we know we are definitely parsing a module.
        self.module_depth += 1;
        self.modules.push(Module::new(module_name.clone(), is_module));

        module_name
    }

    pub fn end_module(&mut self, instance_type: u32, constructor_type: u32) {
        let module = self.modules.pop().unwrap().bake(instance_type, constructor_type);
        let id: u32 = self.declare_const(module);

        self.push(Constant(id));

        // Decrement module depth
        self.module_depth -= 1;
    }

    /// Declares a function with a given name and arguments.
    /// Returns the constant identifier for this function, however the function itself is currently located in `self.functions`, not `self.constants`.
    /// Instead, we push a dummy `Nil` into the constants array, and store the constant index on our parser function. During teardown, we inject these into the right spots.
    pub fn declare_function(&mut self, name: String, params: &ParserFunctionParameters) -> u32 {
        let constant_id: u32 = self.constants.len() as u32;

        self.constants.push(ValuePtr::nil());
        self.functions.push(ParserFunctionImpl {
            name,
            args: params.args.iter().map(|u| u.to_code_str()).collect(),
            default_args: Vec::new(),
            variadic: params.variadic,
            native: false,
            instance: params.instance,
            code: Code::new(),
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
    /// - If `init` is true, this local will be immediately initialized (if declared) afterwards.
    pub fn declare_local(&mut self, name: String, init: bool) -> Option<usize> {

        // Lookup the name as a binding - if it is, it will be denied as we don't allow shadowing global native functions
        if core::NativeFunction::find(&name).is_some() {
            self.semantic_error(LocalVariableConflictWithNativeFunction(name.clone()));
            return None
        }

        // Ensure there are no conflicts within the current scope, as we don't allow shadowing in the same scope.
        for local in &self.locals.last().unwrap().locals {
            if local.scope_depth == self.scope_depth && local.name.is_same(&name) {
                self.semantic_error(LocalVariableConflict(name.clone()));
                return None
            }
        }

        let index = self.declare_local_internal(Reference::Named(name.clone()));
        let local = &self.locals.last().unwrap().locals[index];

        if local.is_global() {
            // Resolve any late bound references to this global
            self.resolve_late_bindings(name.clone(), LateBound::Local(local.index));

            // Declare this global variable's name
            self.global_references.push(name);
        }

        if init {
            self.init_local(index);
        }

        Some(index)
    }

    /// Declares and initializes the local variable for a `self` keyword.
    pub fn declare_self_local(&mut self) {
        let local = self.declare_local_internal(Reference::This);
        self.init_local(local);
    }

    /// Declares a synthetic local variable. Unlike `declare_local()`, this can never fail.
    /// Returns the index of the local variable in `locals`.
    pub fn declare_synthetic_local(&mut self) -> usize {
        let name = Reference::Named(self.synthetic_name());
        self.declare_local_internal(name)
    }

    /// Declares a local variable by the name `name` in the current scope.
    fn declare_local_internal(&mut self, name: Reference) -> usize {
        let local: Local = Local::new(name, self.locals.last().unwrap().locals.len(), self.scope_depth, self.function_depth);
        self.locals.last_mut().unwrap().locals.push(local);
        self.locals.last().unwrap().locals.len() - 1
    }

    /// Upon declaring a global variable, struct field or method, this will resolve any references to this variable if they exist.
    /// It may raise errors due to the variable not being assignable (i.e. a module method).
    ///
    /// - `name` is the name of the newly declared variable.
    /// - `result` is the type of the newly declared variable.
    fn resolve_late_bindings(&mut self, name: String, result: LateBound) {
        let mut i = 0;
        while i < self.late_bindings.len() {
            let binding = &self.late_bindings[i];

            // If the binding matches, it will be declared here
            // The first declaration is always the matching one, as the precedence for methods/fields -> globals is always maintained
            if binding.name == name {
                let binding = self.late_bindings.swap_remove(i); // Remove this binding - this drops the borrow on self.bindings

                // Update the binding if possible
                match (binding.ty, &result) {
                    (ReferenceType::Load(at), LateBound::Local(index)) => self.insert(at, binding.loc, PushGlobal(*index)),
                    (ReferenceType::Store(at), LateBound::Local(index)) => self.insert(at, binding.loc, StoreGlobal(*index, false)),
                    (ReferenceType::StorePattern(pattern_index), LateBound::Local(local)) => {
                        // Need to update all the matching late bound globals in the pattern
                        let pattern = &mut self.patterns[pattern_index];
                        pattern.visit(&mut |op| {
                            if let ParserStoreOp::LateBoundGlobal(index) = op {
                                if *index == binding.index {
                                    *op = ParserStoreOp::Bound(StoreOp::Global(*local))
                                }
                            }
                        });
                    },
                    (ReferenceType::Load(at), LateBound::Method { function_id, instance }) => {
                        if *instance {
                            // If the current method is a self method, it can only be called through other self methods, and they need to do a `GetMethod` to do so
                            // Otherwise, it needs to raise a parser error
                            match binding.this {
                                LValueReference::Invalid => self.error_at(binding.loc, UndeclaredIdentifier(String::from("self"))),

                                // Since we have a single reference point for insert, we need to insert in reverse order from the same point
                                // Slightly different opcodes for `LocalThis` vs. `UpValueThis`
                                LValueReference::LocalThis(index) => {
                                    self.insert(at.clone(), binding.loc, GetMethod(*function_id));
                                    self.insert(at, binding.loc, PushLocal(index))
                                }
                                LValueReference::UpValueThis(index) => {
                                    self.insert(at.clone(), binding.loc, GetMethod(*function_id));
                                    self.insert(at, binding.loc, PushUpValue(index))
                                },

                                lvalue => panic!("`self` should not be bound to {:?}", lvalue)
                            }
                        } else {
                            // If the current method is a non-self method, then references to it just use the function ID
                            // Both self methods and non-self methods can call it this way.
                            self.insert(at, binding.loc, Constant(*function_id));
                        }
                    },
                    (ReferenceType::Store(_), LateBound::Method { .. }) => self.error_at(binding.loc, InvalidAssignmentTarget),
                    (_, _) => panic!("Invalid reference type set!"),
                }
                continue; // Skip the below i += 1, since this binding got swap-removed
            }

            i += 1;
        }
    }

    /// Resolves any outstanding late bindings present at teardown, by either falling back to a global, or raises an error.
    pub fn resolve_remaining_late_bindings(&mut self) {
        for binding in self.late_bindings.drain(..) {
            if let Some(fallback) = binding.fallback {
                match binding.ty {
                    ReferenceType::Load(at) => Parser::do_insert(at, binding.loc, PushGlobal(fallback), &mut self.functions),
                    ReferenceType::Store(at) => Parser::do_insert(at, binding.loc, StoreGlobal(fallback, false), &mut self.functions),
                    _ => panic!()
                }
            } else if binding.error {
                self.errors.insert(ParserError::new(UndeclaredIdentifier(binding.name), binding.loc));
            }
        }

        // Also emit errors for any missing used-but-not-declared fields
        for (field_name, field) in &self.fields.fields {
            if let Some(loc) = field.loc {
                Parser::do_error(loc, InvalidFieldName(field_name.clone()), false, &mut self.errors, &mut self.error_recovery);
            }
        }
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

    /// Returns a mutable reference to the current `ParserFunctionImpl`. Will panic if a function is currently not being parsed.
    pub fn current_function_mut(&mut self) -> &mut ParserFunctionImpl {
        let func: usize = self.current_locals().func.unwrap();
        &mut self.functions[func]
    }

    /// Resolves a reference like `resolve_reference()`, but requires that the reference returned be mutable (i.e. can be assigned to).
    /// If the bound reference is not mutable, it will return `Invalid` instead.
    ///
    /// - Native Functions cannot be assigned to as they cannot be shadowed
    /// - Methods in enclosing modules are not assignable
    pub fn resolve_mutable_reference(&mut self, loc: Location, name: Reference) -> LValueReference {
        match self.resolve_reference(name) {
            LValueReference::NativeFunction(_) |
            LValueReference::Method(_) |
            LValueReference::LocalThis(_) => {
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
    ///    - `self` method and fields can be referenced by a combination of referencing `self`, plus a reference to a `GetField` or `GetMethod` opcode.
    ///    - This is indicated by the `ThisField` and `ThisMethod` references.
    /// 5. Global Variables (with both function depth == 0 and scope depth == 0)
    ///
    /// ### `self` References
    ///
    /// Note that when `Reference::This` is provided, the return types are limited to:
    /// - `LValueReference::LocalThis` -> indicating the `self` parameter is present in a local
    /// - `LValueReference::UpValueThis` -> indicating the `self` parameter is present in an upvalue
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
    pub fn resolve_reference(&mut self, name: Reference) -> LValueReference {
        if let Reference::Named(ref name) = name {
            if let Some(b) = core::NativeFunction::find(name) {
                return LValueReference::NativeFunction(b);
            }
        }

        // 1. Search for locals in the current function. This may return `Local`, or `Global` based on the scope of the variable.
        //   - Locals that are captured as upvalues, but are now being referenced as locals again, emit upvalue references, as the stack stops getting updated after a value is lifted into an upvalue.
        for local in self.current_locals().locals.iter().rev() {
            if local.name.is_equivalent(&name) && local.initialized {
                return local.as_reference();
            }
        }

        // 2. If we are in function depth > 0, we search in enclosing functions (and global scope), for values that can be captured by this function.
        // 3. Locals in an enclosing function can be captured.
        if self.function_depth > 0 {
            for depth in (0..self.function_depth).rev() { // Iterate through the range of [function_depth - 1, ... 0]
                for local in self.locals[depth as usize].locals.iter().rev() { // In reverse, as we go inner -> outer scopes
                    if local.name.is_equivalent(&name) && local.initialized && !local.is_global() { // Note that it must **not** be a global, anything else can be captured as an upvalue
                        let index = local.index;
                        let is_self = local.name.is_self();
                        self.locals[depth as usize].locals[index as usize].captured = true;
                        return self.resolve_upvalue(depth, index, is_self);
                    }
                }
            }
        }

        // The rest of the resolution methods will not use a `self` reference, so we discard that possibility here
        let name = match name {
            Reference::Named(name) => name,
            Reference::This => {
                self.semantic_error(UndeclaredIdentifier(String::from("self")));
                return LValueReference::Invalid
            }
            // Return `Invalid` without raising an error
            Reference::LateThis => return LValueReference::Invalid,
        };

        // 4. If we are within a function and module, we may bind to functions defined on this, or enclosing modules
        // Note that we can only bind to named references via this mechanism, `self` is only present as a local
        if self.function_depth > 0 && self.module_depth > 0 {
            // Resolve reference to fields or other `self` methods
            //
            // 1. A reference to a field or `self` method, without an explicit `self` is different than a non-self method, in that it cannot fully elide the `self`.
            //    That is to say, in `fn f(self) { g() }`, we _must_ interpret `g()` as `self->g()`
            // 2. Fields or `self` methods can only be resolved on the immediate enclosing struct. Parent enclosing structs are unrelated.
            //    To use the analogy from Java, inner structs are "static" inner classes, and don't have a reference to their parent.
            // 3. In the case where a field or `self` method is in a higher nested struct than a non-self method in an outer struct, we consider it hidden,
            //    even if this is illegal to reference (i.e. via a non-self method):
            // ```
            // module Outer {
            //     fn a() {}
            //     struct Inner(a) {
            //         fn b() { a }  <- this raises an error because 'self' is not accessible (3.)
            // ```
            //
            // - In practice, (1.) means if we first detect we may resolve a field or `self` method (on the immediately enclosing struct, see (2.)), we then have to resolve `self` _as a local variable_.
            // - Then, we can insert said local variable reference (which may error - i.e. if this is not a `self` method), and emit the `GetField` corresponding to this reference.
            // - This necessitates a new kind of `LValueReference`, which combines the `self` reference (a `Local`), the method reference (a `Method`), and will emit the corresponding `GetField`

            let module = self.modules.last().unwrap(); // Only search the top module for self methods and fields
            if !module.is_module { // fields and `self` are only allowed on structs

                // We can check fields and methods in any order because they are required to be disjoint already
                if let Some(method) = module.methods.get(&name) {
                    if method.instance() { // Only check for instance methods here - other methods will be resolved below
                        let function_id = method.function_id();
                        return match self.resolve_reference(Reference::This) {
                            // `self` was successful, so we can create the composite reference
                            LValueReference::LocalThis(index) => LValueReference::ThisMethod { upvalue: false, index, function_id },
                            LValueReference::UpValueThis(index) => LValueReference::ThisMethod { upvalue: true, index, function_id },

                            LValueReference::Invalid => LValueReference::Invalid,
                            lvalue => panic!("Invalid `self` for method load {:?}", lvalue),
                        }
                    }
                }

                if let Some(field_index) = module.fields.iter().position(|field| field == &name) {
                    return match self.resolve_reference(Reference::This) {
                        // `self` was successful, so we can create the composite reference
                        LValueReference::LocalThis(index) => LValueReference::ThisField { upvalue: false, index, field_index: field_index as u32 },
                        LValueReference::UpValueThis(index) => LValueReference::ThisField { upvalue: true, index, field_index: field_index as u32 },

                        LValueReference::Invalid => LValueReference::Invalid,
                        lvalue => panic!("Invalid `self` for field load {:?}", lvalue)
                    }
                }
            }

            // Search for method next - this maintains (3.) from above, since we would've errored above
            for module in self.modules.iter().rev() {
                if let Some(method) = module.methods.get(&name) {
                    if !method.instance() { // Don't allow binding to instance methods here
                        return LValueReference::Method(method.function_id())
                    }
                }
            }
        }

        // 5. If we are in a function depth > 0, then we can also resolve true globals
        // If we are in function depth == 0, any true globals will be caught and resolved as locals by (1.) (but the opcodes for global load/store will still be emitted)
        if self.function_depth > 0 {
            for local in self.locals[0].locals.iter().rev() {
                if local.name.is_same(&name) && local.initialized && local.is_global() {
                    if self.module_depth > 0 {
                        // Note that module methods should take priority over globals, and both can be late bound
                        // In order to handle this, we create a late binding here, but with a fallback instead of an error
                        // If unbound, this will default to assuming the fallback
                        return LValueReference::LateBinding(LateBinding::with_fallback(name, local.index, self));
                    }
                    return LValueReference::Global(local.index)
                }
            }

            // Late Bindings, but with an error
            return LValueReference::LateBinding(LateBinding::with_error(name, self));
        }

        // In global scope if we still could not resolve a variable, we return `None`
        self.semantic_error(UndeclaredIdentifier(name));
        LValueReference::Invalid
    }

    /// Resolves an `UpValue` reference.
    /// For a given reference to a local, defined at a function depth `local_depth` at index `local_index`, this will
    /// bubble up the upvalue through each of the enclosing functions between here and `self.function_depth`, and ensure the variable is added as an `UpValue`.
    fn resolve_upvalue(&mut self, local_depth: u32, local_index: u32, is_self: bool) -> LValueReference {

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
        if is_self { LValueReference::UpValueThis(index) } else { LValueReference::UpValue(index) }
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
    pub fn declare_method(&mut self, type_index: u32, name: String, loc: Location, params: &ParserFunctionParameters) {
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
        let function_id: u32 = self.declare_function(name.clone(), params);
        let methods = &mut self.modules.last_mut().unwrap().methods;
        let offset = methods.len();

        methods.insert(name.clone(), Method::new(function_id, params.instance));
        self.declare_field_offset(name.clone(), type_index, offset);
        self.resolve_late_bindings(name, LateBound::Method { function_id, instance: params.instance });
    }

    fn declare_field_offset(&mut self, name: String, type_index: u32, field_offset: usize) {
        let next_field_index: u32 = self.fields.fields.len() as u32;
        let field_index: u32 = match self.fields.fields.entry(name) {
            Entry::Occupied(it) => {
                let it = it.into_mut();
                it.loc = None; // The field has been used, so clear the possible error
                it.field_index
            }
            Entry::Vacant(it) => {
                it.insert(FieldReference { loc: None, field_index: next_field_index });
                next_field_index
            }
        };

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

    /// Resolves a field name to a specific field index.
    ///
    /// Note that while this always returns a `field_index`, it records the field as not-declared, and an error will be raised at teardown.
    pub fn resolve_field(&mut self, name: String) -> u32 {
        let loc = if self.error_recovery { None } else { Some(self.prev_location()) };
        let next_field_index: u32 = self.fields.fields.len() as u32;

        match self.fields.fields.entry(name) {
            Entry::Occupied(it) => it.into_mut().field_index,
            Entry::Vacant(it) => {
                // Insert a new field reference, but with a `Some(loc)` to flag the possible error
                it.insert(FieldReference { loc, field_index: next_field_index });
                next_field_index
            }
        }
    }
}