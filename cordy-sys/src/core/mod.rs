use std::collections::{BinaryHeap, VecDeque};
use std::default::Default;
use std::fs;
use std::hash::Hash;
use fxhash::FxBuildHasher;
use indexmap::{IndexMap, IndexSet};

use crate::trace;
use crate::vm::{ErrorResult, IntoIterableValue, IntoValue, MAX_INT, MIN_INT, operator, RuntimeError, Type, ValueOption, ValuePtr, ValueResult, VirtualInterface};
use crate::vm::operator::BinaryOp;

pub use crate::core::collections::{get_index, get_slice, set_index, to_index};
pub use crate::core::strings::format_string;
pub use crate::core::pattern::Pattern;

use Argument::{*};
use NativeFunction::{*};
use RuntimeError::{*};


mod math;
mod pattern;
mod strings;
mod collections;


/// An enum representing all possible native functions implemented in Cordy
#[repr(u8)]
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub enum NativeFunction {
    Read,
    ReadLine,
    Print,
    ReadText,
    WriteText,
    Env,
    Argv,
    Bool,
    Int,
    Complex,
    Str,
    List,
    Set,
    Dict,
    Heap,
    Vector,
    Function,
    Iterable,
    Repr,
    Eval,
    TypeOf,
    Monitor,

    // Native Operators
    OperatorSub,
    OperatorUnaryNot,

    OperatorMul,
    OperatorDiv,
    OperatorDivSwap,
    OperatorPow,
    OperatorPowSwap,
    OperatorMod,
    OperatorModSwap,
    OperatorIs,
    OperatorIsSwap,
    OperatorIsNot,
    OperatorIsNotSwap,
    OperatorIn,
    OperatorInSwap,
    OperatorNotIn,
    OperatorNotInSwap,
    OperatorAdd,
    OperatorAddSwap,
    OperatorLeftShift,
    OperatorLeftShiftSwap,
    OperatorRightShift,
    OperatorRightShiftSwap,
    OperatorBitwiseAnd,
    OperatorBitwiseOr,
    OperatorBitwiseXor,
    OperatorLessThan,
    OperatorLessThanSwap,
    OperatorLessThanEqual,
    OperatorLessThanEqualSwap,
    OperatorGreaterThan,
    OperatorGreaterThanSwap,
    OperatorGreaterThanEqual,
    OperatorGreaterThanEqualSwap,
    OperatorEqual,
    OperatorNotEqual,

    // strings
    ToLower,
    ToUpper,
    Replace,
    Search,
    Trim,
    Split,
    Join,
    Char,
    Ord,
    Hex,
    Bin,

    // collections
    Len,
    Range,
    Enumerate,
    Sum,
    Min,
    Max,
    MinBy,
    MaxBy,
    Map,
    Filter,
    FlatMap,
    Concat, // Native optimized version of flatMap(fn(x) -> x)
    Zip,
    Reduce,
    Sort,
    SortBy,
    GroupBy,
    Reverse,
    Permutations,
    Combinations,
    Any,
    All,
    Memoize,
    Union,
    Intersect,
    Difference,

    Peek, // Peek first value
    Pop, // Remove value at end
    PopFront, // Remove value at front
    Push, // Insert value at end
    PushFront, // Insert value at front
    Insert, // Insert value at index
    Remove, // Remove (list: by index, set: by value, dict: by key)
    Clear, // Remove all values - shortcut for `retain(fn(_) -> false)`
    Find, // Find first value (list, set) or key (dict) by predicate
    RightFind, // Find last index of value (list, set), or key (dict) by predicate
    IndexOf, // Find first index of value, or index by predicate
    RightIndexOf, // Find last index of a value, or index by predicate
    Default, // For a `Dict`, sets the default value
    Keys, // `Dict.keys` -> returns a set of all keys
    Values, // `Dict.values` -> returns a list of all values

    // math
    Abs,
    Sqrt,
    Gcd,
    Lcm,
    CountOnes,
    CountZeros,
    Real,
    Imag,
}


impl NativeFunction {

    /// Return the total number of `NativeFunction`s
    ///
    /// Uses unsafe rust to use `variant_count`, and if that ever breaks, we can always replace with a constant.
    pub const fn total() -> usize {
        std::mem::variant_count::<NativeFunction>()
    }

    /// Find a native function with the given name, that is not hidden.
    pub fn find(name: &str) -> Option<NativeFunction> {
        NATIVE_FUNCTIONS.iter()
            .find(|info| info.name == name && !info.hidden)
            .map(|info| info.native)
    }

    /// Returns the minimum amount of arguments needed to evaluate this function, where below this number it will return a partial function
    pub fn min_nargs(&self) -> u32 { self.info().arg.min_nargs() }

    /// Returns the name of the function
    pub fn name(&self) -> &'static str { self.info().name }

    /// Returns the `repr` string of the function, which is the form `fn <name>(<args> ...)`
    pub fn repr(&self) -> String { let info = self.info(); format!("fn {}({})", info.name, info.args) }

    pub fn is_operator(&self) -> bool { self.swap() != *self }

    /// An `operator` refers to an operator which has a direct opcode representation.
    /// Note this excludes asymmetric 'swap' operators
    pub fn is_binary_operator(&self) -> bool { self.as_binary_operator().is_some() }
    pub fn is_binary_operator_swap(&self) -> bool { self.swap().as_binary_operator().is_some() }

    pub fn as_binary_operator(&self) -> Option<BinaryOp> {
        match self {
            OperatorMul => Some(BinaryOp::Mul),
            OperatorDiv => Some(BinaryOp::Div),
            OperatorPow => Some(BinaryOp::Pow),
            OperatorMod => Some(BinaryOp::Mod),
            OperatorIs => Some(BinaryOp::Is),
            OperatorIsNot => Some(BinaryOp::IsNot),
            OperatorIn => Some(BinaryOp::In),
            OperatorNotIn => Some(BinaryOp::NotIn),
            OperatorAdd => Some(BinaryOp::Add),
            OperatorLeftShift => Some(BinaryOp::LeftShift),
            OperatorRightShift => Some(BinaryOp::RightShift),
            OperatorBitwiseAnd => Some(BinaryOp::And),
            OperatorBitwiseOr => Some(BinaryOp::Or),
            OperatorBitwiseXor => Some(BinaryOp::Xor),
            OperatorLessThan => Some(BinaryOp::LessThan),
            OperatorLessThanEqual => Some(BinaryOp::LessThanEqual),
            OperatorGreaterThan => Some(BinaryOp::GreaterThan),
            OperatorGreaterThanEqual => Some(BinaryOp::GreaterThanEqual),
            OperatorEqual => Some(BinaryOp::Equal),
            OperatorNotEqual => Some(BinaryOp::NotEqual),
            _ => None
        }
    }

    pub fn swap(self) -> NativeFunction {
        match self {
            OperatorDiv => OperatorDivSwap,
            OperatorDivSwap => OperatorDiv,
            OperatorPow => OperatorPowSwap,
            OperatorPowSwap => OperatorPow,
            OperatorMod => OperatorModSwap,
            OperatorModSwap => OperatorMod,
            OperatorIs => OperatorIsSwap,
            OperatorIsSwap => OperatorIs,
            OperatorIsNot => OperatorIsNotSwap,
            OperatorIsNotSwap => OperatorIsNot,
            OperatorIn => OperatorInSwap,
            OperatorInSwap => OperatorIn,
            OperatorNotIn => OperatorNotInSwap,
            OperatorNotInSwap => OperatorNotIn,
            OperatorAdd => OperatorAddSwap,
            OperatorAddSwap => OperatorAdd,
            OperatorLeftShift => OperatorLeftShiftSwap,
            OperatorLeftShiftSwap => OperatorLeftShift,
            OperatorRightShift => OperatorRightShiftSwap,
            OperatorRightShiftSwap => OperatorRightShift,
            OperatorLessThan => OperatorLessThanSwap,
            OperatorLessThanSwap => OperatorLessThan,
            OperatorLessThanEqual => OperatorLessThanEqualSwap,
            OperatorLessThanEqualSwap => OperatorLessThanEqual,
            OperatorGreaterThan => OperatorGreaterThanSwap,
            OperatorGreaterThanSwap => OperatorGreaterThan,
            OperatorGreaterThanEqual => OperatorGreaterThanEqualSwap,
            OperatorGreaterThanEqualSwap => OperatorGreaterThanEqual,
            op => op,
        }
    }

    fn info(self) -> &'static NativeFunctionInfo {
        &NATIVE_FUNCTIONS[self as usize]
    }
}


/// Information associated with a native function, including compiler-visible data (such as name, hidden), and argument information.
#[derive(Debug)]
struct NativeFunctionInfo {
    native: NativeFunction,
    name: &'static str,
    args: &'static str,
    arg: Argument,
    hidden: bool,
}

impl NativeFunctionInfo {
    const fn new(native: NativeFunction, name: &'static str, args: &'static str, arg: Argument, hidden: bool) -> NativeFunctionInfo {
        NativeFunctionInfo { native, name, args, arg, hidden }
    }
}

static NATIVE_FUNCTIONS: [NativeFunctionInfo; NativeFunction::total()] = load_native_functions();


const fn load_native_functions() -> [NativeFunctionInfo; NativeFunction::total()] {

    const fn op1(f: NativeFunction, name: &'static str, args: &'static str, arg: Argument) -> NativeFunctionInfo { NativeFunctionInfo::new(f, name, args, arg, true) }
    const fn op2(f: NativeFunction, name: &'static str) -> NativeFunctionInfo { NativeFunctionInfo::new(f, name, "lhs, rhs", Arg2, true) }
    const fn new(f: NativeFunction, name: &'static str, args: &'static str, arg: Argument) -> NativeFunctionInfo { NativeFunctionInfo::new(f, name, args, arg, false) }

    [
        new(Read, "read", "", Arg0),
        new(ReadLine, "read_line", "", Arg0),
        new(Print, "print", "...", Unique),
        new(ReadText, "read_text", "file", Arg1),
        new(WriteText, "write_text", "file, text", Arg2),
        new(Env, "env", "...", Arg0To1),
        new(Argv, "argv", "", Arg0),
        new(Bool, "bool", "x", Arg1),
        new(Int, "int", "x, default?", Arg1To2),
        new(Complex, "complex", "", Invalid),
        new(Str, "str", "x", Arg1),
        new(List, "list", "...", Iter),
        new(Set, "set", "...", Iter),
        new(Dict, "dict", "...", Iter),
        new(Heap, "heap", "...", Iter),
        new(Vector, "vector", "...", Unique),
        new(Function, "function", "", Invalid),
        new(Iterable, "iterable", "", Invalid),
        new(Repr, "repr", "x", Arg1),
        new(Eval, "eval", "expr", Arg1),
        new(TypeOf, "typeof", "x", Arg1),
        new(Monitor, "monitor", "cmd", Arg1),

        // operator
        op1(OperatorSub, "(-)", "x", Arg1To2),
        op1(OperatorUnaryNot, "(!)", "x", Arg1),

        op2(OperatorMul, "(*)"),
        op2(OperatorDiv, "(/)"),
        op2(OperatorDivSwap, "(/)"),
        op2(OperatorPow, "(**)"),
        op2(OperatorPowSwap, "(**)"),
        op2(OperatorMod, "(%)"),
        op2(OperatorModSwap, "(%)"),
        op2(OperatorIs, "(is)"),
        op2(OperatorIsSwap, "(is)"),
        op2(OperatorIsNot, "(is not)"),
        op2(OperatorIsNotSwap, "(is not)"),
        op2(OperatorIn, "(in)"),
        op2(OperatorInSwap, "(in)"),
        op2(OperatorNotIn, "(not in)"),
        op2(OperatorNotInSwap, "(not in)"),
        op2(OperatorAdd, "(+)"),
        op2(OperatorAddSwap, "(+)"),
        op2(OperatorLeftShift, "(<<)"),
        op2(OperatorLeftShiftSwap, "(<<)"),
        op2(OperatorRightShift, "(>>)"),
        op2(OperatorRightShiftSwap, "(>>)"),
        op2(OperatorBitwiseAnd, "(&)"),
        op2(OperatorBitwiseOr, "(|)"),
        op2(OperatorBitwiseXor, "(^)"),
        op2(OperatorLessThan, "(<)"),
        op2(OperatorLessThanSwap, "(<)"),
        op2(OperatorLessThanEqual, "(<=)"),
        op2(OperatorLessThanEqualSwap, "(<=)"),
        op2(OperatorGreaterThan, "(>)"),
        op2(OperatorGreaterThanSwap, "(>)"),
        op2(OperatorGreaterThanEqual, "(>=)"),
        op2(OperatorGreaterThanEqualSwap, "(>=)"),
        op2(OperatorEqual, "(==)"),
        op2(OperatorNotEqual, "(!=)"),

        // strings
        new(ToLower, "to_lower", "x", Arg1),
        new(ToUpper, "to_upper", "x", Arg1),
        new(Replace, "replace", "pattern, replacer, x", Arg3),
        new(Search, "search", "pattern, x", Arg2),
        new(Trim, "trim", "x", Arg1),
        new(Split, "split", "pattern, x", Arg2),
        new(Join, "join", "joiner, iter", Arg2),
        new(Char, "char", "x", Arg1),
        new(Ord, "ord", "x", Arg1),
        new(Hex, "hex", "x", Arg1),
        new(Bin, "bin", "x", Arg1),

        new(Len, "len", "x", Arg1),
        new(Range, "range", "start, stop, step", Arg1To3),
        new(Enumerate, "enumerate", "iter", Arg1),
        new(Sum, "sum", "...", IterNonEmpty),
        new(Min, "min", "...", UniqueNonEmpty),
        new(Max, "max", "...", UniqueNonEmpty),
        new(MinBy, "min_by", "key_or_cmp, iter", Arg2),
        new(MaxBy, "max_by", "key_or_cmp, iter", Arg2),
        new(Map, "map", "f, iter", Arg2),
        new(Filter, "filter", "f, iter", Arg2),
        new(FlatMap, "flat_map", "f, iter", Arg2),
        new(Concat, "concat", "iter", Arg1),
        new(Zip, "zip", "...", IterNonEmpty),
        new(Reduce, "reduce", "f, iter", Arg2),
        new(Sort, "sort", "...", IterNonEmpty),
        new(SortBy, "sort_by", "f, iter", Arg2),
        new(GroupBy, "group_by", "f, iter", Arg2),
        new(Reverse, "reverse", "...", IterNonEmpty),
        new(Permutations, "permutations", "n, iter", Arg2),
        new(Combinations, "combinations", "n, iter", Arg2),
        new(Any, "any", "f, it", Arg2),
        new(All, "all", "f, it", Arg2),
        new(Memoize, "memoize", "f", Arg1),
        new(Union, "union", "other, self", Arg2),
        new(Intersect, "intersect", "other, self", Arg2),
        new(Difference, "difference", "other, self", Arg2),

        new(Peek, "peek", "collection", Arg1),
        new(Pop, "pop", "collection", Arg1),
        new(PopFront, "pop_front", "collection", Arg1),
        new(Push, "push", "value, collection", Arg2),
        new(PushFront, "push_front", "value, collection", Arg2),
        new(Insert, "insert", "index, value, collection", Arg3),
        new(Remove, "remove", "param, collection", Arg2),
        new(Clear, "clear", "collection", Arg1),
        new(Find, "find", "predicate, collection", Arg2),
        new(RightFind, "rfind", "predicate, collection", Arg2),
        new(IndexOf, "index_of", "value_or_predicate, collection", Arg2),
        new(RightIndexOf, "rindex_of", "value_or_predicate, collection", Arg2),
        new(Default, "default", "value, dictionary", Arg2),
        new(Keys, "keys", "dictionary", Arg1),
        new(Values, "values", "dictionary", Arg1),

        // math
        new(Abs, "abs", "x", Arg1),
        new(Sqrt, "sqrt", "x", Arg1),
        new(Gcd, "gcd", "...", IterNonEmpty),
        new(Lcm, "lcm", "...", IterNonEmpty),
        new(CountOnes, "count_ones", "x", Arg1),
        new(CountZeros, "count_zeros", "x", Arg1),
        new(Real, "real", "x", Arg1),
        new(Imag, "imag", "x", Arg1),
    ]
}



/// A representation of the argument type semantics of a function.
///
/// Some base rules that all functions must obey w.r.t partial evaluation:
///
/// - If `f(a1, ... aN)` is partial, then for all `M < N`, `f(a1, ... aM)` is also partial.
/// - `f()`, if partial, is *implicitly* partial, in that it returns itself.
///
/// With this, we have the following conventions:
///
/// - `Arg<N>` is partial on `[0, N)` arguments, evaluated on `N` arguments, and errors on `> N` arguments.
/// - `Arg<N>To<M>` is partial on `[0, N)` arguments, evaluated on `[N, M)` arguments, and errors on `> M` arguments.
///
/// We also define a few unique types which are never partial, but have unique invocation patterns:
///
/// - `Unique` has unique evaluations for `0`, `1`, and `> 1` arguments.
/// - `Iter` has unique evaluations for `0` and `> 1` arguments, and expands a single argument as an iterable `> 1` argument.
/// - `IterNonEmpty` only evaluates `> 1` arguments, but expands a single argument as an iterable `> 1` argument, and errors on `0` arguments.
///
/// Finally, `Invalid` is a `NativeFunction` which is not evaluable under any arguments, and always errors.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Argument {
    Arg0,
    Arg0To1,
    Arg1,
    Arg1To2,
    Arg1To3,
    Arg2,
    Arg3,
    Unique,
    UniqueNonEmpty,
    Iter,
    IterNonEmpty,
    Invalid,
}

impl Argument {
    /// Returns the minimum amount of arguments needed to evaluate this function, where below this number it will return a partial function
    fn min_nargs(self) -> u32 {
        match self {
            Arg1 | Arg1To2 | Arg1To3 => 1,
            Arg2 => 2,
            Arg3 => 3,
            _ => 0,
        }
    }
}

/// The data structure representing a partially evaluated function.
///
/// Partial function evaluation is slow, in general, but it *can* be fast. The primary slowness comes from having to box arguments into a partial value, unbox them on the stack, then (in the native case), perform a full dispatch against the function, only to pop arguments off the stack again.
///
/// This *should be* optimize-able, given that we already perform a dispatch against the function once, when creating the partial function. So, we use `PartialArgument` to represent that state, post-dispatch.
/// We can also create partial argument types that have an explicit number of partial arguments
#[derive(Debug, Clone)]
pub enum PartialArgument {
    Arg2Par1(ValuePtr),
    Arg3Par1(ValuePtr),
    Arg3Par2(ValuePtr, ValuePtr),
}

impl PartialArgument {
    /// Returns the minimum amount of arguments needed to evaluate this function, where below this number it will return a partial function
    pub fn min_nargs(&self) -> u32 {
        match self {
            PartialArgument::Arg2Par1(_) | PartialArgument::Arg3Par2(_, _) => 1,
            PartialArgument::Arg3Par1(_) => 2,
        }
    }

    #[inline]
    fn to_value(self, f: NativeFunction) -> ValueResult {
        ValuePtr::partial_native(f, self).ok()
    }
}


/// An `InvokeArg` is an optimized structure for a function that is meant to be invoked multiple times, from native code.
/// It pre-dispatches the function against the expected number of arguments, and will either throw an error, or return a `InvokeArg<N>`.
/// This elides having to push **anything** onto the stack for native and partial native functions.
///
/// We expose the following types in `InvokeArg0`, `InvokeArg1`, and `InvokeArg2`:
///
/// - `User`           : User functions are always invoked on the stack, and so they get no special treatment other than categorizing.
/// - `Native`         : For a `InvokeArg<N>`, this represents a native function with a compatible argument type, that can dispatch to `core::invoke_arg<N>`
/// - `NativePar<M>`   : Similar to the above, but where `InvokeArg<N>` dispatches to `core::invoke_arg<N+M>`
/// - `Arg<M>Par<N-1>` : Indicates that a new partial native function will be created based on `PartialArgument::Arg<M>Par<N>`
///
/// `NativeVar` is also a unique type for `InvokeArg1` and `InvokeArg2`:
///
/// - With `InvokeArg1`, it indicates the single argument needs to be iterable-expanded and passed to `core::invoke_var`
/// - With `InvokeArg2`, it indicates the two arguments need to be *boxed* into an iterable and passed to `core::invoke_var`
///
/// Note that `InvokeArg0` can not, definitionally, have partially evaluated arguments, as it would no-op.
/// We do still have to handle that case, however, since it is technically legal code (if contrived).
#[derive(Debug, Clone)]
pub enum InvokeArg0 {
    Noop(ValuePtr),
    User(ValuePtr),
    Native(NativeFunction),
}

#[derive(Debug, Clone)]
enum InvokeArg1 {
    User(ValuePtr),
    Native(NativeFunction),
    NativePar1(NativeFunction, ValuePtr),
    NativePar2(NativeFunction, ValuePtr, ValuePtr),
    NativeVar(NativeFunction),
    Arg2Par1(NativeFunction),
    Arg3Par1(NativeFunction),
    Arg3Par2(NativeFunction, ValuePtr),
}

#[derive(Debug, Clone)]
enum InvokeArg2 {
    User(ValuePtr),
    Native(NativeFunction),
    NativePar1(NativeFunction, ValuePtr),
    NativeVar(NativeFunction),
    Arg3Par1(NativeFunction),
}

impl InvokeArg0 {
    fn from(f: ValuePtr) -> ErrorResult<InvokeArg0> {
        match f.ty() {
            Type::Function | Type::Closure | Type::PartialFunction | Type::StructType | Type::Memoized => Ok(InvokeArg0::User(f)),
            Type::NativeFunction => match f.as_native().info().arg {
                Arg0 | Arg0To1 | Unique | Iter => Ok(InvokeArg0::Native(f.as_native())),
                Arg1 | Arg1To2 | Arg1To3 | Arg2 | Arg3 => Ok(InvokeArg0::Noop(f)), // Partial with zero arg = no-op
                UniqueNonEmpty | IterNonEmpty => IncorrectArgumentsNativeFunction(f.as_native(), 0).err(),
                Invalid => ValueIsNotFunctionEvaluable(f).err(),
            },
            Type::PartialNativeFunction => Ok(InvokeArg0::Noop(f)),
            _ => ValueIsNotFunctionEvaluable(f).err()
        }
    }

    fn invoke<VM : VirtualInterface>(self, vm: &mut VM) -> ValueResult {
        match self {
            InvokeArg0::Noop(f) => f.ok(),
            InvokeArg0::User(f) => vm.invoke_func0(f),
            InvokeArg0::Native(f) => invoke_arg0(f, vm),
        }
    }
}

impl InvokeArg1 {
    fn from(f: ValuePtr) -> ErrorResult<InvokeArg1> {
        match f.ty() {
            Type::Function | Type::Closure | Type::PartialFunction | Type::List | Type::Slice | Type::StructType | Type::GetField | Type::Memoized => Ok(InvokeArg1::User(f)),
            Type::NativeFunction => match f.as_native().info().arg {
                Arg0To1 | Arg1 | Arg1To2 | Arg1To3 | Unique => Ok(InvokeArg1::Native(f.as_native())),
                Iter | UniqueNonEmpty | IterNonEmpty => Ok(InvokeArg1::NativeVar(f.as_native())),
                Arg2 => Ok(InvokeArg1::Arg2Par1(f.as_native())),
                Arg3 => Ok(InvokeArg1::Arg3Par1(f.as_native())),
                Arg0 => IncorrectArgumentsNativeFunction(f.as_native(), 1).err(),
                Invalid => ValueIsNotFunctionEvaluable(f).err(),
            },
            Type::PartialNativeFunction => {
                let it = f.as_partial_native().value;
                match it.partial {
                    PartialArgument::Arg2Par1(arg) => Ok(InvokeArg1::NativePar1(it.func, arg)),
                    PartialArgument::Arg3Par1(arg) => Ok(InvokeArg1::Arg3Par2(it.func, arg)),
                    PartialArgument::Arg3Par2(arg1, arg2) => Ok(InvokeArg1::NativePar2(it.func, arg1, arg2)),
                }
            },
            _ => ValueIsNotFunctionEvaluable(f).err()
        }
    }

    fn invoke<VM: VirtualInterface>(&self, arg: ValuePtr, vm: &mut VM) -> ValueResult {
        match self {
            InvokeArg1::User(f) => vm.invoke_func1(f.clone(), arg),
            InvokeArg1::Native(f) => invoke_arg1(*f, arg, vm),
            InvokeArg1::NativePar1(f, a1) => invoke_arg2(*f, a1.clone(), arg, vm),
            InvokeArg1::NativePar2(f, a1, a2) => invoke_arg3(*f, a1.clone(), a2.clone(), arg, vm),
            InvokeArg1::NativeVar(f) => invoke_var(*f, arg.to_iter()?, vm),
            InvokeArg1::Arg2Par1(f) => PartialArgument::Arg2Par1(arg).to_value(*f),
            InvokeArg1::Arg3Par1(f) => PartialArgument::Arg3Par1(arg).to_value(*f),
            InvokeArg1::Arg3Par2(f, a1) => PartialArgument::Arg3Par2(a1.clone(), arg).to_value(*f),
        }
    }
}

impl InvokeArg2 {
    fn from(f: ValuePtr) -> ErrorResult<InvokeArg2> {
        match f.ty() {
            Type::Function | Type::Closure | Type::PartialFunction | Type::List | Type::Slice | Type::StructType | Type::GetField | Type::Memoized => Ok(InvokeArg2::User(f)),
            Type::NativeFunction => match f.as_native().info().arg {
                Arg1To2 | Arg1To3 | Arg2 | Unique => Ok(InvokeArg2::Native(f.as_native())),
                Iter | UniqueNonEmpty | IterNonEmpty => Ok(InvokeArg2::NativeVar(f.as_native())),
                Arg3 => Ok(InvokeArg2::Arg3Par1(f.as_native())),
                Arg0 | Arg0To1 | Arg1 => IncorrectArgumentsNativeFunction(f.as_native(), 2).err(),
                Invalid => ValueIsNotFunctionEvaluable(f).err(),
            },
            Type::PartialNativeFunction => {
                let it = f.as_partial_native().value;
                match it.partial {
                    PartialArgument::Arg2Par1(_) |
                    PartialArgument::Arg3Par2(_, _) => IncorrectArgumentsNativeFunction(it.func, 3).err(),
                    PartialArgument::Arg3Par1(arg) => Ok(InvokeArg2::NativePar1(it.func, arg)),
                }
            },
            _ => ValueIsNotFunctionEvaluable(f).err()
        }
    }

    fn invoke<VM : VirtualInterface>(&self, arg1: ValuePtr, arg2: ValuePtr, vm: &mut VM) -> ValueResult {
        match self {
            InvokeArg2::User(f) => vm.invoke_func2(f.clone(), arg1, arg2),
            InvokeArg2::Native(f) => invoke_arg2(*f, arg1, arg2, vm),
            InvokeArg2::NativePar1(f, a1) => invoke_arg3(*f, a1.clone(), arg1, arg2, vm),
            InvokeArg2::NativeVar(f) => invoke_var(*f, vec![arg1, arg2].into_iter(), vm),
            InvokeArg2::Arg3Par1(f) => PartialArgument::Arg3Par2(arg1, arg2).to_value(*f),
        }
    }
}



/// Invokes a function with arguments laid out on the stack.
pub fn invoke_stack<VM : VirtualInterface>(f: NativeFunction, nargs: u32, vm: &mut VM) -> ValueResult {
    trace::trace_interpreter!("core::invoke_stack f={}, nargs={}", f.name(), nargs);
    match f.info().arg {
        Arg0 => match nargs {
            0 => invoke_arg0(f, vm),
            _ => IncorrectArgumentsNativeFunction(f, nargs).err()
        },
        Arg0To1 => match nargs {
            0 => invoke_arg0(f, vm),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_arg1(f, a1, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, nargs).err()
        },
        Arg1 => match nargs {
            0 => f.to_value().ok(),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_arg1(f, a1, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, nargs).err()
        },
        Arg1To2 => match nargs {
            0 => f.to_value().ok(),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_arg1(f, a1, vm)
            },
            2 => {
                let a2: ValuePtr = vm.pop();
                let a1: ValuePtr = vm.pop();
                invoke_arg2(f, a1, a2, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, nargs).err()
        },
        Arg1To3 => match nargs {
            0 => f.to_value().ok(),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_arg1(f, a1, vm)
            },
            2 => {
                let a2: ValuePtr = vm.pop();
                let a1: ValuePtr = vm.pop();
                invoke_arg2(f, a1, a2, vm)
            },
            3 => {
                let a3: ValuePtr = vm.pop();
                let a2: ValuePtr = vm.pop();
                let a1: ValuePtr = vm.pop();
                invoke_arg3(f, a1, a2, a3, vm)
            }
            _ => IncorrectArgumentsNativeFunction(f, nargs).err()
        },
        Arg2 => match nargs {
            0 => f.to_value().ok(),
            1 => {
                let a1: ValuePtr = vm.pop();
                PartialArgument::Arg2Par1(a1).to_value(f)
            }
            2 => {
                let a2: ValuePtr = vm.pop();
                let a1: ValuePtr = vm.pop();
                invoke_arg2(f, a1, a2, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, nargs).err()
        },
        Arg3 => match nargs {
            0 => f.to_value().ok(),
            1 => {
                let a1: ValuePtr = vm.pop();
                PartialArgument::Arg3Par1(a1).to_value(f)
            },
            2 => {
                let a2: ValuePtr = vm.pop();
                let a1: ValuePtr = vm.pop();
                PartialArgument::Arg3Par2(a1, a2).to_value(f)
            },
            3 => {
                let a3: ValuePtr = vm.pop();
                let a2: ValuePtr = vm.pop();
                let a1: ValuePtr = vm.pop();
                invoke_arg3(f, a1, a2, a3, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, nargs).err()
        },
        Unique => match nargs {
            0 => invoke_arg0(f, vm),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_arg1(f, a1, vm)
            },
            _ => {
                let args = vm.popn(nargs).into_iter();
                invoke_var(f, args, vm)
            }
        },
        UniqueNonEmpty => match nargs {
            0 => IncorrectArgumentsNativeFunction(f, nargs).err(),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_arg1(f, a1, vm)
            },
            _ => {
                let args = vm.popn(nargs).into_iter();
                invoke_var(f, args, vm)
            }
        },
        Iter => match nargs {
            0 => invoke_arg0(f, vm),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_var(f, a1.to_iter()?, vm)
            },
            _ => {
                let args = vm.popn(nargs).into_iter();
                invoke_var(f, args, vm)
            }
        },
        IterNonEmpty => match nargs {
            0 => IncorrectArgumentsNativeFunction(f, nargs).err(),
            1 => {
                let a1: ValuePtr = vm.pop();
                invoke_var(f, a1.to_iter()?, vm)
            },
            _ => {
                let args = vm.popn(nargs).into_iter();
                invoke_var(f, args, vm)
            }
        },
        Invalid => ValueIsNotFunctionEvaluable(f.to_value()).err(),
    }
}


/// Invokes a partial native function with partial arguments held in the unique `PartialArgument` structure, and additional arguments present on the stack.
pub fn invoke_partial<VM : VirtualInterface>(f: NativeFunction, partial: PartialArgument, nargs: u32, vm: &mut VM) -> ValueResult {
    trace::trace_interpreter!("core::invoke_partial f={}, nargs={}", f.name(), nargs);
    match partial {
        PartialArgument::Arg2Par1(a1) => match nargs {
            0 => PartialArgument::Arg2Par1(a1).to_value(f),
            1 => {
                let a2: ValuePtr = vm.pop();
                invoke_arg2(f, a1, a2, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, 1 + nargs).err(),
        },
        PartialArgument::Arg3Par1(a1) => match nargs {
            0 => PartialArgument::Arg3Par1(a1).to_value(f),
            1 => {
                let a2: ValuePtr = vm.pop();
                PartialArgument::Arg3Par2(a1, a2).to_value(f)
            },
            2 => {
                let a3: ValuePtr = vm.pop();
                let a2: ValuePtr = vm.pop();
                invoke_arg3(f, a1, a2, a3, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, 1 + nargs).err(),
        },
        PartialArgument::Arg3Par2(a1, a2) => match nargs {
            0 => PartialArgument::Arg3Par2(a1, a2).to_value(f),
            1 => {
                let a3: ValuePtr = vm.pop();
                invoke_arg3(f, a1, a2, a3, vm)
            },
            _ => IncorrectArgumentsNativeFunction(f, 2 + nargs).err(),
        },
    }
}


fn invoke_arg0<VM : VirtualInterface>(f: NativeFunction, vm: &mut VM) -> ValueResult {
    match f {
        Read => vm.read().to_value().ok(),
        ReadLine => vm.read_line().to_value().ok(),
        Print => {
            vm.println0();
            ValuePtr::nil().ok()
        },
        Env => vm.get_envs().ok(),
        Argv => vm.get_args().ok(),

        List => VecDeque::new().to_value().ok(),
        Set => IndexSet::with_hasher(FxBuildHasher::default()).to_value().ok(),
        Dict => IndexMap::with_hasher(FxBuildHasher::default()).to_value().ok(),
        Heap => BinaryHeap::new().to_value().ok(),
        Vector => Vec::new().to_value().ok(),

        _ => panic!("core::invoke_arg0() not supported for {:?}", f),
    }
}

fn invoke_arg1<VM : VirtualInterface>(f: NativeFunction, a1: ValuePtr, vm: &mut VM) -> ValueResult {
    match f {
        Print => {
            vm.println(a1.to_str());
            ValuePtr::nil().ok()
        },
        ReadText => {
            let path = a1.check_str()?;
            match fs::read_to_string::<&str>(path.as_str().borrow_const().as_ref()) {
                Ok(text) => text.replace('\r', "").to_value().ok(),
                Err(err) => IOError(err.to_string()).err(),
            }
        },
        Env => vm.get_env(a1.check_str()?.as_str().borrow_const()).ok(),

        Bool => a1.to_bool().to_value().ok(),
        Int => math::convert_to_int(a1, ValueOption::none()),
        Str => a1.to_str().to_value().ok(),
        Vector => if a1.is_precise_complex() {  // Handle `a + bi . vector` as a special case here
            let it = a1.as_precise_complex().value.inner;
            (it.re.to_value(), it.im.to_value()).to_value().ok()
        } else {
            a1.to_iter()?.to_vector().ok()
        },
        Repr => a1.to_repr_str().to_value().ok(),
        Eval => vm.invoke_eval(a1.check_str()?.as_str().borrow_const()),
        TypeOf => type_of(a1).ok(),
        Monitor => vm.invoke_monitor(a1.check_str()?.as_str().borrow_const()),

        OperatorSub => operator::unary_sub(a1),
        OperatorUnaryNot => operator::unary_not(a1),

        ToLower => strings::to_lower(a1),
        ToUpper => strings::to_upper(a1),
        Trim => strings::trim(a1),
        Char => strings::to_char(a1),
        Ord => strings::to_ord(a1),
        Hex => strings::to_hex(a1),
        Bin => strings::to_bin(a1),

        Len => a1.len()?.to_value().ok(),
        Range => ValuePtr::range(0, a1.check_int()?.as_int(), 1),
        Enumerate => ValuePtr::enumerate(a1).ok(),
        Min => match a1.is_native() {
            true if a1.as_native() == Int => MIN_INT.to_value().ok(),
            _ => collections::min(a1.to_iter()?),
        },
        Max => match a1.is_native() {
            true if a1.as_native() == Int => MAX_INT.to_value().ok(),
            _ => collections::max(a1.to_iter()?),
        },
        Concat => collections::flat_map(vm, None, a1),
        Memoize => collections::create_memoized(a1),

        Peek => collections::peek(a1),
        Pop => collections::pop(a1),
        PopFront => collections::pop_front(a1),
        Clear => collections::clear(a1),

        Keys => collections::dict_keys(a1),
        Values => collections::dict_values(a1),

        Abs => math::abs(a1),
        Sqrt => math::sqrt(a1),
        CountOnes => math::count_ones(a1),
        CountZeros => math::count_zeros(a1),
        Real => math::get_real(a1),
        Imag => math::get_imag(a1),

        _ => panic!("core::invoke_arg1() not supported for {:?}", f),
    }
}

fn invoke_arg2<VM : VirtualInterface>(f: NativeFunction, a1: ValuePtr, a2: ValuePtr, vm: &mut VM) -> ValueResult {
    match f {
        WriteText => {
            let path = a1.check_str()?;
            let text = a2.check_str()?;
            match fs::write(path.as_str().borrow_const(), text.as_str().borrow_const()) {
                Ok(_) => ValuePtr::nil().ok(),
                Err(err) => IOError(err.to_string()).err(),
            }
        },
        Int => math::convert_to_int(a1, ValueOption::some(a2)),

        OperatorSub => operator::binary_sub(a1, a2),
        OperatorMul => operator::binary_mul(a1, a2),
        OperatorDiv => operator::binary_div(a1, a2),
        OperatorDivSwap => operator::binary_div(a2, a1),
        OperatorPow => operator::binary_pow(a1, a2),
        OperatorPowSwap => operator::binary_pow(a2, a1),
        OperatorMod => operator::binary_mod(a1, a2),
        OperatorModSwap => operator::binary_mod(a2, a1),
        OperatorIs => operator::binary_is(a1, a2, false),
        OperatorIsSwap => operator::binary_is(a2, a1, false),
        OperatorIsNot => operator::binary_is(a1, a2, true),
        OperatorIsNotSwap => operator::binary_is(a2, a1, true),
        OperatorIn => operator::binary_in(a1, a2, false),
        OperatorInSwap => operator::binary_in(a2, a1, false),
        OperatorNotIn => operator::binary_in(a1, a2, true),
        OperatorNotInSwap => operator::binary_in(a2, a1, true),
        OperatorAdd => operator::binary_add(a1, a2),
        OperatorAddSwap => operator::binary_add(a2, a1),
        OperatorLeftShift => operator::binary_left_shift(a1, a2),
        OperatorLeftShiftSwap => operator::binary_left_shift(a2, a1),
        OperatorRightShift => operator::binary_right_shift(a1, a2),
        OperatorRightShiftSwap => operator::binary_right_shift(a2, a1),
        OperatorBitwiseAnd => operator::binary_bitwise_and(a1, a2),
        OperatorBitwiseOr => operator::binary_bitwise_or(a1, a2),
        OperatorBitwiseXor => operator::binary_bitwise_xor(a1, a2),
        OperatorLessThan => (a1 < a2).to_value().ok(),
        OperatorLessThanSwap => (a2 < a1).to_value().ok(),
        OperatorLessThanEqual => (a1 <= a2).to_value().ok(),
        OperatorLessThanEqualSwap => (a2 <= a1).to_value().ok(),
        OperatorGreaterThan => (a1 > a2).to_value().ok(),
        OperatorGreaterThanSwap => (a2 > a1).to_value().ok(),
        OperatorGreaterThanEqual => (a1 >= a2).to_value().ok(),
        OperatorGreaterThanEqualSwap => (a2 >= a1).to_value().ok(),
        OperatorEqual => (a1 == a2).to_value().ok(),
        OperatorNotEqual => (a1 != a2).to_value().ok(),

        Search => strings::search(a1, a2),
        Split => strings::split(a1, a2),
        Join => strings::join(a1, a2),

        Range => ValuePtr::range(a1.check_int()?.as_int(), a2.check_int()?.as_int(), 1),
        MinBy => collections::min_by(vm, a1, a2),
        MaxBy => collections::max_by(vm, a1, a2),
        Map => collections::map(vm, a1, a2),
        Filter => collections::filter(vm, a1, a2),
        FlatMap => collections::flat_map(vm, Some(a1), a2),
        Reduce => collections::reduce(vm, a1, a2),
        SortBy => collections::sort_by(vm, a1, a2),
        GroupBy => collections::group_by(vm, a1, a2),
        Permutations => collections::permutations(a1, a2),
        Combinations => collections::combinations(a1, a2),
        Any => collections::any(vm, a1, a2),
        All => collections::all(vm, a1, a2),
        Union => collections::set_union(a1, a2),
        Intersect => collections::set_intersect(a1, a2),
        Difference => collections::set_difference(a1, a2),

        Push => collections::push(a1, a2),
        PushFront => collections::push_front(a1, a2),
        Remove => collections::remove(a1, a2),
        Find => collections::left_find(vm, a1, a2, false),
        RightFind => collections::right_find(vm, a1, a2, false),
        IndexOf => collections::left_find(vm, a1, a2, true),
        RightIndexOf => collections::right_find(vm, a1, a2, true),
        Default => collections::dict_set_default(a1, a2),

        _ => panic!("core::invoke_arg2() not supported for {:?}", f),
    }
}

fn invoke_arg3<VM : VirtualInterface>(f: NativeFunction, a1: ValuePtr, a2: ValuePtr, a3: ValuePtr, vm: &mut VM) -> ValueResult {
    match f {
        Replace => strings::replace(vm, a1, a2, a3),
        Range => ValuePtr::range(a1.check_int()?.as_int(), a2.check_int()?.as_int(), a3.check_int()?.as_int()),
        Insert => collections::insert(a1, a2, a3),

        _ => panic!("core::invoke_arg3() not supported for {:?}", f),
    }
}

fn invoke_var<VM : VirtualInterface, I : Iterator<Item=ValuePtr>>(f: NativeFunction, mut an: I, vm: &mut VM) -> ValueResult {
    match f {
        Print => {
            vm.print(an.next().unwrap().to_str());
            for ai in an {
                vm.print(format!(" {}", ai.to_str()));
            }
            vm.println0();
            ValuePtr::nil().ok()
        },

        List => an.to_list().ok(),
        Set => an.to_set().ok(),
        Dict => collections::collect_into_dict(an),
        Heap => an.to_heap().ok(),
        Vector => an.to_vector().ok(),

        Sum => collections::sum(an),
        Min => collections::min(an),
        Max => collections::max(an),
        Zip => collections::zip(an),
        Sort => collections::sort(an).ok(),
        Reverse => collections::reverse(an).ok(),

        Gcd => math::gcd(an),
        Lcm => math::lcm(an),

        _ => panic!("core::invoke_var() not supported for {:?}", f),
    }
}


/// Invokes a `Memoized()` function wrapper from the stack. This assumes the stack is already setup a priori with the memoized wrapper, and arguments in place.
pub fn invoke_memoized<VM : VirtualInterface>(vm: &mut VM, nargs: u32) -> ValueResult {
    let args: Vec<ValuePtr> = vm.popn(nargs);
    let func: ValuePtr = vm.pop();
    let memoized = func.as_memoized();

    let func: ValuePtr = {
        // We cannot use the `.entry()` API, as that requires we mutably borrow the cache during the call to `vm.invoke_func()`
        // We only lookup by key once (in the cached case), and twice (in the uncached case)
        let borrow = memoized.borrow();
        if let Some(ret) = borrow.cache.get(&args) {
            return ret.clone().ok();
        }
        borrow.func.clone()
        // `borrow` is dropped here
    };

    let ret: ValuePtr = vm.invoke_func(func, &args)?;

    // The above computation might've entered a value into the cache - so we have to go through `.entry()` again
    return memoized.borrow_mut().cache
        .entry(args)
        .or_insert(ret)
        .clone()
        .ok();
}


fn type_of(value: ValuePtr) -> ValuePtr {
    match value.ty() {
        Type::Nil => ValuePtr::nil(),
        Type::Bool => Bool.to_value(),
        Type::Int => Int.to_value(),
        Type::Complex => Complex.to_value(),
        Type::Str => Str.to_value(),

        Type::List => List.to_value(),
        Type::Set => Set.to_value(),
        Type::Dict => Dict.to_value(),
        Type::Heap => Heap.to_value(),
        Type::Vector => Vector.to_value(),

        Type::Struct => value.as_struct().borrow().type_impl.get().clone().to_value(), // Structs return their type constructor
        Type::StructType => Function.to_value(), // And the type constructor returns `function`

        Type::Range => Range.to_value(),
        Type::Enumerate => Enumerate.to_value(),
        Type::Slice => Function.to_value(),

        Type::Iter | Type::Memoized | Type::Error | Type::None | Type::Never => panic!("{:?} is synthetic and cannot have type_of() called on it", value),

        Type::Function | Type::PartialFunction | Type::NativeFunction | Type::PartialNativeFunction | Type::Closure | Type::GetField => Function.to_value(),
    }
}


#[cfg(test)]
mod tests {
    use crate::{compiler, core, SourceView};
    use crate::core::{Argument, NativeFunction};
    use crate::vm::{IntoValue, ValuePtr, VirtualInterface, VirtualMachine};

    #[test]
    fn test_native_functions_are_declared_in_order() {
        // Tests `NATIVE_FUNCTIONS[i].native = i`
        for (i, info) in core::NATIVE_FUNCTIONS.iter().enumerate() {
            assert_eq!(info.native as usize, i, "Native function {} / {:?} declared in order {} != {}", info.name, info.native, i, info.native as usize)
        }
    }

    #[test]
    fn test_native_functions_arg_matches_args() {
        // Tests various conventions about the `args` field, based on `arg`
        for info in &core::NATIVE_FUNCTIONS {
            match info.arg {
                Argument::Arg0 => assert_eq!(info.args, "", "in {:?}", info),
                Argument::Arg1 => assert!(!info.args.contains(","), "in {:?}", info),
                Argument::Arg2 => assert_eq!(info.args.split(",").count(), 2, "in {:?}", info),
                Argument::Arg3 => assert_eq!(info.args.split(",").count(), 3, "in {:?}", info),
                Argument::Iter | Argument::IterNonEmpty => assert_eq!(info.args, "...", "in {:?}", info),
                _ => {}
            }
        }
    }

    /// Asserts that no panics are generated from calling all supported combinations of argument types.
    #[test]
    fn test_native_functions_support_from_arg() {
        let mut vm = VirtualMachine::new(compiler::default(), SourceView::empty(), &b""[..], vec![], vec![]);

        for info in &core::NATIVE_FUNCTIONS {
            match info.arg {
                Argument::Arg0 => {
                    let _ = core::invoke_arg0(info.native, &mut vm);
                },
                Argument::Arg0To1 => {
                    let _ = core::invoke_arg0(info.native, &mut vm);
                    let _ = core::invoke_arg1(info.native, ValuePtr::nil(), &mut vm);
                },
                Argument::Arg1 => {
                    let _ = core::invoke_arg1(info.native, ValuePtr::nil(), &mut vm);
                },
                Argument::Arg1To2 => {
                    let _ = core::invoke_arg1(info.native, ValuePtr::nil(), &mut vm);
                    let _ = core::invoke_arg2(info.native, ValuePtr::nil(), ValuePtr::nil(), &mut vm);
                },
                Argument::Arg1To3 => {
                    let _ = core::invoke_arg1(info.native, ValuePtr::nil(), &mut vm);
                    let _ = core::invoke_arg2(info.native, ValuePtr::nil(), ValuePtr::nil(), &mut vm);
                    let _ = core::invoke_arg3(info.native, ValuePtr::nil(), ValuePtr::nil(), ValuePtr::nil(), &mut vm);
                },
                Argument::Arg2 => {
                    let _ = core::invoke_arg2(info.native, ValuePtr::nil(), ValuePtr::nil(), &mut vm);
                },
                Argument::Arg3 => {
                    let _ = core::invoke_arg3(info.native, ValuePtr::nil(), ValuePtr::nil(), ValuePtr::nil(), &mut vm);
                },
                Argument::Unique => {
                    let _ = core::invoke_arg0(info.native, &mut vm);
                    let _ = core::invoke_arg1(info.native, ValuePtr::nil(), &mut vm);
                    let _ = core::invoke_var(info.native, vec![ValuePtr::nil()].into_iter(), &mut vm);
                },
                Argument::UniqueNonEmpty => {
                    let _ = core::invoke_arg1(info.native, ValuePtr::nil(), &mut vm);
                    let _ = core::invoke_var(info.native, vec![ValuePtr::nil()].into_iter(), &mut vm);
                },
                Argument::Iter => {
                    let _ = core::invoke_arg0(info.native, &mut vm);
                    let _ = core::invoke_var(info.native, vec![].into_iter(), &mut vm);
                },
                Argument::IterNonEmpty => {
                    let _ = core::invoke_var(info.native, vec![].into_iter(), &mut vm);
                },
                Argument::Invalid => {},
            }
        }
    }

    /// Asserts that `nargs < native.nargs()` is a sufficient condition for declaring a function is consistent
    #[test]
    fn test_consistency_condition() {
        let mut vm = VirtualMachine::new(compiler::default(), SourceView::empty(), &b""[..], vec![], vec![]);

        fn is_partial(v: &ValuePtr, f: NativeFunction) -> bool {
            v.is_partial_native() && v.as_partial_native_ref().func == f
        }

        for f in core::NATIVE_FUNCTIONS.iter() {
            let min_nargs: u32 = f.arg.min_nargs();
            if min_nargs > 1 {
                for nargs in 1..min_nargs {
                    // Prepare stack arguments
                    let args: Vec<ValuePtr> = (0..nargs).map(|arg| (arg as i64).to_value()).collect();
                    vm.push( f.native.to_value());
                    for arg in args.iter().rev().cloned() {
                        vm.push(arg);
                    }

                    let ret = core::invoke_stack(f.native, nargs, &mut vm).as_result().ok();

                    assert!(ret.is_some(), "Error invoking {:?}", f.native);
                    assert!(is_partial(ret.as_ref().unwrap(), f.native), "Return value of invoking {:?} with nargs={} is not partial: {:?}", f.native, nargs, ret.unwrap());

                    vm.pop();
                }
            }
        }
    }
}
