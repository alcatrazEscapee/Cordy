use std::collections::{BinaryHeap, VecDeque};
use std::fs;
use indexmap::{IndexMap, IndexSet};

use crate::vm::{operator, IntoIterableValue, IntoValue, IntoValueResult, Value, VirtualInterface, RuntimeError, ValueResult};
use crate::vm::operator::BinaryOp;
use crate::{trace, vm};

pub use crate::core::collections::{list_slice, literal_slice, to_index};

use NativeFunction::{*};
use RuntimeError::{*};

mod strings;
mod collections;
mod math;


/// Looks up a global native function, or reserved keyword, by name.
pub fn find_native_function(name: &String) -> Option<NativeFunction> {
    NATIVE_FUNCTIONS.iter()
        .find(|info| info.name == name && !info.hidden)
        .map(|info| info.native)
}


/// A native function implemented in Cordy
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

    // Native Operators
    OperatorUnarySub,
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
    OperatorSub, // Cannot be referenced as (- <expr>)
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
    SyntheticMemoizedFunction,

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
}

/// Information associated with a native function, including compiler-visible data (such as name, hidden), and argument information.
struct NativeFunctionInfo {
    native: NativeFunction,
    name: &'static str,
    args: &'static str,
    nargs: Option<u32>,
    hidden: bool,
}


impl NativeFunction {
    pub fn nargs(self: &Self) -> Option<u32> { self.info().nargs }
    pub fn name(self: &Self) -> &'static str { self.info().name }
    pub fn args(self: &Self) -> &'static str { self.info().args }

    pub fn is_operator(self: &Self) -> bool { self.swap() != *self }

    /// An `operator` refers to an operator which has a direct opcode representation.
    /// Note this excludes asymmetric 'swap' operators
    pub fn is_binary_operator(self: &Self) -> bool { self.as_binary_operator().is_some() }
    pub fn is_binary_operator_swap(self: &Self) -> bool { self.swap().as_binary_operator().is_some() }

    pub fn as_binary_operator(self: &Self) -> Option<BinaryOp> {
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
            OperatorSub => Some(BinaryOp::Sub),
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

    pub fn swap(self: Self) -> NativeFunction {
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

    fn info(self: Self) -> &'static NativeFunctionInfo {
        &NATIVE_FUNCTIONS[self as usize]
    }
}


const N_NATIVE_FUNCTIONS: usize = 114; //std::mem::variant_count::<NativeFunction>(); // unstable
static NATIVE_FUNCTIONS: [NativeFunctionInfo; N_NATIVE_FUNCTIONS] = load_native_functions();


const fn load_native_functions() -> [NativeFunctionInfo; N_NATIVE_FUNCTIONS] {
    macro_rules! declare {
        ($native:expr, $name:expr, $args:expr) => { declare!($native, $name, $args, None, false) };
        ($native:expr, $name:expr, $args:expr, $nargs:expr) => { declare!($native, $name, $args, Some($nargs), false) };
        ($native:expr, $name:expr, $args:expr, $nargs:expr, $hidden:expr) => { NativeFunctionInfo { native: $native, name: $name, args: $args, nargs: $nargs, hidden: $hidden } };
    }
    [
        declare!(Read, "read", "", 1),
        declare!(ReadLine, "read_line", "", 0),
        declare!(Print, "print", "..."),
        declare!(ReadText, "read_text", "file", 1),
        declare!(WriteText, "write_text", "file, text", 2),
        declare!(Env, "env", "..."),
        declare!(Argv, "argv", "", 0),
        declare!(Bool, "bool", "x", 1),
        declare!(Int, "int", "x", 1),
        declare!(Complex, "complex", "x", 1),
        declare!(Str, "str", "x", 1),
        declare!(List, "list", "..."),
        declare!(Set, "set", "..."),
        declare!(Dict, "dict", "..."),
        declare!(Heap, "heap", "..."),
        declare!(Vector, "vector", "..."),
        declare!(Function, "function", "", None, false),
        declare!(Iterable, "iterable", "", None, false),
        declare!(Repr, "repr", "x", 1),
        declare!(Eval, "eval", "expr", 1),
        declare!(TypeOf, "typeof", "x", 1),

        // operator
        declare!(OperatorUnarySub, "(-)", "x", Some(1), true),
        declare!(OperatorUnaryNot, "(!)", "x", Some(1), true),

        declare!(OperatorMul, "(*)", "lhs, rhs", Some(2), true),
        declare!(OperatorDiv, "(/)", "lhs, rhs", Some(2), true),
        declare!(OperatorDivSwap, "(/)", "lhs, rhs", Some(2), true),
        declare!(OperatorPow, "(**)", "lhs, rhs", Some(2), true),
        declare!(OperatorPowSwap, "(**)", "lhs, rhs", Some(2), true),
        declare!(OperatorMod, "(%)", "lhs, rhs", Some(2), true),
        declare!(OperatorModSwap, "(%)", "lhs, rhs", Some(2), true),
        declare!(OperatorIs, "(is)", "lhs, rhs", Some(2), true),
        declare!(OperatorIsSwap, "(is)", "lhs, rhs", Some(2), true),
        declare!(OperatorIsNot, "(is not)", "lhs, rhs", Some(2), true),
        declare!(OperatorIsNotSwap, "(is not)", "lhs, rhs", Some(2), true),
        declare!(OperatorIn, "(in)", "lhs, rhs", Some(2), true),
        declare!(OperatorInSwap, "(in)", "lhs, rhs", Some(2), true),
        declare!(OperatorNotIn, "(not in)", "lhs, rhs", Some(2), true),
        declare!(OperatorNotInSwap, "(not in)", "lhs, rhs", Some(2), true),
        declare!(OperatorAdd, "(+)", "lhs, rhs", Some(2), true),
        declare!(OperatorAddSwap, "(+)", "lhs, rhs", Some(2), true),
        declare!(OperatorSub, "(-)", "lhs, rhs", Some(2), true),
        declare!(OperatorLeftShift, "(<<)", "lhs, rhs", Some(2), true),
        declare!(OperatorLeftShiftSwap, "(<<)", "lhs, rhs", Some(2), true),
        declare!(OperatorRightShift, "(>>)", "lhs, rhs", Some(2), true),
        declare!(OperatorRightShiftSwap, "(>>)", "lhs, rhs", Some(2), true),
        declare!(OperatorBitwiseAnd, "(&)", "lhs, rhs", Some(2), true),
        declare!(OperatorBitwiseOr, "(|)", "lhs, rhs", Some(2), true),
        declare!(OperatorBitwiseXor, "(^)", "lhs, rhs", Some(2), true),
        declare!(OperatorLessThan, "(<)", "lhs, rhs", Some(2), true),
        declare!(OperatorLessThanSwap, "(<)", "lhs, rhs", Some(2), true),
        declare!(OperatorLessThanEqual, "(<=)", "lhs, rhs", Some(2), true),
        declare!(OperatorLessThanEqualSwap, "(<=)", "lhs, rhs", Some(2), true),
        declare!(OperatorGreaterThan, "(>)", "lhs, rhs", Some(2), true),
        declare!(OperatorGreaterThanSwap, "(>)", "lhs, rhs", Some(2), true),
        declare!(OperatorGreaterThanEqual, "(>=)", "lhs, rhs", Some(2), true),
        declare!(OperatorGreaterThanEqualSwap, "(>=)", "lhs, rhs", Some(2), true),
        declare!(OperatorEqual, "(==)", "lhs, rhs", Some(2), true),
        declare!(OperatorNotEqual, "(!=)", "lhs, rhs", Some(2), true),

        // strings
        declare!(ToLower, "to_lower", "x", 1),
        declare!(ToUpper, "to_upper", "x", 1),
        declare!(Replace, "replace", "pattern, replacer, x", 3),
        declare!(Search, "search", "pattern, x", 2),
        declare!(Trim, "trim", "x", 1),
        declare!(Split, "split", "pattern, x", 2),
        declare!(Join, "join", "joiner, iter", 2),
        declare!(Char, "char", "x", 1),
        declare!(Ord, "ord", "x", 1),
        declare!(Hex, "hex", "x", 1),
        declare!(Bin, "bin", "x", 1),

        declare!(Len, "len", "x", 1),
        declare!(Range, "range", "start, stop, step", None, false),
        declare!(Enumerate, "enumerate", "iter", 1),
        declare!(Sum, "sum", "..."),
        declare!(Min, "min", "..."),
        declare!(Max, "max", "..."),
        declare!(MinBy, "min_by", "key_or_cmp, iter", 2),
        declare!(MaxBy, "max_by", "key_or_cmp, iter", 2),
        declare!(Map, "map", "f, iter", 2),
        declare!(Filter, "filter", "f, iter", 2),
        declare!(FlatMap, "flat_map", "f, iter", 2),
        declare!(Concat, "concat", "iter", 1),
        declare!(Zip, "zip", "..."),
        declare!(Reduce, "reduce", "f, iter", 2),
        declare!(Sort, "sort", "..."),
        declare!(SortBy, "sort_by", "f, iter", 2),
        declare!(GroupBy, "group_by", "f, iter", 2),
        declare!(Reverse, "reverse", "iter", 1),
        declare!(Permutations, "permutations", "n, iter", 2),
        declare!(Combinations, "combinations", "n, iter", 2),
        declare!(Any, "any", "..."),
        declare!(All, "all", "..."),
        declare!(Memoize, "memoize", "f", 1),
        declare!(SyntheticMemoizedFunction, "<synthetic memoized>", "f, ...", None, true),

        declare!(Peek, "peek", "collection", 1),
        declare!(Pop, "pop", "collection", 1),
        declare!(PopFront, "pop_front", "collection", 1),
        declare!(Push, "push", "value, collection", 2),
        declare!(PushFront, "push_front", "value, collection", 2),
        declare!(Insert, "insert", "index, value, collection", 3),
        declare!(Remove, "remove", "param, collection", 2),
        declare!(Clear, "clear", "collection", 1),
        declare!(Find, "find", "predicate, collection", 2),
        declare!(RightFind, "rfind", "predicate, collection", 2),
        declare!(IndexOf, "index_of", "value_or_predicate, collection", 2),
        declare!(RightIndexOf, "rindex_of", "value_or_predicate, collection", 2),
        declare!(Default, "default", "value, dictionary", 2),
        declare!(Keys, "keys", "dictionary", 1),
        declare!(Values, "values", "dictionary", 1),

        // math
        declare!(Abs, "abs", "x", 1),
        declare!(Sqrt, "sqrt", "x", 1),
        declare!(Gcd, "gcd", "..."),
        declare!(Lcm, "lcm", "..."),
        declare!(CountOnes, "count_ones", "x", 1),
        declare!(CountZeros, "count_zeros", "x", 1),
    ]
}



/// `allow` usages here are due to macro expansion of the `dispatch!()` cases.
#[allow(unused_mut)]
#[allow(unused_variables)]
pub fn invoke<VM>(native: NativeFunction, nargs: u32, vm: &mut VM) -> ValueResult where VM : VirtualInterface
{
    trace::trace_interpreter!("core::invoke native={}, nargs={}", native.name(), nargs);

    // Dispatch macros for 0, 1, 2, 3 and variadic functions.
    // Most of these will support partial evaluation where the function allows it.
    //
    // dispatch!() handles the following signatures:
    //
    // <expr>                             : f()
    // a1, <expr>                         : f(a1)
    // a1, a2, <expr>                     : f(a1, a2)
    // a1, a2, a3, <expr>                 : f(a1, a2, a3)
    //
    // <expr>, a1, <expr>                 : f(), f(a1)
    // <expr>, a1, <expr>, a2, <expr>     : f(), f(a1), f(a1, a2)
    //
    // a1, <expr>, a2, <expr>             : f(a1), f(a1, a2)
    // a1, <expr>, a2, <expr>, a3, <expr> : f(a1), f(a1, a2), f(a1, a2, a3)
    //
    // dispatch_varargs!() handles the following signatures:
    //
    // a1 an <expr>                       : f(a1, an, ...)  *for synthetic partial methods only
    //
    // <expr> a1 <expr> an <expr>         : f(), f(a1), f(an, ...)
    //
    // an <expr>                          : f(an), f(an, ...)  *single argument is expanded as iterable
    // <expr> an <expr>                   : f(), f(an), f(an, ...)  *single argument is expanded as iterable
    macro_rules! dispatch {
        ($ret:expr) => { // f()
            match nargs {
                0 => {
                    let ret = $ret;
                    ret
                },
                _ => IncorrectArgumentsNativeFunction(native, nargs).err()
            }
        };
        ($a1:ident, $ret:expr) => { // f(a1)
            match nargs {
                0 => Ok(wrap_as_partial(native, nargs, vm)),
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectArgumentsNativeFunction(native, nargs).err()
            }
        };
        ($a1:ident, $a2:ident, $ret:expr) => { // f(a1, a2)
            match nargs {
                0 | 1 => Ok(wrap_as_partial(native, nargs, vm)),
                2 => {
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectArgumentsNativeFunction(native, nargs).err()
            }
        };
        ($a1:ident, $a2:ident, $a3:ident, $ret:expr) => { // f(a1, a2, a3)
            match nargs {
                0 | 1 | 2 => Ok(wrap_as_partial(native, nargs, vm)),
                3 => {
                    let $a3: Value = vm.pop();
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectArgumentsNativeFunction(native, nargs).err()
            }
        };
        ($ret0:expr, $a1:ident, $ret1:expr) => { // f(), f(a1)
            match nargs {
                0 => {
                    let ret = $ret0;
                    ret
                }
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret1;
                    ret
                },
                _ => IncorrectArgumentsNativeFunction(native, nargs).err()
            }
        };
        ($a1:ident, $ret1:expr, $a2:ident, $ret2:expr) => { // f(a1), f(a1, a2)
            match nargs {
                0 => Ok(wrap_as_partial(native, nargs, vm)),
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret1;
                    ret
                },
                2 => {
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret2;
                    ret
                },
                _ => IncorrectArgumentsNativeFunction(native, nargs).err()
            }
        };
        ($a1:ident, $ret1:expr, $a2:ident, $ret2:expr, $a3:ident, $ret3:expr) => { // f(a1), f(a1, a2), f(a1, a2, a3)
            match nargs {
                0 => Ok(wrap_as_partial(native, nargs, vm)),
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret1;
                    ret
                },
                2 => {
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret2;
                    ret
                },
                3 => {
                    let $a3: Value = vm.pop();
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret3;
                    ret
                }
                _ => IncorrectArgumentsNativeFunction(native, nargs).err()
            }
        };
    }
    macro_rules! dispatch_varargs {
        ($a1:ident, $an:ident, $ret:expr) => {
            match nargs {
                0 => panic!("Illegal invoke of synthetic method with no primary argument"),
                _ => {
                    let $an = vm.popn(nargs - 1);
                    let $a1 = vm.pop();
                    $ret
                }
            }
        };
        ($ret0:expr, $a1:ident, $ret1:expr, $an:ident, $ret_n:expr) => { // f(), f(a1), f(an, ...)
            match nargs {
                0 => {
                    let ret = $ret0;
                    ret
                },
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret1;
                    ret
                },
                _ => {
                    let varargs = vm.popn(nargs);
                    let mut $an = varargs.iter().cloned();
                    let ret = $ret_n;
                    ret
                }
            }
        };
        ($an:ident, $ret_n:expr) => { // f(an), f(an, ...)  *single argument is expanded as iterable
            match nargs {
                0 => IncorrectArgumentsNativeFunction(native, 0).err(),
                1 => match vm.pop().as_iter() {
                    Ok($an) => $ret_n,
                    Err(e) => Err(e),
                },
                _ => {
                    let varargs = vm.popn(nargs);
                    let mut $an = varargs.iter().cloned();
                    $ret_n
                },
            }
        };
        ($ret0:expr, $an:ident, $ret_n:expr) => { // f(), f(an), f(an, ...)  *single argument is expanded as iterable
            match nargs {
                0 => $ret0,
                1 => match vm.pop().as_iter() {
                    Ok($an) => $ret_n,
                    Err(e) => Err(e),
                },
                _ => {
                    let varargs = vm.popn(nargs);
                    let mut $an = varargs.iter().cloned();
                    $ret_n
                },
            }
        };
    }

    match native {
        Read => dispatch!(Ok(vm.read().to_value())),
        ReadLine => dispatch!(Ok(vm.read_line().to_value())),
        Print => {
            dispatch_varargs!(vm.println0(), a1, vm.println(a1.to_str()), an, {
                vm.print(an.next().unwrap().to_str());
                for ai in an {
                    vm.print(format!(" {}", ai.to_str()));
                }
                vm.println0()
            });
            Ok(Value::Nil)
        },
        ReadText => dispatch!(a1, match a1 {
            Value::Str(s1) => Ok(fs::read_to_string(s1.as_ref()).unwrap().replace("\r", "").to_value()), // todo: error handling?
            _ => TypeErrorArgMustBeStr(a1).err(),
        }),
        WriteText => dispatch!(a1, a2, {
            fs::write(a1.as_str()?, a2.as_str()?).unwrap();
            Ok(Value::Nil)
        }),
        Env => dispatch!(Ok(vm.get_envs()), a1, Ok(vm.get_env(a1.as_str()?))),
        Argv => dispatch!(Ok(vm.get_args())),
        Bool => dispatch!(a1, Ok(Value::Bool(a1.as_bool()))),
        Int => dispatch!(a1, math::convert_to_int(a1, None), a2, math::convert_to_int(a1, Some(a2))),
        Str => dispatch!(Ok("".to_value()), a1, Ok(a1.to_str().to_value())),
        List => dispatch_varargs!(Ok(VecDeque::new().to_value()), an, Ok(an.to_list())),
        Set => dispatch_varargs!(Ok(IndexSet::new().to_value()), an, Ok(an.to_set())),
        Dict => dispatch_varargs!(Ok(IndexMap::new().to_value()), an, collections::collect_into_dict(an)),
        Heap => dispatch_varargs!(Ok(BinaryHeap::new().to_value()), an, Ok(an.to_heap())),
        Vector => match nargs {
            0 => Ok(Vec::new().to_value()),
            1 => {
                let arg: Value = vm.pop(); // Handle `a + bi . vector` as a special case here
                if let Value::Complex(it) = arg {
                    return Ok(vec![Value::Int(it.re), Value::Int(it.im)].to_value())
                }
                match arg.as_iter() {
                    Ok(an) => Ok(an.to_vector()),
                    Err(e) => Err(e),
                }
            },
            _ => Ok(vm.popn(nargs).to_value()), // Optimization: we can skip iterating, and instead just create a vector from `popn`
        },
        Repr => dispatch!(a1, Ok(a1.to_repr_str().to_value())),
        Eval => dispatch!(a1, vm.invoke_eval(a1.as_str()?)),
        TypeOf => dispatch!(a1, Ok(type_of(a1))),

        // operator
        OperatorUnarySub => dispatch!(a1, operator::unary_sub(a1), a2, operator::binary_sub(a1, a2)),
        OperatorUnaryNot => dispatch!(a1, operator::unary_not(a1)),
        OperatorMul => dispatch!(a1, a2, operator::binary_mul(a1, a2)),
        OperatorDiv => dispatch!(a1, a2, operator::binary_div(a1, a2)),
        OperatorDivSwap => dispatch!(a1, a2, operator::binary_div(a2, a1)),
        OperatorPow => dispatch!(a1, a2, operator::binary_pow(a1, a2)),
        OperatorPowSwap => dispatch!(a1, a2, operator::binary_pow(a2, a1)),
        OperatorMod => dispatch!(a1, a2, operator::binary_mod(a1, a2)),
        OperatorModSwap => dispatch!(a1, a2, operator::binary_mod(a2, a1)),
        OperatorIs => dispatch!(a1, a2, operator::binary_is(a1, a2).to_value()),
        OperatorIsSwap => dispatch!(a1, a2, operator::binary_is(a2, a1).to_value()),
        OperatorIsNot => dispatch!(a1, a2, operator::binary_is(a1, a2).map(|u| !u).to_value()),
        OperatorIsNotSwap => dispatch!(a1, a2, operator::binary_is(a2, a1).map(|u| !u).to_value()),
        OperatorIn => dispatch!(a1, a2, operator::binary_in(a1, a2).to_value()),
        OperatorInSwap => dispatch!(a1, a2, operator::binary_in(a2, a1).to_value()),
        OperatorNotIn => dispatch!(a1, a2, operator::binary_in(a1, a2).map(|u| !u).to_value()),
        OperatorNotInSwap => dispatch!(a1, a2, operator::binary_in(a2, a1).map(|u| !u).to_value()),
        OperatorAdd => dispatch!(a1, a2, operator::binary_add(a1, a2)),
        OperatorAddSwap => dispatch!(a1, a2, operator::binary_add(a2, a1)),
        OperatorSub => dispatch!(a1, a2, operator::binary_sub(a1, a2)),
        OperatorLeftShift => dispatch!(a1, a2, operator::binary_left_shift(a1, a2)),
        OperatorLeftShiftSwap => dispatch!(a1, a2, operator::binary_left_shift(a2, a1)),
        OperatorRightShift => dispatch!(a1, a2, operator::binary_right_shift(a1, a2)),
        OperatorRightShiftSwap => dispatch!(a1, a2, operator::binary_right_shift(a2, a1)),
        OperatorBitwiseAnd => dispatch!(a1, a2, operator::binary_bitwise_and(a1, a2)),
        OperatorBitwiseOr => dispatch!(a1, a2, operator::binary_bitwise_or(a1, a2)),
        OperatorBitwiseXor => dispatch!(a1, a2, operator::binary_bitwise_xor(a1, a2)),
        OperatorLessThan => dispatch!(a1, a2, Ok((a1 < a2).to_value())),
        OperatorLessThanSwap => dispatch!(a1, a2, Ok((a2 < a1).to_value())),
        OperatorLessThanEqual => dispatch!(a1, a2, Ok((a1 <= a2).to_value())),
        OperatorLessThanEqualSwap => dispatch!(a1, a2, Ok((a2 <= a1).to_value())),
        OperatorGreaterThan => dispatch!(a1, a2, Ok((a1 > a2).to_value())),
        OperatorGreaterThanSwap => dispatch!(a1, a2, Ok((a2 > a1).to_value())),
        OperatorGreaterThanEqual => dispatch!(a1, a2, Ok((a1 >= a2).to_value())),
        OperatorGreaterThanEqualSwap => dispatch!(a1, a2, Ok((a2 >= a1).to_value())),
        OperatorEqual => dispatch!(a1, a2, Ok((a1 == a2).to_value())),
        OperatorNotEqual => dispatch!(a1, a2, Ok((a1 != a2).to_value())),

        // strings
        ToLower => dispatch!(a1, strings::to_lower(a1)),
        ToUpper => dispatch!(a1, strings::to_upper(a1)),
        Replace => dispatch!(a1, a2, a3, strings::replace(vm, a1, a2, a3)),
        Search => dispatch!(a1, a2, strings::search(a1, a2)),
        Trim => dispatch!(a1, strings::trim(a1)),
        Split => dispatch!(a1, a2, strings::split(a1, a2)),
        Join => dispatch!(a1, a2, strings::join(a1, a2)),
        Char => dispatch!(a1, strings::to_char(a1)),
        Ord => dispatch!(a1, strings::to_ord(a1)),
        Hex => dispatch!(a1, strings::to_hex(a1)),
        Bin => dispatch!(a1, strings::to_bin(a1)),

        // collections
        Len => dispatch!(a1, a1.len().map(|u| Value::Int(u as i64))),
        Range => dispatch!(a1, Value::range(0, a1.as_int()?, 1), a2, Value::range(a1.as_int()?, a2.as_int()?, 1), a3, Value::range(a1.as_int()?, a2.as_int()?, a3.as_int()?)),
        Enumerate => dispatch!(a1, Ok(Value::Enumerate(Box::new(a1)))),
        Sum => dispatch_varargs!(an, collections::sum(an)),
        Min => match nargs {
            0 => IncorrectArgumentsNativeFunction(native, 0).err(),
            1 => match vm.pop() {
                Value::NativeFunction(Int) => Ok(Value::Int(i64::MIN)),
                an => collections::min(an.as_iter()?),
            },
            _ => collections::min(vm.popn(nargs).iter().cloned()),
        },
        MinBy => dispatch!(a1, a2, collections::min_by(vm, a1, a2)),
        Max => match nargs {
            0 => IncorrectArgumentsNativeFunction(native, 0).err(),
            1 => match vm.pop() {
                Value::NativeFunction(Int) => Ok(Value::Int(i64::MAX)),
                an => collections::max(an.as_iter()?),
            },
            _ => collections::max(vm.popn(nargs).iter().cloned()),
        },
        MaxBy => dispatch!(a1, a2, collections::max_by(vm, a1, a2)),
        Map => dispatch!(a1, a2, collections::map(vm, a1, a2)),
        Filter => dispatch!(a1, a2, collections::filter(vm, a1, a2)),
        FlatMap => dispatch!(a1, a2, collections::flat_map(vm, Some(a1), a2)),
        Concat => dispatch!(a1, collections::flat_map(vm, None, a1)),
        Zip => dispatch_varargs!(an, collections::zip(an)),
        Reduce => dispatch!(a1, a2, collections::reduce(vm, a1, a2)),
        Sort => dispatch_varargs!(an, collections::sort(an)),
        SortBy => dispatch!(a1, a2, collections::sort_by(vm, a1, a2)),
        GroupBy => dispatch!(a1, a2, collections::group_by(vm, a1, a2)),
        Reverse => dispatch_varargs!(an, collections::reverse(an)),
        Permutations => dispatch!(a1, a2, collections::permutations(a1, a2)),
        Combinations => dispatch!(a1, a2, collections::combinations(a1, a2)),
        Any => dispatch!(a1, a2, collections::any(vm, a1, a2)),
        All => dispatch!(a1, a2, collections::all(vm, a1, a2)),
        Memoize => dispatch!(a1, collections::create_memoized(a1)),
        SyntheticMemoizedFunction => dispatch_varargs!(a1, an, collections::invoke_memoized(vm, a1, an)),

        Peek => dispatch!(a1, collections::peek(a1)),
        Pop => dispatch!(a1, collections::pop(a1)),
        PopFront => dispatch!(a1, collections::pop_front(a1)),
        Push => dispatch!(a1, a2, collections::push(a1, a2)),
        PushFront => dispatch!(a1, a2, collections::push_front(a1, a2)),
        Insert => dispatch!(a1, a2, a3, collections::insert(a1, a2, a3)),
        Remove => dispatch!(a1, a2, collections::remove(a1, a2)),
        Clear => dispatch!(a1, collections::clear(a1)),
        Find => dispatch!(a1, a2, collections::left_find(vm, a1, a2, false)),
        RightFind => dispatch!(a1, a2, collections::right_find(vm, a1, a2, false)),
        IndexOf => dispatch!(a1, a2, collections::left_find(vm, a1, a2, true)),
        RightIndexOf => dispatch!(a1, a2, collections::right_find(vm, a1, a2, true)),
        Default => dispatch!(a1, a2, collections::dict_set_default(a1, a2)),
        Keys => dispatch!(a1, collections::dict_keys(a1)),
        Values => dispatch!(a1, collections::dict_values(a1)),

        // math
        Abs => dispatch!(a1, math::abs(a1)),
        Sqrt => dispatch!(a1, math::sqrt(a1)),
        Gcd => dispatch_varargs!(an, math::gcd(an)),
        Lcm => dispatch_varargs!(an, math::lcm(an)),
        CountOnes => dispatch!(a1, math::count_ones(a1)),
        CountZeros => dispatch!(a1, math::count_zeros(a1)),

        Function | Iterable | Complex => ValueIsNotFunctionEvaluable(Value::NativeFunction(native)).err()
    }
}


pub fn get_index<VM>(vm: &mut VM, target: Value, index: Value) -> ValueResult where VM : VirtualInterface {
    if target.is_dict() {
        return get_dict_index(vm, target, index);
    }

    let indexable = target.as_index()?;
    let index: usize = indexable.check_index(index)?;

    Ok(indexable.get_index(index))
}

fn get_dict_index<VM>(vm: &mut VM, dict: Value, key: Value) -> ValueResult where VM : VirtualInterface {
    // Dict objects have their own overload of indexing to mean key-value lookups, that doesn't fit with ValueAsIndex (as it doesn't take integer keys, always)
    // The handling for this is a bit convoluted due to `clone()` issues, and possible cases of default / no default / functional default

    // Initially unbox (non mutable) to clone out the default value.
    // If the default is a function, we can't have a reference out of the dict while we're accessing the default.

    let dict = match dict { Value::Dict(it) => it, _ => panic!() };
    let default_factory: Value;
    {
        let mut dict = dict.unbox_mut(); // mutable as we might insert in the immutable default case
        match dict.default.clone() {
            Some(default) if default.is_function() => match dict.dict.get(&key) {
                Some(existing_value) => return Ok(existing_value.clone()),
                None => {
                    // We need to insert, so fallthrough as we need to drop the borrow on `dict`
                    default_factory = default;
                },
            },
            Some(default_value) => {
                return Ok(dict.dict.entry(key).or_insert(default_value).clone())
            }
            None => return match dict.dict.get(&key) {
                Some(existing_value) => Ok(existing_value.clone()),
                None => ValueErrorKeyNotPresent(key).err()
            },
        }
    }

    // Invoke the new value supplier - this might modify the dict
    // We go through the `.entry()` API again in this case
    let new_value: Value = vm.invoke_func0(default_factory)?;
    let mut dict = dict.unbox_mut();
    Ok(dict.dict.entry(key).or_insert(new_value).clone())
}

pub fn set_index(target: &Value, index: Value, value: Value) -> Result<(), Box<RuntimeError>> {

    if let Value::Dict(it) = target {
        match vm::guard_recursive_hash(|| it.unbox_mut().dict.insert(index, value)) {
            Err(_) => ValueErrorRecursiveHash(target.clone()).err(),
            Ok(_) => Ok(())
        }
    } else {
        let mut indexable = target.as_index()?;
        let index: usize = indexable.check_index(index)?;

        indexable.set_index(index, value)
    }
}

pub fn format_string(string: &String, args: Value) -> ValueResult {
    strings::format_string(string, args)
}



/// Not unused - invoked via the dispatch!() macro above
fn wrap_as_partial<VM>(native: NativeFunction, nargs: u32, vm: &mut VM) -> Value where VM : VirtualInterface {
    if nargs == 0 {
        // Special case for 0-arg invoke, don't create a partial wrapper
        return Value::NativeFunction(native)
    }
    // vm stack will contain [..., arg1, arg2, ... argN]
    // popping in order will populate the vector with [argN, argN-1, ... arg1]
    let mut args: Vec<Value> = Vec::with_capacity(nargs as usize);
    for _ in 0..nargs {
        args.push(vm.pop().clone());
    }
    Value::PartialNativeFunction(native, Box::new(args))
}

fn type_of(value: Value) -> Value {
    // This function is here because we don't want `Value::{*}` to be imported, rather `NativeFunction::{*}` due to shadowing issues.
    match value {
        Value::Nil => Value::Nil,
        Value::Bool(_) => Bool.to_value(),
        Value::Int(_) => Int.to_value(),
        Value::Complex(_) => Complex.to_value(),
        Value::Str(_) => Str.to_value(),

        Value::List(_) => List.to_value(),
        Value::Set(_) => Set.to_value(),
        Value::Dict(_) => Dict.to_value(),
        Value::Heap(_) => Heap.to_value(),
        Value::Vector(_) => Vector.to_value(),

        Value::Struct(it) => Value::StructType(it.unbox().type_impl.clone()), // Structs return their type constructor
        Value::StructType(_) => Function.to_value(), // And the type constructor returns `function`

        Value::Range(_) => Range.to_value(),
        Value::Enumerate(_) => Enumerate.to_value(),
        Value::Slice(_) => Function.to_value(),

        x @ (Value::Iter(_) | Value::Memoized(_)) => panic!("{:?} is synthetic and cannot have type_of() called on it", x),

        Value::Function(_) | Value::PartialFunction(_) | Value::NativeFunction(_) | Value::PartialNativeFunction(_, _) | Value::Closure(_) | Value::GetField(_) => Function.to_value(),
    }
}


#[cfg(test)]
mod tests {
    use crate::{compiler, SourceView, core};
    use crate::vm::{IntoValue, Value, VirtualInterface, VirtualMachine};

    #[test]
    fn test_native_functions_are_declared_in_order() {
        for (i, info) in core::NATIVE_FUNCTIONS.iter().enumerate() {
            assert_eq!(info.native as usize, i, "Native function {} / {:?} declared in order {} != {}", info.name, info.native, i, info.native as usize)
        }
    }

    #[test]
    fn test_consistency_condition() {
        // Asserts that `nargs < native.nargs()` is a sufficient condition for declaring a function is consistent
        let mut buffer = Vec::new();
        let mut vm = VirtualMachine::new(compiler::default(), SourceView::new(String::new(), String::new()), &b""[..], &mut buffer, vec![]);

        for native in core::NATIVE_FUNCTIONS.iter() {
            match native.nargs {
                Some(n) => {
                    for nargs in 1..n {
                        // Prepare stack arguments
                        let args: Vec<Value> = (0..nargs).map(|arg| (arg as i64).to_value()).collect();
                        vm.push( native.native.to_value());
                        for arg in args.iter().rev().cloned() {
                            vm.push(arg);
                        }

                        let ret = core::invoke(native.native, nargs, &mut vm);

                        assert_eq!(ret.clone().ok(), Some(Value::PartialNativeFunction(native.native, Box::new(args))), "Native function {:?} with nargs={:?}, args={:?} is not consistent, returned result {:?} instead", native.name, native.nargs, nargs, ret);

                        vm.pop();
                    }
                },
                None => {},
            }
        }
    }
}
