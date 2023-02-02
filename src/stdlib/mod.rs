use std::collections::{BinaryHeap, VecDeque};
use std::fs;
use indexmap::{IndexMap, IndexSet};
use lazy_static::lazy_static;

use crate::vm::{operator, IntoIterableValue, IntoValue, Value, VirtualInterface, RuntimeError};
use crate::trace;

use NativeFunction::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;

mod strings;
mod collections;
mod math;


/// Looks up a global native function, or reserved keyword, by name.
pub fn find_native_function(name: &String) -> Option<NativeFunction> {
    NATIVE_FUNCTIONS.iter()
        .find(|info| info.name == name && !info.hidden)
        .map(|info| info.native)
}


#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub enum NativeFunction {
    Print,
    ReadText,
    WriteText,
    Bool,
    Int,
    Str,
    List,
    Set,
    Dict,
    Heap,
    Vector,
    Function,
    Repr,
    Eval,

    // Native Operators
    OperatorUnarySub,
    OperatorUnaryLogicalNot,
    OperatorUnaryBitwiseNot,

    OperatorMul,
    OperatorDiv,
    OperatorDivSwap,
    OperatorPow,
    OperatorPowSwap,
    OperatorMod,
    OperatorModSwap,
    OperatorIs,
    OperatorIsSwap,
    OperatorIn,
    OperatorInSwap,
    OperatorAdd,
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
    Trim,
    Split,
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
    Reverse,
    Permutations,
    Combinations,
    Any,
    All,
    Memoize,
    SyntheticMemoizedFunction,

    Peek,
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
}

impl NativeFunction {
    pub fn nargs(self: &Self) -> Option<u8> {
        NATIVE_FUNCTIONS[*self as usize].nargs
    }

    pub fn name(self: &Self) -> &'static str {
        NATIVE_FUNCTIONS[*self as usize].name
    }

    pub fn args(self: &Self) -> &'static str {
        NATIVE_FUNCTIONS[*self as usize].args
    }
}


struct NativeFunctionInfo {
    native: NativeFunction,
    name: &'static str,
    args: &'static str,
    nargs: Option<u8>,
    hidden: bool,
}

impl NativeFunctionInfo {
    fn new(native: NativeFunction, name: &'static str, args: &'static str, nargs: Option<u8>, hidden: bool) -> NativeFunctionInfo {
        NativeFunctionInfo { native, name, args, nargs, hidden }
    }
}

lazy_static! {
    static ref NATIVE_FUNCTIONS: Vec<NativeFunctionInfo> = load_native_functions();
}


fn load_native_functions() -> Vec<NativeFunctionInfo> {
    let mut natives: Vec<NativeFunctionInfo> = Vec::new();
    macro_rules! declare {
        ($native:expr, $name:expr, $args:expr) => {
            declare!($native, $name, $args, None, false);
        };
        ($native:expr, $name:expr, $args:expr, $nargs:expr) => {
            declare!($native, $name, $args, Some($nargs), false);
        };
        ($native:expr, $name:expr, $args:expr, $nargs:expr, $hidden:expr) => {
            let info = NativeFunctionInfo::new($native, $name, $args, $nargs, $hidden);
            assert_eq!(natives.len(), $native as usize, "Native Function {:?} (index = {}) declared in order {}", $native, $native as usize, natives.len());
            natives.push(info);
        };
    }

    // core
    declare!(Print, "print", "...");
    declare!(ReadText, "read_text", "file", 1);
    declare!(WriteText, "write_text", "file, text", 2);
    declare!(Bool, "bool", "x", 1);
    declare!(Int, "int", "x", 1);
    declare!(Str, "str", "x", 1);
    declare!(List, "list", "...");
    declare!(Set, "set", "...");
    declare!(Dict, "dict", "...");
    declare!(Heap, "heap", "...");
    declare!(Vector, "vector", "...");
    declare!(Function, "function", "", None, true);
    declare!(Repr, "repr", "x", 1);
    declare!(Eval, "eval", "expr", 1);

    // operator
    declare!(OperatorUnarySub, "(-)", "x", Some(1), true);
    declare!(OperatorUnaryLogicalNot, "(!)", "x", Some(1), true);
    declare!(OperatorUnaryBitwiseNot, "(~)", "x", Some(1), true);

    declare!(OperatorMul, "(*)", "lhs, rhs", Some(2), true);
    declare!(OperatorDiv, "(/)", "lhs, rhs", Some(2), true);
    declare!(OperatorDivSwap, "(/)", "lhs, rhs", Some(2), true);
    declare!(OperatorPow, "(**)", "lhs, rhs", Some(2), true);
    declare!(OperatorPowSwap, "(**)", "lhs, rhs", Some(2), true);
    declare!(OperatorMod, "(%)", "lhs, rhs", Some(2), true);
    declare!(OperatorModSwap, "(%)", "lhs, rhs", Some(2), true);
    declare!(OperatorIs, "(is)", "lhs, rhs", Some(2), true);
    declare!(OperatorIsSwap, "(is)", "lhs, rhs", Some(2), true);
    declare!(OperatorIn, "(in)", "lhs, rhs", Some(2), true);
    declare!(OperatorInSwap, "(in)", "lhs, rhs", Some(2), true);
    declare!(OperatorAdd, "(+)", "lhs, rhs", Some(2), true);
    declare!(OperatorSub, "(-)", "lhs, rhs", Some(2), true); // Cannot be referenced as (- <expr>)
    declare!(OperatorLeftShift, "(<<)", "lhs, rhs", Some(2), true);
    declare!(OperatorLeftShiftSwap, "(<<)", "lhs, rhs", Some(2), true);
    declare!(OperatorRightShift, "(>>)", "lhs, rhs", Some(2), true);
    declare!(OperatorRightShiftSwap, "(>>)", "lhs, rhs", Some(2), true);
    declare!(OperatorBitwiseAnd, "(&)", "lhs, rhs", Some(2), true);
    declare!(OperatorBitwiseOr, "(|)", "lhs, rhs", Some(2), true);
    declare!(OperatorBitwiseXor, "(^)", "lhs, rhs", Some(2), true);
    declare!(OperatorLessThan, "(<)", "lhs, rhs", Some(2), true);
    declare!(OperatorLessThanSwap, "(<)", "lhs, rhs", Some(2), true);
    declare!(OperatorLessThanEqual, "(<=)", "lhs, rhs", Some(2), true);
    declare!(OperatorLessThanEqualSwap, "(<=)", "lhs, rhs", Some(2), true);
    declare!(OperatorGreaterThan, "(>)", "lhs, rhs", Some(2), true);
    declare!(OperatorGreaterThanSwap, "(>)", "lhs, rhs", Some(2), true);
    declare!(OperatorGreaterThanEqual, "(>=)", "lhs, rhs", Some(2), true);
    declare!(OperatorGreaterThanEqualSwap, "(>=)", "lhs, rhs", Some(2), true);
    declare!(OperatorEqual, "(==)", "lhs, rhs", Some(2), true);
    declare!(OperatorNotEqual, "(!=)", "lhs, rhs", Some(2), true);

    // strings
    declare!(ToLower, "to_lower", "x", 1);
    declare!(ToUpper, "to_upper", "x", 1);
    declare!(Replace, "replace", "search, with, target", 3);
    declare!(Trim, "trim", "x", 1);
    declare!(Split, "split", "delim, x", 2);
    declare!(Char, "char", "x", 1);
    declare!(Ord, "ord", "x", 1);
    declare!(Hex, "hex", "x", 1);
    declare!(Bin, "bin", "x", 1);

    declare!(Len, "len", "x", 1);
    declare!(Range, "range", "start, stop, step", None, false);
    declare!(Enumerate, "enumerate", "iter", 1);
    declare!(Sum, "sum", "...");
    declare!(Min, "min", "...");
    declare!(Max, "max", "...");
    declare!(MinBy, "min_by", "key_or_cmp, iter", 2);
    declare!(MaxBy, "max_by", "key_or_cmp, iter", 2);
    declare!(Map, "map", "f, iter", 2);
    declare!(Filter, "filter", "f, iter", 2);
    declare!(FlatMap, "flat_map", "f, iter", 2);
    declare!(Concat, "concat", "iter", 1);
    declare!(Zip, "zip", "...");
    declare!(Reduce, "reduce", "f, iter", 2);
    declare!(Sort, "sort", "...");
    declare!(SortBy, "sort_by", "f, iter", 2);
    declare!(Reverse, "reverse", "iter", 1);
    declare!(Permutations, "permutations", "n, iter", 2);
    declare!(Combinations, "combinations", "n, iter", 2);
    declare!(Any, "any", "...");
    declare!(All, "all", "...");
    declare!(Memoize, "memoize", "f", 1);
    declare!(SyntheticMemoizedFunction, "<synthetic memoized>", "f, ...", None, true);

    declare!(Peek, "peek", "collection", 1); // Peek first value
    declare!(Pop, "pop", "collection", 1); // Remove value at end
    declare!(PopFront, "pop_front", "collection", 1); // Remove value at front
    declare!(Push, "push", "value, collection", 2); // Insert value at end
    declare!(PushFront, "push_front", "value, collection", 2); // Insert value at front
    declare!(Insert, "insert", "index, value, collection", 3); // Insert value at index
    declare!(Remove, "remove", "param, collection", 2); // Remove (list: by index, set: by value, dict: by key)
    declare!(Clear, "clear", "collection", 1); // Remove all values - shortcut for `retain(fn(_) -> false)`
    declare!(Find, "find", "predicate, collection", 2); // Find first value (list, set) or key (dict) by predicate
    declare!(RightFind, "rfind", "predicate, collection", 2); // Find last index of value (list, set), or key (dict) by predicate
    declare!(IndexOf, "index_of", "value_or_predicate, collection", 2); // Find first index of value, or index by predicate
    declare!(RightIndexOf, "rindex_of", "value_or_predicate, collection", 2); // Find last index of a value, or index by predicate
    declare!(Default, "default", "value, dictionary", 2); // For a `Dict`, sets the default value
    declare!(Keys, "keys", "dictionary", 1); // `Dict.keys` -> returns a set of all keys
    declare!(Values, "values", "dictionary", 1); // `Dict.values` -> returns a list of all values

    // math
    declare!(Abs, "abs", "x", 1);
    declare!(Sqrt, "sqrt", "x", 1);
    declare!(Gcd, "gcd", "...");
    declare!(Lcm, "lcm", "...");

    natives
}



/// `allow` usages here are due to macro expansion of the `dispatch!()` cases.
#[allow(unused_mut)]
#[allow(unused_variables)]
pub fn invoke<VM>(native: NativeFunction, nargs: u8, vm: &mut VM) -> ValueResult where VM : VirtualInterface
{
    trace::trace_interpreter!("stdlib::invoke() func={}, nargs={}", native.name(), nargs);

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
                _ => IncorrectNumberOfArguments(native, nargs, 0).err()
            }
        };
        ($a1:ident, $ret:expr) => { // f(a1)
            match nargs {
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(native, nargs, 1).err()
            }
        };
        ($a1:ident, $a2:ident, $ret:expr) => { // f(a1, a2)
            match nargs {
                1 => Ok(wrap_as_partial(native, nargs, vm)),
                2 => {
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(native, nargs, 2).err()
            }
        };
        ($a1:ident, $a2:ident, $a3:ident, $ret:expr) => { // f(a1, a2, a3)
            match nargs {
                1 | 2 => Ok(wrap_as_partial(native, nargs, vm)),
                3 => {
                    let $a3: Value = vm.pop();
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(native, nargs, 3).err()
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
                _ => IncorrectNumberOfArguments(native, nargs, 1).err()
            }
        };
        ($a1:ident, $ret1:expr, $a2:ident, $ret2:expr) => { // f(a1), f(a1, a2)
            match nargs {
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
                _ => IncorrectNumberOfArguments(native, nargs, 1).err()
            }
        };
        ($a1:ident, $ret1:expr, $a2:ident, $ret2:expr, $a3:ident, $ret3:expr) => { // f(a1), f(a1, a2), f(a1, a2, a3)
            match nargs {
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
                _ => IncorrectNumberOfArguments(native, nargs, 1).err()
            }
        };
    }
    macro_rules! dispatch_varargs {
        ($a1:ident, $an:ident, $ret:expr) => {
            match nargs {
                0 => panic!("Illegal invoke of synthetic method with no primary argument"),
                _ => {
                    let $an = vm.popn(nargs as usize - 1);
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
                    let varargs = vm.popn(nargs as usize);
                    let mut $an = varargs.iter().cloned();
                    let ret = $ret_n;
                    ret
                }
            }
        };
        ($an:ident, $ret_n:expr) => { // f(an), f(an, ...)  *single argument is expanded as iterable
            match nargs {
                0 => IncorrectNumberOfArgumentsVariadicAtLeastOne(native).err(),
                1 => match vm.pop().as_iter() {
                    Ok($an) => $ret_n,
                    Err(e) => Err(e),
                },
                _ => {
                    let varargs = vm.popn(nargs as usize);
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
                    let varargs = vm.popn(nargs as usize);
                    let mut $an = varargs.iter().cloned();
                    $ret_n
                },
            }
        };
    }

    match native {
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
        WriteText => dispatch!(a1, a2, match (&a1, &a2) {
            (Value::Str(s1), Value::Str(s2)) => {
                fs::write(s1.as_ref(), s2.as_ref()).unwrap();
                Ok(Value::Nil)
            },
            (l, r) => {
                if !l.is_str() {
                    return TypeErrorArgMustBeStr(l.clone()).err()
                } else {
                    return TypeErrorArgMustBeStr(r.clone()).err()
                }
            },
        }),
        Bool => dispatch!(a1, Ok(Value::Bool(a1.as_bool()))),
        Int => dispatch!(a1, math::convert_to_int(a1, None), a2, math::convert_to_int(a1, Some(a2))),
        Str => dispatch!(Ok("".to_value()), a1, Ok(a1.to_str().to_value())),
        List => dispatch_varargs!(Ok(VecDeque::new().to_value()), an, Ok(an.to_list())),
        Set => dispatch_varargs!(Ok(IndexSet::new().to_value()), an, Ok(an.to_set())),
        Dict => dispatch_varargs!(Ok(IndexMap::new().to_value()), an, collections::collect_into_dict(an)),
        Heap => dispatch_varargs!(Ok(BinaryHeap::new().to_value()), an, Ok(an.to_heap())),
        Vector => dispatch_varargs!(Ok(vec![].to_value()), an, Ok(an.to_vector())),
        Repr => dispatch!(a1, Ok(a1.to_repr_str().to_value())),
        Eval => dispatch!(a1, vm.invoke_eval(a1.as_str()?)),

        // operator
        OperatorUnarySub => dispatch!(a1, operator::unary_sub(a1)),
        OperatorUnaryLogicalNot => dispatch!(a1, operator::unary_logical_not(a1)),
        OperatorUnaryBitwiseNot => dispatch!(a1, operator::unary_bitwise_not(a1)),
        OperatorMul => dispatch!(a1, a2, operator::binary_mul(a1, a2)),
        OperatorDiv => dispatch!(a1, a2, operator::binary_div(a1, a2)),
        OperatorDivSwap => dispatch!(a1, a2, operator::binary_div(a2, a1)),
        OperatorPow => dispatch!(a1, a2, operator::binary_pow(a1, a2)),
        OperatorPowSwap => dispatch!(a1, a2, operator::binary_pow(a2, a1)),
        OperatorMod => dispatch!(a1, a2, operator::binary_mod(a1, a2)),
        OperatorModSwap => dispatch!(a1, a2, operator::binary_mod(a2, a1)),
        OperatorIs => dispatch!(a1, a2, operator::binary_is(a1, a2)),
        OperatorIsSwap => dispatch!(a1, a2, operator::binary_is(a2, a1)),
        OperatorIn => dispatch!(a1, a2, operator::binary_in(a1, a2)),
        OperatorInSwap => dispatch!(a1, a2, operator::binary_in(a2, a1)),
        OperatorAdd => dispatch!(a1, a2, operator::binary_add(a1, a2)),
        OperatorSub => dispatch!(a1, a2, operator::binary_sub(a1, a2)),
        OperatorLeftShift => dispatch!(a1, a2, operator::binary_left_shift(a1, a2)),
        OperatorLeftShiftSwap => dispatch!(a1, a2, operator::binary_left_shift(a2, a1)),
        OperatorRightShift => dispatch!(a1, a2, operator::binary_right_shift(a1, a2)),
        OperatorRightShiftSwap => dispatch!(a1, a2, operator::binary_right_shift(a2, a1)),
        OperatorBitwiseAnd => dispatch!(a1, a2, operator::binary_bitwise_and(a1, a2)),
        OperatorBitwiseOr => dispatch!(a1, a2, operator::binary_bitwise_or(a1, a2)),
        OperatorBitwiseXor => dispatch!(a1, a2, operator::binary_bitwise_xor(a1, a2)),
        OperatorLessThan => dispatch!(a1, a2, Ok(operator::binary_less_than(a1, a2))),
        OperatorLessThanSwap => dispatch!(a1, a2, Ok(operator::binary_less_than(a2, a1))),
        OperatorLessThanEqual => dispatch!(a1, a2, Ok(operator::binary_less_than_or_equal(a1, a2))),
        OperatorLessThanEqualSwap => dispatch!(a1, a2, Ok(operator::binary_less_than_or_equal(a2, a1))),
        OperatorGreaterThan => dispatch!(a1, a2, Ok(operator::binary_greater_than(a1, a2))),
        OperatorGreaterThanSwap => dispatch!(a1, a2, Ok(operator::binary_greater_than(a2, a1))),
        OperatorGreaterThanEqual => dispatch!(a1, a2, Ok(operator::binary_greater_than_or_equal(a1, a2))),
        OperatorGreaterThanEqualSwap => dispatch!(a1, a2, Ok(operator::binary_greater_than_or_equal(a2, a1))),
        OperatorEqual => dispatch!(a1, a2, Ok(operator::binary_equals(a1, a2))),
        OperatorNotEqual => dispatch!(a1, a2, Ok(operator::binary_not_equals(a1, a2))),

        // strings
        ToLower => dispatch!(a1, strings::to_lower(a1)),
        ToUpper => dispatch!(a1, strings::to_upper(a1)),
        Replace => dispatch!(a1, a2, a3, strings::replace(a1, a2, a3)),
        Trim => dispatch!(a1, strings::trim(a1)),
        Split => dispatch!(a1, a2, strings::split(a1, a2)),
        Char => dispatch!(a1, strings::to_char(a1)),
        Ord => dispatch!(a1, strings::to_ord(a1)),
        Hex => dispatch!(a1, strings::to_hex(a1)),
        Bin => dispatch!(a1, strings::to_bin(a1)),

        // collections
        Len => dispatch!(a1, a1.len().map(|u| Value::Int(u as i64))),
        Range => dispatch!(a1, Value::range(0, a1.as_int()?, 1), a2, Value::range(a1.as_int()?, a2.as_int()?, 1), a3, Value::range(a1.as_int()?, a2.as_int()?, a3.as_int()?)),
        Enumerate => dispatch!(a1, Ok(Value::Enumerate(Box::new(a1)))),
        Sum => dispatch_varargs!(an, collections::sum(an)),
        Min => dispatch_varargs!(an, collections::min(an)),
        MinBy => dispatch!(a1, a2, collections::min_by(vm, a1, a2)),
        Max => dispatch_varargs!(an, collections::max(an)),
        MaxBy => dispatch!(a1, a2, collections::max_by(vm, a1, a2)),
        Map => dispatch!(a1, a2, collections::map(vm, a1, a2)),
        Filter => dispatch!(a1, a2, collections::filter(vm, a1, a2)),
        FlatMap => dispatch!(a1, a2, collections::flat_map(vm, Some(a1), a2)),
        Concat => dispatch!(a1, collections::flat_map(vm, None, a1)),
        Zip => dispatch_varargs!(an, collections::zip(an)),
        Reduce => dispatch!(a1, a2, collections::reduce(vm, a1, a2)),
        Sort => dispatch_varargs!(an, collections::sort(an)),
        SortBy => dispatch!(a1, a2, collections::sort_by(vm, a1, a2)),
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

        _ => ValueIsNotFunctionEvaluable(Value::NativeFunction(native)).err(),
    }
}


pub fn get_index(a1: &Value, a2: &Value) -> ValueResult {

    // Dict objects have their own overload of indexing to mean key-value lookups, that doesn't fit with ValueAsIndex (as it doesn't take integer keys, always)
    if let Value::Dict(it) = a1 {
        let it = it.unbox();
        return if let Some(v) = it.dict.get(a2).or(it.default.as_ref()) {
            Ok(v.clone())
        } else {
            ValueErrorKeyNotPresent(a2.clone()).err()
        }
    }

    let indexable = a1.as_index()?;
    let index: usize = collections::get_checked_index(indexable.len(), a2.as_int()?)?;

    Ok(indexable.get_index(index))
}

pub fn get_slice(a1: Value, a2: Value, a3: Value, a4: Value) -> ValueResult {
    collections::list_slice(a1, a2, a3, a4)
}

pub fn set_index(a1: &Value, a2: Value, a3: Value) -> Result<(), Box<RuntimeError>> {

    if let Value::Dict(it) = a1 {
        it.unbox_mut().dict.insert(a2, a3);
        Ok(())
    } else {
        let mut indexable = a1.as_index()?;
        let index: usize = collections::get_checked_index(indexable.len(), a2.as_int()?)?;

        indexable.set_index(index, a3)
    }
}

pub fn format_string(string: &String, args: Value) -> ValueResult {
    strings::format_string(string, args)
}



/// Not unused - invoked via the dispatch!() macro above
fn wrap_as_partial<VM>(native: NativeFunction, nargs: u8, vm: &mut VM) -> Value where VM : VirtualInterface {
    // vm stack will contain [..., arg1, arg2, ... argN]
    // popping in order will populate the vector with [argN, argN-1, ... arg1]
    let mut args: Vec<Value> = Vec::with_capacity(nargs as usize);
    for _ in 0..nargs {
        args.push(vm.pop().clone());
    }
    Value::PartialNativeFunction(native, Box::new(args))
}
