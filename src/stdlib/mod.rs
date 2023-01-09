use std::collections::{BinaryHeap, VecDeque};
use std::fs;
use hashlink::{LinkedHashMap, LinkedHashSet};
use lazy_static::lazy_static;

use crate::vm::{operator, VirtualInterface};
use crate::vm::value::{Mut, Value};
use crate::vm::error::RuntimeError;
use crate::trace;

use StdBinding::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;

mod strings;
mod collections;
mod math;


/// Looks up a `StdBinding`'s string name
pub fn lookup_name(b: StdBinding) -> &'static str {
    NATIVE_BINDINGS.iter()
        .find(|info| info.binding == b)
        .unwrap()
        .name
}

/// Looks up a `StdBinding` by a name.
/// This is used in the parser to resolve a top-level name.
pub fn lookup_named_binding(name: &String) -> Option<StdBinding> {
    NATIVE_BINDINGS.iter()
        .find(|info| info.parent == None && info.name == name)
        .map(|info| info.binding)
}

/// Looks up a named child `StdBinding` via the parent and child name.
/// This is used in the parser to resolve `->` references.
pub fn lookup_named_sub_binding(parent: StdBinding, child: String) -> Option<StdBinding> {
    NATIVE_BINDINGS.iter()
        .find(|info| info.parent == Some(parent) && info.name == child)
        .map(|info| info.binding)
}



struct StdBindingInfo {
    name: &'static str,
    binding: StdBinding,
    parent: Option<StdBinding>,
}


/// The enum containing all bindings as they are represented at runtime.
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub enum StdBinding {
    Void, // A dummy binding that parents itself. Any children of this binding cannot be referenced in the parser

    Print,
    Read,
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

    // Native Operators
    OperatorUnarySub,
    OperatorUnaryLogicalNot,
    OperatorUnaryBitwiseNot,

    OperatorMul,
    OperatorDiv,
    OperatorPow,
    OperatorMod,
    OperatorIs,
    OperatorIn,
    OperatorAdd,
    OperatorSub, // Cannot be referenced as (- <expr>)
    OperatorLeftShift,
    OperatorRightShift,
    OperatorBitwiseAnd,
    OperatorBitwiseOr,
    OperatorBitwiseXor,
    OperatorLessThan,
    OperatorLessThanEqual,
    OperatorGreaterThan,
    OperatorGreaterThanEqual,
    OperatorEqual,
    OperatorNotEqual,

    // strings
    ToLower,
    ToUpper,
    Replace,
    Trim,
    Split,

    // collections
    Len,
    Range,
    Enumerate,
    Sum,
    Min,
    Max,
    Map,
    Filter,
    FlatMap,
    Concat, // Native optimized version of flatMap(fn(x) -> x)
    Zip,
    Reduce,
    Sorted,
    Reversed,
    Permutations,
    Combinations,

    Pop, // Remove last, or at index
    Push, // Inverse of `pop` (alias)
    Last, // Just last element
    Head, // Just first element
    Tail, // Compliment of `head`
    Init, // Compliment of `last`
    Default, // For a `Dict`, sets the default value
    Keys, // `Dict.keys` -> returns a set of all keys
    Values, // `Dict.values` -> returns a list of all values
    Find, // Find index of first occurrence matching a value. Value can be a function (which will be checked for equality), or a target value
    RightFind, // Find index of last occurrence matching a value
    FindCount, // Find the count of elements

    // math
    Abs,
    Sqrt,
    Gcd,
    Lcm,
}

impl StdBindingInfo {
    fn new(name: &'static str, binding: StdBinding, parent: Option<StdBinding>) -> StdBindingInfo {
        StdBindingInfo { name, binding, parent }
    }
}

lazy_static! {
    static ref NATIVE_BINDINGS: Vec<StdBindingInfo> = load_bindings();
}

fn load_bindings() -> Vec<StdBindingInfo> {
    macro_rules! of {
        ($b:expr, $name:expr) => { StdBindingInfo::new($name, $b, None) };
        ($p: expr, $b:expr, $name:expr) => { StdBindingInfo::new($name, $b, Some($p)) };
    }
    vec![
        of!(Void, ""),

        of!(Print, "print"),
        of!(Read, "read"),
        of!(ReadText, "read_text"),
        of!(WriteText, "write_text"),

        of!(Bool, "bool"),
        of!(Int, "int"),
        of!(Str, "str"),
        of!(List, "list"),
        of!(Set, "set"),
        of!(Dict, "dict"),
        of!(Heap, "heap"),
        of!(Vector, "vector"),
        of!(Function, "function"),

        of!(Void, OperatorUnarySub, "(-)"),
        of!(Void, OperatorUnaryLogicalNot, "(!)"),
        of!(Void, OperatorUnaryBitwiseNot, "(~)"),

        of!(Void, OperatorMul, "(*)"),
        of!(Void, OperatorDiv, "(/)"),
        of!(Void, OperatorPow, "(**)"),
        of!(Void, OperatorMod, "(%)"),
        of!(Void, OperatorIs, "(is)"),
        of!(Void, OperatorIn, "(in)"),
        of!(Void, OperatorAdd, "(+)"),
        of!(Void, OperatorSub, "(-)"), // Cannot be referenced as (- <expr>)
        of!(Void, OperatorLeftShift, "(<<)"),
        of!(Void, OperatorRightShift, "(>>)"),
        of!(Void, OperatorBitwiseAnd, "(&)"),
        of!(Void, OperatorBitwiseOr, "(|)"),
        of!(Void, OperatorBitwiseXor, "(^)"),
        of!(Void, OperatorLessThan, "(<)"),
        of!(Void, OperatorLessThanEqual, "(<=)"),
        of!(Void, OperatorGreaterThan, "(>)"),
        of!(Void, OperatorGreaterThanEqual, "(>=)"),
        of!(Void, OperatorEqual, "(==)"),
        of!(Void, OperatorNotEqual, "(!=)"),

        of!(Repr, "repr"),

        of!(ToLower, "to_lower"),
        of!(ToUpper, "to_upper"),
        of!(Replace, "replace"),
        of!(Trim, "trim"),
        of!(Split, "split"),

        of!(Sum, "sum"),
        of!(Max, "max"),
        of!(Min, "min"),

        of!(Len, "len"),
        of!(Range, "range"),
        of!(Enumerate, "enumerate"),
        of!(Map, "map"),
        of!(Filter, "filter"),
        of!(FlatMap, "flat_map"),
        of!(Concat, "concat"),
        of!(Zip, "zip"),
        of!(Reduce, "reduce"),
        of!(Sorted, "sorted"),
        of!(Reversed, "reversed"),
        of!(Permutations, "permutations"),
        of!(Combinations, "combinations"),

        of!(Pop, "pop"),
        of!(Push, "push"),
        of!(Last, "last"),
        of!(Head, "head"),
        of!(Tail, "tail"),
        of!(Init, "init"),
        of!(Default, "default"),
        of!(Keys, "keys"),
        of!(Values, "values"),
        of!(Find, "find"),
        of!(RightFind, "rfind"),
        of!(FindCount, "findn"),

        of!(Abs, "abs"),
        of!(Sqrt, "sqrt"),
        of!(Gcd, "gcd"),
        of!(Lcm, "lcm"),
    ]
}


/// `allow` usages here are due to macro expansion of the `dispatch!()` cases.
#[allow(unused_mut)]
#[allow(unused_variables)]
pub fn invoke<VM>(binding: StdBinding, nargs: u8, vm: &mut VM) -> ValueResult where VM : VirtualInterface
{
    trace::trace_interpreter!("stdlib::invoke() func={}, nargs={}", lookup_name(binding), nargs);

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
    // a1, <expr>, a2, <expr>, a3, <expr> : f(a1), f(a1, a2), f(a1, a2, a3)
    //
    // dispatch_varargs!() handles the following signatures:
    //
    // <expr> a1 <expr> an <expr>         : f(), f(a1), f(an, ...)
    //
    // an <expr>                          : f(an), f(an, ...)  *single argument is expanded as iterable
    // <expr> an <expr>                   : f(), f(an), f(an, ...)  * single argument is expanded as iterable
    macro_rules! dispatch {
        ($ret:expr) => { // f()
            match nargs {
                0 => {
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(binding, nargs, 0).err()
            }
        };
        ($a1:ident, $ret:expr) => { // f(a1)
            match nargs {
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(binding, nargs, 1).err()
            }
        };
        ($a1:ident, $a2:ident, $ret:expr) => { // f(a1, a2)
            match nargs {
                1 => Ok(wrap_as_partial(binding, nargs, vm)),
                2 => {
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(binding, nargs, 2).err()
            }
        };
        ($a1:ident, $a2:ident, $a3:ident, $ret:expr) => { // f(a1, a2, a3)
            match nargs {
                1 | 2 => Ok(wrap_as_partial(binding, nargs, vm)),
                3 => {
                    let $a3: Value = vm.pop();
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(binding, nargs, 3).err()
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
                _ => IncorrectNumberOfArguments(binding, nargs, 1).err()
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
                _ => IncorrectNumberOfArguments(binding, nargs, 1).err()
            }
        };
    }
    macro_rules! dispatch_varargs {
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
                    let mut $an = varargs.iter();
                    let ret = $ret_n;
                    ret
                }
            }
        };
        ($an:ident, $ret_n:expr) => { // f(an), f(an, ...)  *single argument is expanded as iterable
            match nargs {
                0 => IncorrectNumberOfArgumentsVariadicAtLeastOne(binding).err(),
                1 => match vm.pop().as_iter() {
                    Ok($an) => $ret_n,
                    Err(e) => Err(e),
                },
                _ => {
                    let varargs = vm.popn(nargs as usize);
                    let mut $an = varargs.iter();
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
                    let mut $an = varargs.iter();
                    $ret_n
                },
            }
        };
    }

    match binding {
        Print => {
            dispatch_varargs!(vm.println0(), a1, vm.println(a1.as_str()), an, {
                vm.print(an.next().unwrap().as_str());
                for ai in an {
                    vm.print(format!(" {}", ai.as_str()));
                }
                vm.println0()
            });
            Ok(Value::Nil)
        },
        Read => dispatch!({
            panic!("Not Implemented");
        }),
        ReadText => dispatch!(a1, match a1 {
            Value::Str(s1) => Ok(Value::Str(Box::new(fs::read_to_string(s1.as_ref()).unwrap().replace("\r", "")))), // todo: error handling?
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
        Int => dispatch!(a1, match &a1 {
            Value::Nil => Ok(Value::Int(0)),
            Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
            Value::Int(_) => Ok(a1),
            Value::Str(s) => match s.parse::<i64>() {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => TypeErrorCannotConvertToInt(a1).err(),
            },
            _ => TypeErrorCannotConvertToInt(a1).err(),
        }),
        Str => dispatch!(Ok(Value::Str(Box::new(String::new()))), a1, Ok(Value::Str(Box::new(a1.as_str())))),
        List => dispatch_varargs!(Ok(Value::list(VecDeque::new())), an, Ok(Value::iter_list(an.into_iter().cloned()))),
        Set => dispatch_varargs!(Ok(Value::set(LinkedHashSet::new())), an, Ok(Value::iter_set(an.into_iter().cloned()))),
        Dict => dispatch_varargs!(Ok(Value::dict(LinkedHashMap::new())), an, collections::collect_into_dict(an.into_iter().cloned())),
        Heap => dispatch_varargs!(Ok(Value::heap(BinaryHeap::new())), an, Ok(Value::iter_heap(an.into_iter().cloned()))),
        Vector => dispatch_varargs!(Ok(Value::vector(vec![])), an, Ok(Value::iter_vector(an.into_iter().cloned()))),
        Repr => dispatch!(a1, Ok(Value::Str(Box::new(a1.as_repr_str())))),

        // operator
        OperatorUnarySub => dispatch!(a1, operator::unary_sub(a1)),
        OperatorUnaryLogicalNot => dispatch!(a1, operator::unary_logical_not(a1)),
        OperatorUnaryBitwiseNot => dispatch!(a1, operator::unary_bitwise_not(a1)),
        OperatorMul => dispatch!(a1, a2, operator::binary_mul(a2, a1)),
        OperatorDiv => dispatch!(a1, a2, operator::binary_div(a2, a1)),
        OperatorPow => dispatch!(a1, a2, operator::binary_pow(a2, a1)),
        OperatorMod => dispatch!(a1, a2, operator::binary_mod(a2, a1)),
        OperatorIs => dispatch!(a1, a2, operator::binary_is(a2, a1)),
        OperatorIn => dispatch!(a1, a2, operator::binary_in(a2, a1)),
        OperatorAdd => dispatch!(a1, a2, operator::binary_add(a2, a1)),
        OperatorSub => dispatch!(a1, a2, operator::binary_sub(a2, a1)),
        OperatorLeftShift => dispatch!(a1, a2, operator::binary_left_shift(a2, a1)),
        OperatorRightShift => dispatch!(a1, a2, operator::binary_right_shift(a2, a1)),
        OperatorBitwiseAnd => dispatch!(a1, a2, operator::binary_bitwise_and(a2, a1)),
        OperatorBitwiseOr => dispatch!(a1, a2, operator::binary_bitwise_or(a2, a1)),
        OperatorBitwiseXor => dispatch!(a1, a2, operator::binary_bitwise_xor(a2, a1)),
        OperatorLessThan => dispatch!(a1, a2, Ok(operator::binary_less_than(a2, a1))),
        OperatorLessThanEqual => dispatch!(a1, a2, Ok(operator::binary_less_than_or_equal(a2, a1))),
        OperatorGreaterThan => dispatch!(a1, a2, Ok(operator::binary_greater_than(a2, a1))),
        OperatorGreaterThanEqual => dispatch!(a1, a2, Ok(operator::binary_greater_than_or_equal(a2, a1))),
        OperatorEqual => dispatch!(a1, a2, Ok(operator::binary_equals(a2, a1))),
        OperatorNotEqual => dispatch!(a1, a2, Ok(operator::binary_not_equals(a2, a1))),

        // lib_str
        ToLower => dispatch!(a1, strings::to_lower(a1)),
        ToUpper => dispatch!(a1, strings::to_upper(a1)),
        Replace => dispatch!(a1, a2, a3, strings::replace(a1, a2, a3)),
        Trim => dispatch!(a1, strings::trim(a1)),
        Split => dispatch!(a1, a2, strings::split(a1, a2)),

        // lib_list
        Len => dispatch!(a1, a1.len().map(|u| Value::Int(u as i64))),
        Range => dispatch!(a1, collections::range_1(a1), a2, collections::range_2(a1, a2), a3, collections::range_3(a1, a2, a3)),
        Enumerate => dispatch!(a1, collections::enumerate(a1)),
        Sum => dispatch_varargs!(an, collections::sum(an.into_iter())),
        Max => dispatch_varargs!(an, collections::max(an.into_iter())),
        Min => dispatch_varargs!(an, collections::min(an.into_iter())),
        Map => dispatch!(a1, a2, collections::map(vm, a1, a2)),
        Filter => dispatch!(a1, a2, collections::filter(vm, a1, a2)),
        FlatMap => dispatch!(a1, a2, collections::flat_map(vm, Some(a1), a2)),
        Concat => dispatch!(a1, collections::flat_map(vm, None, a1)),
        Zip => dispatch_varargs!(an, collections::zip(an.into_iter())),
        Reduce => dispatch!(a1, a2, collections::reduce(vm, a1, a2)),
        Sorted => dispatch_varargs!(an, collections::sorted(an.into_iter())),
        Reversed => dispatch_varargs!(an, collections::reversed(an.into_iter())),
        Permutations => dispatch!(a1, a2, collections::permutations(a1, a2)),
        Combinations => dispatch!(a1, a2, collections::combinations(a1, a2)),

        Pop => dispatch!(a1, collections::pop(a1)),
        Push => dispatch!(a1, a2, collections::push(a1, a2)),
        Last => dispatch!(a1, collections::last(a1)),
        Head => dispatch!(a1, collections::head(a1)),
        Init => dispatch!(a1, collections::init(a1)),
        Tail => dispatch!(a1, collections::tail(a1)),
        Default => dispatch!(a1, a2, collections::dict_set_default(a1, a2)),
        Keys => dispatch!(a1, collections::dict_keys(a1)),
        Values => dispatch!(a1, collections::dict_values(a1)),
        Find => dispatch!(a1, a2, collections::left_find(vm, a1, a2)),
        RightFind => dispatch!(a1, a2, collections::right_find(vm, a1, a2)),
        FindCount => dispatch!(a1, a2, collections::find_count(vm, a1, a2)),

        // lib_math
        Abs => dispatch!(a1, math::abs(a1)),
        Sqrt => dispatch!(a1, math::sqrt(a1)),
        Gcd => dispatch_varargs!(an, math::gcd(an.into_iter())),
        Lcm => dispatch_varargs!(an, math::lcm(an.into_iter())),

        _ => ValueIsNotFunctionEvaluable(Value::NativeFunction(binding)).err(),
    }
}


pub fn get_index(a1: &Value, a2: &Value) -> ValueResult {

    // Dict objects have their own overload of indexing to mean key-value lookups, that doesn't fit with ValueAsIndex (as it doesn't take integer keys, always)
    if let Value::Dict(it) = a1 {
        let it = it.unbox();
        return if let Some(v) = it.dict.get(&a2).or(it.default.as_ref()) {
            Ok(v.clone())
        } else {
            ValueErrorKeyNotPresent(a2.clone()).err()
        }
    }

    let indexable = a1.to_index()?;
    let index: usize = collections::get_checked_index(a1.len()?, a2.as_int()?)?;

    Ok(indexable.get_index(index))
}

pub fn get_slice(a1: Value, a2: Value, a3: Value, a4: Value) -> ValueResult {
    collections::list_slice(a1, a2, a3, a4)
}

pub fn list_set_index(list_ref: Mut<VecDeque<Value>>, index: i64, value: Value) -> Result<(), Box<RuntimeError>> {
    collections::list_set_index(list_ref, index, value)
}



/// Not unused - invoked via the dispatch!() macro above
fn wrap_as_partial<VM>(binding: StdBinding, nargs: u8, vm: &mut VM) -> Value where VM : VirtualInterface {
    // vm stack will contain [..., arg1, arg2, ... argN]
    // popping in order will populate the vector with [argN, argN-1, ... arg1]
    let mut args: Vec<Box<Value>> = Vec::with_capacity(nargs as usize);
    for _ in 0..nargs {
        args.push(Box::new(vm.pop().clone()));
    }
    Value::PartialNativeFunction(binding, Box::new(args))
}
