use std::collections::VecDeque;
use std::fs;
use lazy_static::lazy_static;

use crate::vm::{operator, VirtualInterface};
use crate::vm::value::{Mut, Value};
use crate::vm::error::RuntimeError;
use crate::trace;

use StdBinding::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;

mod lib_str;
mod lib_list;
mod lib_math;


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
    Nil,
    Bool,
    Int,
    Str,
    List,
    Set,
    Dict,
    Function,
    Repr,
    Len,

    // Native Operators
    OperatorUnarySub,
    OperatorUnaryLogicalNot,
    OperatorUnaryBitwiseNot,

    OperatorMul,
    OperatorDiv,
    OperatorPow,
    OperatorMod,
    OperatorIs,
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

    // lib_str
    ToLower,
    ToUpper,
    Replace,
    Trim,
    IndexOf,
    CountOf,
    Split,

    // lib_list
    Sum,
    Min,
    Max,
    Map,
    Filter,
    Reduce,
    Sorted,
    Reversed,

    Pop, // Remove last, or at index
    Push, // Inverse of `pop` (alias)
    Last, // Just last element
    Head, // Just first element
    Tail, // Compliment of `head`
    Init, // Compliment of `last`
    Insert, // Insert by key (or index)
    Remove, // Remove by key (or index)
    Merge, // Merges two collections (append right onto left)

    // lib_math
    Abs,
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

        of!(Nil, "nil"),
        of!(Bool, "bool"),
        of!(Int, "int"),
        of!(Str, "str"),
        of!(List, "list"),
        of!(Set, "set"),
        of!(Dict, "dict"),
        of!(Function, "function"),

        of!(Void, OperatorUnarySub, "(-)"),
        of!(Void, OperatorUnaryLogicalNot, "(!)"),
        of!(Void, OperatorUnaryBitwiseNot, "(~)"),

        of!(Void, OperatorMul, "(*)"),
        of!(Void, OperatorDiv, "(/)"),
        of!(Void, OperatorPow, "(**)"),
        of!(Void, OperatorMod, "(%)"),
        of!(Void, OperatorIs, "(is)"),
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
        of!(IndexOf, "index_of"),
        of!(CountOf, "count_of"),
        of!(Split, "split"),

        of!(Sum, "sum"),
        of!(Max, "max"),
        of!(Min, "min"),

        of!(Len, "len"),
        of!(Map, "map"),
        of!(Filter, "filter"),
        of!(Reduce, "reduce"),
        of!(Sorted, "sorted"),
        of!(Reversed, "reversed"),

        of!(Pop, "pop"),
        of!(Push, "push"),
        of!(Last, "last"),
        of!(Head, "head"),
        of!(Tail, "tail"),
        of!(Init, "init"),

        of!(Abs, "abs"),
        of!(Gcd, "gcd"),
        of!(Lcm, "lcm"),
    ]
}


/// `allow` usages here are due to macro expansion of the `dispatch!()` cases.
#[allow(unused_mut)]
#[allow(unused_variables)]
pub fn invoke<VM>(bound: StdBinding, nargs: u8, vm: &mut VM) -> ValueResult where VM : VirtualInterface
{
    trace::trace_interpreter!("stdlib::invoke() func={}, nargs={}", lookup_name(&bound), nargs);

    // Dispatch macros for 0, 1, 2 and 3 argument functions
    // All dispatch!() cases support partial evaluation (where 0 < nargs < required args)
    // This includes variants for default argument functions
    macro_rules! dispatch {
        ($ret:expr) => { // f() -> T
            match nargs {
                0 => {
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(bound.clone(), nargs, 0).err()
            }
        };
        ($a1:ident, $ret:expr) => { // f(T1) -> T
            match nargs {
                1 => {
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(bound.clone(), nargs, 1).err()
            }
        };
        ($a1:ident, $a2:ident, $ret:expr) => { // f(T1, T2) -> T
            match nargs {
                1 => Ok(wrap_as_partial(bound, nargs, vm)),
                2 => {
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(bound.clone(), nargs, 2).err()
            }
        };
        ($a1:ident, $a2:ident, $a3:ident, $ret:expr) => { // f(T1, T2, T3) -> T
            match nargs {
                1 | 2 => Ok(wrap_as_partial(bound, nargs, vm)),
                3 => {
                    let $a3: Value = vm.pop();
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    let ret = $ret;
                    ret
                },
                _ => IncorrectNumberOfArguments(bound.clone(), nargs, 3).err()
            }
        };
        ($ret0:expr, $a1:ident, $ret1:expr) => { // f(), f(T1) -> T
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
                _ => IncorrectNumberOfArguments(bound.clone(), nargs, 1).err()
            }
        };
        ($a1:ident, $ret1:expr, $a2:ident, $ret2:expr) => { // f(T1), f(T1, T2) -> T
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
                _ => IncorrectNumberOfArguments(bound.clone(), nargs, 1).err()
            }
        };
        ($a_var:ident, $ret_var:expr, $op:expr) => { // f(...) -> T
            match nargs {
                0 => IncorrectNumberOfArgumentsVariadicAtLeastOne($op).err(),
                1 => match vm.pop().as_iter() {
                    Ok($a_var) => $ret_var,
                    Err(e) => Err(e),
                },
                _ => {
                    let varargs = vm.popn(nargs as usize);
                    let mut $a_var = varargs.iter();
                    $ret_var
                },
            }
        };
        ($ret0:expr, $a1:ident, $ret1:expr, $a_var:ident, $ret_var:expr) => { // f(), f(T1), f(...) -> T
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
                    let mut $a_var = varargs.iter();
                    let ret = $ret_var;
                    ret
                }
            }
        };
    }

    match bound {
        Print => {
            dispatch!(vm.println0(), a1, vm.println(a1.as_str()), an, {
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
        Str => dispatch!(a1, Ok(Value::Str(Box::new(a1.as_str())))),
        List => dispatch!(a1, match a1.as_iter() {
            Ok(it) => Ok(Value::iter_list(it.into_iter().cloned())),
            Err(e) => Err(e),
        }),
        Set => dispatch!(a1, match a1.as_iter() {
            Ok(it) => Ok(Value::iter_set(it.into_iter().cloned())),
            Err(e) => Err(e),
        }),
        Repr => dispatch!(a1, Ok(Value::Str(Box::new(a1.as_repr_str())))),
        Len => dispatch!(a1, a1.len().map(|u| Value::Int(u as i64))),

        // operator
        OperatorUnarySub => dispatch!(a1, operator::unary_sub(a1)),
        OperatorUnaryLogicalNot => dispatch!(a1, operator::unary_logical_not(a1)),
        OperatorUnaryBitwiseNot => dispatch!(a1, operator::unary_bitwise_not(a1)),
        OperatorMul => dispatch!(a1, a2, operator::binary_mul(a2, a1)),
        OperatorDiv => dispatch!(a1, a2, operator::binary_div(a2, a1)),
        OperatorPow => dispatch!(a1, a2, operator::binary_pow(a2, a1)),
        OperatorMod => dispatch!(a1, a2, operator::binary_mod(a2, a1)),
        OperatorIs => dispatch!(a1, a2, operator::binary_is(a2, a1)),
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
        ToLower => dispatch!(a1, lib_str::to_lower(a1)),
        ToUpper => dispatch!(a1, lib_str::to_upper(a1)),
        Replace => dispatch!(a1, a2, a3, lib_str::replace(a1, a2, a3)),
        Trim => dispatch!(a1, lib_str::trim(a1)),
        IndexOf => dispatch!(a1, a2, lib_str::index_of(a1, a2)),
        CountOf => dispatch!(a1, a2, lib_str::count_of(a1, a2)),
        Split => dispatch!(a1, a2, lib_str::split(a1, a2)),

        // lib_list
        Sum => dispatch!(an, lib_list::sum(an.into_iter()), Sum),
        Max => dispatch!(an, lib_list::max(an.into_iter()), Max),
        Min => dispatch!(an, lib_list::min(an.into_iter()), Min),
        Map => dispatch!(a1, a2, lib_list::map(vm, a1, a2)),
        Filter => dispatch!(a1, a2, lib_list::filter(vm, a1, a2)),
        Reduce => dispatch!(a1, a2, lib_list::reduce(vm, a1, a2)),
        Sorted => dispatch!(an, lib_list::sorted(an.into_iter()), Sorted),
        Reversed => dispatch!(an, lib_list::reversed(an.into_iter()), Reversed),

        Pop => dispatch!(a1, lib_list::pop(a1)),
        Push => dispatch!(a1, a2, lib_list::push(a1, a2)),
        Last => dispatch!(a1, lib_list::last(a1)),
        Head => dispatch!(a1, lib_list::head(a1)),
        Init => dispatch!(a1, lib_list::init(a1)),
        Tail => dispatch!(a1, lib_list::tail(a1)),

        // lib_math
        Abs => dispatch!(a1, lib_math::abs(a1)),

        _ => BindingIsNotFunctionEvaluable(bound.clone()).err(),
    }
}

pub fn list_index(list_ref: Mut<VecDeque<Value>>, r: i64) -> ValueResult {
    lib_list::list_index(list_ref, r)
}

pub fn list_slice(a1: Value, a2: Value, a3: Value, a4: Value) -> ValueResult {
    lib_list::list_slice(a1, a2, a3, a4)
}


/// Not unused - invoked via the dispatch!() macro above
fn wrap_as_partial<VM>(bound: StdBinding, nargs: u8, vm: &mut VM) -> Value where VM : VirtualInterface {
    // vm stack will contain [..., arg1, arg2, ... argN]
    // popping in order will populate the vector with [argN, argN-1, ... arg1]
    let mut args: Vec<Box<Value>> = Vec::with_capacity(nargs as usize);
    for _ in 0..nargs {
        args.push(Box::new(vm.pop().clone()));
    }
    Value::PartialBinding(bound, Box::new(args))
}
