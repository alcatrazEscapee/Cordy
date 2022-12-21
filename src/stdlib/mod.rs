use std::collections::HashMap;
use std::{fs, io};

use crate::vm::{operator, VirtualInterface};
use crate::vm::value::Value;
use crate::vm::error::RuntimeErrorType;

use StdBinding::{*};
use crate::trace;

mod lib_str;
pub mod lib_list; // VM to directly call slice, index
mod lib_math;


/// Build a `Node` containing all native bindings for the interpreter runtime.
/// This is used in the parser in order to replace raw identifiers with their bound enum value.
pub fn bindings() -> HashMap<&'static str, StdBinding> {
    HashMap::from([
        ("print", Print),
        ("read", Read),
        ("read_text", ReadText),
        ("write_text", WriteText),
        ("nil", Nil),
        ("bool", Bool),
        ("int", Int),
        ("str", Str),
        ("function", Function),
        ("repr", Repr),
        ("len", Len),

        // lib_str
        ("to_lower", ToLower),
        ("to_upper", ToUpper),
        ("replace", Replace),
        ("trim", Trim),
        ("index_of", IndexOf),
        ("count_of", CountOf),
        ("split", Split),

        // lib_list
        ("sum", Sum),
        ("max", Max),
        ("min", Min),

        ("map", Map),

        // lib_math
        ("abs", Abs),
    ])
}

/// Looks up the semantic name of a binding. This is the result of calling `print->out . str` for example
pub fn lookup_binding(b: &StdBinding) -> &'static str {
    match b {
        Print => "print",
        Read => "read",
        ReadText => "read_text",
        WriteText => "write_text",
        Nil => "nil",
        Bool => "bool",
        Int => "int",
        Str => "str",
        Function => "function",
        Repr => "repr",
        Len => "len",

        OperatorUnarySub => "(-)",
        OperatorUnaryLogicalNot => "(!)",
        OperatorUnaryBitwiseNot => "(~)",

        OperatorMul => "(*)",
        OperatorDiv => "(/)",
        OperatorPow => "(**)",
        OperatorMod => "(%)",
        OperatorIs => "(is)",
        OperatorAdd => "(+)",
        OperatorSub => "(-)",
        OperatorLeftShift => "(<<)",
        OperatorRightShift => "(>>)",
        OperatorBitwiseAnd => "(&)",
        OperatorBitwiseOr => "(|)",
        OperatorBitwiseXor => "(^)",
        OperatorLessThan => "(<)",
        OperatorLessThanEqual => "(<=)",
        OperatorGreaterThan => "(>)",
        OperatorGreaterThanEqual => "(>=)",
        OperatorEqual => "(==)",
        OperatorNotEqual => "(!=)",

        // lib_str
        ToLower => "to_lower",
        ToUpper => "to_upper",
        Replace => "replace",
        Trim => "trim",
        IndexOf => "index_of",
        CountOf => "count_of",
        Split => "split",

        // lib_list
        Sum => "sum",
        Min => "min",
        Max => "max",
        Map => "map",

        // lib_math
        Abs => "abs",
    }
}

/// The enum containing all bindings as they are represented at runtime.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum StdBinding {
    Print,
    Read,
    ReadText,
    WriteText,
    Nil,
    Bool,
    Int,
    Str,
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
    // Filter,
    // Reduce,

    // lib_math
    Abs,
}


pub fn invoke<VM>(bound: StdBinding, nargs: u8, vm: &mut VM) -> Result<Value, RuntimeErrorType> where VM : VirtualInterface
{
    trace::trace_interpreter!("stdlib::invoke() func={}, nargs={}", lookup_binding(&bound), nargs);
    // Dispatch macros for 0, 1, 2 and 3 argument functions
    // All dispatch!() cases support partial evaluation (where 0 < nargs < required args)
    macro_rules! dispatch {
        ($ret:expr) => {
            match nargs {
                0 => $ret,
                _ => Err(RuntimeErrorType::IncorrectNumberOfArguments(bound.clone(), nargs, 0))
            }
        };
        ($a1:ident, $ret:expr) => {
            match nargs {
                1 => {
                    let $a1: Value = vm.pop();
                    $ret
                },
                _ => Err(RuntimeErrorType::IncorrectNumberOfArguments(bound.clone(), nargs, 1))
            }
        };
        ($a1:ident, $a2:ident, $ret:expr) => {
            match nargs {
                1 => wrap_as_partial(bound, nargs, vm),
                2 => {
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    $ret
                },
                _ => Err(RuntimeErrorType::IncorrectNumberOfArguments(bound.clone(), nargs, 2))
            }
        };
        ($a1:ident, $a2:ident, $a3:ident, $ret:expr) => {
            match nargs {
                1 | 2 => wrap_as_partial(bound, nargs, vm),
                3 => {
                    let $a3: Value = vm.pop();
                    let $a2: Value = vm.pop();
                    let $a1: Value = vm.pop();
                    $ret
                },
                _ => Err(RuntimeErrorType::IncorrectNumberOfArguments(bound.clone(), nargs, 3))
            }
        };
    }

    macro_rules! dispatch_varargs {
        ($iter:path, $list:path) => {
            match nargs {
                0 => Err(RuntimeErrorType::IncorrectNumberOfArgumentsVariadicAtLeastOne(Sum)),
                1 => {
                    let a1: Value = vm.pop();
                    $iter(a1)
                },
                _ => {
                    let args: Vec<Value> = vm.popn(nargs as usize);
                    $list(&args)
                }
            }
        };
    }

    match bound {
        Print => {
            match nargs {
                0 => vm.println0(),
                1 => {
                    let s = vm.pop().as_str();
                    vm.println(s)
                },
                _ => {
                    let mut rev = Vec::with_capacity(nargs as usize);
                    for _ in 0..nargs {
                        rev.push(vm.pop().as_str());
                    }
                    vm.print(rev.pop().unwrap());
                    for _ in 1..nargs {
                        vm.print(format!(" {}", rev.pop().unwrap()));
                    }
                    vm.println0();
                }
            }
            Ok(Value::Nil)
        },
        Read => dispatch!({
            // todo: this doesn't work
            let stdin = io::stdin();
            let it = stdin.lines();
            Ok(Value::Str(Box::new(it.map(|t| String::from(t.unwrap())).collect::<Vec<String>>().join("\n"))))
        }),
        ReadText => dispatch!(a1, match a1 {
            Value::Str(s1) => Ok(Value::Str(Box::new(fs::read_to_string(s1.as_ref()).unwrap().replace("\r", "")))), // todo: error handling?
            _ => Err(RuntimeErrorType::TypeErrorFunc1("read_text(str) -> str", a1)),
        }),
        WriteText => dispatch!(a1, a2, match (&a1, &a2) {
            (Value::Str(s1), Value::Str(s2)) => {
                fs::write(s1.as_ref(), s2.as_ref()).unwrap();
                Ok(Value::Nil)
            },
            _ => Err(RuntimeErrorType::TypeErrorFunc1("write_text(str, str) -> str", a1.clone())),
        }),
        Bool => dispatch!(a1, Ok(Value::Bool(a1.as_bool()))),
        Int => dispatch!(a1, match &a1 {
            Value::Nil => Ok(Value::Int(0)),
            Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
            Value::Int(_) => Ok(a1),
            Value::Str(s) => match s.parse::<i64>() {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => Err(RuntimeErrorType::TypeErrorCannotConvertToInt(a1)),
            },
            _ => Err(RuntimeErrorType::TypeErrorCannotConvertToInt(a1)),
        }),
        Str => dispatch!(a1, Ok(Value::Str(Box::new(a1.as_str())))),
        Repr => dispatch!(a1, Ok(Value::Str(Box::new(a1.as_repr_str())))),
        Len => dispatch!(a1, match &a1 {
            Value::Str(s) => Ok(Value::Int(s.len() as i64)),
            Value::List(l) => Ok(Value::Int((*l).borrow().len() as i64)),
            _ => Err(RuntimeErrorType::TypeErrorFunc1("len([T] | str): int", a1))
        }),

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
        OperatorLessThan => dispatch!(a1, a2, operator::binary_less_than(a2, a1)),
        OperatorLessThanEqual => dispatch!(a1, a2, operator::binary_less_than_or_equal(a2, a1)),
        OperatorGreaterThan => dispatch!(a1, a2, operator::binary_greater_than(a2, a1)),
        OperatorGreaterThanEqual => dispatch!(a1, a2, operator::binary_greater_than_or_equal(a2, a1)),
        OperatorEqual => dispatch!(a1, a2, operator::binary_equals(a2, a1)),
        OperatorNotEqual => dispatch!(a1, a2, operator::binary_not_equals(a2, a1)),

        // lib_str
        ToLower => dispatch!(a1, lib_str::to_lower(a1)),
        ToUpper => dispatch!(a1, lib_str::to_upper(a1)),
        Replace => dispatch!(a1, a2, a3, lib_str::replace(a1, a2, a3)),
        Trim => dispatch!(a1, lib_str::trim(a1)),
        IndexOf => dispatch!(a1, a2, lib_str::index_of(a1, a2)),
        CountOf => dispatch!(a1, a2, lib_str::count_of(a1, a2)),
        Split => dispatch!(a1, a2, lib_str::split(a1, a2)),

        // lib_list
        Sum => dispatch_varargs!(lib_list::sum_iter, lib_list::sum_list),
        Max => dispatch_varargs!(lib_list::max_iter, lib_list::max_list),
        Min => dispatch_varargs!(lib_list::min_iter, lib_list::min_list),
        Map => dispatch!(a1, a2, lib_list::map(vm, a1, a2)),

        // lib_math
        Abs => dispatch!(a1, lib_math::abs(a1)),

        _ => Err(RuntimeErrorType::BindingIsNotFunctionEvaluable(bound.clone())),
    }
}


fn wrap_as_partial<VM>(bound: StdBinding, nargs: u8, vm: &mut VM) -> Result<Value, RuntimeErrorType> where VM : VirtualInterface {
    // vm stack will contain [..., arg1, arg2, ... argN]
    // popping in order will populate the vector with [argN, argN-1, ... arg1]
    let mut args: Vec<Box<Value>> = Vec::with_capacity(nargs as usize);
    for _ in 0..nargs {
        args.push(Box::new(vm.pop().clone()));
    }
    Ok(Value::PartialBinding(bound, Box::new(args)))
}
