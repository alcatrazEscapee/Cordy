use std::collections::HashMap;
use std::{fs, io};

use crate::vm::{IO, Stack};
use crate::vm::value::Value;
use crate::vm::error::RuntimeErrorType;

use StdBinding::{*};

pub mod lib_str;
pub mod lib_list;


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
        ("repr", Repr),
        ("len", Len),

        // lib_io
        ("to_lower", ToLower),
        ("to_upper", ToUpper),
        ("replace", Replace),
        ("trim", Trim),
        ("index_of", IndexOf),
        ("count_of", CountOf),
        ("split", Split),
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
        Repr => "repr",
        Len => "len",

        // lib_str
        ToLower => "to_lower",
        ToUpper => "to_upper",
        Replace => "replace",
        Trim => "trim",
        IndexOf => "index_of",
        CountOf => "count_of",
        Split => "split",

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
    Repr,
    Len,

    // lib_str
    ToLower,
    ToUpper,
    Replace,
    Trim,
    IndexOf,
    CountOf,
    Split,
}

/*
pub struct StdBindingTree {
    pub binding: Option<StdBinding>,
    pub children: Option<HashMap<&'static str, StdBindingTree>>
}

fn root<const N: usize>(children: [(&'static str, StdBindingTree); N]) -> StdBindingTree { StdBindingTree::new(None, Some(HashMap::from(children))) }
fn node<const N: usize>(node: StdBinding, children: [(&'static str, StdBindingTree); N]) -> StdBindingTree { StdBindingTree::new(Some(node), Some(HashMap::from(children))) }
fn leaf(node: StdBinding) -> StdBindingTree { StdBindingTree::new(Some(node), None) }

impl StdBindingTree {
    fn new(binding: Option<StdBinding>, children: Option<HashMap<&'static str, StdBindingTree>>) -> StdBindingTree {
        StdBindingTree { binding, children }
    }
}*/


pub fn invoke<S>(bound: StdBinding, nargs: u8, vm: &mut S) -> Result<Value, RuntimeErrorType> where
    S : Stack,
    S : IO,
{
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

        // lib_str
        ToLower => dispatch!(a1, lib_str::to_lower(a1)),
        ToUpper => dispatch!(a1, lib_str::to_upper(a1)),
        Replace => dispatch!(a1, a2, a3, lib_str::replace(a1, a2, a3)),
        Trim => dispatch!(a1, lib_str::trim(a1)),
        IndexOf => dispatch!(a1, a2, lib_str::index_of(a1, a2)),
        CountOf => dispatch!(a1, a2, lib_str::count_of(a1, a2)),
        Split => dispatch!(a1, a2, lib_str::split(a1, a2)),

        _ => Err(RuntimeErrorType::BindingIsNotFunctionEvaluable(bound.clone()))
    }
}


fn wrap_as_partial<S>(bound: StdBinding, nargs: u8, vm: &mut S) -> Result<Value, RuntimeErrorType> where
    S : Stack,
{
    // vm stack will contain [..., arg1, arg2, ... argN]
    // popping in order will populate the vector with [argN, argN-1, ... arg1]
    let mut args: Vec<Box<Value>> = Vec::with_capacity(nargs as usize);
    for _ in 0..nargs {
        args.push(Box::new(vm.pop().clone()));
    }
    Ok(Value::PartialBinding(bound, Box::new(args)))
}
