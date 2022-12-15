use std::collections::HashMap;

use crate::vm::{IO, RuntimeErrorType, Stack};
use crate::vm::value::Value;

use crate::stdlib::StdBinding::{*};

/// Build a `Node` containing all native bindings for the interpreter runtime.
/// This is used in the parser in order to replace raw identifiers with their bound enum value.
pub fn bindings() -> StdBindingTree {
    root([
        ("print", node(PrintOut, [
            ("out", leaf(PrintOut)),
            ("err", leaf(PrintErr)),
        ])),
        ("nil", leaf(Nil)),
        ("bool", leaf(Bool)),
        ("int", leaf(Int)),
        ("str", leaf(Str)),
        ("repr", leaf(Repr)),
    ])
}

/// Looks up the semantic name of a binding. This is the result of calling `print->out . str` for example
pub fn lookup_binding(b: &StdBinding) -> &'static str {
    match b {
        PrintOut => "print->out",
        PrintErr => "print->err",
        Nil => "nil",
        Bool => "bool",
        Int => "int",
        Str => "str",
        Repr => "repr"
    }
}

/// The enum containing all bindings as they are represented at runtime.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum StdBinding {
    PrintOut,
    PrintErr,
    Nil,
    Bool,
    Int,
    Str,
    Repr,
}


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
}

pub fn invoke_type_binding(bound: StdBinding, arg: Value) -> Result<Value, RuntimeErrorType> {
    match bound {
        Nil => Ok(Value::Bool(arg.is_nil())),
        Bool => Ok(Value::Bool(arg.is_bool())),
        Int => Ok(Value::Bool(arg.is_int())),
        Str => Ok(Value::Bool(arg.is_str())),
        _ => Err(RuntimeErrorType::TypeErrorBinaryIs(arg, Value::Binding(bound)))
    }
}


pub fn invoke_func_binding<S>(bound: StdBinding, nargs: u8, vm: &mut S) -> Result<Value, RuntimeErrorType> where
    S : Stack,
    S : IO
{
    macro_rules! pop_args {
        ($a1:ident) => {
            if nargs != 1 { return Err(RuntimeErrorType::IncorrectNumberOfArguments(bound.clone(), nargs, 1)); }
            let $a1: Value = vm.pop();
        };
    }

    match bound {
        PrintOut => {
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
        PrintErr => {
            match nargs {
                0 => eprintln!(),
                1 => eprintln!("{}", vm.pop().as_str()),
                _ => {
                    let mut rev = Vec::with_capacity(nargs as usize);
                    for _ in 0..nargs {
                        rev.push(vm.pop().as_str());
                    }
                    eprint!("{}", rev.pop().unwrap());
                    for _ in 1..nargs {
                        eprint!(" {}", rev.pop().unwrap());
                    }
                    eprintln!();
                }
            }
            Ok(Value::Nil)
        },
        Bool => {
            pop_args!(a1);
            Ok(Value::Bool(a1.as_bool()))
        },
        Int => {
            pop_args!(a1);
            match &a1 {
                Value::Nil => Ok(Value::Int(0)),
                Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                Value::Int(_) => Ok(a1),
                Value::Str(s) => match s.parse::<i64>() {
                    Ok(i) => Ok(Value::Int(i)),
                    Err(_) => Err(RuntimeErrorType::TypeErrorCannotConvertToInt(a1)),
                },
                _ => Err(RuntimeErrorType::TypeErrorCannotConvertToInt(a1)),
            }
        },
        Str => {
            pop_args!(a1);
            Ok(Value::Str(a1.as_str()))
        },
        Repr => {
            pop_args!(a1);
            Ok(Value::Str(a1.as_repr_str()))
        }
        _ => Err(RuntimeErrorType::BindingIsNotFunctionEvaluable(bound.clone()))
    }
}
