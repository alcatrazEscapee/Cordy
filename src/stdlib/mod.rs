use std::collections::HashMap;

use crate::vm::{IO, RuntimeErrorType, Stack};
use crate::vm::value::Value;

use crate::stdlib::StdBinding::{*};


mod lib_str;


/// Build a `Node` containing all native bindings for the interpreter runtime.
/// This is used in the parser in order to replace raw identifiers with their bound enum value.
pub fn bindings() -> HashMap<&'static str, StdBinding> {
    HashMap::from([
        ("print", Print),
        ("nil", Nil),
        ("bool", Bool),
        ("int", Int),
        ("str", Str),
        ("repr", Repr),
        ("to_lower", ToLower),
        ("to_upper", ToUpper),
        ("replace", Replace),
        ("trim", Trim),
        ("index_of", IndexOf),
        ("count_of", CountOf),
    ])
}

/// Looks up the semantic name of a binding. This is the result of calling `print->out . str` for example
pub fn lookup_binding(b: &StdBinding) -> &'static str {
    match b {
        Print => "print",
        Nil => "nil",
        Bool => "bool",
        Int => "int",
        Str => "str",
        Repr => "repr",
        ToLower => "to_lower",
        ToUpper => "to_upper",
        Replace => "replace",
        Trim => "trim",
        IndexOf => "index_of",
        CountOf => "count_of",

    }
}

/// The enum containing all bindings as they are represented at runtime.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum StdBinding {
    Print,
    Nil,
    Bool,
    Int,
    Str,
    Repr,

    // lib_str
    ToLower,
    ToUpper,
    Replace,
    Trim,
    IndexOf,
    CountOf,
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
        ($a1:ident,$a2:ident) => {
            if nargs != 2 { return Err(RuntimeErrorType::IncorrectNumberOfArguments(bound.clone(), nargs, 2)); }
            let $a2: Value = vm.pop();
            let $a1: Value = vm.pop();
        };
        ($a1:ident,$a2:ident,$a3:ident) => {
            if nargs != 3 { return Err(RuntimeErrorType::IncorrectNumberOfArguments(bound.clone(), nargs, 3)); }
            let $a3: Value = vm.pop();
            let $a2: Value = vm.pop();
            let $a1: Value = vm.pop();
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
        },

        // lib_str
        ToLower => { pop_args!(a1); lib_str::to_lower(a1) },
        ToUpper => { pop_args!(a1); lib_str::to_upper(a1) },
        Replace => { pop_args!(a1, a2, a3); lib_str::replace(a1, a2, a3) },
        Trim => { pop_args!(a1); lib_str::trim(a1) },
        IndexOf => { pop_args!(a1, a2); lib_str::index_of(a1, a2) },
        CountOf => { pop_args!(a1, a2); lib_str::count_of(a1, a2) },

        _ => Err(RuntimeErrorType::BindingIsNotFunctionEvaluable(bound.clone()))
    }
}
