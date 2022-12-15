use std::collections::HashMap;

use crate::stdlib::StdBinding::{*};
use crate::vm::Value;

/// Build a `Node` containing all native bindings for the interpreter runtime.
/// This is used in the parser in order to replace raw identifiers with their bound enum value.
pub fn bindings() -> StdBindingTree {
    root([
        ("print", node(PrintOut, [
            ("out", leaf(PrintOut)),
            ("err", leaf(PrintErr)),
        ])),
    ])
}

/// Looks up the semantic name of a binding. This is the result of calling `print->out . str` for example
pub fn lookup_binding(b: StdBinding) -> &'static str {
    match b {
        PrintOut => "print->out",
        PrintErr => "print->err",
    }
}

/// The enum containing all bindings as they are represented at runtime.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum StdBinding {
    PrintOut,
    PrintErr,
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


pub fn invoke_binding_0(binding: StdBinding) -> Value {
    match binding {
        PrintOut => {
            println!();
            Value::Nil
        },
        PrintErr => {
            eprintln!();
            Value::Nil
        },
    }
}

/// **Implementation Note:** This takes `args` in reverse order, so `args[0]` will be the last argument, `args[args.len() - 1]` will be the first, etc.
pub fn invoke_binding_n(binding: StdBinding, args: Vec<Value>) -> Value {
    match binding {
        PrintOut => {
            let mut it = args.iter();
            if let Some(v) = it.next() {
                println!("{}", v.as_str());
                for v0 in it {
                    println!(" {}", v0.as_str());
                }
            }
            Value::Nil
        },
        PrintErr => {
            let mut it = args.iter();
            if let Some(v) = it.next() {
                eprintln!("{}", v.as_str());
                for v0 in it {
                    eprintln!(" {}", v0.as_str());
                }
            }
            Value::Nil
        }
    }
}
