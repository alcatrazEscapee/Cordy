use std::borrow::Borrow;
use std::cell::RefCell;
use std::rc::Rc;

use crate::stdlib;
use crate::stdlib::StdBinding;
use crate::vm::error::RuntimeError;

use Value::{*};
use RuntimeError::{*};


type BoolResult = Result<bool, Box<RuntimeError>>;


/// The runtime sum type used by the virtual machine
/// All `Value` type objects must be cloneable, and so mutable objects must be reference counted to ensure memory safety
#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Value {
    // Primitive (Immutable) Types
    Nil,
    Bool(bool),
    Int(i64),
    Str(Box<String>),
    Function(Rc<FunctionImpl>),
    PartialFunction(Box<PartialFunctionImpl>),

    // Reference (Mutable) Types
    // Really shitty memory management for now just using RefCell... in the future maybe we implement a garbage collected system
    // mark and sweep or something... but for now, RefCell will do!
    List(Rc<RefCell<Vec<Value>>>),

    // Functions
    Binding(StdBinding),
    PartialBinding(StdBinding, Box<Vec<Box<Value>>>),
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct FunctionImpl {
    pub head: usize, // Pointer to the first opcode of the function's execution
    pub nargs: u8, // The number of arguments the function takes
    name: String, // The name of the function, useful to show in stack traces
    args: Vec<String>, // Names of the arguments
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct PartialFunctionImpl {
    pub func: Rc<FunctionImpl>,
    pub args: Vec<Box<Value>>,
}


impl Value {

    // Constructors
    pub fn list(vec: Vec<Value>) -> Value { List(Rc::new(RefCell::new(vec))) }
    pub fn partial1(func: Rc<FunctionImpl>, arg: Value) -> Value { PartialFunction(Box::new(PartialFunctionImpl { func, args: vec![Box::new(arg)] }))}
    pub fn partial(func: Rc<FunctionImpl>, args: Vec<Value>) -> Value { PartialFunction(Box::new(PartialFunctionImpl { func, args: args.into_iter().map(|v| Box::new(v)).collect() }))}

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn as_str(self: &Self) -> String {
        match self {
            Str(s) => *s.clone(),
            List(v) => format!("[{}]", (*v).as_ref().borrow().iter().map(|t| t.as_str()).collect::<Vec<String>>().join(", ")),
            Function(f) => f.name.clone(),
            PartialFunction(f) => f.func.name.clone(),
            Binding(b) => String::from(stdlib::lookup_binding(b)),
            PartialBinding(b, _) => String::from(stdlib::lookup_binding(b)),
            _ => self.as_repr_str(),
        }
    }

    /// Converts the `Value` to a representative `String. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn as_repr_str(self: &Self) -> String {
        match self {
            Nil => String::from("nil"),
            Bool(b) => b.to_string(),
            Int(i) => i.to_string(),
            Str(s) => {
                let escaped = format!("{:?}", s);
                format!("'{}'", &escaped[1..escaped.len() - 1])
            },
            List(v) => format!("[{}]", (*v).as_ref().borrow().iter().map(|t| t.as_repr_str()).collect::<Vec<String>>().join(", ")),
            Function(f) => {
                let f = (*f).as_ref().borrow();
                format!("fn {}({})", f.name, f.args.join(", "))
            },
            PartialFunction(f) => {
                let f = (*f).as_ref().borrow();
                format!("fn {}({})", f.func.name, f.func.args.join(", "))
            },
            Binding(b) => format!("fn {}()", stdlib::lookup_binding(b)),
            PartialBinding(b, _) => format!("fn {}()", stdlib::lookup_binding(b)),
        }
    }

    /// Represents the type of this `Value`. This is used for runtime error messages,
    pub fn as_type_str(self: &Self) -> String {
        String::from(match self {
            Nil => "nil",
            Bool(_) => "bool",
            Int(_) => "int",
            Str(_) => "str",
            List(_) => "list",
            Function(_) => "function",
            PartialFunction(_) => "partial function",
            Binding(_) => "native function",
            PartialBinding(_, _) => "partial native function",
        })
    }

    #[cfg(any(trace_interpreter = "on", trace_interpreter_stack = "on"))]
    pub fn as_debug_str(self: &Self) -> String {
        format!("{}: {}", self.as_repr_str(), self.as_type_str())
    }

    pub fn as_bool(self: &Self) -> bool {
        match self {
            Nil => false,
            Bool(b) => *b,
            Int(i) => *i != 0,
            Str(s) => !s.is_empty(),
            List(l) => !l.as_ref().borrow().is_empty(),
            Function(_) => true,
            PartialFunction(_) => true,
            Binding(_) => true,
            PartialBinding(_, _) => true,
        }
    }

    pub fn is_nil(self: &Self) -> bool {
        match self {
            Nil => true,
            _ => false
        }
    }

    pub fn is_bool(self: &Self) -> bool {
        match self {
            Bool(_) => true,
            _ => false
        }
    }

    pub fn is_int(self: &Self) -> bool {
        match self {
            Int(_) => true,
            _ => false
        }
    }

    pub fn is_str(self: &Self) -> bool {
        match self {
            Str(_) => true,
            _ => false
        }
    }

    pub fn is_function(self: &Self) -> bool {
        match self {
            Function(_) => true,
            _ => false
        }
    }

    pub fn is_equal(self: &Self, other: &Value) -> bool {
        match (self, other) {
            (Nil, Nil) => true,
            (Bool(l), Bool(r)) => l == r,
            (Int(l), Int(r)) => l == r,
            (Str(l), Str(r)) => l == r,
            (Binding(l), Binding(r)) => l == r,
            _ => false,
        }
    }

    pub fn is_less_than(self: &Self, other: &Value) -> BoolResult {
        match (self, other) {
            (Bool(l), Bool(r)) => Ok(!*l && *r),
            (Int(l), Int(r)) => Ok(l < r),
            (Str(l), Str(r)) => Ok(l < r),
            (l, r) => TypeErrorCannotCompare(l.clone(), r.clone()).err()
        }
    }

    pub fn is_less_than_or_equal(self: &Self, other: &Value) -> BoolResult {
        match (self, other) {
            (Bool(l), Bool(r)) => Ok(!*l || *r),
            (Int(l), Int(r)) => Ok(l <= r),
            (Str(l), Str(r)) => Ok(l <= r),
            (l, r) => TypeErrorCannotCompare(l.clone(), r.clone()).err()
        }
    }
}


impl FunctionImpl {
    pub fn new(head: usize, name: String, args: Vec<String>) -> FunctionImpl {
        FunctionImpl {
            head,
            nargs: args.len() as u8,
            name,
            args
        }
    }
}


#[cfg(test)]
mod test {
    use crate::vm::error::RuntimeError;
    use crate::vm::value::Value;

    #[test] fn test_layout() { assert_eq!(16, std::mem::size_of::<Value>()); }
    #[test] fn test_result_box_layout() { assert_eq!(16, std::mem::size_of::<Result<Value, Box<RuntimeError>>>()); }
}
