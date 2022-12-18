use std::cell::RefCell;
use std::rc::Rc;
use crate::stdlib;
use crate::stdlib::StdBinding;
use crate::vm::error::RuntimeErrorType;

use Value::{*};

/// The runtime sum type used by the virtual machine
/// All `Value` type objects must be cloneable, and so mutable objects must be reference counted to ensure memory safety
#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Value {
    // Primitive (Immutable) Types
    Nil,
    Bool(bool),
    Int(i64),
    Str(Box<String>),

    // Reference (Mutable) Types
    // Really shitty memory management for now just using RefCell... in the future maybe we implement a garbage collected system
    // mark and sweep or something... but for now, RefCell will do!
    List(Rc<RefCell<Vec<Value>>>),

    // Functions
    Binding(StdBinding),
    PartialBinding(StdBinding, Box<Vec<Box<Value>>>),
}


impl Value {

    // Constructors
    pub fn list(vec: Vec<Value>) -> Value {
        List(Rc::new(RefCell::new(vec)))
    }

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn as_str(self: &Self) -> String {
        match self {
            Str(s) => *s.clone(),
            List(v) => format!("[{}]", (*v).borrow().iter().map(|t| t.as_str()).collect::<Vec<String>>().join(", ")),
            Binding(b) => String::from(stdlib::lookup_binding(b)),
            _ => self.as_repr_str(),
        }
    }

    /// Converts the `Value` to a representative `String. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn as_repr_str(self: &Self) -> String {
        match self {
            Nil => String::from("nil"),
            Bool(b) => b.to_string(),
            Int(i) => i.to_string(),
            Str(s) => format!("'{}'", s),
            List(v) => format!("[{}]", (*v).borrow().iter().map(|t| t.as_repr_str()).collect::<Vec<String>>().join(", ")),
            Binding(b) => String::from(stdlib::lookup_binding(b)),
            PartialBinding(b, v) => format!("{}({})", stdlib::lookup_binding(b), v.iter().rev().map(|a| a.as_repr_str()).collect::<Vec<String>>().join(", "))
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
            Binding(_) => "function",
            PartialBinding(_, _) => "partial function",
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

    pub fn is_less_than(self: &Self, other: &Value) -> Result<bool, RuntimeErrorType> {
        match (self, other) {
            (Bool(l), Bool(r)) => Ok(!*l && *r),
            (Int(l), Int(r)) => Ok(l < r),
            (Str(l), Str(r)) => Ok(l < r),
            (l, r) => Err(RuntimeErrorType::TypeErrorCannotCompare(l.clone(), r.clone()))
        }
    }

    pub fn is_less_than_or_equal(self: &Self, other: &Value) -> Result<bool, RuntimeErrorType> {
        match (self, other) {
            (Bool(l), Bool(r)) => Ok(!*l || *r),
            (Int(l), Int(r)) => Ok(l <= r),
            (Str(l), Str(r)) => Ok(l <= r),
            (l, r) => Err(RuntimeErrorType::TypeErrorCannotCompare(l.clone(), r.clone()))
        }
    }
}


#[cfg(test)]
mod test {
    use crate::vm::value::Value;

    #[test] fn test_layout() { assert_eq!(16, std::mem::size_of::<Value>()); }
}
