use crate::stdlib;
use crate::stdlib::StdBinding;
use crate::vm::RuntimeErrorType;

use Value::{*};

/// The runtime sum type used by the virtual machine
#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Str(String),
    Binding(StdBinding),
}


impl Value {
    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn as_str(self: &Self) -> String {
        match self {
            Nil => String::from("nil"),
            Bool(b) => b.to_string(),
            Int(i) => i.to_string(),
            Str(s) => s.clone(),
            Binding(b) => String::from(stdlib::lookup_binding(b)),
        }
    }

    /// Converts the `Value` to a representative `String. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn as_repr_str(self: &Self) -> String {
        match self {
            Nil => String::from("nil"),
            Bool(b) => b.to_string(),
            Int(i) => i.to_string(),
            Str(s) => format!("'{}'", s),
            Binding(b) => String::from(stdlib::lookup_binding(b)),
        }
    }

    /// Represents the type of this `Value`. This is used for runtime error messages,
    pub fn as_type_str(self: &Self) -> String {
        String::from(match self {
            Nil => "nil",
            Bool(_) => "bool",
            Int(_) => "int",
            Str(_) => "str",
            Binding(_) => "{binding}",
        })
    }

    pub fn as_bool(self: &Self) -> bool {
        match self {
            Nil => false,
            Bool(b) => *b,
            Int(i) => *i != 0,
            Str(s) => !s.is_empty(),
            Binding(_) => true,
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
