use crate::stdlib;
use crate::stdlib::StdBinding;

/// The runtime sum type used by the virtual machine
#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Str(String),
    Binding(StdBinding)
}


impl Value {
    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn as_str(self: &Self) -> String {
        match self {
            Value::Nil => String::from("nil"),
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Str(s) => s.clone(),
            Value::Binding(b) => String::from(stdlib::lookup_binding(b))
        }
    }

    /// Represents the type of this `Value`. This is used for runtime error messages,
    pub fn as_type_str(self: &Self) -> String {
        String::from(match self {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Str(_) => "str",
            Value::Binding(_) => "{binding}"
        })
    }

    pub fn as_bool(self: &Self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Str(s) => !s.is_empty(),
            Value::Binding(_) => true,
        }
    }

    pub fn is_nil(self: &Self) -> bool {
        match self {
            Value::Nil => true,
            _ => false
        }
    }

    pub fn is_bool(self: &Self) -> bool {
        match self {
            Value::Bool(_) => true,
            _ => false
        }
    }

    pub fn is_int(self: &Self) -> bool {
        match self {
            Value::Int(_) => true,
            _ => false
        }
    }

    pub fn is_str(self: &Self) -> bool {
        match self {
            Value::Str(_) => true,
            _ => false
        }
    }
}
