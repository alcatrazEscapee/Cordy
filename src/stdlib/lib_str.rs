use crate::vm::error::RuntimeErrorType;
use crate::vm::value::Value;

use Value::{*};

macro_rules! fn_str_to_str {
    ($func:ident, $s:ident, $expr:expr) => {
        pub fn $func(v: Value) -> Result<Value, RuntimeErrorType> {
            match &v {
                Str($s) => Ok($expr),
                _ => Err(RuntimeErrorType::TypeErrorFunc1(concat!(stringify!($ident), "(str): str"), v))
            }
        }
    };
}


fn_str_to_str!(to_lower, s, Str(Box::new(s.to_lowercase())));
fn_str_to_str!(to_upper, s, Str(Box::new(s.to_uppercase())));
fn_str_to_str!(trim, s, Str(Box::new(String::from(s.trim()))));

pub fn replace(v0: Value, v1: Value, v2: Value) -> Result<Value, RuntimeErrorType> {
    match (&v0, &v1, &v2) {
        (Str(a0), Str(a1), Str(a2)) => Ok(Str(Box::new(a2.replace(a0.as_ref(), a1.as_str())))),
        _ => Err(RuntimeErrorType::TypeErrorFunc3("replace(str, str, str): str", v0, v1, v2))
    }
}

pub fn index_of(v0: Value, v1: Value) -> Result<Value, RuntimeErrorType> {
    match (&v0, &v1) {
        (Str(a0), Str(a1)) => Ok(Int(a1.find(a0.as_ref()).map(|i| i as i64).unwrap_or(-1))),
        _ => Err(RuntimeErrorType::TypeErrorFunc2("index_of(str, str): int", v0, v1))
    }
}

pub fn count_of(v0: Value, v1: Value) -> Result<Value, RuntimeErrorType> {
    match (&v0, &v1) {
        (Str(a0), Str(a1)) => Ok(Int(a1.matches(a0.as_ref()).count() as i64)),
        _ => Err(RuntimeErrorType::TypeErrorFunc2("count_of(str, str): int", v0, v1))
    }
}

pub fn split(v0: Value, v1: Value) -> Result<Value, RuntimeErrorType> {
    match (&v0, &v1) {
        (Str(a0), Str(a1)) => Ok(Value::list(a1.split(a0.as_ref()).map(|v| Str(Box::new(String::from(v)))).collect::<Vec<Value>>())),
        _ => Err(RuntimeErrorType::TypeErrorFunc2("split(str, str): [str]", v0, v1))
    }
}

