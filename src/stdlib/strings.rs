use crate::vm::error::RuntimeError;
use crate::vm::value::Value;

use Value::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


macro_rules! fn_str_to_str {
    ($func:ident, $s:ident, $expr:expr) => {
        pub fn $func(v: Value) -> ValueResult {
            match &v {
                Str($s) => Ok($expr),
                _ => TypeErrorFunc1(concat!(stringify!($ident), "(str): str"), v).err()
            }
        }
    };
}


fn_str_to_str!(to_lower, s, Str(Box::new(s.to_lowercase())));
fn_str_to_str!(to_upper, s, Str(Box::new(s.to_uppercase())));
fn_str_to_str!(trim, s, Str(Box::new(String::from(s.trim()))));

pub fn replace(v0: Value, v1: Value, v2: Value) -> ValueResult {
    match (&v0, &v1, &v2) {
        (Str(a0), Str(a1), Str(a2)) => Ok(Str(Box::new(a2.replace(a0.as_ref(), a1.as_str())))),
        _ => TypeErrorFunc3("replace(str, str, str): str", v0, v1, v2).err()
    }
}

pub fn split(v0: Value, v1: Value) -> ValueResult {
    match (&v0, &v1) {
        (Str(a0), Str(a1)) => Ok(Value::iter_list(a1.split(a0.as_ref()).map(|v| Str(Box::new(String::from(v)))))),
        _ => TypeErrorFunc2("split(str, str): [str]", v0, v1).err()
    }
}

pub fn to_char(a1: Value) -> ValueResult {
    match &a1 {
        Int(i) if i > &0 => match char::from_u32(*i as u32) {
            Some(c) => Ok(Value::str(c)),
            None => ValueErrorInvalidCharacterOrdinal(*i).err()
        },
        Int(i) => ValueErrorInvalidCharacterOrdinal(*i).err(),
        _ => TypeErrorArgMustBeInt(a1).err()
    }
}

pub fn to_ord(a1: Value) -> ValueResult {
    match &a1 {
        Str(s) if s.len() == 1 => Ok(Int(s.chars().next().unwrap() as u32 as i64)),
        _ => TypeErrorArgMustBeChar(a1).err()
    }
}

pub fn to_hex(a1: Value) -> ValueResult {
    Ok(Str(Box::new(format!("{:x}", a1.as_int()?))))
}

pub fn to_bin(a1: Value) -> ValueResult {
    Ok(Str(Box::new(format!("{:b}", a1.as_int()?))))
}


