use crate::vm::error::RuntimeErrorType;
use crate::vm::value::Value;

use RuntimeErrorType::{*};
use Value::{*};


pub fn abs(a1: Value) -> Result<Value, RuntimeErrorType> {
    match a1 {
        Int(i) => Ok(Int(i.abs())),
        v => Err(TypeErrorArgMustBeInt(v.clone())),
    }
}