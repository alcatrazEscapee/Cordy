use crate::vm::error::RuntimeError;
use crate::vm::value::Value;

use RuntimeError::{*};
use Value::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


pub fn abs(a1: Value) -> ValueResult {
    match a1 {
        Int(i) => Ok(Int(i.abs())),
        v => TypeErrorArgMustBeInt(v.clone()).err(),
    }
}