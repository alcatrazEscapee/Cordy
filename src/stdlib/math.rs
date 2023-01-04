use crate::vm::error::RuntimeError;
use crate::vm::value::Value;
use num_integer::Roots;

use RuntimeError::{*};
use Value::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


pub fn abs(a1: Value) -> ValueResult {
    Ok(Int(a1.as_int()?.abs()))
}

pub fn sqrt(a1: Value) -> ValueResult {
    let i = a1.as_int()?;
    if i < 0 {
        ValueErrorValueMustBeNonNegative(i).err()
    } else {
        Ok(Int(i.sqrt()))
    }
}

pub fn gcd<'a>(a1: impl Iterator<Item=&'a Value>) -> ValueResult {
    a1.map(|v| v.as_int())
        .collect::<Result<Vec<i64>, Box<RuntimeError>>>()?
        .into_iter()
        .reduce(|a, b| num_integer::gcd(a, b))
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| Ok(Int(v)))
}

pub fn lcm<'a>(a1: impl Iterator<Item=&'a Value>) -> ValueResult {
    a1.map(|v| v.as_int())
        .collect::<Result<Vec<i64>, Box<RuntimeError>>>()?
        .into_iter()
        .reduce(|a, b| num_integer::lcm(a, b))
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| Ok(Int(v)))
}

