use num_integer::Roots;

use crate::vm::{Value, RuntimeError};

use RuntimeError::{*};
use Value::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;

pub fn convert_to_int(a1: Value, a2: Option<Value>) -> ValueResult {
    match &a1 {
        Nil => Ok(Int(0)),
        Bool(b) => Ok(Int(if *b { 1 } else { 0 })),
        Int(_) => Ok(a1),
        Str(s) => match s.parse::<i64>() {
            Ok(i) => Ok(Int(i)),
            Err(_) => match a2 {
                Some(a2) => Ok(a2),
                None => TypeErrorCannotConvertToInt(a1).err(),
            },
        },
        _ => TypeErrorCannotConvertToInt(a1).err(),
    }
}

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

pub fn gcd(a1: impl Iterator<Item=Value>) -> ValueResult {
    a1.map(|v| v.as_int())
        .collect::<Result<Vec<i64>, Box<RuntimeError>>>()?
        .into_iter()
        .reduce(|a, b| num_integer::gcd(a, b))
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| Ok(Int(v)))
}

pub fn lcm(a1: impl Iterator<Item=Value>) -> ValueResult {
    a1.map(|v| v.as_int())
        .collect::<Result<Vec<i64>, Box<RuntimeError>>>()?
        .into_iter()
        .reduce(|a, b| num_integer::lcm(a, b))
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| Ok(Int(v)))
}

