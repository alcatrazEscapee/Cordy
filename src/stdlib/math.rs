use num_integer::Roots;

use crate::vm::{Value, RuntimeError};

use RuntimeError::{*};
use Value::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;

pub fn convert_to_int(target: Value, default: Option<Value>) -> ValueResult {
    match &target {
        Nil => Ok(Int(0)),
        Bool(b) => Ok(Int(if *b { 1 } else { 0 })),
        Int(_) => Ok(target),
        Str(s) => match s.parse::<i64>() {
            Ok(i) => Ok(Int(i)),
            Err(_) => match default {
                Some(a2) => Ok(a2),
                None => TypeErrorCannotConvertToInt(target).err(),
            },
        },
        _ => TypeErrorCannotConvertToInt(target).err(),
    }
}

pub fn abs(value: Value) -> ValueResult {
    Ok(Int(value.as_int()?.abs()))
}

pub fn sqrt(value: Value) -> ValueResult {
    let i = value.as_int()?;
    if i < 0 {
        ValueErrorValueMustBeNonNegative(i).err()
    } else {
        Ok(Int(i.sqrt()))
    }
}

pub fn gcd(args: impl Iterator<Item=Value>) -> ValueResult {
    args.map(|v| v.as_int())
        .collect::<Result<Vec<i64>, Box<RuntimeError>>>()?
        .into_iter()
        .reduce(|a, b| num_integer::gcd(a, b))
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| Ok(Int(v)))
}

pub fn lcm(args: impl Iterator<Item=Value>) -> ValueResult {
    args.map(|v| v.as_int())
        .collect::<Result<Vec<i64>, Box<RuntimeError>>>()?
        .into_iter()
        .reduce(|a, b| num_integer::lcm(a, b))
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| Ok(Int(v)))
}

pub fn count_ones(value: Value) -> ValueResult {
    Ok(Int(value.as_int()?.count_ones() as i64))
}

pub fn count_zeros(value: Value) -> ValueResult {
    Ok(Int(value.as_int()?.count_zeros() as i64))
}

