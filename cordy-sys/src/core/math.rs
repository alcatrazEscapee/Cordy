use num_integer::Roots;

use crate::vm::{ErrorResult, IntoValue, RuntimeError, Type, ValueOption, ValuePtr, ValueResult};

use RuntimeError::{*};


pub fn convert_to_int(target: ValuePtr, default: ValueOption) -> ValueResult {
    match target.ty() {
        Type::Nil => 0i64.to_value().ok(),
        Type::Bool => target.as_int().to_value().ok(),
        Type::Int => target.ok(),
        Type::ShortStr | Type::LongStr => match target.as_str_slice().parse::<i64>() {
            Ok(i) => i.to_value().ok(),
            Err(_) => match default.as_option() {
                Some(a2) => a2.ok(),
                None => TypeErrorCannotConvertToInt(target).err(),
            },
        },
        _ => TypeErrorCannotConvertToInt(target).err(),
    }
}

pub fn abs(value: ValuePtr) -> ValueResult {
    value.check_int()?
        .as_int()
        .abs()
        .to_value()
        .ok()
}

pub fn sqrt(value: ValuePtr) -> ValueResult {
    let i = value.check_int()?.as_int();
    if i < 0 {
        ValueErrorValueMustBeNonNegative(i).err()
    } else {
        i.sqrt().to_value().ok()
    }
}

pub fn gcd(args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    args.map(|v| v.check_int())
        .collect::<ErrorResult<Vec<ValuePtr>>>()?
        .into_iter()
        .map(|u| u.as_int())
        .reduce(num_integer::gcd)
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| v.to_value().ok())
}

pub fn lcm(args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    args.map(|v| v.check_int())
        .collect::<ErrorResult<Vec<ValuePtr>>>()?
        .into_iter()
        .map(|u| u.as_int())
        .reduce(num_integer::lcm)
        .map_or_else(|| ValueErrorValueMustBeNonEmpty.err(), |v| v.to_value().ok())
}

pub fn count_ones(value: ValuePtr) -> ValueResult {
    (value.check_int()?
        .as_int()
        .count_ones() as i64)
        .to_value()
        .ok()
}

pub fn count_zeros(value: ValuePtr) -> ValueResult {
    (value.check_int()?
        .as_int()
        .count_zeros() as i64)
        .to_value()
        .ok()
}

pub fn get_real(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Complex => value.as_precise_complex().value.inner.re.to_value().ok(),
        Type::Bool | Type::Int => value.as_int().to_value().ok(),
        _ => TypeErrorArgMustBeComplex(value).err()
    }
}

pub fn get_imag(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Complex => value.as_precise_complex().value.inner.im.to_value().ok(),
        Type::Bool | Type::Int => 0i64.to_value().ok(),
        _ => TypeErrorArgMustBeComplex(value).err()
    }
}
