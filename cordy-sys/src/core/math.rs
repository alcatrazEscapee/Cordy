use num_integer::Roots;

use crate::vm::{ComplexType, IntoValue, operator, RuntimeError, Type, ValuePtr, ValueResult};

use RuntimeError::{*};


pub fn convert_to_int(target: ValuePtr, default: Option<ValuePtr>) -> ValueResult {
    match target.ty() {
        Type::Nil => 0i64.to_value().ok(),
        Type::Bool => target.as_int().to_value().ok(),
        Type::Int => target.ok(),
        Type::ShortStr | Type::LongStr => match target.as_str_slice().parse::<i64>() {
            Ok(i) => i.to_value().ok(),
            Err(_) => match default {
                Some(a2) => a2.ok(),
                None => TypeErrorCannotConvertToInt(target).err(),
            },
        },
        _ => TypeErrorCannotConvertToInt(target).err(),
    }
}

pub fn convert_to_rational(numer: ValuePtr, denom: Option<ValuePtr>) -> ValueResult {
    match numer.is_rational() {
        true => match denom {
            Some(denom) if denom.is_rational() => {
                let denom = denom.to_rational();
                match denom.is_zero() {
                    true => ValueErrorValueMustBeNonZero.err(),
                    false => (numer.to_rational() / denom).to_value().ok()
                }
            },
            Some(denom) => TypeErrorCannotConvertToRational(denom).err(),
            None => numer.to_rational().to_value().ok(),
        }
        false => TypeErrorCannotConvertToRational(numer).err(),
    }
}

pub fn abs(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Bool | Type::Int => value.as_int().abs().to_value().ok(),
        Type::Complex => {
            let c = value.as_complex();
            ComplexType::new(c.re.abs(), c.im.abs()).to_value().ok()
        },
        Type::Vector => {
            operator::apply_vector_unary(value, abs)
        }
        _ => TypeErrorCannotConvertToInt(value).err()
    }
}

pub fn sqrt(value: ValuePtr) -> ValueResult {
    let i = value.as_int_checked()?;
    if i < 0 {
        ValueErrorValueMustBeNonNegative(i).err()
    } else {
        i.sqrt().to_value().ok()
    }
}

pub fn gcd(mut args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    let mut acc = match args.next() {
        Some(it) => it.as_int_checked()?,
        None => return ValueErrorValueMustBeNonEmpty.err()
    };

    for arg in args {
        acc = num_integer::gcd(acc, arg.as_int_checked()?);
    }

    acc.to_value().ok()
}

pub fn lcm(mut args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    let mut acc = match args.next() {
        Some(it) => it.as_int_checked()?,
        None => return ValueErrorValueMustBeNonEmpty.err()
    };

    for arg in args {
        acc = num_integer::lcm(acc, arg.as_int_checked()?);
    }

    acc.to_value().ok()
}

pub fn count_ones(value: ValuePtr) -> ValueResult {
    (value.as_int_checked()?.count_ones() as i64)
        .to_value()
        .ok()
}

pub fn count_zeros(value: ValuePtr) -> ValueResult {
    (value.as_int_checked()?.count_zeros() as i64)
        .to_value()
        .ok()
}

pub fn get_real(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Complex => value.as_complex().re.to_value().ok(),
        Type::Bool | Type::Int => value.as_int().to_value().ok(),
        _ => TypeErrorArgMustBeComplex(value).err()
    }
}

pub fn get_imag(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Complex => value.as_complex().im.to_value().ok(),
        Type::Bool | Type::Int => 0i64.to_value().ok(),
        _ => TypeErrorArgMustBeComplex(value).err()
    }
}
