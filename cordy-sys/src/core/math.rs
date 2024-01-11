use num_integer::Roots;

#[cfg(feature = "rational")] use rug::integer::SmallInteger;
#[cfg(feature = "rational")] use rug::{Complete, Rational};

#[cfg(feature = "rational")] use crate::vm::RationalType;

use crate::vm::{ComplexType, IntoValue, operator, RuntimeError, Type, ValuePtr, ValueResult};
use crate::core::NativeFunction::{Gcd, Lcm};

use RuntimeError::{*};


pub fn convert_to_int(target: ValuePtr, default: Option<ValuePtr>) -> ValueResult {
    match target.ty() {
        Type::Nil => 0i64.to_value().ok(),
        Type::Bool => target.as_int().to_value().ok(),
        Type::Int => target.ok(),
        Type::ShortStr | Type::LongStr => match target.as_str_slice().parse::<i64>() {
            Ok(i) => i.to_value().ok(),
            Err(err) => match default {
                Some(a2) => a2.ok(),
                None => ValueErrorCannotConvertStringToIntBadParse(target, err.to_string()).err(),
            },
        },
        #[cfg(feature = "rational")]
        Type::Rational => match target.as_rational().denom() == &SmallInteger::from(1) {
            true => match target.as_rational().numer().to_i64() {
                Some(value) if ValuePtr::MIN_INT <= value && value <= ValuePtr::MAX_INT => value.to_value().ok(),
                _ => ValueErrorCannotConvertRationalToIntTooLarge(target).err()
            },
            false => ValueErrorCannotConvertRationalToIntNotAnInteger(target).err()
        }
        _ => TypeErrorCannotConvertToInt(target).err(),
    }
}

#[cfg(feature = "rational")]
pub fn convert_to_rational(numer: ValuePtr, denom: Option<ValuePtr>) -> ValueResult {
    match numer.ty() {
        Type::Bool | Type::Int | Type::Rational => match denom {
            Some(denom) if denom.is_rational() => {
                let denom = denom.to_rational();
                match denom.is_zero() {
                    true => ValueErrorDivideByZero.err(),
                    false => (numer.to_rational() / denom).to_value().ok()
                }
            },
            Some(denom) => TypeErrorCannotConvertToRational(denom).err(),
            None => numer.to_rational().to_value().ok(),
        }
        Type::ShortStr | Type::LongStr => match Rational::parse(numer.as_str_slice()) {
            Ok(value) => value.complete().to_value().ok(),
            Err(err) => ValueErrorCannotConvertStringToRationalBadParse(numer, err.to_string()).err()
        }
        _ => TypeErrorCannotConvertToRational(numer).err(),
    }
}

pub fn abs(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Bool | Type::Int => value.as_int().abs().to_value().ok(),
        Type::Complex => {
            let c = value.as_complex();
            ComplexType::new(c.re.abs(), c.im.abs()).to_value().ok()
        },
        #[cfg(feature = "rational")]
        Type::Rational => value.as_rational().clone().abs().to_value().ok(),
        Type::Vector => {
            operator::apply_vector_unary(value, abs)
        }
        _ => TypeErrorCannotConvertToInt(value).err()
    }
}

pub fn sqrt(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Bool => value.as_int().to_value().ok(),
        Type::Int => {
            let n = value.as_precise_int();
            match n < 0 {
                true => ValueErrorValueMustBeNonNegative(value).err(),
                false => n.sqrt().to_value().ok()
            }
        },
        #[cfg(feature = "rational")]
        Type::Rational => {
            let n = value.as_rational();
            match n.is_negative() {
                true => ValueErrorValueMustBeNonNegative(value).err(),
                false => Rational::from((n.numer().clone().sqrt(), n.denom().clone().sqrt())).to_value().ok()
            }
        },
        _ => TypeErrorArgMustBeRational(value).err()
    }
}

pub fn gcd(mut args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    let mut acc = match args.next() {
        Some(it) => it.as_int_checked()?,
        None => return ValueErrorArgMustBeNonEmpty(Gcd).err()
    };

    for arg in args {
        acc = num_integer::gcd(acc, arg.as_int_checked()?);
    }

    acc.to_value().ok()
}

pub fn lcm(mut args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    let mut acc = match args.next() {
        Some(it) => it.as_int_checked()?,
        None => return ValueErrorArgMustBeNonEmpty(Lcm).err()
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

#[cfg(feature = "rational")]
pub fn get_numer(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Bool | Type::Int | Type::Rational => RationalType::from(value.to_rational().numer()).to_value().ok(),
        _ => TypeErrorArgMustBeRational(value).err()
    }
}

#[cfg(feature = "rational")]
pub fn get_denom(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Bool | Type::Int | Type::Rational => RationalType::from(value.to_rational().denom()).to_value().ok(),
        _ => TypeErrorArgMustBeRational(value).err()
    }
}
