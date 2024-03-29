use std::collections::VecDeque;

#[cfg(feature = "rational")] use rug::ops::Pow;

use crate::core;
use crate::core::NativeFunction;
use crate::vm::{ComplexType, ErrorPtr, ErrorResult, Type, ValuePtr, ValueResult};
use crate::vm::error::RuntimeError;
use crate::vm::value::{IntoIterableValue, IntoValue};

use RuntimeError::{*};
use Type::{*};


#[repr(u8)]
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum UnaryOp {
    Neg, Not, LogicalNot
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum BinaryOp {
    Mul, Div, Pow, Mod, Is, IsNot, Add, Sub, LeftShift, RightShift, And, Or, Xor, In, NotIn, LessThan, GreaterThan, LessThanEqual, GreaterThanEqual, Equal, NotEqual, Max, Min
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CompareOp {
    LessThan, GreaterThan, LessThanEqual, GreaterThanEqual, Equal, NotEqual
}

impl UnaryOp {
    pub fn apply(self, arg: ValuePtr) -> ValueResult {
        match self {
            UnaryOp::Neg => unary_sub(arg),
            UnaryOp::Not => unary_not(arg),
            UnaryOp::LogicalNot => unary_logical_not(arg)
        }
    }
}

impl BinaryOp {
    pub fn apply(self, lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
        match self {
            BinaryOp::Mul => binary_mul(lhs, rhs),
            BinaryOp::Div => binary_div(lhs, rhs),
            BinaryOp::Pow => binary_pow(lhs, rhs),
            BinaryOp::Mod => binary_mod(lhs, rhs),
            BinaryOp::Is => binary_is(lhs, rhs, false),
            BinaryOp::IsNot => binary_is(lhs, rhs, true),
            BinaryOp::Add => binary_add(lhs, rhs),
            BinaryOp::Sub => binary_sub(lhs, rhs),
            BinaryOp::LeftShift => binary_left_shift(lhs, rhs),
            BinaryOp::RightShift => binary_right_shift(lhs, rhs),
            BinaryOp::And => binary_bitwise_and(lhs, rhs),
            BinaryOp::Or => binary_bitwise_or(lhs, rhs),
            BinaryOp::Xor => binary_bitwise_xor(lhs, rhs),
            BinaryOp::In => binary_in(lhs, rhs, false),
            BinaryOp::NotIn => binary_in(lhs, rhs, true),
            BinaryOp::LessThan => (lhs < rhs).to_value().ok(),
            BinaryOp::GreaterThan => (lhs > rhs).to_value().ok(),
            BinaryOp::LessThanEqual => (lhs <= rhs).to_value().ok(),
            BinaryOp::GreaterThanEqual => (lhs >= rhs).to_value().ok(),
            BinaryOp::Equal => (lhs == rhs).to_value().ok(),
            BinaryOp::NotEqual => (lhs != rhs).to_value().ok(),
            BinaryOp::Max => std::cmp::max(lhs, rhs).ok(),
            BinaryOp::Min => std::cmp::min(lhs, rhs).ok(),
        }
    }
}

impl From<CompareOp> for BinaryOp {
    fn from(value: CompareOp) -> Self {
        match value {
            CompareOp::LessThan => BinaryOp::LessThan,
            CompareOp::LessThanEqual => BinaryOp::LessThanEqual,
            CompareOp::GreaterThan => BinaryOp::GreaterThan,
            CompareOp::GreaterThanEqual => BinaryOp::GreaterThanEqual,
            CompareOp::Equal => BinaryOp::Equal,
            CompareOp::NotEqual => BinaryOp::NotEqual,
        }
    }
}

impl CompareOp {
    pub fn apply(self, lhs: &ValuePtr, rhs: &ValuePtr) -> bool {
        match self {
            CompareOp::LessThan => lhs < rhs,
            CompareOp::GreaterThan => lhs > rhs,
            CompareOp::LessThanEqual => lhs <= rhs,
            CompareOp::GreaterThanEqual => lhs >= rhs,
            CompareOp::Equal => lhs == rhs,
            CompareOp::NotEqual => lhs != rhs,
        }
    }
}

impl From<BinaryOp> for CompareOp {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::LessThan => CompareOp::LessThan,
            BinaryOp::LessThanEqual => CompareOp::LessThanEqual,
            BinaryOp::GreaterThan => CompareOp::GreaterThan,
            BinaryOp::GreaterThanEqual => CompareOp::GreaterThanEqual,
            BinaryOp::Equal => CompareOp::Equal,
            BinaryOp::NotEqual => CompareOp::NotEqual,
            op => panic!("Cannot convert {:?} to a CompareOp", op)
        }
    }
}


pub fn unary_sub(a1: ValuePtr) -> ValueResult {
    match a1.ty() {
        Bool | Int => (-a1.as_int()).to_value().ok(),
        Complex => (-a1.as_complex()).to_value().ok(),
        #[cfg(feature = "rational")]
        Rational => (-a1.as_rational().clone()).to_value().ok(),
        Vector => apply_vector_unary(a1, unary_sub),
        _ => TypeErrorUnaryOp(UnaryOp::Neg, a1).err(),
    }
}

pub fn unary_not(a1: ValuePtr) -> ValueResult {
    match a1.ty() {
        Bool => (!a1.as_bool()).to_value().ok(),
        Int => (!a1.as_int()).to_value().ok(),
        Complex => a1.as_complex().conj().to_value().ok(),
        Vector => apply_vector_unary(a1, unary_not),
        _ => TypeErrorUnaryOp(UnaryOp::Not, a1).err(),
    }
}

pub fn unary_logical_not(a1: ValuePtr) -> ValueResult {
    (!a1.to_bool()).to_value().ok()
}


pub fn binary_mul(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => (lhs.as_int() * rhs.as_int()).to_value().ok(),
        (Bool | Int | Complex, Bool | Int | Complex) => (lhs.to_complex() * rhs.to_complex()).to_value().ok(),
        #[cfg(feature = "rational")]
        (Bool | Int | Rational, Bool | Int | Rational) => (lhs.to_rational() * rhs.to_rational()).to_value().ok(),
        (ShortStr | LongStr, Int) => binary_str_repeat(lhs, rhs),
        (Int, ShortStr | LongStr) => binary_str_repeat(rhs, lhs),
        (List, Int) => binary_list_repeat(lhs, rhs, false),
        (Int, List) => binary_list_repeat(rhs, lhs, false),
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_mul),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_mul),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_mul),
        _ => TypeErrorBinaryOp(BinaryOp::Mul, lhs, rhs).err()
    }
}

fn binary_str_repeat(string: ValuePtr, repeat: ValuePtr) -> ValueResult {
    let i = repeat.as_int();
    if i < 0 {
        ValueErrorBinaryOpMustNotBeNegative(string, BinaryOp::Mul, i).err()
    } else {
        string.as_str_slice().repeat(i as usize).to_value().ok()
    }
}

fn binary_list_repeat(list: ValuePtr, repeat: ValuePtr, deep: bool) -> ValueResult {
    let i = repeat.as_int();
    if i < 0 {
        ValueErrorBinaryOpMustNotBeNegative(list, if deep { BinaryOp::Pow } else { BinaryOp::Mul }, i).err()
    } else {
        let list = list.as_list().borrow();
        let iter = list.iter()
            .cycle()
            .take(i as usize * list.len())
            .cloned();

        match deep {
            true => iter.map(core::copy).to_list(),
            false => iter.to_list()
        }.ok()
    }
}

pub fn binary_div(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => {
            match rhs.as_int() == 0 {
                true => ValueErrorDivideByZero.err(),
                false => num_integer::div_floor(lhs.as_int(), rhs.as_int()).to_value().ok()
            }
        },
        (Bool | Int | Complex, Bool | Int | Complex) => {
            let rhs = rhs.to_complex();
            match rhs.norm_sqr() == 0 {
                true => ValueErrorDivideByZero.err(),
                false => c64_div_floor(lhs.to_complex(), rhs).to_value().ok()
            }
        },
        #[cfg(feature = "rational")]
        (Bool | Int | Rational, Bool | Int | Rational) => {
            let rhs = rhs.to_rational();
            match rhs.is_zero() {
                true => ValueErrorDivideByZero.err(),
                false => (lhs.to_rational() / rhs).to_value().ok(),
            }
        }
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_div),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_div),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_div),
        _ => TypeErrorBinaryOp(BinaryOp::Div, lhs, rhs).err()
    }
}

/// The `C64` type provided by `num-complex` defines `div()` using regular rust division.
/// This is a clone of that but using `floor_div` provided by `num-integer`, which keeps consistency with how we define division for `complex / int`
#[inline]
fn c64_div_floor(lhs: ComplexType, rhs: ComplexType) -> ComplexType {
    let norm_sqr = rhs.norm_sqr();
    let re = lhs.re * rhs.re + lhs.im * rhs.im;
    let im = lhs.im * rhs.re - lhs.re * rhs.im;
    ComplexType::new(num_integer::div_floor(re, norm_sqr), num_integer::div_floor(im, norm_sqr))
}

pub fn binary_mod(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => {
            let rhs = rhs.as_int();
            match rhs == 0 {
                true => ValueErrorModuloByZero.err(),
                false => num_integer::mod_floor(lhs.as_int(), rhs).to_value().ok()
            }
        },
        (Complex, Bool | Int) => {
            let rhs = rhs.as_int();
            match rhs == 0 {
                true => ValueErrorModuloByZero.err(),
                false => {
                    let lhs = lhs.as_complex();
                    ComplexType::new(num_integer::mod_floor(lhs.re, rhs), num_integer::mod_floor(lhs.im, rhs)).to_value().ok()
                }
            }
        }
        (ShortStr | LongStr, _) => core::format_string(lhs.as_str_slice(), rhs),
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_mod),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_mod),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_mod),
        _ => TypeErrorBinaryOp(BinaryOp::Mod, lhs, rhs).err()
    }
}

pub fn binary_pow(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => {
            let rhs = rhs.as_int();
            match rhs >= 0 {
                true => lhs.as_int().pow(rhs as u32).to_value().ok(),
                false => ValueErrorPowerByNegative(rhs).err()
            }
        },
        (Complex, Bool | Int) => {
            let rhs = rhs.as_int();
            match rhs >= 0 {
                true => lhs.as_complex().powu(rhs as u32).to_value().ok(),
                false => ValueErrorPowerByNegative(rhs).err()
            }
        },
        #[cfg(feature = "rational")]
        (Rational, Bool | Int) => {
            let lhs = lhs.to_rational();
            match i32::try_from(rhs.as_int()) {
                Ok(rhs) => lhs.pow(rhs).to_value().ok(),
                Err(_) => ValueErrorValueWouldBeTooLarge.err()
            }
        }
        (List, Int) => binary_list_repeat(lhs, rhs, true),
        (Int, List) => binary_list_repeat(rhs, lhs, true),
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_pow),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_pow),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_pow),
        _ => TypeErrorBinaryOp(BinaryOp::Pow, lhs, rhs).err()
    }
}

/// When `invert = true`, this is a `binary_is_not` operator
pub fn binary_is(lhs: ValuePtr, rhs: ValuePtr, invert: bool) -> ValueResult {
    (match rhs.ty() {
        Nil => lhs.is_nil(),
        StructType => lhs.is_struct() && lhs.as_struct().borrow().is_instance_of(rhs.as_struct_type()),
        NativeFunction => match rhs.as_native() {
            NativeFunction::Bool => lhs.is_bool(),
            NativeFunction::Int => lhs.is_int(),
            NativeFunction::Complex => lhs.is_complex(),
            NativeFunction::Rational => lhs.is_rational(),
            NativeFunction::Str => lhs.is_str(),
            NativeFunction::Function => lhs.is_evaluable(),
            NativeFunction::List => lhs.is_list(),
            NativeFunction::Set => lhs.is_set(),
            NativeFunction::Dict => lhs.is_dict(),
            NativeFunction::Vector => lhs.is_vector(),
            NativeFunction::Iterable => lhs.is_iterable(),
            NativeFunction::Heap => lhs.is_heap(),
            NativeFunction::Any => true,
            _ => return TypeErrorBinaryIs(lhs, rhs).err()
        },
        _ => return TypeErrorBinaryIs(lhs, rhs).err()
    } != invert).to_value().ok()
}

/// When `invert = true`, this is a `binary_not_in` operator
pub fn binary_in(lhs: ValuePtr, rhs: ValuePtr, invert: bool) -> ValueResult {
    (match (lhs.ty(), rhs.ty()) {
        (ShortStr | LongStr, ShortStr | LongStr) => rhs.as_str_slice().contains(lhs.as_str_slice()),
        (Int | Bool, Range) => rhs.as_range().contains(lhs.as_int()),
        (_, List) => rhs.as_list().borrow().contains(&lhs),
        (_, Set) => rhs.as_set().borrow().contains(&lhs),
        (_, Dict) => rhs.as_dict().borrow().contains_key(&lhs),
        (_, Heap) => rhs.as_heap().borrow().iter().any(|v|v.0 == lhs),
        (_, Vector) => rhs.as_vector().borrow().contains(&lhs),
        _ => return TypeErrorBinaryOp(BinaryOp::In, lhs, rhs).err()
    } != invert).to_value().ok()
}

pub fn binary_add(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => (lhs.as_int() + rhs.as_int()).to_value().ok(),
        (Bool | Int | Complex, Bool | Int | Complex) => (lhs.to_complex() + rhs.to_complex()).to_value().ok(),
        #[cfg(feature = "rational")]
        (Bool | Int | Rational, Bool | Int | Rational) => (lhs.to_rational() + rhs.to_rational()).to_value().ok(),
        (List, List) => {
            let lhs = lhs.as_list().borrow();
            let rhs = rhs.as_list().borrow();
            let mut ret: VecDeque<ValuePtr> = VecDeque::with_capacity(lhs.len() + rhs.len());
            ret.extend(lhs.iter().cloned());
            ret.extend(rhs.iter().cloned());
            ret.to_value().ok()
        }
        (ShortStr | LongStr, _) => format!("{}{}", lhs.as_str_slice(), rhs.to_str()).to_value().ok(),
        (_, ShortStr | LongStr) => format!("{}{}", lhs.to_str(), rhs.as_str_slice()).to_value().ok(),
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_add),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_add),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_add),
        _ => TypeErrorBinaryOp(BinaryOp::Add, lhs, rhs).err(),
    }
}

pub fn binary_sub(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => (lhs.as_int() - rhs.as_int()).to_value().ok(),
        (Bool | Int | Complex, Bool | Int | Complex) => (lhs.to_complex() - rhs.to_complex()).to_value().ok(),
        #[cfg(feature = "rational")]
        (Bool | Int | Rational, Bool | Int | Rational) => (lhs.to_rational() - rhs.to_rational()).to_value().ok(),
        (Set, Set) => {
            let lhs = lhs.as_set().borrow();
            let rhs = rhs.as_set().borrow();
            lhs.difference(&rhs).cloned().to_set().ok()
        },
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_sub),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_sub),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_sub),
        _ => TypeErrorBinaryOp(BinaryOp::Sub, lhs, rhs).err()
    }
}

/// Left shifts by negative values are defined as right shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_left_shift(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => i64_left_shift(lhs.as_int(), rhs.as_int()).to_value().ok(),
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_left_shift),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_left_shift),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_left_shift),
        _ => TypeErrorBinaryOp(BinaryOp::LeftShift, lhs, rhs).err(),
    }
}

/// Right shifts by negative values are defined as left shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_right_shift(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool | Int, Bool | Int) => i64_left_shift(lhs.as_int(), -rhs.as_int()).to_value().ok(),
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_right_shift),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_right_shift),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_right_shift),
        _ => TypeErrorBinaryOp(BinaryOp::RightShift, lhs, rhs).err(),
    }
}

/// Performs a binary shift either left or right, based on the sign of `rhs`. Positive values shift left, negative values shift right.
#[inline]
fn i64_left_shift(lhs: i64, rhs: i64) -> i64 {
    if rhs >= 0 {
        lhs << rhs
    } else {
        lhs >> (-rhs)
    }
}


pub fn binary_bitwise_and(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool, Bool) => (lhs.as_bool() & rhs.as_bool()).to_value().ok(),
        (Bool | Int, Bool | Int) => (lhs.as_int() & rhs.as_int()).to_value().ok(),
        (Set, Set) => {
            let lhs = lhs.as_set().borrow();
            let rhs = rhs.as_set().borrow();
            lhs.intersection(&rhs).cloned().to_set().ok()
        }
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_bitwise_and),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_bitwise_and),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_bitwise_and),
        _ => TypeErrorBinaryOp(BinaryOp::And, lhs, rhs).err()
    }
}

pub fn binary_bitwise_or(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool, Bool) => (lhs.as_bool() | rhs.as_bool()).to_value().ok(),
        (Bool | Int, Bool | Int) => (lhs.as_int() | rhs.as_int()).to_value().ok(),
        (Set, Set) => {
            let lhs = lhs.as_set().borrow();
            let rhs = rhs.as_set().borrow();
            lhs.union(&rhs).cloned().to_set().ok()
        },
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_bitwise_or),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_bitwise_or),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_bitwise_or),
        _ => TypeErrorBinaryOp(BinaryOp::Or, lhs, rhs).err()
    }
}

pub fn binary_bitwise_xor(lhs: ValuePtr, rhs: ValuePtr) -> ValueResult {
    match (lhs.ty(), rhs.ty()) {
        (Bool, Bool) => (lhs.as_bool() ^ rhs.as_bool()).to_value().ok(),
        (Bool | Int, Bool | Int) => (lhs.as_int() ^ rhs.as_int()).to_value().ok(),
        (Set, Set) => {
            let lhs = lhs.as_set().borrow();
            let rhs = rhs.as_set().borrow();
            lhs.symmetric_difference(&rhs).cloned().to_set().ok()
        }
        (Vector, Vector) => apply_vector_binary(lhs, rhs, binary_bitwise_xor),
        (Vector, _) => apply_vector_binary_scalar_rhs(lhs, rhs, binary_bitwise_xor),
        (_, Vector) => apply_vector_binary_scalar_lhs(lhs, rhs, binary_bitwise_xor),
        _ => TypeErrorBinaryOp(BinaryOp::Xor, lhs, rhs).err()
    }
}


type UnaryFn = fn(ValuePtr) -> ValueResult;
type BinaryFn = fn(ValuePtr, ValuePtr) -> ValueResult;

/// Helpers for `Vector` operations, which all apply elementwise
pub fn apply_vector_unary(vector: ValuePtr, unary_op: UnaryFn) -> ValueResult {
    vector.as_vector().borrow().iter()
        .map(|v| unary_op(v.clone()))
        .collect::<Result<Vec<ValuePtr>, ErrorPtr>>()?
        .to_value()
        .ok()
}

fn apply_vector_binary(lhs: ValuePtr, rhs: ValuePtr, binary_op: BinaryFn) -> ValueResult {
    lhs.as_vector().borrow().iter()
        .zip(rhs.as_vector().borrow().iter())
        .map(|(l, r)| binary_op(l.clone(), r.clone()))
        .collect::<Result<Vec<ValuePtr>, ErrorPtr>>()?
        .to_value()
        .ok()
}

fn apply_vector_binary_scalar_lhs(scalar_lhs: ValuePtr, rhs: ValuePtr, binary_op: BinaryFn) -> ValueResult {
    rhs.as_vector().borrow().iter()
        .map(|r| binary_op(scalar_lhs.clone(), r.clone()))
        .collect::<ErrorResult<Vec<ValuePtr>>>()?
        .to_value()
        .ok()
}

fn apply_vector_binary_scalar_rhs(lhs: ValuePtr, scalar_rhs: ValuePtr, binary_op: BinaryFn) -> ValueResult {
    lhs.as_vector().borrow().iter()
        .map(|l| binary_op(l.clone(), scalar_rhs.clone()))
        .collect::<ErrorResult<Vec<ValuePtr>>>()?
        .to_value()
        .ok()
}


#[cfg(test)]
mod test {
    use crate::vm::{ComplexType, IntoValue, operator, RuntimeError};

    #[test]
    fn test_binary_int_div() {
        for (a, b, c) in vec![
            (1i64, 3i64, 0i64), (2, 3, 0), (3, 3, 1), (4, 3, 1), (5, 3, 1), (6, 3, 2), (7, 3, 2), // + / +
            (1, -3, -1), (2, -3, -1), (3, -3, -1), (4, -3, -2), (5, -3, -2), (6, -3, -2), (7, -3, -3), // + / -
            (-1, 3, -1), (-2, 3, -1), (-3, 3, -1), (-4, 3, -2), (-5, 3, -2), (-6, 3, -2), (-7, 3, -3), // - / +
            (-1, -3, 0), (-2, -3, 0), (-3, -3, 1), (-4, -3, 1), (-5, -3, 1), (-6, -3, 2), (-7, -3, 2) // - / -
        ] {
            assert_eq!(operator::binary_div(a.to_value(), b.to_value()), c.to_value().ok(), "{} / {}", a, b)
        }

        for a in -5i64..=-5 {
            assert_eq!(operator::binary_div(a.to_value(), 0i64.to_value()), RuntimeError::ValueErrorDivideByZero.err())
        }
    }

    #[test]
    fn test_binary_complex_by_int_div() {
        for ((a, ai), b, (c, ci)) in vec![
            ((3, 3), 2i64, (1, 1)), ((2, 3), 2, (1, 1)), ((1, 3), 2, (0, 1)), ((0, 3), 2, (0, 1)), ((-1, 3), 2, (-1, 1)), ((-2, 3), 2, (-1, 1)), ((-3, 3), 2, (-2, 1)),
            ((3, 2), 2, (1, 1)), ((2, 2), 2, (1, 1)), ((1, 2), 2, (0, 1)), ((0, 2), 2, (0, 1)), ((-1, 2), 2, (-1, 1)), ((-2, 2), 2, (-1, 1)), ((-3, 2), 2, (-2, 1)),
            ((3, 1), 2, (1, 0)), ((2, 1), 2, (1, 0)), ((1, 1), 2, (0, 0)), ((0, 1), 2, (0, 0)), ((-1, 1), 2, (-1, 0)), ((-2, 1), 2, (-1, 0)), ((-3, 1), 2, (-2, 0)),
            ((3, 0), 2, (1, 0)), ((2, 0), 2, (1, 0)), ((1, 0), 2, (0, 0)), ((0, 0), 2, (0, 0)), ((-1, 0), 2, (-1, 0)), ((-2, 0), 2, (-1, 0)), ((-3, 0), 2, (-2, 0)),
            ((3, 1), 2, (1, 0)), ((2, 1), 2, (1, 0)), ((1, 1), 2, (0, 0)), ((0, 1), 2, (0, 0)), ((-1, 1), 2, (-1, 0)), ((-2, 1), 2, (-1, 0)), ((-3, 1), 2, (-2, 0)),
            ((3, 2), 2, (1, 1)), ((2, 2), 2, (1, 1)), ((1, 2), 2, (0, 1)), ((0, 2), 2, (0, 1)), ((-1, 2), 2, (-1, 1)), ((-2, 2), 2, (-1, 1)), ((-3, 2), 2, (-2, 1)),
            ((3, 3), 2, (1, 1)), ((2, 3), 2, (1, 1)), ((1, 3), 2, (0, 1)), ((0, 3), 2, (0, 1)), ((-1, 3), 2, (-1, 1)), ((-2, 3), 2, (-1, 1)), ((-3, 3), 2, (-2, 1)),
        ] {
            assert_eq!(operator::binary_div(ComplexType::new(a, ai).to_value(), b.to_value()), ComplexType::new(c, ci).to_value().ok(), "{:?} / {}", (a, ai), b)
        }
    }

    #[test]
    fn test_binary_int_mod() {
        for (a, b, c) in vec![
            (1i64, 3i64, 1i64), (2, 3, 2), (3, 3, 0), (4, 3, 1), (5, 3, 2), (6, 3, 0), (7, 3, 1), // + % +
            (1, -3, -2), (2, -3, -1), (3, -3, 0), (4, -3, -2), (5, -3, -1), (6, -3, 0), (7, -3, -2), // + % -
            (-1, 3, 2), (-2, 3, 1), (-3, 3, 0), (-4, 3, 2), (-5, 3, 1), (-6, 3, 0), (-7, 3, 2), // - % +
            (-1, -3, -1), (-2, -3, -2), (-3, -3, 0), (-4, -3, -1), (-5, -3, -2), (-6, -3, 0), (-7, -3, -1), // - % -
        ] {
            assert_eq!(operator::binary_mod(a.to_value(), b.to_value()), c.to_value().ok(), "{} % {}", a, b);
        }
    }
}