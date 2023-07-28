use std::collections::VecDeque;

use crate::core;
use crate::core::NativeFunction;
use crate::vm::ValueResult;
use crate::vm::error::RuntimeError;
use crate::vm::value::{C64, IntoIterableValue, IntoValue, IntoValueResult, Mut, Value};

use RuntimeError::{*};
use Value::{*};


#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum UnaryOp {
    Neg, Not
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BinaryOp {
    Mul, Div, Pow, Mod, Is, IsNot, Add, Sub, LeftShift, RightShift, And, Or, Xor, In, NotIn, LessThan, GreaterThan, LessThanEqual, GreaterThanEqual, Equal, NotEqual, Max, Min
}

impl UnaryOp {
    pub fn apply(self, arg: Value) -> ValueResult {
        match self {
            UnaryOp::Neg => unary_sub(arg),
            UnaryOp::Not => unary_not(arg),
        }
    }
}

impl BinaryOp {
    pub fn apply(self, lhs: Value, rhs: Value) -> ValueResult {
        match self {
            BinaryOp::Mul => binary_mul(lhs, rhs),
            BinaryOp::Div => binary_div(lhs, rhs),
            BinaryOp::Pow => binary_pow(lhs, rhs),
            BinaryOp::Mod => binary_mod(lhs, rhs),
            BinaryOp::Is => binary_is(lhs, rhs).to_value(),
            BinaryOp::IsNot => binary_is(lhs, rhs).to_value(),
            BinaryOp::Add => binary_add(lhs, rhs),
            BinaryOp::Sub => binary_sub(lhs, rhs),
            BinaryOp::LeftShift => binary_left_shift(lhs, rhs),
            BinaryOp::RightShift => binary_right_shift(lhs, rhs),
            BinaryOp::And => binary_bitwise_and(lhs, rhs),
            BinaryOp::Or => binary_bitwise_or(lhs, rhs),
            BinaryOp::Xor => binary_bitwise_xor(lhs, rhs),
            BinaryOp::In => binary_in(lhs, rhs).to_value(),
            BinaryOp::NotIn => binary_in(lhs, rhs).map(|u| !u).to_value(),
            BinaryOp::LessThan => Ok(Bool(lhs < rhs)),
            BinaryOp::GreaterThan => Ok(Bool(lhs > rhs)),
            BinaryOp::LessThanEqual => Ok(Bool(lhs <= rhs)),
            BinaryOp::GreaterThanEqual => Ok(Bool(lhs >= rhs)),
            BinaryOp::Equal => Ok(Bool(lhs == rhs)),
            BinaryOp::NotEqual => Ok(Bool(lhs != rhs)),
            BinaryOp::Max => Ok(std::cmp::max(lhs, rhs)),
            BinaryOp::Min => Ok(std::cmp::min(lhs, rhs)),
        }
    }
}


pub fn unary_sub(a1: Value) -> ValueResult {
    match a1 {
        Int(it) => Ok(Int(-it)),
        Complex(it) => Ok((-*it).to_value()),
        Vector(it) => apply_vector_unary(it, unary_sub),
        v => TypeErrorUnaryOp(UnaryOp::Neg, v).err(),
    }
}

pub fn unary_not(a1: Value) -> ValueResult {
    match a1 {
        Bool(b1) => Ok(Bool(!b1)),
        Int(i1) => Ok(Int(!i1)),
        Complex(it) => Ok(it.conj().to_value()),
        Vector(it) => apply_vector_unary(it, unary_not),
        v => TypeErrorUnaryOp(UnaryOp::Not, v).err(),
    }
}


pub fn binary_mul(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(l.as_int_unchecked() * r.as_int_unchecked())),
        (l @ (Bool(_) | Int(_) | Complex(_)), r @ (Bool(_) | Int(_) | Complex(_))) => Ok((l.as_complex_unchecked() * r.as_complex_unchecked()).to_value()),
        (Str(s1), Int(i2)) if i2 >= 0 => Ok(s1.repeat(i2 as usize).to_value()),
        (Int(i1), Str(s2)) if i1 >= 0 => Ok(s2.repeat(i1 as usize).to_value()),
        (List(l1), Int(i1)) if i1 > 0 => {
            let l1 = l1.unbox();
            let len: usize = l1.len();
            Ok(l1.iter().cycle().take(i1 as usize * len).cloned().to_list())
        },
        (l @ Int(_), r @ List(_)) => binary_mul(r, l),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_mul),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_mul),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_mul),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Mul, l, r).err()
    }
}

pub fn binary_div(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => {
            let l = l.as_int_unchecked();
            let r = r.as_int_unchecked();
            if r == 0 {
                ValueErrorValueMustBeNonZero.err()
            } else {
                Ok(Int(num_integer::div_floor(l, r)))
            }
        },
        (l @ (Bool(_) | Int(_) | Complex(_)), r @ (Bool(_) | Int(_) | Complex(_))) => {
            let l = l.as_complex_unchecked();
            let r = r.as_complex_unchecked();
            if r.norm_sqr() == 0 {
                ValueErrorValueMustBeNonZero.err()
            } else {
                Ok(c64_div_floor(l, r).to_value())
            }
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_div),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_div),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_div),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Div, l, r).err()
    }
}

/// The `C64` type provided by `num-complex` defines `div()` using regular rust division.
/// This is a clone of that but using `floor_div` provided by `num-integer`, which keeps consistency with how we define division for `complex / int`
#[inline]
fn c64_div_floor(lhs: C64, rhs: C64) -> C64 {
    let norm_sqr = rhs.norm_sqr();
    let re = lhs.re * rhs.re + lhs.im * rhs.im;
    let im = lhs.im * rhs.re - lhs.re * rhs.im;
    C64::new(num_integer::div_floor(re, norm_sqr), num_integer::div_floor(im, norm_sqr))
}

pub fn binary_mod(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(num_integer::mod_floor(l.as_int_unchecked(), r.as_int_unchecked()))),
        (Str(l), r) => core::format_string(&l, r),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_mod),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_mod),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_mod),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Mod, l, r).err()
    }
}

pub fn binary_pow(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => {
            let r = r.as_int_unchecked();
            if r >= 0 {
                Ok(Int(l.as_int_unchecked().pow(r as u32)))
            } else {
                ValueErrorValueMustBeNonNegative(r).err()
            }
        },
        (Complex(l), r @ (Bool(_) | Int(_))) => {
            let r = r.as_int_unchecked();
            if r >= 0 {
                Ok(l.powu(r as u32).to_value())
            } else {
                ValueErrorValueMustBeNonNegative(r).err()
            }
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_pow),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_pow),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_pow),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Mod, l, r).err()
    }
}

pub fn binary_is(lhs: Value, rhs: Value) -> Result<bool, Box<RuntimeError>> {
    match rhs {
        Nil => Ok(lhs == Nil),
        StructType(it) => Ok(if let Struct(instance) = lhs {
            instance.unbox().type_index == it.type_index
        } else { false }),
        Value::NativeFunction(b) => {
            let ret: bool = match b {
                NativeFunction::Bool => lhs.is_bool(),
                NativeFunction::Int => lhs.is_int(),
                NativeFunction::Complex => lhs.is_complex(),
                NativeFunction::Str => lhs.is_str(),
                NativeFunction::Function => lhs.is_function(),
                NativeFunction::List => lhs.is_list(),
                NativeFunction::Set => lhs.is_set(),
                NativeFunction::Dict => lhs.is_dict(),
                NativeFunction::Vector => lhs.is_vector(),
                NativeFunction::Iterable => lhs.is_iter(),
                NativeFunction::Any => true,
                _ => return TypeErrorBinaryIs(lhs, Value::NativeFunction(b)).err()
            };
            Ok(ret)
        },
        _ => TypeErrorBinaryIs(lhs, rhs).err()
    }
}

pub fn binary_in(a1: Value, a2: Value) -> Result<bool, Box<RuntimeError>> {
    match (a1, a2) {
        (Str(l), Str(r)) => Ok(r.contains(l.as_str())),
        (Int(l), Range(r)) => Ok(r.contains(l)),
        (l, List(it)) => Ok(it.unbox().contains(&l)),
        (l, Set(it)) => Ok(it.unbox().set.contains(&l)),
        (l, Dict(it)) => Ok(it.unbox().dict.contains_key(&l)),
        (l, Heap(it)) => Ok(it.unbox().heap.iter().any(|v|v.0 == l)),
        (l, Vector(it)) => Ok(it.unbox().contains(&l)),
        (l, r) => TypeErrorBinaryOp(BinaryOp::In, l, r).err()
    }
}

pub fn binary_add(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(l.as_int_unchecked() + r.as_int_unchecked())),
        (l @ (Bool(_) | Int(_) | Complex(_)), r @ (Bool(_) | Int(_) | Complex(_))) => Ok((l.as_complex_unchecked() + r.as_complex_unchecked()).to_value()),
        (List(l), List(r)) => {
            let list1 = l.unbox();
            let list2 = r.unbox();
            let mut list3: VecDeque<Value> = VecDeque::with_capacity(list1.len() + list2.len());
            list3.extend(list1.iter().cloned());
            list3.extend(list2.iter().cloned());
            Ok(list3.to_value())
        }
        (Str(l), r) => Ok(format!("{}{}", l, r.to_str()).to_value()),
        (l, Str(r)) => Ok(format!("{}{}", l.to_str(), r).to_value()),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_add),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_add),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_add),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Add, l, r).err(),
    }
}

pub fn binary_sub(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(l.as_int_unchecked() - r.as_int_unchecked())),
        (l @ (Bool(_) | Int(_) | Complex(_)), r @ (Bool(_) | Int(_) | Complex(_))) => Ok((l.as_complex_unchecked() - r.as_complex_unchecked()).to_value()),
        (Set(s1), Set(s2)) => {
            let s1 = s1.unbox();
            let s2 = s2.unbox();
            Ok(s1.set.difference(&s2.set).cloned().to_set())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_sub),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_sub),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_sub),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Sub, l, r).err()
    }
}

/// Left shifts by negative values are defined as right shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_left_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(i64_left_shift(l.as_int_unchecked(), r.as_int_unchecked()))),
        (Complex(l), r @ (Bool(_) | Int(_))) => {
            let r = r.as_int_unchecked();
            Ok(C64::new(i64_left_shift(l.re, r), i64_left_shift(l.im, r)).to_value())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_left_shift),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_left_shift),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_left_shift),
        (l, r) => TypeErrorBinaryOp(BinaryOp::LeftShift, l, r).err(),
    }
}

/// Right shifts by negative values are defined as left shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_right_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(i64_left_shift(l.as_int_unchecked(), -r.as_int_unchecked()))),
        (Complex(l), r @ (Bool(_) | Int(_))) => {
            let r = -r.as_int_unchecked();
            Ok(C64::new(i64_left_shift(l.re, r), i64_left_shift(l.im, r)).to_value())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_right_shift),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_right_shift),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_right_shift),
        (l, r) => TypeErrorBinaryOp(BinaryOp::RightShift, l, r).err(),
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


pub fn binary_bitwise_and(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Bool(l), Bool(r)) => Ok(Bool(l & r)),
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(l.as_int_unchecked() & r.as_int_unchecked())),
        (Complex(l), r @ (Bool(_) | Int(_))) => {
            let r = r.as_int_unchecked();
            Ok(C64::new(l.re & r, l.im & r).to_value())
        }
        (Set(l), Set(r)) => {
            let l = l.unbox();
            let r = r.unbox();
            Ok(l.set.intersection(&r.set).cloned().to_set())
        }
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_bitwise_and),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_and),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_and),
        (l, r) => TypeErrorBinaryOp(BinaryOp::And, l, r).err()
    }
}

pub fn binary_bitwise_or(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Bool(l), Bool(r)) => Ok(Bool(l | r)),
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(l.as_int_unchecked() | r.as_int_unchecked())),
        (Complex(l), r @ (Bool(_) | Int(_))) => {
            let r = r.as_int_unchecked();
            Ok(C64::new(l.re | r, l.im | r).to_value())
        }
        (Set(l), Set(r)) => {
            let l = l.unbox();
            let r = r.unbox();
            Ok(l.set.union(&r.set).cloned().to_set())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_bitwise_or),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_or),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_or),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Or, l, r).err()
    }
}

pub fn binary_bitwise_xor(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Bool(l), Bool(r)) => Ok(Bool(l ^ r)),
        (l @ (Bool(_) | Int(_)), r @ (Bool(_) | Int(_))) => Ok(Int(l.as_int_unchecked() ^ r.as_int_unchecked())),
        (Complex(l), r @ (Bool(_) | Int(_))) => {
            let r = r.as_int_unchecked();
            Ok(C64::new(l.re ^ r, l.im ^ r).to_value())
        }
        (Set(l), Set(r)) => {
            let l = l.unbox();
            let r = r.unbox();
            Ok(l.set.symmetric_difference(&r.set).cloned().to_set())
        }
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_bitwise_xor),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_xor),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_xor),
        (l, r) => TypeErrorBinaryOp(BinaryOp::Xor, l, r).err()
    }
}



/// Helpers for `Vector` operations, which all apply elementwise
fn apply_vector_unary(vector: Mut<Vec<Value>>, unary_op: fn(Value) -> ValueResult) -> ValueResult {
    Ok(vector.unbox()
        .iter()
        .map(|v| unary_op(v.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?
        .to_value())
}

fn apply_vector_binary(lhs: Mut<Vec<Value>>, rhs: Mut<Vec<Value>>, binary_op: fn(Value, Value) -> ValueResult) -> ValueResult {
    Ok(lhs.unbox()
        .iter()
        .zip(rhs.unbox().iter())
        .map(|(l, r)| binary_op(l.clone(), r.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?
        .to_value())
}

fn apply_vector_binary_scalar_lhs(scalar_lhs: Value, rhs: Mut<Vec<Value>>, binary_op: fn(Value, Value) -> ValueResult) -> ValueResult {
    Ok(rhs.unbox()
        .iter()
        .map(|r| binary_op(scalar_lhs.clone(), r.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?
        .to_value())
}

fn apply_vector_binary_scalar_rhs(lhs: Mut<Vec<Value>>, scalar_rhs: Value, binary_op: fn(Value, Value) -> ValueResult) -> ValueResult {
    Ok(lhs.unbox()
        .iter()
        .map(|l| binary_op(l.clone(), scalar_rhs.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?
        .to_value())
}


#[cfg(test)]
mod test {
    use crate::vm::{IntoValue, operator, RuntimeError};
    use crate::vm::value::C64;
    use crate::vm::value::Value::Int;

    #[test]
    fn test_binary_int_div() {
        for (a, b, c) in vec![
            (1, 3, 0), (2, 3, 0), (3, 3, 1), (4, 3, 1), (5, 3, 1), (6, 3, 2), (7, 3, 2), // + / +
            (1, -3, -1), (2, -3, -1), (3, -3, -1), (4, -3, -2), (5, -3, -2), (6, -3, -2), (7, -3, -3), // + / -
            (-1, 3, -1), (-2, 3, -1), (-3, 3, -1), (-4, 3, -2), (-5, 3, -2), (-6, 3, -2), (-7, 3, -3), // - / +
            (-1, -3, 0), (-2, -3, 0), (-3, -3, 1), (-4, -3, 1), (-5, -3, 1), (-6, -3, 2), (-7, -3, 2) // - / -
        ] {
            assert_eq!(operator::binary_div(Int(a), Int(b)), Ok(Int(c)), "{} / {}", a, b)
        }

        for a in -5..=-5 {
            assert_eq!(operator::binary_div(Int(a), Int(0)), RuntimeError::ValueErrorValueMustBeNonZero.err())
        }
    }

    #[test]
    fn test_binary_complex_by_int_div() {
        for ((a, ai), b, (c, ci)) in vec![
            ((3, 3), 2, (1, 1)), ((2, 3), 2, (1, 1)), ((1, 3), 2, (0, 1)), ((0, 3), 2, (0, 1)), ((-1, 3), 2, (-1, 1)), ((-2, 3), 2, (-1, 1)), ((-3, 3), 2, (-2, 1)),
            ((3, 2), 2, (1, 1)), ((2, 2), 2, (1, 1)), ((1, 2), 2, (0, 1)), ((0, 2), 2, (0, 1)), ((-1, 2), 2, (-1, 1)), ((-2, 2), 2, (-1, 1)), ((-3, 2), 2, (-2, 1)),
            ((3, 1), 2, (1, 0)), ((2, 1), 2, (1, 0)), ((1, 1), 2, (0, 0)), ((0, 1), 2, (0, 0)), ((-1, 1), 2, (-1, 0)), ((-2, 1), 2, (-1, 0)), ((-3, 1), 2, (-2, 0)),
            ((3, 0), 2, (1, 0)), ((2, 0), 2, (1, 0)), ((1, 0), 2, (0, 0)), ((0, 0), 2, (0, 0)), ((-1, 0), 2, (-1, 0)), ((-2, 0), 2, (-1, 0)), ((-3, 0), 2, (-2, 0)),
            ((3, 1), 2, (1, 0)), ((2, 1), 2, (1, 0)), ((1, 1), 2, (0, 0)), ((0, 1), 2, (0, 0)), ((-1, 1), 2, (-1, 0)), ((-2, 1), 2, (-1, 0)), ((-3, 1), 2, (-2, 0)),
            ((3, 2), 2, (1, 1)), ((2, 2), 2, (1, 1)), ((1, 2), 2, (0, 1)), ((0, 2), 2, (0, 1)), ((-1, 2), 2, (-1, 1)), ((-2, 2), 2, (-1, 1)), ((-3, 2), 2, (-2, 1)),
            ((3, 3), 2, (1, 1)), ((2, 3), 2, (1, 1)), ((1, 3), 2, (0, 1)), ((0, 3), 2, (0, 1)), ((-1, 3), 2, (-1, 1)), ((-2, 3), 2, (-1, 1)), ((-3, 3), 2, (-2, 1)),
        ] {
            assert_eq!(operator::binary_div(C64::new(a, ai).to_value(), Int(b)), Ok(C64::new(c, ci).to_value()), "{:?} / {}", (a, ai), b)
        }
    }

    #[test]
    fn test_binary_int_mod() {
        for (a, b, c) in vec![
            (1, 3, 1), (2, 3, 2), (3, 3, 0), (4, 3, 1), (5, 3, 2), (6, 3, 0), (7, 3, 1), // + % +
            (1, -3, -2), (2, -3, -1), (3, -3, 0), (4, -3, -2), (5, -3, -1), (6, -3, 0), (7, -3, -2), // + % -
            (-1, 3, 2), (-2, 3, 1), (-3, 3, 0), (-4, 3, 2), (-5, 3, 1), (-6, 3, 0), (-7, 3, 2), // - % +
            (-1, -3, -1), (-2, -3, -2), (-3, -3, 0), (-4, -3, -1), (-5, -3, -2), (-6, -3, 0), (-7, -3, -1), // - % -
        ] {
            assert_eq!(operator::binary_mod(Int(a), Int(b)), Ok(Int(c)), "{} % {}", a, b);
        }
    }
}