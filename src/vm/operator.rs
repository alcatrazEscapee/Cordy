use std::collections::VecDeque;
use crate::stdlib::StdBinding;
use crate::vm::error::RuntimeError;
use crate::vm::value::{Mut, Value};

use crate::vm::error::RuntimeError::{*};
use crate::vm::opcode::Opcode::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


pub fn unary_sub(a1: Value) -> ValueResult {
    match a1 {
        Value::Int(it) => Ok(Value::Int(-it)),
        Value::Vector(it) => apply_vector_unary(it, unary_sub),
        v => TypeErrorUnaryOp(UnarySub, v).err(),
    }
}

pub fn unary_logical_not(a1: Value) -> ValueResult {
    match a1 {
        Value::Bool(b1) => Ok(Value::Bool(!b1)),
        Value::Vector(it) => apply_vector_unary(it, unary_logical_not),
        v => TypeErrorUnaryOp(UnaryLogicalNot, v).err(),
    }
}

pub fn unary_bitwise_not(a1: Value) -> ValueResult {
    match a1 {
        Value::Int(i1) => Ok(Value::Int(!i1)),
        Value::Vector(it) => apply_vector_unary(it, unary_bitwise_not),
        v => TypeErrorUnaryOp(UnaryBitwiseNot, v).err(),
    }
}


pub fn binary_mul(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 * i2)),
        (Value::Str(s1), Value::Int(i2)) if i2 > 0 => Ok(Value::Str(Box::new(s1.repeat(i2 as usize)))),
        (Value::Int(i1), Value::Str(s2)) if i1 > 0 => Ok(Value::Str(Box::new(s2.repeat(i1 as usize)))),
        (Value::List(l1), Value::Int(i1)) if i1 > 0 => {
            let l1 = l1.unbox();
            let len: usize = l1.len();
            Ok(Value::iter_list(l1.iter().cycle().take(i1 as usize * len).cloned()))
        },
        (l @ Value::Int(_), r @ Value::List(_)) => binary_mul(r, l),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_mul),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_mul),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_mul),
        (l, r) => TypeErrorBinaryOp(OpMul, l, r).err()
    }
}

/// Division of (a / b) will be equal to sign(a * b) * floor(abs(a) / abs(b))
/// The sign will be treated independently of the division, so if a, b > 0, then a / b == -a / -b, and -a / b = a / -b = -(a / b)
pub fn binary_div(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) if i2 != 0 => Ok(Value::Int(if i2 < 0 { -(-i1).div_euclid(i2) } else { i1.div_euclid(i2) })),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_div),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_div),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_div),
        (l, r) => TypeErrorBinaryOp(OpDiv, l, r).err()
    }
}

/// Modulo is defined by where a2 > 0, it is equal to x in [0, a2) s.t. x + n*a2 = a1 for some integer n
/// This corresponds to the mathematical definition of a modulus, and matches the operator in Python (for positive integers)
/// Unlike Python, we don't define the behavior for negative modulus.
pub fn binary_mod(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) if i2 > 0 => Ok(Value::Int(i1.rem_euclid(i2))),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_mod),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_mod),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_mod),
        (l, r) => TypeErrorBinaryOp(OpMod, l, r).err()
    }
}

pub fn binary_pow(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) if i2 > 0 => Ok(Value::Int(i1.pow(i2 as u32))),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_pow),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_pow),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_pow),
        (l, r) => TypeErrorBinaryOp(OpMod, l, r).err()
    }
}

pub fn binary_is(a1: Value, a2: Value) -> ValueResult {
    match a2 {
        Value::Nil => Ok(Value::Bool(a1.is_nil())),
        Value::NativeFunction(b) => {
            let ret: bool = match b {
                StdBinding::Bool => a1.is_bool(),
                StdBinding::Int => a1.is_int(),
                StdBinding::Str => a1.is_str(),
                StdBinding::Function => a1.is_function(),
                StdBinding::List => a1.is_list(),
                StdBinding::Set => a1.is_set(),
                StdBinding::Dict => a1.is_dict(),
                StdBinding::Vector => a1.is_vector(),
                _ => return TypeErrorBinaryIs(a1, Value::NativeFunction(b)).err()
            };
            Ok(Value::Bool(ret))
        },
        _ => return TypeErrorBinaryIs(a1, a2).err()
    }
}

pub fn binary_in(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Str(l), Value::Str(r)) => Ok(Value::Bool(r.contains(l.as_str()))),
        (l, Value::List(it)) => Ok(Value::Bool(it.unbox().contains(&l))),
        (l, Value::Set(it)) => Ok(Value::Bool(it.unbox().contains(&l))),
        (l, Value::Dict(it)) => Ok(Value::Bool(it.unbox().dict.contains_key(&l))),
        (l, Value::Heap(it)) => Ok(Value::Bool(it.unbox().heap.iter().any(|v|v.0 == l))),
        (l, Value::Vector(it)) => Ok(Value::Bool(it.unbox().contains(&l))),
        (l, r) => TypeErrorBinaryOp(OpIn, l, r).err()
    }
}

pub fn binary_add(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 + i2)),
        (Value::List(l1), Value::List(l2)) => {
            let list1 = l1.unbox();
            let list2 = l2.unbox();
            let mut list3: VecDeque<Value> = VecDeque::with_capacity(list1.len() + list2.len());
            list3.extend(list1.iter().cloned());
            list3.extend(list2.iter().cloned());
            Ok(Value::list(list3))
        },
        (Value::Str(s1), r) => Ok(Value::Str(Box::new(format!("{}{}", s1, r.as_str())))),
        (l, Value::Str(s2)) => Ok(Value::Str(Box::new(format!("{}{}", l.as_str(), s2)))),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_add),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_add),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_add),
        (l, r) => TypeErrorBinaryOp(OpAdd, l, r).err(),
    }
}

pub fn binary_sub(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 - i2)),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_sub),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_sub),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_sub),
        (l, r) => TypeErrorBinaryOp(OpSub, l, r).err()
    }
}

/// Left shifts by negative values are defined as right shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_left_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(if i2 >= 0 { i1 << i2 } else {i1 >> (-i2)})),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_left_shift),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_left_shift),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_left_shift),
        (l, r) => return TypeErrorBinaryOp(OpLeftShift, l, r).err(),
    }
}

/// Right shifts by negative values are defined as left shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_right_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(if i2 >= 0 { i1 >> i2 } else {i1 << (-i2)})),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_right_shift),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_right_shift),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_right_shift),
        (l, r) => TypeErrorBinaryOp(OpRightShift, l, r).err(),
    }
}


pub fn binary_less_than(a1: Value, a2: Value) -> Value { Value::Bool(a1 < a2) }
pub fn binary_less_than_or_equal(a1: Value, a2: Value) -> Value { Value::Bool(a1 <= a2) }
pub fn binary_greater_than(a1: Value, a2: Value) -> Value { Value::Bool(a1 > a2) }
pub fn binary_greater_than_or_equal(a1: Value, a2: Value) -> Value { Value::Bool(a1 >= a2) }

pub fn binary_equals(a1: Value, a2: Value) -> Value {
    Value::Bool(a1 == a2)
}
pub fn binary_not_equals(a1: Value, a2: Value) -> Value { Value::Bool(a1 != a2) }


pub fn binary_bitwise_and(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 & i2)),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_bitwise_and),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_and),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_and),
        (l, r) => TypeErrorBinaryOp(OpBitwiseAnd, l, r).err()
    }
}

pub fn binary_bitwise_or(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 | i2)),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_bitwise_or),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_or),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_or),
        (l, r) => return TypeErrorBinaryOp(OpBitwiseAnd, l, r).err()
    }
}

pub fn binary_bitwise_xor(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 ^ i2)),
        (Value::Vector(l), Value::Vector(r)) => apply_vector_binary(l, r, binary_bitwise_xor),
        (Value::Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_xor),
        (l, Value::Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_xor),
        (l, r) => TypeErrorBinaryOp(OpBitwiseAnd, l, r).err()
    }
}



/// Helpers for `Vector` operations, which all apply elementwise
fn apply_vector_unary(vector: Mut<Vec<Value>>, unary_op: fn(Value) -> ValueResult) -> ValueResult {
    Ok(Value::vector(vector.unbox()
        .iter()
        .map(|v| unary_op(v.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?))
}

fn apply_vector_binary(lhs: Mut<Vec<Value>>, rhs: Mut<Vec<Value>>, binary_op: fn(Value, Value) -> ValueResult) -> ValueResult {
    Ok(Value::vector(lhs.unbox()
        .iter()
        .zip(rhs.unbox().iter())
        .map(|(l, r)| binary_op(l.clone(), r.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?))
}

fn apply_vector_binary_scalar_lhs(scalar_lhs: Value, rhs: Mut<Vec<Value>>, binary_op: fn(Value, Value) -> ValueResult) -> ValueResult {
    Ok(Value::vector(rhs.unbox()
        .iter()
        .map(|r| binary_op(scalar_lhs.clone(), r.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?))
}

fn apply_vector_binary_scalar_rhs(lhs: Mut<Vec<Value>>, scalar_rhs: Value, binary_op: fn(Value, Value) -> ValueResult) -> ValueResult {
    Ok(Value::vector(lhs.unbox()
        .iter()
        .map(|l| binary_op(l.clone(), scalar_rhs.clone()))
        .collect::<Result<Vec<Value>, Box<RuntimeError>>>()?))
}


#[cfg(test)]
mod test {
    use crate::vm::operator;
    use crate::vm::value::Value::Int;

    #[test]
    fn test_binary_mod() {
        assert_eq!(Int(1), operator::binary_mod(Int(-5), Int(3)).unwrap());
        assert_eq!(Int(2), operator::binary_mod(Int(-4), Int(3)).unwrap());
        assert_eq!(Int(0), operator::binary_mod(Int(-3), Int(3)).unwrap());
        assert_eq!(Int(1), operator::binary_mod(Int(-2), Int(3)).unwrap());
        assert_eq!(Int(2), operator::binary_mod(Int(-1), Int(3)).unwrap());
        assert_eq!(Int(0), operator::binary_mod(Int(0), Int(3)).unwrap());
        assert_eq!(Int(1), operator::binary_mod(Int(1), Int(3)).unwrap());
        assert_eq!(Int(2), operator::binary_mod(Int(2), Int(3)).unwrap());
        assert_eq!(Int(0), operator::binary_mod(Int(3), Int(3)).unwrap());
        assert_eq!(Int(1), operator::binary_mod(Int(4), Int(3)).unwrap());
        assert_eq!(Int(2), operator::binary_mod(Int(5), Int(3)).unwrap());

        assert!(operator::binary_mod(Int(5), Int(0)).is_err());
        assert!(operator::binary_mod(Int(5), Int(-3)).is_err());
    }

    #[test]
    fn test_binary_div() {
        assert_eq!(Int(-2), operator::binary_div(Int(-5), Int(3)).unwrap());
        assert_eq!(Int(-1), operator::binary_div(Int(-2), Int(3)).unwrap());
        assert_eq!(Int(0), operator::binary_div(Int(0), Int(3)).unwrap());
        assert_eq!(Int(0), operator::binary_div(Int(2), Int(3)).unwrap());
        assert_eq!(Int(1), operator::binary_div(Int(5), Int(3)).unwrap());

        assert_eq!(Int(1), operator::binary_div(Int(-5), Int(-3)).unwrap());
        assert_eq!(Int(0), operator::binary_div(Int(-2), Int(-3)).unwrap());
        assert_eq!(Int(0), operator::binary_div(Int(0), Int(-3)).unwrap());
        assert_eq!(Int(-1), operator::binary_div(Int(2), Int(-3)).unwrap());
        assert_eq!(Int(-2), operator::binary_div(Int(5), Int(-3)).unwrap());

        assert!(operator::binary_div(Int(5), Int(0)).is_err());
    }
}