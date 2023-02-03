use std::collections::VecDeque;

use crate::stdlib;
use crate::stdlib::NativeFunction;
use crate::vm::error::RuntimeError;
use crate::vm::opcode::Opcode;
use crate::vm::value::{IntoIterableValue, IntoValue, Mut, Value};

use RuntimeError::{*};
use Value::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


pub fn unary_sub(a1: Value) -> ValueResult {
    match a1 {
        Int(it) => Ok(Int(-it)),
        Vector(it) => apply_vector_unary(it, unary_sub),
        v => TypeErrorUnaryOp(Opcode::UnarySub, v).err(),
    }
}

pub fn unary_logical_not(a1: Value) -> ValueResult {
    match a1 {
        Bool(b1) => Ok(Bool(!b1)),
        Vector(it) => apply_vector_unary(it, unary_logical_not),
        v => TypeErrorUnaryOp(Opcode::UnaryLogicalNot, v).err(),
    }
}

pub fn unary_bitwise_not(a1: Value) -> ValueResult {
    match a1 {
        Int(i1) => Ok(Int(!i1)),
        Vector(it) => apply_vector_unary(it, unary_bitwise_not),
        v => TypeErrorUnaryOp(Opcode::UnaryBitwiseNot, v).err(),
    }
}


pub fn binary_mul(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(i1 * i2)),
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
        (l, r) => TypeErrorBinaryOp(Opcode::OpMul, l, r).err()
    }
}

/// Division of (a / b) will be equal to sign(a * b) * floor(abs(a) / abs(b))
/// The sign will be treated independently of the division, so if a, b > 0, then a / b == -a / -b, and -a / b = a / -b = -(a / b)
pub fn binary_div(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) if i2 != 0 => Ok(Int(if i2 < 0 { -(-i1).div_euclid(i2) } else { i1.div_euclid(i2) })),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_div),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_div),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_div),
        (l, r) => TypeErrorBinaryOp(Opcode::OpDiv, l, r).err()
    }
}

/// Modulo is defined by where a2 > 0, it is equal to x in [0, a2) s.t. x + n*a2 = a1 for some integer n
/// This corresponds to the mathematical definition of a modulus, and matches the operator in Python (for positive integers)
/// Unlike Python, we don't define the behavior for negative modulus.
pub fn binary_mod(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) if i2 > 0 => Ok(Int(i1.rem_euclid(i2))),
        (Str(l), r) => stdlib::format_string(&*l, r),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_mod),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_mod),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_mod),
        (l, r) => TypeErrorBinaryOp(Opcode::OpMod, l, r).err()
    }
}

pub fn binary_pow(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) if i2 > 0 => Ok(Int(i1.pow(i2 as u32))),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_pow),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_pow),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_pow),
        (l, r) => TypeErrorBinaryOp(Opcode::OpMod, l, r).err()
    }
}

pub fn binary_is(a1: Value, a2: Value) -> ValueResult {
    match a2 {
        Nil => Ok(Bool(a1.is_nil())),
        Value::NativeFunction(b) => {
            let ret: bool = match b {
                NativeFunction::Bool => a1.is_bool(),
                NativeFunction::Int => a1.is_int(),
                NativeFunction::Str => a1.is_str(),
                NativeFunction::Function => a1.is_function(),
                NativeFunction::List => a1.is_list(),
                NativeFunction::Set => a1.is_set(),
                NativeFunction::Dict => a1.is_dict(),
                NativeFunction::Vector => a1.is_vector(),
                NativeFunction::Iterable => a1.is_iter(),
                NativeFunction::Any => true,
                _ => return TypeErrorBinaryIs(a1, Value::NativeFunction(b)).err()
            };
            Ok(Bool(ret))
        },
        _ => return TypeErrorBinaryIs(a1, a2).err()
    }
}

pub fn binary_in(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Str(l), Str(r)) => Ok(Bool(r.contains(l.as_str()))),
        (Int(l), Range(r)) => Ok(Bool(r.contains(l))),
        (l, List(it)) => Ok(Bool(it.unbox().contains(&l))),
        (l, Set(it)) => Ok(Bool(it.unbox().set.contains(&l))),
        (l, Dict(it)) => Ok(Bool(it.unbox().dict.contains_key(&l))),
        (l, Heap(it)) => Ok(Bool(it.unbox().heap.iter().any(|v|v.0 == l))),
        (l, Vector(it)) => Ok(Bool(it.unbox().contains(&l))),
        (l, r) => TypeErrorBinaryOp(Opcode::OpIn, l, r).err()
    }
}

pub fn binary_add(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(i1 + i2)),
        (List(l1), List(l2)) => {
            let list1 = l1.unbox();
            let list2 = l2.unbox();
            let mut list3: VecDeque<Value> = VecDeque::with_capacity(list1.len() + list2.len());
            list3.extend(list1.iter().cloned());
            list3.extend(list2.iter().cloned());
            Ok(list3.to_value())
        },
        (Str(s1), r) => Ok(format!("{}{}", s1, r.to_str()).to_value()),
        (l, Str(s2)) => Ok(format!("{}{}", l.to_str(), s2).to_value()),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_add),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_add),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_add),
        (l, r) => TypeErrorBinaryOp(Opcode::OpAdd, l, r).err(),
    }
}

pub fn binary_sub(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(i1 - i2)),
        (Set(s1), Set(s2)) => {
            let s1 = s1.unbox();
            let s2 = s2.unbox();
            Ok(s1.set.difference(&s2.set).cloned().to_set())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_sub),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_sub),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_sub),
        (l, r) => TypeErrorBinaryOp(Opcode::OpSub, l, r).err()
    }
}

/// Left shifts by negative values are defined as right shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_left_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(if i2 >= 0 { i1 << i2 } else {i1 >> (-i2)})),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_left_shift),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_left_shift),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_left_shift),
        (l, r) => return TypeErrorBinaryOp(Opcode::OpLeftShift, l, r).err(),
    }
}

/// Right shifts by negative values are defined as left shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_right_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(if i2 >= 0 { i1 >> i2 } else {i1 << (-i2)})),
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_right_shift),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_right_shift),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_right_shift),
        (l, r) => TypeErrorBinaryOp(Opcode::OpRightShift, l, r).err(),
    }
}


pub fn binary_less_than(a1: Value, a2: Value) -> Value { Bool(a1 < a2) }
pub fn binary_less_than_or_equal(a1: Value, a2: Value) -> Value { Bool(a1 <= a2) }
pub fn binary_greater_than(a1: Value, a2: Value) -> Value { Bool(a1 > a2) }
pub fn binary_greater_than_or_equal(a1: Value, a2: Value) -> Value { Bool(a1 >= a2) }

pub fn binary_equals(a1: Value, a2: Value) -> Value {
    Bool(a1 == a2)
}
pub fn binary_not_equals(a1: Value, a2: Value) -> Value { Bool(a1 != a2) }


pub fn binary_bitwise_and(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(i1 & i2)),
        (Set(s1), Set(s2)) => {
            let s1 = s1.unbox();
            let s2 = s2.unbox();
            Ok(s1.set.intersection(&s2.set).cloned().to_set())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_bitwise_and),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_and),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_and),
        (l, r) => TypeErrorBinaryOp(Opcode::OpBitwiseAnd, l, r).err()
    }
}

pub fn binary_bitwise_or(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(i1 | i2)),
        (Set(s1), Set(s2)) => {
            let s1 = s1.unbox();
            let s2 = s2.unbox();
            Ok(s1.set.union(&s2.set).cloned().to_set())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_bitwise_or),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_or),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_or),
        (l, r) => return TypeErrorBinaryOp(Opcode::OpBitwiseOr, l, r).err()
    }
}

pub fn binary_bitwise_xor(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Int(i1), Int(i2)) => Ok(Int(i1 ^ i2)),
        (Set(s1), Set(s2)) => {
            let s1 = s1.unbox();
            let s2 = s2.unbox();
            Ok(s1.set.symmetric_difference(&s2.set).cloned().to_set())
        },
        (Vector(l), Vector(r)) => apply_vector_binary(l, r, binary_bitwise_xor),
        (Vector(l), r) => apply_vector_binary_scalar_rhs(l, r, binary_bitwise_xor),
        (l, Vector(r)) => apply_vector_binary_scalar_lhs(l, r, binary_bitwise_xor),
        (l, r) => TypeErrorBinaryOp(Opcode::OpBitwiseXor, l, r).err()
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