use crate::stdlib::StdBinding;
use crate::vm::error::RuntimeError;
use crate::vm::value::Value;

use crate::vm::error::RuntimeError::{*};
use crate::vm::opcode::Opcode::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


pub fn unary_sub(a1: Value) -> ValueResult {
    match a1 {
        Value::Int(i1) => Ok(Value::Int(-i1)),
        v => TypeErrorUnaryOp(UnarySub, v).err(),
    }
}

pub fn unary_logical_not(a1: Value) -> ValueResult {
    match a1 {
        Value::Bool(b1) => Ok(Value::Bool(!b1)),
        v => TypeErrorUnaryOp(UnaryLogicalNot, v).err(),
    }
}

pub fn unary_bitwise_not(a1: Value) -> ValueResult {
    match a1 {
        Value::Int(i1) => Ok(Value::Int(!i1)),
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
        (l, r) => TypeErrorBinaryOp(OpMul, l, r).err()
    }
}

/// Division of (a / b) will be equal to sign(a * b) * floor(abs(a) / abs(b))
/// The sign will be treated independently of the division, so if a, b > 0, then a / b == -a / -b, and -a / b = a / -b = -(a / b)
pub fn binary_div(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) if i2 != 0 => Ok(Value::Int(if i2 < 0 { -(-i1).div_euclid(i2) } else { i1.div_euclid(i2) })),
        (l, r) => TypeErrorBinaryOp(OpDiv, l, r).err()
    }
}

/// Modulo is defined by where a2 > 0, it is equal to x in [0, a2) s.t. x + n*a2 = a1 for some integer n
/// This corresponds to the mathematical definition of a modulus, and matches the operator in Python (for positive integers)
/// Unlike Python, we don't define the behavior for negative modulus.
pub fn binary_mod(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) if i2 > 0 => Ok(Value::Int(i1.rem_euclid(i2))),
        (l, r) => TypeErrorBinaryOp(OpMod, l, r).err()
    }
}

pub fn binary_pow(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) if i2 > 0 => Ok(Value::Int(i1.pow(i2 as u32))),
        (l, r) => TypeErrorBinaryOp(OpMod, l, r).err()
    }
}

pub fn binary_is(a1: Value, a2: Value) -> ValueResult {
    match a2 {
        Value::Binding(b) => {
            let ret: bool = match b {
                StdBinding::Nil => a1.is_nil(),
                StdBinding::Bool => a1.is_bool(),
                StdBinding::Int => a1.is_int(),
                StdBinding::Str => a1.is_str(),
                StdBinding::Function => a1.is_function(),
                StdBinding::List => a1.is_list(),
                StdBinding::Set => a1.is_set(),
                StdBinding::Dict => a1.is_dict(),
                _ => return TypeErrorBinaryIs(a1, Value::Binding(b)).err()
            };
            Ok(Value::Bool(ret))
        },
        _ => return TypeErrorBinaryIs(a1, a2).err()
    }
}

pub fn binary_add(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 + i2)),
        (Value::List(l1), Value::List(l2)) => {
            let list1 = l1.unbox();
            let list2 = l2.unbox();
            let mut list3: Vec<Value> = Vec::with_capacity(list1.len() + list2.len());
            list3.extend(list1.iter().cloned());
            list3.extend(list2.iter().cloned());
            Ok(Value::list(list3))
        },
        (Value::Str(s1), r) => Ok(Value::Str(Box::new(format!("{}{}", s1, r.as_str())))),
        (l, Value::Str(s2)) => Ok(Value::Str(Box::new(format!("{}{}", l.as_str(), s2)))),
        (l, r) => TypeErrorBinaryOp(OpAdd, l, r).err(),
    }
}

pub fn binary_sub(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 - i2)),
        (l, r) => TypeErrorBinaryOp(OpSub, l, r).err()
    }
}

/// Left shifts by negative values are defined as right shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_left_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(if i2 >= 0 { i1 << i2 } else {i1 >> (-i2)})),
        (l, r) => return TypeErrorBinaryOp(OpLeftShift, l, r).err(),
    }
}

/// Right shifts by negative values are defined as left shifts by the corresponding positive value. So (a >> -b) == (a << b)
pub fn binary_right_shift(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(if i2 >= 0 { i1 >> i2 } else {i1 << (-i2)})),
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
        (l, r) => TypeErrorBinaryOp(OpBitwiseAnd, l, r).err()
    }
}

pub fn binary_bitwise_or(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 | i2)),
        (l, r) => return TypeErrorBinaryOp(OpBitwiseAnd, l, r).err()
    }
}

pub fn binary_bitwise_xor(a1: Value, a2: Value) -> ValueResult {
    match (a1, a2) {
        (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1 ^ i2)),
        (l, r) => TypeErrorBinaryOp(OpBitwiseAnd, l, r).err()
    }
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