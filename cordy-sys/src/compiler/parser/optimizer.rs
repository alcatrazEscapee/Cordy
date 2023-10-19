use crate::compiler::parser::expr::{Expr, ExprType};
use crate::core::NativeFunction;
use crate::vm::{IntoValue, LiteralType, MAX_INT, MIN_INT, RuntimeError, ValuePtr};
use crate::vm::operator::BinaryOp;

/// A trait for objects which are able to be optimized via a recursive self-transformation
/// This is implemented for `Expr` and `Vec<Expr>`, as those are common forms we encounter during expression optimization.
pub trait Optimize {
    fn optimize(self) -> Self;
}
