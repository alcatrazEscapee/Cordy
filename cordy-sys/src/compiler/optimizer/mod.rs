/// ### Core Optimizing Compiler Routines
///
/// This is broken out into two separate modules:
/// - `expr`, which contains an `Expr` (expression) optimizer.
/// - `block`, which contains a `Block` (basic block) optimizer.
///
/// `Optimize` is a trait for objects which are able to be optimized via a recursive self-transformation. It is implemented for:
///
/// - `Expr` and `Vec<Expr>` by the expression optimizer,
/// - `&mut Code` by the block optimizer.
pub trait Optimize {
    fn optimize(self) -> Self;
}

mod expr;
mod block;