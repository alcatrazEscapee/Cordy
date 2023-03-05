use crate::compiler::parser::expr::{Expr, ExprType, SequenceOp};
use crate::vm::{IntoValue, RuntimeError, Value};
use crate::vm::operator::BinaryOp;

/// A trait for objects which are able to be optimized via a recursive self-transformation
/// This is implemented for `Expr` and `Vec<Expr>`, as those are common forms we encounter during expression optimization.
pub trait Optimize {
    fn optimize(self: Self) -> Self;
}

impl Optimize for Vec<Expr> {
    fn optimize(self: Self) -> Self {
        self.into_iter().map(|u| u.optimize()).collect()
    }
}


impl Optimize for Expr {

    /// Traverses this expression, and returns a new expression which is the optimized form. This is able to apply a number of different optimizations,
    /// with full view of the expression tree. It includes the following passes:
    ///
    /// - Constant Folding + Dead Code Elimination (`1 + 2` -> `3`)
    /// - Compose/List Folding (`a . [b]` -> `a[b]`)
    /// - Compose/Eval Reordering (`a . b` -> `b(a)` where legal)
    /// - Consistent Function Eval Merging (`a(b1, b2, ...)(c1, c2, ...)` -> `a(b1, b2, ... c1, c2, ...)` where legal)
    ///
    fn optimize(self: Self) -> Self {
        match self {
            // Terminals
            e @ Expr(_, ExprType::Nil | ExprType::Exit | ExprType::Bool(_) | ExprType::Int(_) | ExprType::Str(_) | ExprType::LValue(_) | ExprType::Function(_, _) | ExprType::NativeFunction(_)) => e,

            // Unary Operators
            Expr(loc, ExprType::Unary(op, arg)) => {
                let arg: Expr = arg.optimize();
                match arg.as_constant() {
                    Ok(arg) => Expr::value_result(loc, op.apply(arg)),
                    Err(arg) => arg.unary(loc, op)
                }
            },

            // Binary Operators
            Expr(loc, ExprType::Binary(op, lhs, rhs)) => {
                let lhs: Expr = lhs.optimize();
                let rhs: Expr = rhs.optimize();
                match lhs.as_constant() {
                    Ok(lhs) => match rhs.as_constant() {
                        Ok(rhs) => Expr::value_result(loc, op.apply(lhs, rhs)),
                        Err(rhs) => Expr::value(lhs).binary(loc, op, rhs),
                    },
                    Err(lhs) => lhs.binary(loc, op, rhs),
                }
            },

            Expr(loc, ExprType::Sequence(op, args)) => op.apply(loc, args.optimize()),

            Expr(loc, ExprType::Eval(f, args)) => {
                let f: Expr = f.optimize();
                let mut args: Vec<Expr> = args.optimize();

                match f {
                    // If we can assert the inner function is partial, then we can merge the two calls
                    // Note this does not require any reordering of the arguments
                    Expr(_, ExprType::Eval(f_inner, mut args_inner)) if f_inner.is_partial(args_inner.len()) => {
                        args_inner.append(&mut args);
                        f_inner.eval(loc, args_inner)
                    },

                    f => f.eval(loc, args)
                }
            },

            Expr(loc, ExprType::Compose(arg, f)) => {
                let arg: Expr = arg.optimize();
                let f: Expr = f.optimize();

                match f {
                    Expr(_, ExprType::Sequence(SequenceOp::List, mut args)) => {
                        // Found `a . [b, ...]`
                        // We can compile-time check this list contains a single element, and rephrase this as `a[b]`
                        let nargs: usize = args.len();
                        if nargs != 1 {
                            Expr::error(loc, Box::new(RuntimeError::ValueErrorEvalListMustHaveUnitLength(nargs)))
                        } else {
                            let index: Expr = args.swap_remove(0);
                            arg.index(loc, index)
                        }
                    },
                    f => {
                        // If possible, reorder f, arg
                        // Then re-call optimization with the new eval
                        if f.can_reorder(&arg) {
                            f.eval(loc, vec![arg]).optimize()
                        } else {
                            arg.compose(loc, f)
                        }
                    }
                }
            },

            Expr(loc, ExprType::LogicalAnd(lhs, rhs)) => lhs.optimize().logical(loc, BinaryOp::And, rhs.optimize()),
            Expr(loc, ExprType::LogicalOr(lhs, rhs)) => lhs.optimize().logical(loc, BinaryOp::Or, rhs.optimize()),
            Expr(loc, ExprType::Index(array, index)) => array.optimize().index(loc, index.optimize()),
            Expr(loc, ExprType::Slice(array, arg1, arg2)) => array.optimize().slice(loc, arg1.optimize(), arg2.optimize()),
            Expr(loc, ExprType::SliceWithStep(array, arg1, arg2, arg3)) => array.optimize().slice_step(loc, arg1.optimize(), arg2.optimize(), arg3.optimize()),

            // Ternary conditions perform basic dead code elimination, if the condition is constant.
            Expr(_, ExprType::IfThenElse(condition, if_true, if_false)) => {
                let condition = condition.optimize();
                match condition.as_constant() {
                    Ok(condition) => if condition.as_bool() { if_true.optimize() } else { if_false.optimize() },
                    Err(condition) => condition.if_then_else(if_true.optimize(), if_false.optimize()),
                }
            },

            Expr(loc, ExprType::Assignment(lvalue, rhs)) => Expr::assign_lvalue(loc, lvalue, rhs.optimize()),
            Expr(loc, ExprType::ArrayAssignment(array, index, rhs)) => Expr::assign_array(loc, array.optimize(), index.optimize(), rhs.optimize()),

            // Note that `BinaryOp::NotEqual` is used to indicate this is a `Compose()` operation under the hood
            Expr(loc, ExprType::ArrayOpAssignment(array, index, op, rhs)) => Expr::assign_op_array(loc, array.optimize(), index.optimize(), op, rhs.optimize()),
            Expr(loc, ExprType::PatternAssignment(lvalue, rhs)) => Expr::assign_pattern(loc, lvalue, rhs.optimize()),

            e => e,
        }
    }
}




impl Expr {
    /// Attempts to fold this expression into a constant value. Either returns `Ok(constant)` or `Err(self)`
    fn as_constant(self: Self) -> Result<Value, Expr> {
        match self {
            Expr(_, ExprType::Nil) => Ok(Value::Nil),
            Expr(_, ExprType::Bool(it)) => Ok(it.to_value()),
            Expr(_, ExprType::Int(it)) => Ok(it.to_value()),
            Expr(_, ExprType::Str(it)) => Ok(it.to_value()),
            _ => Err(self)
        }
    }

    fn can_reorder(self: &Self, other: &Self) -> bool {
        match self.purity() {
            Purity::Strong => true,
            Purity::Weak => other.purity() != Purity::None,
            Purity::None => other.purity() == Purity::Strong,
        }
    }

    fn purity(self: &Self) -> Purity {
        match &self.1 {
            ExprType::Nil | ExprType::Exit | ExprType::Bool(_) | ExprType::Int(_) | ExprType::Str(_) | ExprType::NativeFunction(_) | ExprType::Function(_, _) => Purity::Strong,
            ExprType::LValue(_) => Purity::Weak,

            ExprType::Unary(_, arg) => arg.purity(),
            ExprType::Binary(_, lhs, rhs) | ExprType::LogicalOr(lhs, rhs) | ExprType::LogicalAnd(lhs, rhs) => lhs.purity().min(rhs.purity()),
            ExprType::Sequence(_, args) => args.iter().map(|u| u.purity()).min().unwrap_or(Purity::Strong),
            ExprType::IfThenElse(condition, if_true, if_false) => condition.purity().min(if_true.purity()).min(if_false.purity()),

            ExprType::Eval(f, args) => match f.is_partial(args.len()) {
                true => f.purity().min(args.iter().map(|u| u.purity()).min().unwrap_or(Purity::Strong)),
                false => Purity::None,
            },

            ExprType::Compose(arg, f) => match f.is_partial(1) {
                true => f.purity().min(arg.purity()),
                false => Purity::None,
            }

            _ => Purity::None,
        }
    }

    fn is_partial(self: &Self, nargs: usize) -> bool {
        match &self.1 {
            ExprType::NativeFunction(native) => match native.nargs() {
                Some(expected_nargs) => expected_nargs > nargs as u8 && nargs > 0,
                None => false,
            },
            _ => false
        }
    }
}


#[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq)]
enum Purity {
    None, Weak, Strong
}
