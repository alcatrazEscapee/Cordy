use crate::compiler::optimizer::Optimize;
use crate::compiler::parser::{Expr, ExprType};
use crate::core::NativeFunction;
use crate::vm::{ErrorPtr, IntoValue, LiteralType, RuntimeError, ValuePtr};
use crate::vm::operator::BinaryOp;


impl Optimize for Vec<Expr> {
    fn optimize(self) -> Self {
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
    /// - Inlining of partially evaluated operators (`(==)(a, b)` -> `a == b`)
    ///
    fn optimize(self) -> Self {
        match self {
            // Terminals
            e @ Expr(_, ExprType::Nil | ExprType::Exit | ExprType::Bool(_) | ExprType::Int(_) | ExprType::Str(_) | ExprType::LValue(_) | ExprType::Function { .. } | ExprType::NativeFunction(_)) => e,

            // Unary Operators
            Expr(loc, ExprType::Unary(op, arg)) => {
                let arg: Expr = arg.optimize();
                match arg.into_const() {
                    Ok(arg) => Expr::value_result(loc, op.apply(arg)),
                    Err(arg) => arg.unary(loc, op)
                }
            },

            // Binary Operators
            Expr(loc, ExprType::Binary(op, lhs, rhs, swap)) => {
                let lhs: Expr = lhs.optimize();
                let rhs: Expr = rhs.optimize();
                match lhs.into_const() {
                    Ok(lhs) => match rhs.into_const() {
                        Ok(rhs) => Expr::value_result(loc, if swap { op.apply(rhs, lhs) } else { op.apply(lhs, rhs) }),
                        Err(rhs) => Expr::value(loc, lhs).binary(loc, op, rhs, swap),
                    },
                    Err(lhs) => lhs.binary(loc, op, rhs, swap),
                }
            },

            Expr(loc, ExprType::Literal(op, args)) => Expr(loc, ExprType::Literal(op, args.optimize())),
            Expr(loc, ExprType::Unroll(arg)) => arg.optimize().unroll(loc),

            Expr(loc, ExprType::Call { f, args, unroll }) => {
                let f: Expr = f.optimize();
                let mut args: Vec<Expr> = args.optimize();
                let nargs: Option<usize> = if unroll { None } else { Some(args.len()) }; // nargs is only valid if no unrolls are present

                match f {
                    // Replace invokes on a binary operator with the operator itself
                    Expr(_, ExprType::NativeFunction(native_f)) if native_f.is_binary_operator() && nargs == Some(2) => {
                        let f_op = native_f.as_binary_operator().unwrap();
                        let rhs = args.pop().unwrap();
                        let lhs = args.pop().unwrap();
                        lhs.binary(loc, f_op, rhs, false).optimize() // Re-call optimize, for constant folding
                    },

                    // Same optimization as above, but with swapped operators, if we can safely reorder them
                    Expr(_, ExprType::NativeFunction(native_f)) if native_f.is_binary_operator_swap() && nargs == Some(2) => {
                        let f_op = native_f.swap().as_binary_operator().unwrap();
                        // lhs and rhs are the arguments, post swap
                        // If we can re-order the arguments, then create just a normal binary(lhs, rhs)
                        // If we *can't*, then we can create a swapped operator instead
                        let lhs = args.pop().unwrap();
                        let rhs = args.pop().unwrap();
                        let swap: bool = !lhs.can_reorder(&rhs);

                        lhs.binary(loc, f_op, rhs, swap).optimize() // Re-call optimize, for constant folding
                    },

                    // This is a special case, for `min(int)` and `max(int)`, we can replace this with a compile time constant
                    Expr(loc, ExprType::NativeFunction(native_f @ (NativeFunction::Min | NativeFunction::Max))) if nargs == Some(1) => {
                        if let Expr(_, ExprType::NativeFunction(NativeFunction::Int)) = args[0] {
                            Expr::int(loc, if native_f == NativeFunction::Min { ValuePtr::MIN_INT } else { ValuePtr::MAX_INT })
                        } else {
                            f.eval(loc, args, unroll)
                        }
                    },

                    // If we can assert the inner function is partial, then we can merge the two calls:
                    // - We know the function being called is partial with the given number of arguments, and
                    // - The call is not unrolled (because then we can never prove it is partial
                    //
                    // Note this does not require any reordering of the arguments
                    // This re-calls `.optimize()` as we might be able to replace the operator on the constant-eval'd function
                    Expr(_, ExprType::Call { f: f_inner, args: mut args_inner, unroll: false }) if f_inner.is_partial(args_inner.len()) => {
                        args_inner.append(&mut args);
                        f_inner.eval(loc, args_inner, unroll).optimize()
                    },

                    f => f.eval(loc, args, unroll)
                }
            }

            Expr(loc, ExprType::Compose(arg, f)) => {
                let arg: Expr = arg.optimize();
                let f: Expr = f.optimize();

                // Check for various optimizations that can be performed without reordering, as those don't depend on purity
                // If we can't optimize compose directly, then try and swap into a eval (if purity allows).
                // At worst, that will eliminate the overhead of a `Swap`, and potentially allow more optimizations like function call merging.
                match f {
                    // Found `arg . [b, ...]`
                    // Compile time check that the list contains a single argument, then inline as `arg[b]`
                    Expr(_, ExprType::Literal(LiteralType::List, mut args)) if !any_unroll(&args) => {
                        let nargs: usize = args.len();
                        if nargs != 1 {
                            Expr::error(loc, ErrorPtr::new(RuntimeError::ValueErrorEvalListMustHaveUnitLength(nargs).to_value()))
                        } else {
                            let index: Expr = args.swap_remove(0);
                            arg.index(loc, index)
                        }
                    },
                    // Found `arg . [a:b:c]`. Inline into `arg[a:b:c]`
                    Expr(_, ExprType::SliceLiteral(a, b, c)) => match *c {
                        Some(c) => arg.slice_step(loc, *a, *b, c),
                        None => arg.slice(loc, *a, *b),
                    },
                    // If possible, replace `arg . f` with `f(arg)`
                    // Then re-optimize the new `eval` expression
                    f if f.can_reorder(&arg) => f.eval(loc, vec![arg], false).optimize(),

                    // If we can't reorder, then we won't fall into optimization cases for binary operators
                    // This hits cases such as `a . (<op> b)` where a and b cannot be re-ordered
                    // We can replace this with `a b <op>`, or in the case of the opposite partial, with a `Swap` opcode as well, in both cases avoiding the function call
                    Expr(_, ExprType::Call { f: f_inner, mut args, unroll: false }) if is_native_operator(&f_inner) && args.len() == 1 => {
                        let native_f = match *f_inner {
                            Expr(_, ExprType::NativeFunction(f)) => f,
                            _ => panic!()
                        };
                        let rhs = args.pop().unwrap();
                        if native_f.is_binary_operator() {
                            rhs.binary(loc, native_f.as_binary_operator().unwrap(), arg, true)
                        } else {
                            arg.binary(loc, native_f.swap().as_binary_operator().unwrap(), rhs, false)
                        }
                    },

                    // Default, return the compose
                    f => arg.compose(loc, f),
                }
            },

            Expr(loc, ExprType::LogicalAnd(lhs, rhs)) => lhs.optimize().logical(loc, BinaryOp::And, rhs.optimize()),
            Expr(loc, ExprType::LogicalOr(lhs, rhs)) => lhs.optimize().logical(loc, BinaryOp::Or, rhs.optimize()),
            Expr(loc, ExprType::Index(array, index)) => array.optimize().index(loc, index.optimize()),
            Expr(loc, ExprType::Slice(array, arg1, arg2)) => array.optimize().slice(loc, arg1.optimize(), arg2.optimize()),
            Expr(loc, ExprType::SliceWithStep(array, arg1, arg2, arg3)) => array.optimize().slice_step(loc, arg1.optimize(), arg2.optimize(), arg3.optimize()),

            // Ternary conditions perform basic dead code elimination, if the condition is constant.
            Expr(loc, ExprType::IfThenElse(condition, if_true, if_false)) => {
                let condition = condition.optimize();
                match condition.into_const() {
                    Ok(condition) => if condition.to_bool() { if_true.optimize() } else { if_false.optimize() },
                    Err(condition) => condition.if_then_else(loc, if_true.optimize(), if_false.optimize()),
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
    ///
    /// **N.B.** This can only be supported for immutable types. If we attempt to const-expr evaluate a non-constant `Value`, like a list, we would have to
    /// un-const-expr it to emit the code - otherwise in `VM.constants` we would have a single instance that gets copied and re-used. This is **very bad**.
    fn into_const(self) -> Result<ValuePtr, Expr> {
        match self {
            Expr(_, ExprType::Nil) => Ok(ValuePtr::nil()),
            Expr(_, ExprType::Bool(it)) => Ok(it.to_value()),
            Expr(_, ExprType::Int(it)) => Ok(it.to_value()),
            Expr(_, ExprType::Complex(it)) => Ok(it.to_value()),
            Expr(_, ExprType::Str(it)) => Ok(it.to_value()),
            _ => Err(self)
        }
    }

    fn can_reorder(&self, other: &Self) -> bool {
        match self.purity() {
            Purity::Strong => true,
            Purity::Weak => other.purity() != Purity::None,
            Purity::None => other.purity() == Purity::Strong,
        }
    }

    fn purity(&self) -> Purity {
        match &self.1 {
            ExprType::Nil | ExprType::Exit | ExprType::Bool(_) | ExprType::Int(_) | ExprType::Str(_) | ExprType::NativeFunction(_) | ExprType::Function { .. } => Purity::Strong,
            ExprType::LValue(_) => Purity::Weak,

            ExprType::Unary(_, arg) => arg.purity(),
            ExprType::Binary(_, lhs, rhs, _) | ExprType::LogicalOr(lhs, rhs) | ExprType::LogicalAnd(lhs, rhs) => lhs.purity().min(rhs.purity()),
            ExprType::Literal(_, args) => args.iter().map(|u| u.purity()).min().unwrap_or(Purity::Strong),
            ExprType::Unroll(arg) => arg.purity(),
            ExprType::IfThenElse(condition, if_true, if_false) => condition.purity().min(if_true.purity()).min(if_false.purity()),

            ExprType::Call { unroll: true, .. } => Purity::None,
            ExprType::Call { f, args, ..} => match f.is_partial(args.len()) {
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

    fn is_partial(&self, nargs: usize) -> bool {
        match &self.1 {
            ExprType::NativeFunction(f) => f.min_nargs() as usize > nargs,
            _ => false
        }
    }
}

fn any_unroll(args: &[Expr]) -> bool {
    args.iter().any(|u| u.is_unroll())
}

fn is_native_operator(expr: &Expr) -> bool {
    match expr {
        Expr(_, ExprType::NativeFunction(native)) => native.is_operator(),
        _ => false,
    }
}


#[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq)]
enum Purity {
    None, Weak, Strong
}