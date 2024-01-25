use crate::compiler::parser::{Expr, ExprType, Parser, Visitable, Visitor};
use crate::core::NativeFunction;
use crate::vm::{ErrorPtr, IntoValue, LiteralType, RuntimeError, ValuePtr};


struct Optimizer;


impl<'a> Parser<'a> {
    pub fn optimize_expr(&mut self, expr: Expr) -> Expr {
        expr.visit(&mut Optimizer)
    }
}

impl Visitor for Optimizer {

    /// Traverses this expression, and returns a new expression which is the optimized form. This is able to apply a number of different optimizations,
    /// with full view of the expression tree. It includes the following passes:
    ///
    /// - Constant Folding + Dead Code Elimination (`1 + 2` -> `3`)
    /// - Compose/List Folding (`a . [b]` -> `a[b]`)
    /// - Compose/Eval Reordering (`a . b` -> `b(a)` where legal)
    /// - Consistent Function Eval Merging (`a(b1, b2, ...)(c1, c2, ...)` -> `a(b1, b2, ... c1, c2, ...)` where legal)
    /// - Inlining of partially evaluated operators (`(==)(a, b)` -> `a == b`)
    ///
    fn visit(&mut self, expr: Expr) -> Expr {
        match expr {
            // Constant Folding
            Expr(loc, ExprType::Unary(op, arg)) => match arg.into_const() {
                Ok(arg) => Expr::result(loc, op.apply(arg)),
                Err(arg) => arg.unary(loc, op)
            }

            Expr(loc, ExprType::Binary(op, lhs, rhs, swap)) => match lhs.into_const() {
                Ok(lhs) => match rhs.into_const() {
                    Ok(rhs) => Expr::result(loc, if swap { op.apply(rhs, lhs) } else { op.apply(lhs, rhs) }),
                    Err(rhs) => Expr::value(loc, lhs).binary(loc, op, rhs, swap),
                },
                Err(lhs) => lhs.binary(loc, op, *rhs, swap),
            }

            Expr(loc, ExprType::Call { f, mut args, unroll }) => {
                let nargs: Option<usize> = if unroll { None } else { Some(args.len()) }; // nargs is only valid if no unrolls are present

                match *f {
                    // Replace invokes on a binary operator with the operator itself
                    Expr(_, ExprType::NativeFunction(native_f)) if native_f.is_binary_operator() && nargs == Some(2) => {
                        let f_op = native_f.as_binary_operator().unwrap();
                        let rhs = args.pop().unwrap();
                        let lhs = args.pop().unwrap();
                        self.visit(lhs.binary(loc, f_op, rhs, false)) // Re-call visit, for constant folding
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

                        self.visit(lhs.binary(loc, f_op, rhs, swap)) // Re-call visit, for constant folding
                    },

                    // This is a special case, for `min(int)` and `max(int)`, we can replace this with a compile time constant
                    Expr(loc, ExprType::NativeFunction(native_f @ (NativeFunction::Min | NativeFunction::Max))) if nargs == Some(1) => {
                        if let Expr(_, ExprType::NativeFunction(NativeFunction::Int)) = args[0] {
                            Expr::int(loc, if native_f == NativeFunction::Min { ValuePtr::MIN_INT } else { ValuePtr::MAX_INT })
                        } else {
                            f.call(loc, args)
                        }
                    },

                    // If we can assert the inner function is partial, then we can merge the two calls:
                    // - We know the function being called is partial with the given number of arguments, and
                    // - The call is not unrolled (because then we can never prove it is partial
                    //
                    // Note this does not require any reordering of the arguments
                    // This re-calls `visit` as we might be able to replace the operator on the constant-eval'd function
                    Expr(_, ExprType::Call { f: f_inner, args: mut args_inner, unroll: false }) if f_inner.is_partial(args_inner.len()) => {
                        args_inner.append(&mut args);
                        self.visit(f_inner.call(loc, args_inner))
                    },

                    f => f.call(loc, args)
                }
            }

            // Check for various optimizations that can be performed without reordering, as those don't depend on purity
            // If we can't optimize compose directly, then try and swap into a eval (if purity allows).
            // At worst, that will eliminate the overhead of a `Swap`, and potentially allow more optimizations like function call merging.
            Expr(loc, ExprType::Compose(arg, f)) => match *f {
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
                f if f.can_reorder(&arg) => self.visit(f.call(loc, vec![*arg])),

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
                        rhs.binary(loc, native_f.as_binary_operator().unwrap(), *arg, true)
                    } else {
                        arg.binary(loc, native_f.swap().as_binary_operator().unwrap(), rhs, false)
                    }
                },

                // Default, return the compose
                f => arg.compose(loc, f),
            }

            // Dead code elimination for constant expression conditions
            Expr(loc, ExprType::IfThenElse(condition, if_true, if_false)) => match condition.into_const() {
                Ok(condition) => if condition.to_bool() { *if_true } else { *if_false },
                Err(condition) => condition.if_then_else(loc, *if_true, *if_false),
            }

            _ => expr
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