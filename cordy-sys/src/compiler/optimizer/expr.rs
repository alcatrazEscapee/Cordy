use crate::compiler::optimizer::Optimize;
use crate::compiler::parser::{Expr, ExprType};
use crate::core::NativeFunction;
use crate::vm::{IntoValue, LiteralType, MAX_INT, MIN_INT, RuntimeError, ValuePtr};
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
            e @ Expr(_, ExprType::Nil | ExprType::Exit | ExprType::Bool(_) | ExprType::Int(_) | ExprType::Str(_) | ExprType::LValue(_) | ExprType::Function(_, _) | ExprType::NativeFunction(_)) => e,

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
                        Err(rhs) => Expr::value(lhs).binary(loc, op, rhs, swap),
                    },
                    Err(lhs) => lhs.binary(loc, op, rhs, swap),
                }
            },

            Expr(loc, ExprType::Literal(op, args)) => Expr(loc, ExprType::Literal(op, args.optimize())),
            Expr(loc, ExprType::Unroll(arg, first)) => arg.optimize().unroll(loc, first),

            Expr(loc, ExprType::Eval(f, args, any_unroll)) => {
                let f: Expr = f.optimize();
                let mut args: Vec<Expr> = args.optimize();
                let nargs: Option<usize> = if any_unroll { None } else { Some(args.len()) }; // nargs is only valid if no unrolls are present

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
                    Expr(_, ExprType::NativeFunction(native_f @ (NativeFunction::Min | NativeFunction::Max))) if nargs == Some(1) => {
                        if let Expr(_, ExprType::NativeFunction(NativeFunction::Int)) = args[0] {
                            Expr::int(if native_f == NativeFunction::Min { MIN_INT } else { MAX_INT })
                        } else {
                            f.eval(loc, args, any_unroll)
                        }
                    },

                    // If we can assert the inner function is partial, then we can merge the two calls:
                    // - We know the function being called is partial with the given number of arguments, and
                    // - The call is not unrolled (because then we can never prove it is partial
                    //
                    // Note this does not require any reordering of the arguments
                    // This re-calls `.optimize()` as we might be able to replace the operator on the constant-eval'd function
                    Expr(_, ExprType::Eval(f_inner, mut args_inner, false)) if f_inner.is_partial(args_inner.len()) => {
                        args_inner.append(&mut args);
                        f_inner.eval(loc, args_inner, any_unroll).optimize()
                    },

                    f => f.eval(loc, args, any_unroll)
                }
            },

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
                            Expr::error(loc, Box::new(RuntimeError::ValueErrorEvalListMustHaveUnitLength(nargs)))
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
                    Expr(_, ExprType::Eval(f_inner, mut args, false)) if is_native_operator(&f_inner) && args.len() == 1 => {
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
            ExprType::Nil | ExprType::Exit | ExprType::Bool(_) | ExprType::Int(_) | ExprType::Str(_) | ExprType::NativeFunction(_) | ExprType::Function(_, _) => Purity::Strong,
            ExprType::LValue(_) => Purity::Weak,

            ExprType::Unary(_, arg) => arg.purity(),
            ExprType::Binary(_, lhs, rhs, _) | ExprType::LogicalOr(lhs, rhs) | ExprType::LogicalAnd(lhs, rhs) => lhs.purity().min(rhs.purity()),
            ExprType::Literal(_, args) => args.iter().map(|u| u.purity()).min().unwrap_or(Purity::Strong),
            ExprType::Unroll(arg, _) => arg.purity(),
            ExprType::IfThenElse(condition, if_true, if_false) => condition.purity().min(if_true.purity()).min(if_false.purity()),

            ExprType::Eval(f, args, any_unroll) => match !*any_unroll && f.is_partial(args.len()) {
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


#[cfg(test)]
mod tests {
    use crate::{compiler, SourceView, util};

    #[test] fn test_constant_folding_int_add() { run_expr("1 + 2", "Int(3) Pop") }
    #[test] fn test_constant_folding_bool_add() { run_expr("1 + true - 4", "Int(-2) Pop") }
    #[test] fn test_constant_folding_int_complex_add() { run_expr("1 + 1i + (2 + 2j)", "Complex(3+3i) Pop") }
    #[test] fn test_constant_folding_constant_ternary_if_true() { run_expr("(if 1 > 0 then 'yes' else 'no')", "Str('yes') Pop") }
    #[test] fn test_constant_folding_constant_ternary_if_false() { run_expr("(if 1 + 1 == 3 then 'yes' else 'no')", "Str('no') Pop") }
    #[test] fn test_constant_folding_constant_ternary_top_level_if_true() { run_expr("if 1 + 1 == 3 then 'yes' else 'no'", "Str('no') Pop") }
    #[test] fn test_constant_folding_constant_ternary_top_level_if_false() { run_expr("if 1 + 1 == 3 then 'yes' else 'no'", "Str('no') Pop") }
    #[test] fn test_compose_list_inlining() { run_expr("1 . [2]", "Int(1) Int(2) OpIndex Pop") }
    #[test] fn test_compose_slice_inlining_1() { run_expr("1 . [2:3]", "Int(1) Int(2) Int(3) OpSlice Pop") }
    #[test] fn test_compose_slice_inlining_2() { run_expr("1 . [2:3:4]", "Int(1) Int(2) Int(3) Int(4) OpSliceWithStep Pop") }
    #[test] fn test_compose_reordering_pure_strong_strong() { run_expr("1 . 2", "Int(2) Int(1) Call(1) Pop") }
    #[test] fn test_compose_reordering_both_strong_weak() { run_expr("do { let x ; 1 . x }", "Nil PushLocal(0)->x Int(1) Call(1) PopN(2)") }
    #[test] fn test_compose_reordering_both_strong_impure() { run_expr("do { let x ; 1 . (x = 2) }", "Nil Int(2) StoreLocal(0)->x Int(1) Call(1) PopN(2)") }
    #[test] fn test_compose_reordering_both_weak_weak() { run_expr("do { let x, y ; x . y }", "Nil Nil PushLocal(1)->y PushLocal(0)->x Call(1) PopN(3)") }
    #[test] fn test_compose_reordering_both_weak_impure() { run_expr("do { let x, y ; x . (y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Swap Call(1) PopN(3)") }
    #[test] fn test_compose_reordering_both_impure_impure() { run_expr("do { let x, y ; (x = 1) . (y = 2) }", "Nil Nil Int(1) StoreLocal(0)->x Int(2) StoreLocal(1)->y Swap Call(1) PopN(3)") }
    #[test] fn test_operator_function_inlining_constant_1() { run_expr("(+)(1)(2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_2() { run_expr("1 . (+2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_3() { run_expr("1 . (2+)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_4() { run_expr("(+1)(2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_5() { run_expr("(1+)(2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_non_constant_1() { run_expr("do { let x ; (+)(x)(2) }", "Nil PushLocal(0)->x Int(2) Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_2() { run_expr("do { let x ; x . (+2) }", "Nil PushLocal(0)->x Int(2) Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_3() { run_expr("do { let x ; x . (2+) }", "Nil Int(2) PushLocal(0)->x Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_4() { run_expr("do { let x ; (+x)(2) }", "Nil Int(2) PushLocal(0)->x Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_5() { run_expr("do { let x ; (x+)(2) }", "Nil PushLocal(0)->x Int(2) Add PopN(2)") }
    #[test] fn test_operator_function_inlining_asymmetric_1() { run_expr("(/)(2)(5)", "Int(0) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_2() { run_expr("2 . (/5)", "Int(0) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_3() { run_expr("2 . (5/)", "Int(2) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_4() { run_expr("(/2)(5)", "Int(2) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_5() { run_expr("(2/)(5)", "Int(0) Pop") }
    #[test] fn test_operator_function_inlining_impure_1() { run_expr("do { let x, y ; (/)(x)(y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_2() { run_expr("do { let x, y ; x . (/(y = 2)) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_3() { run_expr("do { let x, y ; x . ((y = 2)/) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Swap Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_4() { run_expr("do { let x, y ; (/x)(y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Swap Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_5() { run_expr("do { let x, y ; (x/)(y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Div PopN(3)") }
    #[test] fn test_operator_function_inlining_with_unroll() { run_expr("(/)(...1, 2)", "OperatorDiv Int(1) Unroll Int(2) Call...(2) Pop") }
    #[test] fn test_inline_int_min() { run_expr("min(int)", "Int(-4611686018427387904) Pop") }
    #[test] fn test_inline_int_max() { run_expr("int.max", "Int(4611686018427387903) Pop") }
    #[test] fn test_partial_function_call_merge_no_args_1() { run_expr("vector()()", "Vector Call(0) Call(0) Pop"); }
    #[test] fn test_partial_function_call_merge_no_args_2() { run_expr("vector()(1)", "Vector Call(0) Int(1) Call(1) Pop"); }
    #[test] fn test_partial_function_call_merge_one_arg_1() { run_expr("int()()", "Int Call(0) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_2() { run_expr("int()(1)", "Int Int(1) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_3() { run_expr("int()(1)(2)", "Int Int(1) Call(1) Int(2) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_4() { run_expr("int()(1)()(2)", "Int Int(1) Call(1) Call(0) Int(2) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_5() { run_expr("int()()(1)()()", "Int Int(1) Call(1) Call(0) Call(0) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_1() { run_expr("map(1)(2)", "Map Int(1) Int(2) Call(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_2() { run_expr("map()(1)(2)", "Map Int(1) Int(2) Call(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_3() { run_expr("map()(1)()(2)", "Map Int(1) Int(2) Call(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_4() { run_expr("map(1)(2)()", "Map Int(1) Int(2) Call(2) Call(0) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_5() { run_expr("map(1)()(2, 3)", "Map Int(1) Int(2) Int(3) Call(3) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_6() { run_expr("map(1, 2)()(3)", "Map Int(1) Int(2) Call(2) Call(0) Int(3) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_7() { run_expr("map(1)()()", "Map Int(1) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_unroll_1() { run_expr("map()(...1)", "Map Int(1) Unroll Call...(1) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_unroll_2() { run_expr("map(1)(...2)", "Map Int(1) Int(2) Unroll Call...(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_unroll_3() { run_expr("map(...1)()", "Map Int(1) Unroll Call...(1) Call(0) Pop") }
    #[test] fn test_store_global_pop_is_merged() { run_expr("let x; x = 1", "InitGlobal Nil Int(1) StoreGlobalPop(0)->x Pop") }

    fn run_expr(text: &'static str, expected: &'static str) {
        let expected: String = format!("{}\nExit", expected.replace(" ", "\n"));
        let actual: String = compiler::compile(true, &SourceView::new(String::new(), String::from(text)))
            .expect("Failed to compile")
            .raw_disassembly();

        util::assert_eq(actual, expected);
    }
}