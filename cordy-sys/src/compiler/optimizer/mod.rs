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


#[cfg(test)]
mod tests {
    use crate::{compiler, SourceView, util};

    #[test] fn test_constant_folding_int_add() { run("1 + 2", "Int(3) Pop") }
    #[test] fn test_constant_folding_bool_add() { run("1 + true - 4", "Int(-2) Pop") }
    #[test] fn test_constant_folding_int_complex_add() { run("1 + 1i + (2 + 2j)", "Complex(3+3i) Pop") }
    #[test] fn test_constant_folding_constant_ternary_if_true() { run("(if 1 > 0 then 'yes' else 'no')", "Str('yes') Pop") }
    #[test] fn test_constant_folding_constant_ternary_if_false() { run("(if 1 + 1 == 3 then 'yes' else 'no')", "Str('no') Pop") }
    #[test] fn test_constant_folding_constant_ternary_top_level_if_true() { run("if 1 + 1 == 3 then 'yes' else 'no'", "Str('no') Pop") }
    #[test] fn test_constant_folding_constant_ternary_top_level_if_false() { run("if 1 + 1 == 3 then 'yes' else 'no'", "Str('no') Pop") }
    #[test] fn test_compose_list_inlining() { run("1 . [2]", "Int(1) Int(2) OpIndex Pop") }
    #[test] fn test_compose_slice_inlining_1() { run("1 . [2:3]", "Int(1) Int(2) Int(3) OpSlice Pop") }
    #[test] fn test_compose_slice_inlining_2() { run("1 . [2:3:4]", "Int(1) Int(2) Int(3) Int(4) OpSliceWithStep Pop") }
    #[test] fn test_compose_reordering_pure_strong_strong() { run("1 . 2", "Int(2) Int(1) Call(1) Pop") }
    #[test] fn test_compose_reordering_both_strong_weak() { run("do { let x ; 1 . x }", "Nil PushLocal(0)->x Int(1) Call(1) PopN(2)") }
    #[test] fn test_compose_reordering_both_strong_impure() { run("do { let x ; 1 . (x = 2) }", "Nil Int(2) StoreLocal(0)->x Int(1) Call(1) PopN(2)") }
    #[test] fn test_compose_reordering_both_weak_weak() { run("do { let x, y ; x . y }", "Nil Nil PushLocal(1)->y PushLocal(0)->x Call(1) PopN(3)") }
    #[test] fn test_compose_reordering_both_weak_impure() { run("do { let x, y ; x . (y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Swap Call(1) PopN(3)") }
    #[test] fn test_compose_reordering_both_impure_impure() { run("do { let x, y ; (x = 1) . (y = 2) }", "Nil Nil Int(1) StoreLocal(0)->x Int(2) StoreLocal(1)->y Swap Call(1) PopN(3)") }
    #[test] fn test_operator_function_inlining_constant_1() { run("(+)(1)(2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_2() { run("1 . (+2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_3() { run("1 . (2+)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_4() { run("(+1)(2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_constant_5() { run("(1+)(2)", "Int(3) Pop") }
    #[test] fn test_operator_function_inlining_non_constant_1() { run("do { let x ; (+)(x)(2) }", "Nil PushLocal(0)->x Int(2) Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_2() { run("do { let x ; x . (+2) }", "Nil PushLocal(0)->x Int(2) Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_3() { run("do { let x ; x . (2+) }", "Nil Int(2) PushLocal(0)->x Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_4() { run("do { let x ; (+x)(2) }", "Nil Int(2) PushLocal(0)->x Add PopN(2)") }
    #[test] fn test_operator_function_inlining_non_constant_5() { run("do { let x ; (x+)(2) }", "Nil PushLocal(0)->x Int(2) Add PopN(2)") }
    #[test] fn test_operator_function_inlining_asymmetric_1() { run("(/)(2)(5)", "Int(0) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_2() { run("2 . (/5)", "Int(0) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_3() { run("2 . (5/)", "Int(2) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_4() { run("(/2)(5)", "Int(2) Pop") }
    #[test] fn test_operator_function_inlining_asymmetric_5() { run("(2/)(5)", "Int(0) Pop") }
    #[test] fn test_operator_function_inlining_impure_1() { run("do { let x, y ; (/)(x)(y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_2() { run("do { let x, y ; x . (/(y = 2)) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_3() { run("do { let x, y ; x . ((y = 2)/) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Swap Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_4() { run("do { let x, y ; (/x)(y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Swap Div PopN(3)") }
    #[test] fn test_operator_function_inlining_impure_5() { run("do { let x, y ; (x/)(y = 2) }", "Nil Nil PushLocal(0)->x Int(2) StoreLocal(1)->y Div PopN(3)") }
    #[test] fn test_operator_function_inlining_with_unroll() { run("(/)(...1, 2)", "OperatorDiv Int(1) Unroll Int(2) Call...(2) Pop") }
    #[test] fn test_inline_int_min() { run("min(int)", "Int(-4611686018427387904) Pop") }
    #[test] fn test_inline_int_max() { run("int.max", "Int(4611686018427387903) Pop") }
    #[test] fn test_partial_function_call_merge_no_args_1() { run("vector()()", "Vector Call(0) Call(0) Pop"); }
    #[test] fn test_partial_function_call_merge_no_args_2() { run("vector()(1)", "Vector Call(0) Int(1) Call(1) Pop"); }
    #[test] fn test_partial_function_call_merge_one_arg_1() { run("int()()", "Int Call(0) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_2() { run("int()(1)", "Int Int(1) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_3() { run("int()(1)(2)", "Int Int(1) Call(1) Int(2) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_4() { run("int()(1)()(2)", "Int Int(1) Call(1) Call(0) Int(2) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_one_arg_5() { run("int()()(1)()()", "Int Int(1) Call(1) Call(0) Call(0) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_1() { run("map(1)(2)", "Map Int(1) Int(2) Call(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_2() { run("map()(1)(2)", "Map Int(1) Int(2) Call(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_3() { run("map()(1)()(2)", "Map Int(1) Int(2) Call(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_4() { run("map(1)(2)()", "Map Int(1) Int(2) Call(2) Call(0) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_5() { run("map(1)()(2, 3)", "Map Int(1) Int(2) Int(3) Call(3) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_6() { run("map(1, 2)()(3)", "Map Int(1) Int(2) Call(2) Call(0) Int(3) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_7() { run("map(1)()()", "Map Int(1) Call(1) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_unroll_1() { run("map()(...1)", "Map Int(1) Unroll Call...(1) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_unroll_2() { run("map(1)(...2)", "Map Int(1) Int(2) Unroll Call...(2) Pop") }
    #[test] fn test_partial_function_call_merge_two_arg_unroll_3() { run("map(...1)()", "Map Int(1) Unroll Call...(1) Call(0) Pop") }
    #[test] fn test_store_global_pop_is_merged() { run("let x; x = 1", "InitGlobal Nil Int(1) StoreGlobalPop(0)->x Pop") }
    #[test] fn test_store_global_is_not_pop() { run("let x; print(x = 1)", "InitGlobal Nil Print Int(1) StoreGlobal(0)->x Call(1) PopN(2)") }

    fn run(text: &'static str, expected: &'static str) {
        let expected: String = format!("{}\nExit", expected.replace(" ", "\n"));
        let actual: String = compiler::compile(true, &SourceView::new(String::new(), String::from(text)))
            .expect("Failed to compile")
            .raw_disassembly();

        util::assert_eq(actual, expected);
    }
}