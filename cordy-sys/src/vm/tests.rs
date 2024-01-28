use crate::{compiler, util};
use crate::reporting::SourceView;
use crate::vm::{ExitType, VirtualMachine};


fn run(text: &'static str, expected: &'static str) {
    run_with(true, text, expected)
}

fn run_unopt(text: &'static str, expected: &'static str) {
    run_with(false, text, expected)
}

macro_rules! run {
    ($path:literal) => {
        run(
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/test/compiler/", $path, ".cor")),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/test/compiler/", $path, ".cor.trace"))
        )
    };
    ($path:literal, $expected:literal) => {
        run(
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/test/compiler/", $path, ".cor")),
            $expected
        )
    }
}

fn run_with(opt: bool, text: &'static str, expected: &'static str) {
    let view: SourceView = SourceView::new(String::from("<test>"), String::from(text));
    let expected = expected.replace("\r", "");

    let compile = match compiler::compile(opt, &view) {
        Err(e) => {
            util::assert_eq(format!("Compile Error:\n\n{}", e.join("\n")), expected);
            return
        }
        Ok(c) => c
    };

    println!("[-d] === Compiled ===");
    for line in compile.disassemble(&view, true) {
        println!("[-d] {}", line);
    }

    let mut vm = VirtualMachine::default(compile, view);

    let result: ExitType = vm.run_until_completion();
    assert!(vm.stack.is_empty() || result.is_early_exit());

    let view: SourceView = vm.view;
    let mut output: String = String::from_utf8(vm.write).unwrap();

    if let ExitType::Error(ref error) = result {
        output.push_str(view.format(error).as_str());
    }
    util::assert_eq(output, expected);
}



#[test] fn empty() { run("", "") }
#[test] fn compose_1() { run("print . print", "print\n") }
#[test] fn compose_2() { run("'hello world' . print", "hello world\n") }
#[test] fn if_01() { run("if 1 < 2 { print('yes') } else { print ('no') }", "yes\n") }
#[test] fn if_02() { run("if 1 < -2 { print('yes') } else { print ('no') }", "no\n") }
#[test] fn if_03() { run("if true { print('yes') } print('and also')", "yes\nand also\n") }
#[test] fn if_04() { run("if 1 < -2 { print('yes') } print('and also')", "and also\n") }
#[test] fn if_05() { run("if 0 { print('yes') }", "") }
#[test] fn if_06() { run("if 1 { print('yes') }", "yes\n") }
#[test] fn if_07() { run("if 'string' { print('yes') }", "yes\n") }
#[test] fn if_08() { run("if 1 < 0 { print('yes') } elif 1 { print('hi') } else { print('hehe') }", "hi\n") }
#[test] fn if_09() { run("if 1 < 0 { print('yes') } elif 2 < 0 { print('hi') } else { print('hehe') }", "hehe\n") }
#[test] fn if_10() { run("if 1 { print('yes') } elif true { print('hi') } else { print('hehe') }", "yes\n") }
#[test] fn if_short_circuiting_1() { run("if true and print('yes') { print('no') }", "yes\n") }
#[test] fn if_short_circuiting_2() { run("if false and print('also no') { print('no') }", "") }
#[test] fn if_short_circuiting_3() { run("if true and (print('yes') or true) { print('also yes') }", "yes\nalso yes\n") }
#[test] fn if_short_circuiting_4() { run("if false or print('yes') { print('no') }", "yes\n") }
#[test] fn if_short_circuiting_5() { run("if true or print('no') { print('yes') }", "yes\n") }
#[test] fn if_then_else_1() { run("(if true then 'hello' else 'goodbye') . print", "hello\n") }
#[test] fn if_then_else_2() { run("(if false then 'hello' else 'goodbye') . print", "goodbye\n") }
#[test] fn if_then_else_3() { run("(if [] then 'hello' else 'goodbye') . print", "goodbye\n") }
#[test] fn if_then_else_4() { run("(if 3 then 'hello' else 'goodbye') . print", "hello\n") }
#[test] fn if_then_else_5() { run("(if false then (fn() -> 'hello' . print)() else 'nope') . print", "nope\n") }
#[test] fn if_then_else_top_level() { run("if true then print('hello') else print('goodbye')", "hello\n") }
#[test] fn if_then_else_top_level_in_loop() { run("for x in range(2) { if x then x else x }", "") }
#[test] fn if_then_elif_else_true_true() { run("print(if true then 1 elif true then 2 else 3)", "1\n") }
#[test] fn if_then_elif_else_true_false() { run("print(if true then 1 elif false then 2 else 3)", "1\n") }
#[test] fn if_then_elif_else_false_true() { run("print(if false then 1 elif true then 2 else 3)", "2\n") }
#[test] fn if_then_elif_else_false_false() { run("print(if false then 1 elif false then 2 else 3)", "3\n") }
#[test] fn if_then_elif_top_level_else_true_true() { run("if true then print 1 elif true then print 2 else print 3", "1\n") }
#[test] fn if_then_elif_top_level_else_true_false() { run("if true then print 1 elif false then print 2 else print 3", "1\n") }
#[test] fn if_then_elif_top_level_else_false_true() { run("if false then print 1 elif true then print 2 else print 3", "2\n") }
#[test] fn if_then_elif_top_level_else_false_false() { run("if false then print 1 elif false then print 2 else print 3", "3\n") }
#[test] fn loop_has_nested_scope() { run("let x ; loop { let x ; break }", "") }
#[test] fn loop_2x_nested_with_break_1() { run("loop { break ; loop { } }", "") }
#[test] fn loop_2x_nested_with_break_2() { run("let x ; loop { break ; loop { } }", "") }
#[test] fn loop_2x_nested_with_break_3() { run("let x ; loop { let y ; break ; loop { } }", "") }
#[test] fn loop_2x_nested_with_break_4() { run("let x ; loop { let y ; break ; let z ; loop { } }", "") }
#[test] fn loop_2x_nested_with_break_5() { run("let x ; loop { break ; loop { } }", "") }
#[test] fn loop_2x_nested_with_break_6() { run("let x ; loop { let y ; break ; loop { } }", "") }
#[test] fn loop_2x_nested_with_break_7() { run("let x ; loop { let y ; break ; loop { let z } }", "") }
#[test] fn loop_2x_nested_with_break_8() { run("let x ; loop { break ; loop { let z } }", "") }
#[test] fn loop_2x_nested_with_break_9() { run("loop { break ; loop { let z } }", "") }
#[test] fn loop_2x_nested_with_break_10() { run("loop { loop { break } ; break }", "") }
#[test] fn while_loop_has_nested_scope() { run("let x ; while nil { let x }", "") }
#[test] fn while_false_if_false() { run("while false { if false { } }", "") }
#[test] fn while_2x_nested_with_break_1() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { break } }", "") }
#[test] fn while_2x_nested_with_break_2() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { break } }", "") }
#[test] fn while_2x_nested_with_break_3() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z ; break } }", "") }
#[test] fn while_2x_nested_with_break_4() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z ; break ; let w } }", "") }
#[test] fn while_2x_nested_with_break_5() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { let z ; break ; let w } }", "") }
#[test] fn while_2x_nested_with_break_6() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { break ; let w } }", "") }
#[test] fn while_2x_nested_with_break_7() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { } break }", "") }
#[test] fn while_2x_nested_with_break_8() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { } break }", "") }
#[test] fn while_2x_nested_with_break_9() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z } break }", "") }
#[test] fn while_2x_nested_with_break_10() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z } let z ; break }", "") }
#[test] fn while_2x_nested_with_break_11() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z } let z ; break ; let w }", "") }
#[test] fn while_2x_nested_with_break_12() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { let z } let z ; break ; let w }", "") }
#[test] fn while_2x_nested_with_break_13() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { } let z ; break ; let w }", "") }
#[test] fn while_2x_nested_with_break_14() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { } break ; let w }", "") }
#[test] fn while_2x_nested_with_continue_1() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { continue } }", "") }
#[test] fn while_2x_nested_with_continue_2() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { continue } }", "") }
#[test] fn while_2x_nested_with_continue_3() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z ; continue } }", "") }
#[test] fn while_2x_nested_with_continue_4() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z ; continue ; let w } }", "") }
#[test] fn while_2x_nested_with_continue_5() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { let z ; continue ; let w } }", "") }
#[test] fn while_2x_nested_with_continue_6() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { continue ; let w } }", "") }
#[test] fn while_2x_nested_with_continue_7() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { } continue }", "") }
#[test] fn while_2x_nested_with_continue_8() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { } continue }", "") }
#[test] fn while_2x_nested_with_continue_9() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z } continue }", "") }
#[test] fn while_2x_nested_with_continue_10() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z } let z ; continue }", "") }
#[test] fn while_2x_nested_with_continue_11() { run("let x = [0,0,1,2] ; while x.pop { let y ; while x.pop { let z } let z ; continue ; let w }", "") }
#[test] fn while_2x_nested_with_continue_12() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { let z } let z ; continue ; let w }", "") }
#[test] fn while_2x_nested_with_continue_13() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { } let z ; continue ; let w }", "") }
#[test] fn while_2x_nested_with_continue_14() { run("let x = [0,0,1,2] ; while x.pop { while x.pop { } continue ; let w }", "") }
#[test] fn while_else_no_loop() { run("while false { break } else { print('hello') }", "hello\n") }
#[test] fn while_else_break() { run("while true { break } else { print('hello') } print('world')", "world\n") }
#[test] fn while_else_no_break() { run("let x = true ; while x { x = false } else { print('hello') }", "hello\n") }
#[test] fn do_loop_has_nested_scope() { run("let x ; do { let x }", "") }
#[test] fn do_while_loop_has_nested_scope() { run("let x ; do { let x } while nil", "") }
#[test] fn do_while_1() { run("do { 'test' . print } while false", "test\n") }
#[test] fn do_while_2() { run("let i = 0 ; do { i . print ; i += 1 } while i < 3", "0\n1\n2\n") }
#[test] fn do_while_3() { run("let i = 0 ; do { i += 1 ; i . print } while i < 3", "1\n2\n3\n") }
#[test] fn do_while_4() { run("let i = 5 ; do { i . print } while i < 3", "5\n") }
#[test] fn do_while_2x_nested_with_break_1() { run("do { break ; do { } while 2 } while 1", "") }
#[test] fn do_while_2x_nested_with_break_2() { run("let x ; do { break ; do { } while 2 } while 1", "") }
#[test] fn do_while_2x_nested_with_break_3() { run("let x ; do { let y ; break ; do { } while 2 } while 1", "") }
#[test] fn do_while_2x_nested_with_break_4() { run("let x ; do { let y ; break ; do { let z ; } while 2 } while 1", "") }
#[test] fn do_while_2x_nested_with_break_5() { run("do { do { break } while 2 ; break } while 1", "") }
#[test] fn do_while_2x_nested_with_break_6() { run("let x ; do { do { break } while 2 ; break } while 1", "") }
#[test] fn do_while_2x_nested_with_break_7() { run("let x ; do { let y ; do { break } while 2 ; break } while 1", "") }
#[test] fn do_while_2x_nested_with_break_8() { run("let x ; do { let y ; do { let z ; break } while 2 ; break } while 1", "") }
#[test] fn do_while_2x_nested_with_break_9() { run("let x ; do { let y ; do { let z ; break ; let w } while 2 ; break } while 1", "") }
#[test] fn do_while_2x_nested_with_break_10() { run("do { do { let z ; break ; let w } while 2 ; break } while 1", "") }
#[test] fn do_while_2x_nested_with_break_11() { run("do { do { break ; let w } while 2 ; break } while 1", "") }
#[test] fn do_while_2x_nested_with_continue_1() { run("let x = [0,0,1,2] ; do { while x.pop { continue } } while x.pop", "") }
#[test] fn do_while_2x_nested_with_continue_2() { run("let x = [0,0,1,2] ; do { let y ; while x.pop { continue } } while x.pop", "") }
#[test] fn do_while_2x_nested_with_continue_3() { run("let x = [0,0,1,2] ; do { let y ; while x.pop { let z ; continue } } while x.pop", "") }
#[test] fn do_while_2x_nested_with_continue_4() { run("let x = [0,0,1,2] ; do { let y ; while x.pop { let z ; continue ; let w } } while x.pop", "") }
#[test] fn do_while_2x_nested_with_continue_5() { run("let x = [0,0,1,2] ; do { while x.pop { let z ; continue ; let w } } while x.pop", "") }
#[test] fn do_while_2x_nested_with_continue_6() { run("let x = [0,0,1,2] ; do { while x.pop { continue ; let w } } while x.pop", "") }
#[test] fn do_while_2x_nested_with_continue_7() { run("let x = [0,0,1,2] ; while x.pop { do { } while x.pop continue }", "") }
#[test] fn do_while_2x_nested_with_continue_8() { run("let x = [0,0,1,2] ; while x.pop { let y ; do { } while x.pop continue }", "") }
#[test] fn do_while_2x_nested_with_continue_9() { run("let x = [0,0,1,2] ; while x.pop { let y ; do { let z } while x.pop continue }", "") }
#[test] fn do_while_2x_nested_with_continue_10() { run("let x = [0,0,1,2] ; while x.pop { let y ; do { let z } while x.pop let z ; continue }", "") }
#[test] fn do_while_2x_nested_with_continue_11() { run("let x = [0,0,1,2] ; while x.pop { let y ; do { let z } while x.pop let z ; continue ; let w }", "") }
#[test] fn do_while_2x_nested_with_continue_12() { run("let x = [0,0,1,2] ; while x.pop { do { let z } while x.pop let z ; continue ; let w }", "") }
#[test] fn do_while_2x_nested_with_continue_13() { run("let x = [0,0,1,2] ; while x.pop { do { } while x.pop let z ; continue ; let w }", "") }
#[test] fn do_while_2x_nested_with_continue_14() { run("let x = [0,0,1,2] ; while x.pop { do { } while x.pop continue ; let w }", "") }
#[test] fn do_without_while() { run("do { 'test' . print }", "test\n") }
#[test] fn do_while_else_1() { run("do { 'loop' . print } while false else { 'else' . print }", "loop\nelse\n") }
#[test] fn do_while_else_2() { run("do { 'loop' . print ; break } while false else { 'else' . print }", "loop\n") }
#[test] fn do_while_else_3() { run("let i = 0 ; do { i . print ; i += 1 ; if i > 2 { break } } while 1 else { 'end' . print }", "0\n1\n2\n") }
#[test] fn do_while_else_4() { run("let i = 0 ; do { i . print ; i += 1 ; if i > 2 { break } } while i < 2 else { 'end' . print }", "0\n1\nend\n") }
#[test] fn for_loop_has_nested_scope() { run("let x ; for x in [] {}", "") }
#[test] fn for_loop_has_2x_nested_scope() { run("let x ; for x in [] { let x }", "") }
#[test] fn for_loop_no_intrinsic_with_list() { run("for x in ['a', 'b', 'c'] { x . print }", "a\nb\nc\n") }
#[test] fn for_loop_no_intrinsic_with_set() { run("for x in 'foobar' . set { x . print }", "f\no\nb\na\nr\n") }
#[test] fn for_loop_no_intrinsic_with_str() { run("for x in 'hello' { x . print }", "h\ne\nl\nl\no\n") }
#[test] fn for_loop_range_stop() { run("for x in range(5) { x . print }", "0\n1\n2\n3\n4\n") }
#[test] fn for_loop_range_start_stop() { run("for x in range(3, 6) { x . print }", "3\n4\n5\n") }
#[test] fn for_loop_range_start_stop_step_positive() { run("for x in range(1, 10, 3) { x . print }", "1\n4\n7\n") }
#[test] fn for_loop_range_start_stop_step_negative() { run("for x in range(11, 0, -4) { x . print }", "11\n7\n3\n") }
#[test] fn for_loop_range_start_stop_step_zero() { run("for x in range(1, 2, 0) { x . print }", "ValueError: 'step' argument cannot be zero\n  at: line 1 (<test>)\n\n1 | for x in range(1, 2, 0) { x . print }\n2 |               ^^^^^^^^^\n") }
#[test] fn for_loop_2x_nested_with_break_1() { run("for _ in 'a' { for _ in 'b' { break } }", "") }
#[test] fn for_loop_2x_nested_with_break_2() { run("for a in 'a' { for b in 'b' { break } }", "") }
#[test] fn for_loop_2x_nested_with_break_3() { run("for a in 'a' { let x ; for b in 'b' { break } }", "") }
#[test] fn for_loop_2x_nested_with_break_4() { run("for a in 'a' { let x ; for b in 'b' { let y ; break } }", "") }
#[test] fn for_loop_2x_nested_with_break_5() { run("for a in 'a' { for b in 'b' { } break }", "") }
#[test] fn for_loop_2x_nested_with_break_6() { run("for a in 'a' { let x ; for b in 'b' { } break }", "") }
#[test] fn for_loop_2x_nested_with_break_7() { run("for a in 'a' { let x ; for b in 'b' { let y ; } break }", "") }
#[test] fn for_loop_2x_nested_with_break_8() { run("let a ; for i in 'a' { let b ; for j in 'b' { break } }", "")}
#[test] fn for_loop_2x_nested_with_continue_1() { run("for _ in 'a' { for _ in 'b' { continue } }", "") }
#[test] fn for_loop_2x_nested_with_continue_2() { run("for a in 'a' { for b in 'b' { continue } }", "") }
#[test] fn for_loop_2x_nested_with_continue_3() { run("for a in 'a' { let x ; for b in 'b' { continue } }", "") }
#[test] fn for_loop_2x_nested_with_continue_4() { run("for a in 'a' { let x ; for b in 'b' { let y ; continue } }", "") }
#[test] fn for_loop_2x_nested_with_continue_5() { run("for a in 'a' { for b in 'b' { } continue }", "") }
#[test] fn for_loop_2x_nested_with_continue_6() { run("for a in 'a' { let x ; for b in 'b' { } continue }", "") }
#[test] fn for_loop_2x_nested_with_continue_7() { run("for a in 'a' { let x ; for b in 'b' { let y } continue }", "") }
#[test] fn for_else_no_loop() { run("for _ in [] { print('hello') ; break } else { print('world') }", "world\n") }
#[test] fn for_else_break() { run("for c in 'abcd' { if c == 'b' { break } } else { print('hello') } print('world')", "world\n") }
#[test] fn for_else_break_2() { run("for x in 'a' { 'hello' break } else { 'world'}", "") }
#[test] fn for_else_no_break() { run("for c in 'abcd' { if c == 'B' { break } } else { print('hello') }", "hello\n") }
#[test] fn for_else_no_break_2() { run("for x in '' { 'hello' break } else { 'world'}", "") }
#[test] fn for_loop_modify_loop_variable() { run("for i in range(5) { i = 5 ; i . print }", "5\n5\n5\n5\n5\n") }
#[test] fn for_loop_range_map() { run("for i in range(5) . map(+3) { i . print }", "3\n4\n5\n6\n7\n") }
#[test] fn for_loop_with_multiple_references() { run!("for_loop_with_multiple_references", "[1, 2, 3, 4]\n") }
#[test] fn struct_str_of_struct_instance() { run("struct Foo(a, b) Foo(1, 2) . print", "Foo(a=1, b=2)\n") }
#[test] fn struct_str_of_struct_constructor() { run("struct Foo(a, b) Foo . print", "struct Foo(a, b)\n") }
#[test] fn struct_get_field_of_struct() { run("struct Foo(a, b) Foo(1, 2) -> a . print", "1\n") }
#[test] fn struct_get_field_of_struct_wrong_name() { run("struct Foo(a, b) struct Bar(c, d) Foo(1, 2) -> c . print", "TypeError: Cannot get field 'c' on struct Foo(a, b)\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) struct Bar(c, d) Foo(1, 2) -> c . print\n2 |                                             ^^^^\n") }
#[test] fn struct_get_field_of_struct_in_self_method_with_other_struct() { run("struct Bad(a) ; struct Good(b) { fn get(self) -> b } ; Good(1)->get() . print", "1\n") }
#[test] fn struct_get_field_of_not_struct() { run("struct Foo(a, b) (1, 2) -> a . print", "TypeError: Cannot get field 'a' on '(1, 2)' of type 'vector'\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) (1, 2) -> a . print\n2 |                         ^^^^\n") }
#[test] fn struct_get_field_with_overlapping_offsets() { run("struct Foo(a, b) struct Bar(b, a) Foo(1, 2) -> b . print", "2\n") }
#[test] fn struct_set_field_of_struct() { run("struct Foo(a, b) let x = Foo(1, 2) ; x->a = 3 ; x->a . print", "3\n") }
#[test] fn struct_set_field_of_struct_wrong_name() { run("struct Foo(a, b) struct Bar(c, d) let x = Foo(1, 2) ; x->c = 3", "TypeError: Cannot set field 'c' on struct Foo(a, b)\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) struct Bar(c, d) let x = Foo(1, 2) ; x->c = 3\n2 |                                                            ^\n") }
#[test] fn struct_set_field_of_not_struct() { run("struct Foo(a, b) (1, 2)->a = 3", "TypeError: Cannot set field 'a' on '(1, 2)' of type 'vector'\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) (1, 2)->a = 3\n2 |                            ^\n") }
#[test] fn struct_op_set_field_of_struct() { run("struct Foo(a, b) let x = Foo(1, 2) ; x->a += 3 ; x->a . print", "4\n") }
#[test] fn struct_partial_get_field_in_bare_method() { run("struct Foo(a, b) let x = Foo(2, 3), f = (->b) ; x . f . print", "3\n") }
#[test] fn struct_partial_get_field_in_function_eval() { run("struct Foo(a, b) [Foo(1, 2), Foo(2, 3)] . map(->b) . print", "[2, 3]\n") }
#[test] fn struct_more_partial_get_field() { run("struct Foo(foo) ; let x = Foo('hello') ; print([x, Foo('')] . filter(->foo) . len)", "1\n") }
#[test] fn struct_recursive_repr() { run("struct S(x) ; let x = S(nil) ; x->x = x ; x.print", "S(x=S(...))\n") }
#[test] fn struct_operator_is() { run("struct A() ; struct B() let a = A(), b = B() ; [a is A, A is function, a is B, A is A, a is function] . print", "[true, true, false, false, false]\n") }
#[test] fn struct_construct_not_enough_arguments() { run("struct Foo(a, b, c) ; Foo(1)(2) . print ; ", "Incorrect number of arguments for struct Foo(a, b, c), got 1\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b, c) ; Foo(1)(2) . print ; \n2 |                          ^^^\n") }
#[test] fn struct_construct_too_many_arguments() { run("struct Foo(a, b, c) ; Foo(1, 2, 3, 4) . print", "Incorrect number of arguments for struct Foo(a, b, c), got 4\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b, c) ; Foo(1, 2, 3, 4) . print\n2 |                          ^^^^^^^^^^^^\n") }
#[test] fn struct_with_method() { run("struct Square(side) { fn area(sq) -> sq->side ** 2 } let x = Square(3) ; Square->area(x) . print", "9\n") }
#[test] fn struct_self_method_const_no_arg() { run("struct A() { fn f(self) -> 123 } ; A()->f() . print", "123\n") }
#[test] fn struct_self_method_const_no_arg_repr() { run("struct A() { fn f(self) -> 123 } ; A()->f . repr . print", "fn f(self)\n") }
#[test] fn struct_self_method_const_no_arg_repr_of_static() { run("struct A() { fn f(self) -> 123 } ; A->f . repr . print", "fn f(self)\n") }
#[test] fn struct_self_method_const_no_arg_call_from_static() { run("struct A() { fn f(self) -> 123 } ; A->f(A()) . print", "123\n") }
#[test] fn struct_self_method_const_one_arg() { run("struct A() { fn f(self, x) -> 123 } ; A()->f(456) . print", "123\n") }
#[test] fn struct_self_method_const_one_arg_repr() { run("struct A() { fn f(self, x) -> 123 } ; A()->f . repr . print", "fn f(self, x)\n") }
#[test] fn struct_self_method_const_one_arg_repr_of_static() { run("struct A() { fn f(self, x) -> 123 } ; A->f . repr . print", "fn f(self, x)\n") }
#[test] fn struct_self_method_const_one_arg_call_from_static() { run("struct A() { fn f(self, x) -> 123 } ; A->f(A(), 456) . print", "123\n") }
#[test] fn struct_self_method_const_one_arg_call_from_static_partial() { run("struct A() { fn f(self, x) -> 123 } ; A->f(A()) . repr . print", "fn f(self, x)\n") }
#[test] fn struct_self_method_this_no_arg() { run("struct A(a) { fn f(self) -> self } ; A(123)->f() . print", "A(a=123)\n") }
#[test] fn struct_self_method_this_no_arg_call_from_static() { run("struct A(a) { fn f(self) -> self } ; A->f(A(123)) . print", "A(a=123)\n") }
#[test] fn struct_self_method_this_one_arg() { run("struct A(a) { fn f(self, a) -> self } ; A(123)->f(456) . print", "A(a=123)\n") }
#[test] fn struct_self_method_this_one_arg_call_from_static() { run("struct A(a) { fn f(self, x) -> self } ; A->f(A(123), 456) . print", "A(a=123)\n") }
#[test] fn struct_self_method_this_one_arg_call_from_static_partial() { run("struct A(a) { fn f(self, x) -> self } ; A->f(A(123)) . repr . print", "fn f(self, x)\n") }
#[test] fn struct_self_method_get_field() { run("struct A(a) { fn f(self) -> self->a } ; A(123)->f() . print", "123\n") }
#[test] fn struct_self_method_get_field_repr() { run("struct A(a) { fn f(self) -> self->a } ; A(123)->f . repr . print", "fn f(self)\n") }
#[test] fn struct_self_method_get_field_repr_of_static() { run("struct A(a) { fn f(self) -> self->a } ; A->f . repr . print", "fn f(self)\n") }
#[test] fn struct_self_method_get_field_call_from_static() { run("struct A(a) { fn f(self) -> self->a } ; A->f(A(123)) . print", "123\n") }
#[test] fn struct_self_method_get_field_method_ref() { run("struct A(a) { fn f(self) -> self->a } ; let q = A(123)->f ; q() . print", "123\n") }
#[test] fn struct_self_method_box_get_repr() { run("struct A(x) { fn get(self) { self->x } fn set(self, y) { self->x = y } } let a = A(123) ; a->get . print", "get\n") }
#[test] fn struct_self_method_box_get_call() { run("struct A(x) { fn get(self) { self->x } fn set(self, y) { self->x = y } } let a = A(123) ; a->get() . print", "123\n") }
#[test] fn struct_self_method_box_set_repr() { run("struct A(x) { fn get(self) { self->x } fn set(self, y) { self->x = y } } let a = A(123) ; a->set . print", "set\n") }
#[test] fn struct_self_method_box_set_call() { run("struct A(x) { fn get(self) { self->x } fn set(self, y) { self->x = y } } let a = A(123) ; a->set() . print ; a.print", "set\nA(x=123)\n") }
#[test] fn struct_self_method_box_set_call_arg() { run("struct A(x) { fn get(self) { self->x } fn set(self, y) { self->x = y } } let a = A(123) ; a->set(456) . print ; a.print", "456\nA(x=456)\n") }
#[test] fn struct_self_method_box_set_call_merge() { run("struct A(x) { fn get(self) { self->x } fn set(self, y) { self->x = y } } let a = A(123) ; a->set()(456) . print ; a.print", "456\nA(x=456)\n") }
#[test] fn struct_self_method_in_function() { run("struct A(x) { fn get(self) { (fn() -> self)() } } A(123)->get() . print", "A(x=123)\n") }
#[test] fn struct_self_method_in_closure() { run("struct A(x) { fn get(self) { (fn() -> self) } } A(123)->get()() . print", "A(x=123)\n") }
#[test] fn struct_self_method_bind_to_self_method_1() { run("struct A() { fn a(self) { 123 } fn b(self) { a() } } A()->b() . print", "123\n") }
#[test] fn struct_self_method_bind_to_self_method_2() { run("fn x() {} struct A() { fn a(self) { 123 } fn b(self) { a() } } A()->b() . print", "123\n") }
#[test] fn struct_self_method_bind_to_self_method_in_upvalue() { run("struct A() { fn a(self) { 123 } fn b(self) { fn c() { a() } c } } A()->b()() . print", "123\n") }
#[test] fn struct_self_method_self_type_call_type_late() { run("struct A() { fn a(self) { self->b() } fn b() { 9 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_self_method_self_type_call_type() { run("struct A() { fn b() { 9 } fn a(self) { self->b() } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_self_method_self_type_call_instance_late() { run("struct A() { fn a(self) { self->b() } fn b() { 9 } } A()->a() . repr . print", "TypeError: Cannot get field 'b' on struct A()\n  at: line 1 (<test>)\n  at: `fn a(self)` (line 1)\n\n1 | struct A() { fn a(self) { self->b() } fn b() { 9 } } A()->a() . repr . print\n2 |                               ^^^\n") }
#[test] fn struct_self_method_self_type_call_instance() { run("struct A() { fn b() { 9 } fn a(self) { self->b() } } A()->a() . repr . print", "TypeError: Cannot get field 'b' on struct A()\n  at: line 1 (<test>)\n  at: `fn a(self)` (line 1)\n\n1 | struct A() { fn b() { 9 } fn a(self) { self->b() } } A()->a() . repr . print\n2 |                                            ^^^\n") }
#[test] fn struct_self_method_self_self_call_type_late() { run("struct A() { fn a(self) { self->b() } fn b(self) { 9 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_self_method_self_self_call_type() { run("struct A() { fn b(self) { 9 } fn a(self) { self->b() } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_self_method_self_self_call_instance_late() { run("struct A() { fn a(self) { self->b() } fn b(self) { 9 } } A()->a() . repr . print", "9\n") }
#[test] fn struct_self_method_self_self_call_instance() { run("struct A() { fn b(self) { 9 } fn a(self) { self->b() } } A()->a() . repr . print", "9\n") }
#[test] fn struct_type_method_type_type_call_type_late() { run("struct A() { fn a() { A->b() } fn b() { 9 } } A->a() . repr . print", "9\n") }
#[test] fn struct_type_method_type_type_call_type() { run("struct A() { fn b() { 9 } fn a() { A->b() } } A->a() . repr . print", "9\n") }
#[test] fn struct_type_method_type_type_call_instance_late() { run("struct A() { fn a() { A->b() } fn b() { 9 } } A()->a() . repr . print", "TypeError: Cannot get field 'a' on struct A()\n  at: line 1 (<test>)\n\n1 | struct A() { fn a() { A->b() } fn b() { 9 } } A()->a() . repr . print\n2 |                                                  ^^^\n") }
#[test] fn struct_type_method_type_type_call_instance() { run("struct A() { fn b() { 9 } fn a() { A->b() } } A()->a() . repr . print", "TypeError: Cannot get field 'a' on struct A()\n  at: line 1 (<test>)\n\n1 | struct A() { fn b() { 9 } fn a() { A->b() } } A()->a() . repr . print\n2 |                                                  ^^^\n") }
#[test] fn struct_type_method_self_type_call_type_late() { run("struct A() { fn a(self) { A->b() } fn b() { 9 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_type_method_self_type_call_type() { run("struct A() { fn b() { 9 } fn a(self) { A->b() } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_type_method_self_type_call_instance_late() { run("struct A() { fn a(self) { A->b() } fn b() { 9 } } A()->a() . repr . print", "9\n") }
#[test] fn struct_type_method_self_type_call_instance() { run("struct A() { fn b() { 9 } fn a(self) { A->b() } } A()->a() . repr . print", "9\n") }
#[test] fn struct_type_method_type_self_call_type_late() { run("struct A() { fn a() { A->b() } fn b(self) { 9 } } A->a() . repr . print", "fn b(self)\n") }
#[test] fn struct_type_method_type_self_call_type() { run("struct A() { fn b(self) { 9 } fn a() { A->b() } } A->a() . repr . print", "fn b(self)\n") }
#[test] fn struct_type_method_type_self_call_instance_late() { run("struct A() { fn a() { A->b() } fn b(self) { 9 } } A()->a() . repr . print", "TypeError: Cannot get field 'a' on struct A()\n  at: line 1 (<test>)\n\n1 | struct A() { fn a() { A->b() } fn b(self) { 9 } } A()->a() . repr . print\n2 |                                                      ^^^\n") }
#[test] fn struct_type_method_type_self_call_instance() { run("struct A() { fn b(self) { 9 } fn a() { A->b() } } A()->a() . repr . print", "TypeError: Cannot get field 'a' on struct A()\n  at: line 1 (<test>)\n\n1 | struct A() { fn b(self) { 9 } fn a() { A->b() } } A()->a() . repr . print\n2 |                                                      ^^^\n") }
#[test] fn struct_type_method_self_self_call_type_late() { run("struct A() { fn a(self) { A->b() } fn b(self) { 9 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_type_method_self_self_call_type() { run("struct A() { fn b(self) { 9 } fn a(self) { A->b() } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_type_method_self_self_call_instance_late() { run("struct A() { fn a(self) { A->b() } fn b(self) { 9 } } A()->a() . repr . print", "fn b(self)\n") }
#[test] fn struct_type_method_self_self_call_instance() { run("struct A() { fn b(self) { 9 } fn a(self) { A->b() } } A()->a() . repr . print", "fn b(self)\n") }
#[test] fn struct_raw_method_type_type_call_type_late() { run("struct A() { fn a() { b() } fn b() { 9 } } A->a() . repr . print", "9\n") }
#[test] fn struct_raw_method_type_type_call_type() { run("struct A() { fn b() { 9 } fn a() { b() } } A->a() . repr . print", "9\n") }
#[test] fn struct_raw_method_type_type_call_instance_late() { run("struct A() { fn a() { b() } fn b() { 9 } } A()->a() . repr . print", "TypeError: Cannot get field 'a' on struct A()\n  at: line 1 (<test>)\n\n1 | struct A() { fn a() { b() } fn b() { 9 } } A()->a() . repr . print\n2 |                                               ^^^\n") }
#[test] fn struct_raw_method_type_type_call_instance() { run("struct A() { fn b() { 9 } fn a() { b() } } A()->a() . repr . print", "TypeError: Cannot get field 'a' on struct A()\n  at: line 1 (<test>)\n\n1 | struct A() { fn b() { 9 } fn a() { b() } } A()->a() . repr . print\n2 |                                               ^^^\n") }
#[test] fn struct_raw_method_self_type_call_type_late() { run("struct A() { fn a(self) { b() } fn b() { 9 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_raw_method_self_type_call_type() { run("struct A() { fn b() { 9 } fn a(self) { b() } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_raw_method_self_type_call_instance_late() { run("struct A() { fn a(self) { b() } fn b() { 9 } } A()->a() . repr . print", "9\n") }
#[test] fn struct_raw_method_self_type_call_instance() { run("struct A() { fn b() { 9 } fn a(self) { b() } } A()->a() . repr . print", "9\n") }
#[test] fn struct_raw_method_self_self_call_type_late() { run("struct A() { fn a(self) { b() } fn b(self) { 9 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_raw_method_self_self_call_type() { run("struct A() { fn b(self) { 9 } fn a(self) { b() } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_raw_method_self_self_call_instance_late() { run("struct A() { fn a(self) { b() } fn b(self) { 9 } } A()->a() . repr . print", "9\n") }
#[test] fn struct_raw_method_self_self_call_instance() { run("struct A() { fn b(self) { 9 } fn a(self) { b() } } A()->a() . repr . print", "9\n") }
#[test] fn struct_self_field_self_call_type() { run("struct A(c) { fn a(self) { self->c } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_self_field_self_call_instance() { run("struct A(c) { fn a(self) { self->c } } A(9)->a() . repr . print", "9\n") }
#[test] fn struct_type_field_type_call_type() { run("struct A(c) { fn a() { A->c } } A->a() . repr . print", "(->)\n") }
#[test] fn struct_type_field_type_call_type_then_eval() { run("struct A(c) { fn a() { A->c } } A->a()(A(5)) . repr . print", "5\n") }
#[test] fn struct_type_field_type_call_instance() { run("struct A(c) { fn a() { A->c } }  A(9)->a() . repr . print", "TypeError: Cannot get field 'a' on struct A(c)\n  at: line 1 (<test>)\n\n1 | struct A(c) { fn a() { A->c } }  A(9)->a() . repr . print\n2 |                                      ^^^\n") }
#[test] fn struct_type_field_self_call_type() { run("struct A(c) { fn a(self) { A->c } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_type_field_self_call_instance() { run("struct A(c) { fn a(self) { A->c } }  A(9)->a() . repr . print", "(->)\n") }
#[test] fn struct_type_field_self_call_instance_then_eval() { run("struct A(c) { fn a(self) { A->c } }  A(9)->a()(A(5)) . repr . print", "5\n") }
#[test] fn struct_func_field_type_call_type() { run("struct A(c) { fn a() { (->c) } } A->a() . repr . print", "(->)\n") }
#[test] fn struct_func_field_type_call_instance() { run("struct A(c) { fn a() { (->c) } }  A(9)->a() . repr . print", "TypeError: Cannot get field 'a' on struct A(c)\n  at: line 1 (<test>)\n\n1 | struct A(c) { fn a() { (->c) } }  A(9)->a() . repr . print\n2 |                                       ^^^\n") }
#[test] fn struct_func_field_self_call_type() { run("struct A(c) { fn a(self) { (->c) } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_func_field_self_call_instance() { run("struct A(c) { fn a(self) { (->c) } }  A(9)->a() . repr . print", "(->)\n") }
#[test] fn struct_raw_field_self_call_type() { run("struct A(c) { fn a(self) { c } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_raw_field_self_call_instance() { run("struct A(c) { fn a(self) { c } }  A(9)->a() . repr . print", "9\n") }
#[test] fn struct_set_field_self_call_type() { run("struct A(c) { fn a(self) { c = 1 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_set_field_self_call_instance() { run("struct A(c) { fn a(self) { c = 1 } } A(9)->a() . repr . print", "1\n") }
#[test] fn struct_operator_set_field_self_call_type() { run("struct A(c) { fn a(self) { c += 1 } } A->a() . repr . print", "fn a(self)\n") }
#[test] fn struct_operator_set_field_self_call_instance() { run("struct A(c) { fn a(self) { c += 1 } } A(9)->a() . repr . print", "10\n") }
#[test] fn module_empty() { run("module Foo module Bar ; (Foo, Bar) . print", "(module Foo, module Bar)\n") }
#[test] fn module_with_method() { run("module Foo { fn bar() -> 123 } ; Foo->bar() . print", "123\n") }
#[test] fn modules_with_same_method() { run("module A { fn a() -> 1 } module B { fn a() -> 2 } ; (A->a(), B->a()) . print", "(1, 2)\n") }
#[test] fn module_with_many_methods() { run("module A { fn a() -> 1 fn b() { 2 } } ; (A->a(), A->b()) . print", "(1, 2)\n") }
#[test] fn module_indirect_method_access() { run("module A { fn a() { print 'hi' } } ; let f = A->a ; f()", "hi\n") }
#[test] fn module_get_field_access() { run("module A { fn a() { print 'hi' } } ; let f = (->a) ; f(A)()", "hi\n") }
#[test] fn module_and_struct() { run("struct A(a) ; module B { fn a() { print 'hi' } } ; let x = A('yes'), f = (->a) ; (f(x), f(B)) . print", "('yes', fn a())\n") }
#[test] fn module_str_and_repr() { run("module A { fn a() {} } A . print ; A . repr . print", "module A\nmodule A\n") }
#[test] fn module_typeof() { run("module A ; typeof A . print", "function\n") }
#[test] fn module_is() { run("module A {} ; A is A . print ; A is function . print", "false\ntrue\n") }
#[test] fn module_field_cannot_be_set() { run("module A { fn a() {} } ; A->a = 123", "TypeError: Cannot set field 'a' on module A\n  at: line 1 (<test>)\n\n1 | module A { fn a() {} } ; A->a = 123\n2 |                               ^\n") }
#[test] fn module_field_not_found() { run("module A { fn a() {} } module B { fn b() -> 1 } A->b() . print", "TypeError: Cannot get field 'b' on module A\n  at: line 1 (<test>)\n\n1 | module A { fn a() {} } module B { fn b() -> 1 } A->b() . print\n2 |                                                  ^^^\n") }
#[test] fn module_cannot_construct() { run("module Foo {} ; Foo()", "Tried to evaluate module Foo but it is not a function.\n  at: line 1 (<test>)\n\n1 | module Foo {} ; Foo()\n2 |                    ^^\n") }
#[test] fn module_cannot_construct_with_args() { run("module Foo {} ; Foo(1, 2, 3)", "Tried to evaluate module Foo but it is not a function.\n  at: line 1 (<test>)\n\n1 | module Foo {} ; Foo(1, 2, 3)\n2 |                    ^^^^^^^^^\n") }
#[test] fn module_method_bound() { run("module A { fn a() { 123 } fn b() { a() } } ; A->b().print", "123\n") }
#[test] fn module_method_bound_late() { run("module A { fn a() { b() } fn b() { 123 } } ; A->a().print", "123\n") }
#[test] fn module_can_be_replaced() { run("module A { fn a() { 'yes' } } ; struct X(a) ; A = X(fn() -> 'no') ; A->a().print", "no\n") }
#[test] fn module_method_bound_cannot_be_replaced() { run("module A { fn a() { 'yes' } fn b() { a() } } ; struct X(a) ; let A1 = A ; A = X(fn() -> 'no') ; A1->b().print", "yes\n") }
#[test] fn module_method_lazy_bound_can_be_replaced() { run("module A { fn a() { 'yes' } fn b() { A->a() } } ; struct X(a) ; let A1 = A ; A = X(fn() -> 'no') ; A1->b().print", "no\n") }
#[test] fn module_native_fn_repr() { run("native module Foo { fn a() } ; Foo->a . repr . print", "native fn a()\n") }
#[test] fn module_method_over_global() { run("fn x() { 1 } module A { fn x() { 2 } fn y() { x() } } ; A->y() . print", "2\n") }
#[test] fn module_late_method_over_global() { run("fn x() { 1 } module A { fn y() { x() } fn x() { 2 } } ; A->y() . print", "2\n") }
#[test] fn module_method_over_late_global() { run("module A { fn x() { 2 } fn y() { x() } } fn x() { 1 } ; A->y() . print", "2\n") }
#[test] fn module_late_method_over_late_global() { run("module A { fn y() { x() } fn x() { 2 } } fn x() { 1 } ; A->y() . print", "2\n") }
#[test] fn module_method_only() { run("module A { fn x() { 1 } fn y() { x() } } ; A->y() . print", "1\n") }
#[test] fn module_late_method_only() { run("module A { fn y() { x() } fn x() { 1 } } ; A->y() . print", "1\n") }
#[test] fn module_global_only() { run("fn x() { 1 } module A { fn y() { x() } } ; A->y() . print", "1\n") }
#[test] fn module_late_global_only() { run("module A { fn y() { x() } } fn x() { 1 } ; A->y() . print", "1\n") }
#[test] fn local_vars_01() { run("let x=0 do { x.print }", "0\n") }
#[test] fn local_vars_02() { run("let x=0 do { let x=1; x.print }", "1\n") }
#[test] fn local_vars_03() { run("let x=0 do { x.print let x=1 }", "0\n") }
#[test] fn local_vars_04() { run("let x=0 do { let x=1 } x.print", "0\n") }
#[test] fn local_vars_05() { run("let x=0 do { x=1 } x.print", "1\n") }
#[test] fn local_vars_06() { run("let x=0 do { x=1 do { x=2; x.print } }", "2\n") }
#[test] fn local_vars_07() { run("let x=0 do { x=1 do { x=2 } x.print }", "2\n") }
#[test] fn local_vars_08() { run("let x=0 do { let x=1 do { x=2 } x.print }", "2\n") }
#[test] fn local_vars_09() { run("let x=0 do { let x=1 do { let x=2 } x.print }", "1\n") }
#[test] fn local_vars_10() { run("let x=0 do { x=1 do { let x=2 } x.print }", "1\n") }
#[test] fn local_vars_11() { run("let x=0 do { x=1 do { let x=2 } } x.print", "1\n") }
#[test] fn local_vars_12() { run("let x=0 do { let x=1 do { let x=2 } } x.print", "0\n") }
#[test] fn local_vars_14() { run("let x=3 do { let x=x; x.print }", "3\n") }
#[test] fn chained_assignments() { run("let a, b, c; a = b = c = 3; [a, b, c] . print", "[3, 3, 3]\n") }
#[test] fn array_assignment_1() { run("let a = [1, 2, 3]; a[0] = 3; a . print", "[3, 2, 3]\n") }
#[test] fn array_assignment_2() { run("let a = [1, 2, 3]; a[2] = 1; a . print", "[1, 2, 1]\n") }
#[test] fn array_assignment_negative_index_1() { run("let a = [1, 2, 3]; a[-1] = 6; a . print", "[1, 2, 6]\n") }
#[test] fn array_assignment_negative_index_2() { run("let a = [1, 2, 3]; a[-3] = 6; a . print", "[6, 2, 3]\n") }
#[test] fn nested_array_assignment_1() { run("let a = [[1, 2], [3, 4]]; a[0][1] = 6; a . print", "[[1, 6], [3, 4]]\n") }
#[test] fn nested_array_assignment_2() { run("let a = [[1, 2], [3, 4]]; a[1][0] = 6; a . print", "[[1, 2], [6, 4]]\n") }
#[test] fn nested_array_assignment_negative_index_1() { run("let a = [[1, 2], [3, 4]]; a[0][-1] = 6; a . print", "[[1, 6], [3, 4]]\n") }
#[test] fn nested_array_assignment_negative_index_2() { run("let a = [[1, 2], [3, 4]]; a[-1][-2] = 6; a . print", "[[1, 2], [6, 4]]\n") }
#[test] fn chained_operator_assignment() { run("let a = 1, b; a += b = 4; [a, b] . print", "[5, 4]\n") }
#[test] fn operator_array_assignment() { run("let a = [12]; a[0] += 4; a[0] . print", "16\n") }
#[test] fn nested_operator_array_assignment() { run("let a = [[12]]; a[0][-1] += 4; a . print", "[[16]]\n") }
#[test] fn weird_assignment() { run("let a = [[12]], b = 3; fn f() -> a; f()[0][-1] += b = 5; [f(), b] . print", "[[[17]], 5]\n") }
#[test] fn mutable_array_in_array_1() { run("let a = [0], b = [a]; b[0] = 'hi'; b. print", "['hi']\n") }
#[test] fn mutable_array_in_array_2() { run("let a = [0], b = [a]; b[0][0] = 'hi'; b. print", "[['hi']]\n") }
#[test] fn mutable_arrays_in_assignments() { run("let a = [0], b = [a, a, a]; b[0][0] = 5; b . print", "[[5], [5], [5]]\n") }
#[test] fn pattern_in_let_works() { run("let x, y = [1, 2] ; [x, y] . print", "[1, 2]\n") }
#[test] fn pattern_in_let_too_long() { run("let x, y, z = [1, 2] ; [x, y] . print", "ValueError: Cannot unpack '[1, 2]' of type 'list' with length 2, expected exactly 3 elements\n  at: line 1 (<test>)\n\n1 | let x, y, z = [1, 2] ; [x, y] . print\n2 |                    ^\n") }
#[test] fn pattern_in_let_too_short() { run("let x, y = [1, 2, 3] ; [x, y] . print", "ValueError: Cannot unpack '[1, 2, 3]' of type 'list' with length 3, expected exactly 2 elements\n  at: line 1 (<test>)\n\n1 | let x, y = [1, 2, 3] ; [x, y] . print\n2 |                    ^\n") }
#[test] fn pattern_in_let_with_var_at_end() { run("let x, *y = [1, 2, 3, 4] ; [x, y] . print", "[1, [2, 3, 4]]\n") }
#[test] fn pattern_in_let_with_var_at_start() { run("let *x, y = [1, 2, 3, 4] ; [x, y] . print", "[[1, 2, 3], 4]\n") }
#[test] fn pattern_in_let_with_var_at_middle() { run("let x, *y, z = [1, 2, 3, 4] ; [x, y, z] . print", "[1, [2, 3], 4]\n") }
#[test] fn pattern_in_let_with_var_zero_len_at_end() { run("let x, *y = [1] ; [x, y] . print", "[1, []]\n") }
#[test] fn pattern_in_let_with_var_zero_len_at_start() { run("let *x, y = [1] ; [x, y] . print", "[[], 1]\n") }
#[test] fn pattern_in_let_with_var_zero_len_at_middle() { run("let x, *y, z = [1, 2] ; [x, y, z] . print", "[1, [], 2]\n") }
#[test] fn pattern_in_let_with_var_one_len_at_end() { run("let x, *y = [1, 2] ; [x, y] . print", "[1, [2]]\n") }
#[test] fn pattern_in_let_with_var_one_len_at_start() { run("let *x, y = [1, 2] ; [x, y] . print", "[[1], 2]\n") }
#[test] fn pattern_in_let_with_var_one_len_at_middle() { run("let x, *y, z = [1, 2, 3] ; [x, y, z] . print", "[1, [2], 3]\n") }
#[test] fn pattern_in_let_with_empty() { run("let a, _, _ = [1, 2, 3] ; a . print", "1\n") }
#[test] fn pattern_in_let_with_empty_too_long() { run("let _, b, _ = [1, 2, 3, 4]", "ValueError: Cannot unpack '[1, 2, 3, 4]' of type 'list' with length 4, expected exactly 3 elements\n  at: line 1 (<test>)\n\n1 | let _, b, _ = [1, 2, 3, 4]\n2 |                          ^\n") }
#[test] fn pattern_in_let_with_empty_too_short() { run("let _, _, c = [1, 2]", "ValueError: Cannot unpack '[1, 2]' of type 'list' with length 2, expected exactly 3 elements\n  at: line 1 (<test>)\n\n1 | let _, _, c = [1, 2]\n2 |                    ^\n") }
#[test] fn pattern_in_let_with_empty_at_end() { run("let _, _, x = [1, 2, 3] ; x . print", "3\n") }
#[test] fn pattern_in_let_with_empty_at_both() { run("let _, x, _ = [1, 2, 3] ; x . print", "2\n") }
#[test] fn pattern_in_let_with_empty_at_start() { run("let x, _, _ = [1, 2, 3] ; x . print", "1\n") }
#[test] fn pattern_in_let_with_empty_at_middle() { run("let x, _, y = [1, 2, 3] ; [x, y] . print", "[1, 3]\n") }
#[test] fn pattern_in_let_with_varargs_empty_at_end() { run("let x, *_ = [1, 2, 3, 4] ; x . print", "1\n") }
#[test] fn pattern_in_let_with_varargs_empty_at_middle() { run("let x, *_, y = [1, 2, 3, 4] ; [x, y] . print", "[1, 4]\n") }
#[test] fn pattern_in_let_with_varargs_empty_at_start() { run("let *_, x = [1, 2, 3, 4] ; x . print", "4\n") }
#[test] fn pattern_in_let_with_varargs_var_at_end() { run("let _, *x = [1, 2, 3, 4] ; x . print", "[2, 3, 4]\n") }
#[test] fn pattern_in_let_with_varargs_var_at_middle() { run("let _, *x, _ = [1, 2, 3, 4] ; x . print", "[2, 3]\n") }
#[test] fn pattern_in_let_with_varargs_var_at_start() { run("let *x, _ = [1, 2, 3, 4] ; x . print", "[1, 2, 3]\n") }
#[test] fn pattern_in_let_with_varargs_empty_to_var_at_end_too_short() { run("let *_, x = []", "ValueError: Cannot unpack '[]' of type 'list' with length 0, expected at least 1 elements\n  at: line 1 (<test>)\n\n1 | let *_, x = []\n2 |              ^\n") }
#[test] fn pattern_in_let_with_varargs_empty_to_var_at_start_too_short() { run("let x, *_ = []", "ValueError: Cannot unpack '[]' of type 'list' with length 0, expected at least 1 elements\n  at: line 1 (<test>)\n\n1 | let x, *_ = []\n2 |              ^\n") }
#[test] fn pattern_in_let_with_varargs_empty_to_var_at_end() { run("let *_, x = [1] ; x . print", "1\n") }
#[test] fn pattern_in_let_with_varargs_empty_to_var_at_start() { run("let x, *_ = [1] ; x . print", "1\n") }
#[test] fn pattern_in_let_with_varargs_empty_to_var_at_middle() { run("let x, *_, y = [1, 2] ; [x, y] . print", "[1, 2]\n") }
#[test] fn pattern_in_let_with_nested_pattern() { run("let x, (y, _) = [[1, 2], [3, 4]] ; [x, y] . print", "[[1, 2], 3]\n") }
#[test] fn pattern_in_let_with_parens_on_one_empty_one_var() { run("let (_, x) = [[1, 2]] ; x . print", "2\n") }
#[test] fn pattern_in_let_with_complex_patterns_1() { run("let *_, (_, x, _), _ = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ; x . print", "5\n") }
#[test] fn pattern_in_let_with_complex_patterns_2() { run("let _, (_, (_, (_, (x, *_)))) = [1, [2, [3, [4, [5, [6, [7, [8, [9, nil]]]]]]]]] ; x . print", "5\n") }
#[test] fn pattern_in_let_with_complex_patterns_3() { run("let ((*x, _), (_, (*y, _), _), *_) = [[[1, 2, 3], [[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1], [2], [3]]]] ; [x, y] . print", "[[1, 2], [2, 3]]\n") }
#[test] fn pattern_in_let_with_1x_nested_lvalue() { run("let (a) = [[1]] ; a . print", "1\n") }
#[test] fn pattern_in_let_with_2x_nested_lvalue() { run("let ((a)) = [[[1]]] ; a . print", "1\n") }
#[test] fn pattern_in_function_1() { run("fn f((a)) -> a ; f([1]) . print", "1\n") }
#[test] fn pattern_in_function_2() { run("fn f((*a)) -> a ; f('hello') . print", "hello\n") }
#[test] fn pattern_in_function_3() { run("fn f((*a, _, b)) -> [a, b] ; f('hello !') . print", "['hello', '!']\n") }
#[test] fn pattern_in_function_4() { run("fn f((a, b)) -> [b, a] . print ; f([1, 2])", "[2, 1]\n") }
#[test] fn pattern_in_function_5() { run("fn f((a, b), (c, d)) -> [a, b, c, d] . print ; f([1, 2], [3, 4])", "[1, 2, 3, 4]\n") }
#[test] fn pattern_in_function_6() { run("fn f(((a))) -> a ; f([[1]]) . print", "1\n") }
#[test] fn pattern_in_function_with_other_args_1() { run("fn f((a, b, c), d, e) -> [a, b, c, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 2, 3, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_2() { run("fn f(a, (b, c, d), e) -> [a, b, c, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 2, 3, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_3() { run("fn f(a, b, (c, d, e)) -> [a, b, c, d, e] . print ; f(1, 2, [3, 4, 5])", "[1, 2, 3, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_and_empty_1() { run("fn f((_, b, _), d, e) -> [1, b, 3, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 2, 3, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_and_empty_2() { run("fn f(a, (_, _, d), e) -> [a, 2, 3, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 2, 3, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_and_empty_3() { run("fn f(a, b, (c, _, _)) -> [a, b, c, 4, 5] . print ; f(1, 2, [3, 4, 5])", "[1, 2, 3, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_and_var_1() { run("fn f((a, *_), d, e) -> [a, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_and_var_2() { run("fn f(a, (*_, d), e) -> [a, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 4, 5]\n") }
#[test] fn pattern_in_function_with_other_args_and_var_3() { run("fn f(a, b, (*c, _, _)) -> [a, b, c] . print ; f(1, 2, [3, 4, 5])", "[1, 2, [3]]\n") }
#[test] fn pattern_in_for_1() { run("for i, x in 'hello' . enumerate { [i, x] . print }", "[0, 'h']\n[1, 'e']\n[2, 'l']\n[3, 'l']\n[4, 'o']\n")}
#[test] fn pattern_in_for_2() { run("for _ in range(5) { 'hello' . print }", "hello\nhello\nhello\nhello\nhello\n") }
#[test] fn pattern_in_for_3() { run("for a, *_, b in ['hello', 'world'] { print(a + b) }", "ho\nwd\n") }
#[test] fn pattern_in_expression_1() { run("let x, y, z ; x, y, z = 'abc' ; print(x, y, z)", "a b c\n") }
#[test] fn pattern_in_expression_2() { run("let x, y, z ; z = x, y = (1, 2) ; print(x, y, z)", "1 2 (1, 2)\n") }
#[test] fn pattern_in_expression_3() { run("do { let x, y, z ; z = x, y = (1, 2) ; print(x, y, z) }", "1 2 (1, 2)\n") }
#[test] fn pattern_in_expression_4() { run("let x, y ; *x, y = 'hello' ; print(x, y)", "hell o\n") }
#[test] fn pattern_in_expression_5() { run("let x, y ; (x, *_), (*_, y) = ('hello', 'world') ; print(x, y)", "h d\n") }
#[test] fn pattern_in_expr_with_array_1() { run("let a = [1, 2] ; a[0], a[1] = 'ab' ; a . print", "['a', 'b']\n") }
#[test] fn pattern_in_expr_with_array_2() { run("let a = [1, 2, 3] ; a[0], a[-1] = 'ab' ; a . print", "['a', 2, 'b']\n") }
#[test] fn pattern_in_expr_with_array_3() { run("let a = [1, 2] ; a[0], _, a[-1] = 'abc' ; a . print", "['a', 'c']\n") }
#[test] fn pattern_in_expr_with_array_4() { run("let a = [1, 2, 3] ; a[0], _, a[1], *_, a[2], _ = 'abc123456' ; a . print", "['a', 'c', '5']\n") }
#[test] fn pattern_in_expr_with_array_and_computed_index() { run("let a = [1, 2, 3, 4] ; _, *_, a[1 + a[-1] * 2 - 7], _ = 'abc123' ; a . print", "[1, 2, '2', 4]\n") }
#[test] fn pattern_in_expr_with_array_and_dependence_1() { run("let a = [1, 2] ; a[1], a[0] = a ; a . print", "[2, 2]\n") }
#[test] fn pattern_in_expr_with_array_and_dependence_2() { run("let a = [1, 2] ; a[0], a[1] = a ; a . print", "[1, 2]\n") }
#[test] fn pattern_in_expr_with_array_and_dependence_3() { run("let a = [1, 2] ; a[2 - a[0]], a[0] = a ; a . print", "[2, 2]\n") }
#[test] fn pattern_in_expr_with_array_and_dependence_4() { run("let a = [1, 2] ; a[2 - a[1]], a[1] = a ; a . print", "[1, 2]\n") }
#[test] fn pattern_in_expr_with_array_and_dependence_5() { run("let a = [2, 1] ; a[2 - a[0]], a[0] = a ; a . print", "[1, 1]\n") }
#[test] fn pattern_in_expr_with_array_and_dependence_6() { run("let a = [2, 1] ; a[2 - a[1]], a[1] = a ; a . print", "[2, 2]\n") }
#[test] fn pattern_in_expr_with_array_and_raw_vector_1() { run("let a = [1, 2] ; a[0], a[1] = a[0], a[1] ; a . print", "[1, 2]\n") }
#[test] fn pattern_in_expr_with_array_and_raw_vector_2() { run("let a = [1, 2] ; a[0], a[1] = a[1], a[0] ; a . print", "[2, 1]\n") }
#[test] fn pattern_in_expr_with_array_and_raw_vector_3() { run("let a = [1, 2] ; a[1], _, a[0] = a[1], a[1], a[0] ; a . print", "[1, 2]\n") }
#[test] fn pattern_in_expr_with_array_and_raw_vector_4() { run("let a = [1, 2] ; a[1], a[0], _ = a[0], 3, a[0] ; a . print", "[3, 1]\n") }
#[test] fn pattern_in_expr_with_bare_vector_1() { run("let a, b = 'xy' ; a, b = a, b ; print(a, b)", "x y\n") }
#[test] fn pattern_in_expr_with_bare_vector_2() { run("let a, b = 'xy' ; a, b = b, a ; print(a, b)", "y x\n") }
#[test] fn pattern_in_expr_with_bare_vector_3() { run("let a, b = 'xy' ; b, a = a, b ; print(a, b)", "y x\n") }
#[test] fn pattern_in_expr_with_bare_vector_4() { run("let a, b = 'xy' ; _, a, b = a, 'z', b ; print(a, b)", "z y\n") }
#[test] fn pattern_in_expr_with_bare_vector_5() { run("let a, b = 'xy' ; _, b, *_, a = a, 'z', b, 'w', 'g' ; print(a, b)", "g z\n") }
#[test] fn pattern_in_expr_with_field_1() { run("struct A(a, b, c) ; let x = A(1, 2, 3) ; x->a, x->b = [5, 6] ; print x", "A(a=5, b=6, c=3)\n") }
#[test] fn pattern_in_expr_with_field_2() { run("struct A(a, b, c) ; let x = A(1, 2, 3) ; x->a, _, x->c = [5, 6, 7] ; print x", "A(a=5, b=2, c=7)\n") }
#[test] fn pattern_in_expr_with_field_3() { run("struct A(a, b, c) ; let x = A(1, 2, 3) ; _, x->a, *_, x->c = [5, 6, 7, 8, 9, 10] ; print x", "A(a=6, b=2, c=10)\n") }
#[test] fn pattern_in_expr_with_field_and_raw_vector_1() { run("struct A(a, b, c) ; let x = A(1, 2, 3) ; x->a, x->b = x->b, x->a ; print x", "A(a=2, b=1, c=3)\n") }
#[test] fn pattern_in_expr_with_field_and_raw_vector_2() { run("struct A(a, b, c) ; let x = A(1, 2, 3) ; x->a, x->c = x->b, x->a ; print x", "A(a=2, b=2, c=1)\n") }
#[test] fn pattern_in_expr_with_field_and_array() { run("struct A(a, b, c) ; let x = A(1, 2, [3, 4]), y = 5 ; _, x->b, y, x->a, _, x->c[0] = [5, 6, 7, 8, 9, 10] ; print(x, y)", "A(a=8, b=6, c=[10, 4]) 7\n") }
#[test] fn function_repr() { run("(fn((_, *_), x) -> nil) . repr . print", "fn _((_, *_), x)\n") }
#[test] fn function_repr_partial() { run("(fn((_, *_), x) -> nil)(1) . repr . print", "fn _((_, *_), x)\n") }
#[test] fn function_closure_repr() { run("fn box(x) -> fn((_, *_), y) -> x ; box(nil) . repr . print", "fn _((_, *_), y)\n") }
#[test] fn functions_01() { run("fn foo() { 'hello' . print } ; foo();", "hello\n") }
#[test] fn functions_02() { run("fn foo() { 'hello' . print } ; foo() ; foo()", "hello\nhello\n") }
#[test] fn functions_03() { run("fn foo(a) { 'hello' . print } ; foo(1)", "hello\n") }
#[test] fn functions_04() { run("fn foo(a) { 'hello ' + a . print } ; foo(1)", "hello 1\n") }
#[test] fn functions_05() { run("fn foo(a, b, c) { a + b + c . print } ; foo(1, 2, 3)", "6\n") }
#[test] fn functions_06() { run("fn foo() { 'hello' . print } ; fn bar() { foo() } bar()", "hello\n") }
#[test] fn functions_07() { run("fn foo() { 'hello' . print } ; fn bar(a) { foo() } bar(1)", "hello\n") }
#[test] fn functions_08() { run("fn foo(a) { a . print } ; fn bar(a, b, c) { foo(a + b + c) } bar(1, 2, 3)", "6\n") }
#[test] fn functions_09() { run("fn foo(h, w) { h + ' ' + w . print } ; fn bar(w) { foo('hello', w) } bar('world')", "hello world\n") }
#[test] fn functions_10() { run("let x = 'hello' ; fn foo(x) { x . print } foo(x)", "hello\n") }
#[test] fn functions_11() { run("do { let x = 'hello' ; fn foo(x) { x . print } foo(x) }", "hello\n") }
#[test] fn functions_12() { run("do { let x = 'hello' ; do { fn foo(x) { x . print } foo(x) } }", "hello\n") }
#[test] fn functions_13() { run("let x = 'hello' ; do { fn foo() { x . print } foo() }", "hello\n") }
#[test] fn functions_14() { run("fn foo(x) { 'hello ' + x . print } 'world' . foo", "hello world\n") }
#[test] fn function_implicit_return_01() { run("fn foo() { } foo() . print", "nil\n") }
#[test] fn function_implicit_return_02() { run("fn foo() { 'hello' } foo() . print", "hello\n") }
#[test] fn function_implicit_return_03() { run("fn foo(x) { if x > 1 then nil else 'nope' } foo(2) . print", "nil\n") }
#[test] fn function_implicit_return_04() { run("fn foo(x) { if x > 1 then true else false } foo(2) . print", "true\n") }
#[test] fn function_implicit_return_05() { run("fn foo(x) { if x > 1 then true else false } foo(0) . print", "false\n") }
#[test] fn function_implicit_return_06() { run("fn foo(x) { if x > 1 then nil else false } foo(2) . print", "nil\n") }
#[test] fn function_implicit_return_07() { run("fn foo(x) { if x > 1 then nil else false } foo(0) . print", "false\n") }
#[test] fn function_implicit_return_08() { run("fn foo(x) { if x > 1 then true else nil } foo(2) . print", "true\n") }
#[test] fn function_implicit_return_09() { run("fn foo(x) { if x > 1 then true else nil } foo(0) . print", "nil\n") }
#[test] fn function_implicit_return_10() { run("fn foo(x) { if x > 1 then 'hello' else nil } foo(2) . print", "hello\n") }
#[test] fn function_implicit_return_11() { run("fn foo(x) { if x > 1 then 'hello' else nil } foo(0) . print", "nil\n") }
#[test] fn function_implicit_return_12() { run("fn foo(x) { if x > 1 then if true then 'hello' else nil else nil } foo(2) . print", "hello\n") }
#[test] fn function_implicit_return_13() { run("fn foo(x) { if x > 1 then if true then 'hello' else nil else nil } foo(0) . print", "nil\n") }
#[test] fn function_implicit_return_14() { run("fn foo(x) { loop { if x > 1 { break } } } foo(2) . print", "nil\n") }
#[test] fn function_implicit_return_15() { run("fn foo(x) { loop { if x > 1 { continue } else { break } } } foo(0) . print", "nil\n") }
#[test] fn closures_01() { run("fn foo() { let x = 'hello' ; fn bar() { x . print } bar() } foo()", "hello\n") }
#[test] fn closures_02() { run("do { fn foo() { 'hello' . print } ; fn bar() { foo() } bar() }", "hello\n") }
#[test] fn closures_03() { run("do { fn foo() { 'hello' . print } ; do { fn bar() { foo() } bar() } }", "hello\n") }
#[test] fn closures_04() { run("do { fn foo() { 'hello' . print } ; fn bar(a) { foo() } bar(1) }", "hello\n") }
#[test] fn closures_05() { run("do { fn foo() { 'hello' . print } ; do { fn bar(a) { foo() } bar(1) } }", "hello\n") }
#[test] fn closures_06() { run("do { let x = 'hello' ; do { fn foo() { x . print } foo() } }", "hello\n") }
#[test] fn closures_07() { run("do { let x = 'hello' ; do { do { fn foo() { x . print } foo() } } }", "hello\n") }
#[test] fn closures_08() { run("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; x } foo() . print", "hello\n") }
#[test] fn closures_09() { run("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; (fn() -> x = 'goodbye')() ; x } foo() . print", "goodbye\n") }
#[test] fn closures_10() { run("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; let y = (fn() -> x)() ; y } foo() . print", "hello\n") }
#[test] fn closures_11() { run("fn foo() { let x = 'hello' ; (fn() -> x += ' world')() ; x } foo() . print", "hello world\n") }
#[test] fn closures_12() { run("fn foo() { let x = 'hello' ; (fn() -> x += ' world')() ; (fn() -> x)() } foo() . print", "hello world\n") }
#[test] fn closure_instead_of_global_variable() { run!("closure_instead_of_global_variable", "outer\n") }
#[test] fn closure_of_loop_variable() { run!("closure_of_loop_variable", "0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n") }
#[test] fn closure_of_partial_function() { run!("closure_of_partial_function", "hello bob and sally\n") }
#[test] fn closure_with_non_unique_values() { run!("closure_with_non_unique_values", "toast\nbread\n") }
#[test] fn closure_without_stack_semantics() { run!("closure_without_stack_semantics", "return from outer\ncreate inner closure\nvalue\n") }
#[test] fn closures_are_poor_mans_classes() { run!("closures_are_poor_mans_classes", "obj1.x = 0 obj2.x = 0\nobj1.x = 5 obj2.x = 0\nobj1.x = 5 obj2.x = 70\n") }
#[test] fn closures_inner_multiple_functions_read_stack() { run!("closures_inner_multiple_functions_read_stack", "hello\nworld\n") }
#[test] fn closures_inner_multiple_functions_read_stack_and_heap() { run!("closures_inner_multiple_functions_read_stack_and_heap", "hello\nworld\n") }
#[test] fn closures_inner_multiple_variables_read_heap() { run!("closures_inner_multiple_variables_read_heap", "('hello', 'world')\n") }
#[test] fn closures_inner_multiple_variables_read_stack() { run!("closures_inner_multiple_variables_read_stack", "('hello', 'world')\n") }
#[test] fn closures_inner_read_heap() { run!("closures_inner_read_heap", "hello\n") }
#[test] fn closures_inner_read_stack() { run!("closures_inner_read_stack", "hello\n") }
#[test] fn closures_inner_write_heap_read_heap() { run!("closures_inner_write_heap_read_heap", "goodbye\n") }
#[test] fn closures_inner_write_stack_read_heap() { run!("closures_inner_write_stack_read_heap", "goodbye\n") }
#[test] fn closures_inner_write_stack_read_return() { run!("closures_inner_write_stack_read_return", "goodbye\n") }
#[test] fn closures_inner_write_stack_read_stack() { run!("closures_inner_write_stack_read_stack", "goodbye\n") }
#[test] fn closures_nested_inner_read_heap() { run!("closures_nested_inner_read_heap", "hello\n") }
#[test] fn closures_nested_inner_read_heap_x2() { run!("closures_nested_inner_read_heap_x2", "hello\n") }
#[test] fn closures_nested_inner_read_stack() { run!("closures_nested_inner_read_stack", "hello\n") }
#[test] fn closure_upvalue_never_captured() { run!("closure_upvalue_never_captured", "") }
#[test] fn function_return_1() { run("fn foo() { return 3 } foo() . print", "3\n") }
#[test] fn function_return_2() { run("fn foo() { let x = 3; return x } foo() . print", "3\n") }
#[test] fn function_return_3() { run("fn foo() { let x = 3; do { return x } } foo() . print", "3\n") }
#[test] fn function_return_4() { run("fn foo() { let x = 3; do { let x; } return x } foo() . print", "3\n") }
#[test] fn function_return_5() { run("fn foo() { let x; do { let x = 3; return x } } foo() . print", "3\n") }
#[test] fn function_return_no_value() { run("fn foo() { print('hello') ; return ; print('world') } foo() . print", "hello\nnil\n") }
#[test] fn partial_func_1() { run("'apples and bananas' . replace ('a', 'o') . print", "opples ond bononos\n") }
#[test] fn partial_func_2() { run("'apples and bananas' . replace ('a') ('o') . print", "opples ond bononos\n") }
#[test] fn partial_func_3() { run("print('apples and bananas' . replace ('a') ('o'))", "opples ond bononos\n") }
#[test] fn partial_func_4() { run("let x = replace ('a', 'o') ; 'apples and bananas' . x . print", "opples ond bononos\n") }
#[test] fn partial_func_5() { run("let x = replace ('a', 'o') ; print(x('apples and bananas'))", "opples ond bononos\n") }
#[test] fn partial_func_6() { run("('o' . replace('a')) ('apples and bananas') . print", "opples ond bononos\n") }
#[test] fn partial_function_composition_1() { run("fn foo(a, b, c) { c . print } (3 . (2 . (1 . foo)))", "3\n") }
#[test] fn partial_function_composition_2() { run("fn foo(a, b, c) { c . print } (2 . (1 . foo)) (3)", "3\n") }
#[test] fn partial_function_composition_3() { run("fn foo(a, b, c) { c . print } (1 . foo) (2) (3)", "3\n") }
#[test] fn partial_function_composition_4() { run("fn foo(a, b, c) { c . print } foo (1) (2) (3)", "3\n") }
#[test] fn partial_function_composition_5() { run("fn foo(a, b, c) { c . print } foo (1, 2) (3)", "3\n") }
#[test] fn partial_function_composition_6() { run("fn foo(a, b, c) { c . print } foo (1) (2, 3)", "3\n") }
#[test] fn partial_function_zero_arg_user_function() { run("fn foo(a, b) {} ; foo() . repr . print", "fn foo(a, b)\n") }
#[test] fn partial_function_zero_arg_native_function() { run("len() . repr . print", "fn len(x)\n") }
#[test] fn partial_function_zero_arg_operator_function() { run("(+)() . repr . print", "fn (+)(lhs, rhs)\n") }
#[test] fn partial_function_zero_arg_partial_user_function() { run("fn foo(a, b) {} ; foo(1)() . repr . print", "fn foo(a, b)\n") }
#[test] fn partial_function_zero_arg_partial_native_function() { run("push(1)() . repr . print", "fn push(value, collection)\n") }
#[test] fn partial_function_zero_arg_partial_operator_function() { run("(+)(1)() . repr . print", "fn (+)(lhs, rhs)\n") }
#[test] fn partial_function_zero_arg_user_not_optimized() { run("fn f(x) -> x() ; f(f(f(f))) . repr . print", "fn f(x)\n") }
#[test] fn partial_function_zero_arg_native_not_optimized() { run("fn f(x) -> x() ; f(f(f(len))) . repr . print", "fn len(x)\n") }
#[test] fn partial_function_zero_arg_operator_not_optimized() { run("fn f(x) -> x() ; f(f(f(+))) . repr . print", "fn (+)(lhs, rhs)\n") }
#[test] fn partial_user_functions_1() { run("fn foo(x) -> print(x) ; foo()('hi')", "hi\n") }
#[test] fn partial_user_functions_2() { run("fn foo(x, y) -> print(x, y) ; foo()('hi', 'there')", "hi there\n") }
#[test] fn partial_user_functions_3() { run("fn foo(x, y) -> print(x, y) ; foo('hi')('there')", "hi there\n") }
#[test] fn partial_user_functions_4() { run("fn foo(x, y) -> print(x, y) ; foo('hi')()('there')", "hi there\n") }
#[test] fn partial_user_functions_5() { run("fn foo(x, y) -> print(x, y) ; [1, 2] . map(foo('hello'))", "hello 1\nhello 2\n") }
#[test] fn partial_user_functions_6() { run("fn add(x, y) -> x + y ; [1, 2, 3] . map(add(3)) . print", "[4, 5, 6]\n") }
#[test] fn partial_user_functions_7() { run("fn add(x, y, z) -> x + y ; [1, 2, 3] . map(add(3)) . print", "[fn add(x, y, z), fn add(x, y, z), fn add(x, y, z)]\n") }
#[test] fn partial_user_functions_8() { run("fn add(x, y) -> x + y ; add(1)(2) . print", "3\n") }
#[test] fn function_with_one_default_arg() { run("fn foo(a, b?) { print(a, b) } ; foo('test') ; foo('test', 'bar')", "test nil\ntest bar\n") }
#[test] fn function_with_one_default_arg_not_enough() { run("fn foo(a, b?) { print(a, b) } ; foo()", "") }
#[test] fn function_with_one_default_arg_too_many() { run("fn foo(a, b?) { print(a, b) } ; foo(1, 2, 3)", "Incorrect number of arguments for fn foo(a, b), got 3\n  at: line 1 (<test>)\n\n1 | fn foo(a, b?) { print(a, b) } ; foo(1, 2, 3)\n2 |                                    ^^^^^^^^^\n") }
#[test] fn function_many_default_args() { run("fn foo(a, b = 1, c = 1 + 1, d = 1 * 3) { print(a, b, c, d) } foo('test') ; foo('and', 11) ; foo('other', 11, 22) ; foo('things', 11, 22, 33)", "test 1 2 3\nand 11 2 3\nother 11 22 3\nthings 11 22 33\n") }
#[test] fn function_unroll_1() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(...['hello', 'the', 'world'])", "hello the world\n") }
#[test] fn function_unroll_2() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, 3, ...[])", "1 2 3\n") }
#[test] fn function_unroll_3() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(1, ...[], 2, ...[], 3)", "1 2 3\n") }
#[test] fn function_unroll_4() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(...'ab', 'c')", "a b c\n") }
#[test] fn function_unroll_5() { run("fn foo(a, b, c, d) -> print(a, b, c, d) ; foo(...'ab', ...'cd')", "a b c d\n") }
#[test] fn function_unroll_6() { run("fn foo(a, b, c, d) -> print(a, b, c, d) ; foo(...'a', ...'bc', ...'d')", "a b c d\n") }
#[test] fn function_unroll_7() { run("fn foo(a, b, c, d) -> print(a, b, c, d) ; foo('a', ...'bc', 'd')", "a b c d\n") }
#[test] fn function_unroll_8() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(1, ...'ab')", "1 a b\n") }
#[test] fn function_unroll_9() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(...'ab', 3)", "a b 3\n") }
#[test] fn function_unroll_10() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, ...[3, 4])", "Incorrect number of arguments for fn foo(a, b, c), got 4\n  at: line 1 (<test>)\n\n1 | fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, ...[3, 4])\n2 |                                        ^^^^^^^^^^^^^^^^^\n") }
#[test] fn function_unroll_11() { run("fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, ...[]) is function . print", "true\n") }
#[test] fn function_unroll_12() { run("sum([1, 2, 3, 4, 5]) . print", "15\n") }
#[test] fn function_unroll_13() { run("sum(...[1, 2, 3, 4, 5]) . print", "15\n") }
#[test] fn function_unroll_14() { run("print(...[1, 2, 3])", "1 2 3\n") }
#[test] fn function_unroll_15() { run("print(...[print(...[1, 2, 3])])", "1 2 3\nnil\n") }
#[test] fn function_unroll_16() { run("print(...[], ...[print(...[], 'second', ...[], ...[print('first', ...[])])], ...[], ...[print('third')])", "first\nsecond nil\nthird\nnil nil\n") }
#[test] fn function_unroll_17() { run("print(1, ...[2, print('a', ...[1, 2, 3], 'e'), -2], 3)", "a 1 2 3 e\n1 2 nil -2 3\n") }
#[test] fn function_var_args_1() { run("fn foo(*a) -> print(a) ; foo()", "()\n") }
#[test] fn function_var_args_2() { run("fn foo(*a) -> print(a) ; foo(1)", "(1)\n") }
#[test] fn function_var_args_3() { run("fn foo(*a) -> print(a) ; foo(1, 2)", "(1, 2)\n") }
#[test] fn function_var_args_4() { run("fn foo(*a) -> print(a) ; foo(1, 2, 3)", "(1, 2, 3)\n") }
#[test] fn function_var_args_5() { run("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1)", "1 nil ()\n") }
#[test] fn function_var_args_6() { run("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2)", "1 2 ()\n") }
#[test] fn function_var_args_7() { run("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2, 3)", "1 2 (3)\n") }
#[test] fn function_var_args_8() { run("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2, 3, 4)", "1 2 (3, 4)\n") }
#[test] fn function_var_args_9() { run("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2, 3, 4, 5)", "1 2 (3, 4, 5)\n") }
#[test] fn function_call_with_over_u8_arguments() { run("sum(...range(1 + 1000)) . print", "500500\n") }
#[test] fn operator_functions_01() { run("(+3) . print", "(+)\n") }
#[test] fn operator_functions_02() { run("4 . (+3) . print", "7\n") }
#[test] fn operator_functions_03() { run("4 . (-) . print", "-4\n") }
#[test] fn operator_functions_04() { run("true . (!) . print", "false\n") }
#[test] fn operator_functions_05() { run("let f = (/5) ; 15 . f . print", "3\n") }
#[test] fn operator_functions_06() { run("let f = (*5) ; f(3) . print", "15\n") }
#[test] fn operator_functions_07() { run("let f = (<5) ; 3 . f . print", "true\n") }
#[test] fn operator_functions_08() { run("let f = (>5) ; 3 . f . print", "false\n") }
#[test] fn operator_functions_09() { run("2 . (**5) . print", "32\n") }
#[test] fn operator_functions_10() { run("7 . (%3) . print", "1\n") }
#[test] fn operator_functions_eval() { run("(+)(1, 2) . print", "3\n") }
#[test] fn operator_functions_partial_eval() { run("(+)(1)(2) . print", "3\n") }
#[test] fn operator_functions_compose_and_eval() { run("2 . (+)(1) . print", "3\n") }
#[test] fn operator_functions_compose() { run("1 . (2 . (+)) . print", "3\n") }
#[test] fn operator_in_expr() { run("(1 < 2) . print", "true\n") }
#[test] fn operator_partial_right() { run("((<2)(1)) . print", "true\n") }
#[test] fn operator_partial_left() { run("((1<)(2)) . print", "true\n") }
#[test] fn operator_partial_twice() { run("((<)(1)(2)) . print", "true\n") }
#[test] fn operator_as_prefix() { run("((<)(1, 2)) . print", "true\n") }
#[test] fn operator_partial_right_with_composition() { run("(1 . (<2)) . print", "true\n") }
#[test] fn operator_partial_left_with_composition() { run("(2 . (1<)) . print", "true\n") }
#[test] fn operator_binary_max_yes() { run("let a = 3 ; a max= 6; a . print", "6\n") }
#[test] fn operator_binary_max_no() { run("let a = 3 ; a max= 2; a . print", "3\n") }
#[test] fn operator_binary_min_yes() { run("let a = 3 ; a min= 1; a . print", "1\n") }
#[test] fn operator_binary_min_no() { run("let a = 3 ; a min= 5; a . print", "3\n") }
#[test] fn operator_dot_equals() { run("let x = 'hello' ; x .= sort ; x .= reduce(+) ; x . print", "ehllo\n") }
#[test] fn operator_dot_equals_operator_function() { run("let x = 3 ; x .= (+4) ; x . print", "7\n") }
#[test] fn operator_dot_equals_anonymous_function() { run("let x = 'hello' ; x .= fn(x) -> x[0] * len(x) ; x . print", "hhhhh\n") }
#[test] fn operator_in() { run("let f = (in) ; f(1, [1]) . print", "true\n") }
#[test] fn operator_in_partial_left() { run("let f = (1 in) ; f([1]) . print", "true\n") }
#[test] fn operator_in_partial_right() { run("let f = (in [1]) ; f(1) . print", "true\n") }
#[test] fn operator_not_in() { run("let f = (not in) ; f(1, []) . print", "true\n") }
#[test] fn operator_not_in_partial_left() { run("let f = (1 not in) ; f([]) . print", "true\n") }
#[test] fn operator_not_in_partial_right() { run("let f = (not in []) ; f(1) . print", "true\n") }
#[test] fn operator_is() { run("let f = (is) ; f(1, int) . print", "true\n") }
#[test] fn operator_is_iterable_yes() { run("[[], '123', set(), dict()] . all(is iterable) . print", "true\n") }
#[test] fn operator_is_iterable_no() { run("[true, false, nil, 123, fn() -> {}] . any(is iterable) . print", "false\n") }
#[test] fn operator_is_any_yes() { run("[[], '123', set(), dict(), 123, true, false, nil, fn() -> nil] . all(is any) . print", "true\n") }
#[test] fn operator_is_function_yes() { run("(fn() -> nil) is function . print", "true\n") }
#[test] fn operator_is_function_no() { run("[nil, true, 123, '123', [], set()] . any(is function) . print", "false\n") }
#[test] fn operator_is_partial_left() { run("let f = (1 is) ; f(int) . print", "true\n") }
#[test] fn operator_is_partial_right() { run("let f = (is int) ; f(1) . print", "true\n") }
#[test] fn operator_not_is() { run("let f = (is not) ; f(1, str) . print", "true\n") }
#[test] fn operator_not_is_partial_left() { run("let f = (1 is not) ; f(str) . print", "true\n") }
#[test] fn operator_not_is_partial_right() { run("let f = (is not str) ; f(1) . print", "true\n") }
#[test] fn operator_sub_as_unary() { run("(-)(3) . print", "-3\n") }
#[test] fn operator_sub_as_binary() { run("(-)(5, 2) . print", "3\n") }
#[test] fn operator_sub_as_partial_not_allowed() { run("(-3) . print", "-3\n") }
#[test] fn operator_lt_chained_yes() { run("(1 < 2 < 3) . print", "true\n") }
#[test] fn operator_lt_chained_yes_long() { run("let x = 5 ; (1 < 5 <= x >= x > 3 <= x < 7) . print", "true\n") }
#[test] fn operator_lt_chained_no() { run("(1 < 5 < 2) . print", "false\n") }
#[test] fn operator_lt_chained_no_long() { run("let x = 9 ; (1 < 5 <= x >= x > 3 <= x < 7) . print", "false\n") }
#[test] fn operator_eq_chained_short_circuit_yes_yes() { run("(print(1) == print(2) == print(3)) . print", "1\n2\n3\ntrue\n") }
#[test] fn operator_eq_chained_short_circuit_yes_no() { run("(print(1) == print(2) != print(3)) . print", "1\n2\n3\nfalse\n") }
#[test] fn operator_eq_chained_short_circuit_no_yes() { run("(print(1) != print(2) == print(3)) . print", "1\n2\nfalse\n") }
#[test] fn operator_eq_chained_short_circuit_no_no() { run("(print(1) != print(2) != print(3)) . print", "1\n2\nfalse\n") }
#[test] fn operator_logical_not_of_bool() { run("not true . print ; not false . print", "false\ntrue\n") }
#[test] fn operator_logical_not_coerces_to_bool() { run("not [] . print ; not [1, 2, 3] . print ; not 123 . print ; not 0 . print", "true\nfalse\nfalse\ntrue\n") }
#[test] fn operator_logical_not_is_not_elementwise_on_vector() { run("not ( vector() ) . print ; not (1,) . print ; not (1, 2, 3) . print", "true\nfalse\nfalse\n") }
#[test] fn operator_logical_not_as_operator_function_raw() { run("let f = (not) ; (1 + 1i) . f . print", "false\n") }
#[test] fn operator_logical_not_as_operator_function_argument() { run("['', 'a', 'ab', 'abc'] . map(not) . print", "[true, false, false, false]\n") }
#[test] fn arrow_functions_01() { run("fn foo() -> 3 ; foo() . print", "3\n") }
#[test] fn arrow_functions_02() { run("fn foo() -> 3 ; foo() . print", "3\n") }
#[test] fn arrow_functions_03() { run("fn foo(a) -> 3 * a ; 5 . foo . print", "15\n") }
#[test] fn arrow_functions_04() { run("fn foo(x, y, z) -> x + y + z ; foo(1, 2, 4) . print", "7\n") }
#[test] fn arrow_functions_05() { run("fn foo() -> (fn() -> 123) ; foo() . print", "_\n") }
#[test] fn arrow_functions_06() { run("fn foo() -> (fn() -> 123) ; foo() . repr . print", "fn _()\n") }
#[test] fn arrow_functions_07() { run("fn foo() -> (fn(a, b, c) -> 123) ; foo() . print", "_\n") }
#[test] fn arrow_functions_08() { run("fn foo() -> (fn(a, b, c) -> 123) ; foo() . repr . print", "fn _(a, b, c)\n") }
#[test] fn arrow_functions_09() { run("fn foo() -> (fn() -> 123) ; foo()() . print", "123\n") }
#[test] fn arrow_functions_10() { run("let x = fn() -> 3 ; x() . print", "3\n") }
#[test] fn arrow_functions_11() { run("let x = fn() -> fn() -> 4 ; x() . print", "_\n") }
#[test] fn arrow_functions_12() { run("let x = fn() -> fn() -> 4 ; x() . repr .print", "fn _()\n") }
#[test] fn arrow_functions_13() { run("let x = fn() -> fn() -> 4 ; x()() . print", "4\n") }
#[test] fn arrow_functions_14() { run("fn foo() { if true { return 123 } else { return 321 } } ; foo() . print", "123\n") }
#[test] fn arrow_functions_15() { run("fn foo() { if false { return 123 } else { return 321 } } ; foo() . print", "321\n") }
#[test] fn arrow_functions_16() { run("fn foo() { let x = 1234; x } ; foo() . print", "1234\n") }
#[test] fn arrow_functions_17() { run("fn foo() { let x, y; x + ' and ' + y } ; foo() . print", "nil and nil\n") }
#[test] fn arrow_functions_18() { run("fn foo() { fn bar() -> 3 ; bar } ; foo() . print", "bar\n") }
#[test] fn arrow_functions_19() { run("fn foo() { fn bar() -> 3 ; bar } ; foo() . repr . print", "fn bar()\n") }
#[test] fn arrow_functions_20() { run("fn foo() { fn bar() -> 3 ; bar } ; foo()() . print", "3\n") }
#[test] fn annotation_named_func_with_name() { run("fn par(f) -> (fn(x) -> f('hello')) ; @par fn foo(x) -> print(x) ; foo('goodbye')", "hello\n") }
#[test] fn annotation_named_func_with_expression() { run("fn par(a, f) -> (fn(x) -> f(a)) ; @par('hello') fn foo(x) -> print(x) ; foo('goodbye')", "hello\n") }
#[test] fn annotation_expression_func_with_name() { run("fn par(f) -> (fn(x) -> f('hello')) ; par(fn(x) -> print(x))('goodbye')", "hello\n") }
#[test] fn annotation_expression_func_with_expression() { run("fn par(a, f) -> (fn(x) -> f(a)) ; par('hello', fn(x) -> print(x))('goodbye')", "hello\n") }
#[test] fn annotation_iife() { run("fn iife(f) -> f() ; @iife fn foo() -> print('hello')", "hello\n") }
#[test] fn function_call_on_list() { run("'hello' . [0] . print", "h\n") }
#[test] fn function_compose_on_list() { run("[-1]('hello') . print", "o\n") }
#[test] fn slice_literal_2_no_nil() { run("let x = [1:2] ; x . print", "[1:2]\n") }
#[test] fn slice_literal_2_all_nil() { run("let x = [:] ; x . print", "[:]\n") }
#[test] fn slice_literal_3_no_nil() { run("let x = [1:2:3] ; x . print", "[1:2:3]\n") }
#[test] fn slice_literal_3_all_nil() { run("let x = [::] ; x . print", "[:]\n") }
#[test] fn slice_literal_3_last_not_nil() { run("let x = [::-1] ; x . print", "[::-1]\n") }
#[test] fn slice_literal_not_int() { run("let x = ['hello':'world'] ; x . print", "TypeError: Expected 'hello' of type 'str' to be a int\n  at: line 1 (<test>)\n\n1 | let x = ['hello':'world'] ; x . print\n2 |         ^^^^^^^^^^^^^^^^^\n") }
#[test] fn slice_in_expr_1() { run("'1234' . [::-1] . print", "4321\n") }
#[test] fn slice_in_expr_2() { run("let x = [::-1] ; '1234' . x . print", "4321\n") }
#[test] fn slice_in_expr_3() { run("'hello the world!' . split(' ') . map([2:]) . print", "['llo', 'e', 'rld!']\n") }
#[test] fn bool_comparisons_1() { run("print(false < false, false < true, true < false, true < true)", "false true false false\n") }
#[test] fn bool_comparisons_2() { run("print(false <= false, false >= true, true >= false, true <= true)", "true false true true\n") }
#[test] fn bool_operator_add() { run("true + true + false + false . print", "2\n") }
#[test] fn bool_sum() { run("range(10) . map(>3) . sum . print", "6\n") }
#[test] fn bool_reduce_add() { run("range(10) . map(>3) . reduce(+) . print", "6\n") }
#[test] fn int_operators() { run("print(5 - 3, 12 + 5, 3 * 9, 16 / 3)", "2 17 27 5\n") }
#[test] fn int_div_mod() { run("print(3 / 2, 3 / 3, -3 / 2, 10 % 3, 11 % 3, 12 % 3)", "1 1 -2 1 2 0\n") }
#[test] fn int_mod_by_zero() { run("1 % 0", "Compile Error:\n\nValueError: Modulo by zero\n  at: line 1 (<test>)\n\n1 | 1 % 0\n2 |   ^\n") }
#[test] fn int_mod_by_negative() { run("5 % -2 . print", "-1\n") }
#[test] fn int_div_by_zero() { run("print(15 / 0)", "Compile Error:\n\nValueError: Division by zero\n  at: line 1 (<test>)\n\n1 | print(15 / 0)\n2 |          ^\n") }
#[test] fn int_left_right_shift() { run("print(1 << 10, 16 >> 1, 16 << -1, 1 >> -10)", "1024 8 8 1024\n") }
#[test] fn int_comparisons_1() { run("print(1 < 3, -5 < -10, 6 > 7, 6 > 4)", "true false false true\n") }
#[test] fn int_comparisons_2() { run("print(1 <= 3, -5 < -10, 3 <= 3, 2 >= 2, 6 >= 7, 6 >= 4, 6 <= 6, 8 >= 8)", "true false true true false true true true\n") }
#[test] fn int_equality() { run("print(1 == 3, -5 == -10, 3 != 3, 2 == 2, 6 != 7)", "false false false true true\n") }
#[test] fn int_bitwise_operators() { run("print(0b111 & 0b100, 0b1100 | 0b1010, 0b1100 ^ 0b1010)", "4 14 6\n") }
#[test] fn int_to_hex() { run("1234 . hex . repr . print", "'4d2'\n") }
#[test] fn int_from_hex() { run("'4d2' . hex . repr . print", "1234\n") }
#[test] fn int_from_hex_error() { run("'4x2' . hex . repr . print", "TypeError: Cannot convert '4x2' of type 'str' to an int\n  at: line 1 (<test>)\n\n1 | '4x2' . hex . repr . print\n2 |       ^^^^^\n") }
#[test] fn int_from_hex_error_leading_0x() { run("'0xff' . hex . repr . print", "TypeError: Cannot convert '0xff' of type 'str' to an int\n  at: line 1 (<test>)\n\n1 | '0xff' . hex . repr . print\n2 |        ^^^^^\n") }
#[test] fn int_to_bin() { run("1234 . bin . repr . print", "'10011010010'\n") }
#[test] fn int_from_bin() { run("'10011010010' . bin . repr . print", "1234\n") }
#[test] fn int_from_bin_error() { run("'100110210010' . bin . repr . print", "TypeError: Cannot convert '100110210010' of type 'str' to an int\n  at: line 1 (<test>)\n\n1 | '100110210010' . bin . repr . print\n2 |                ^^^^^\n") }
#[test] fn int_from_bin_error_leading_0b() { run("'0b11' . bin . repr . print", "TypeError: Cannot convert '0b11' of type 'str' to an int\n  at: line 1 (<test>)\n\n1 | '0b11' . bin . repr . print\n2 |        ^^^^^\n") }
#[test] fn int_default_value_yes() { run("int('123', 567) . print", "123\n") }
#[test] fn int_default_value_no() { run("int('yes', 567) . print", "567\n") }
#[test] fn int_min_and_max() { run("[int.min, max(int)] . print", "[-4611686018427387904, 4611686018427387903]\n") }
#[test] fn int_min_and_max_indirect() { run("let i = int ; [min(i), i.max] . print", "[-4611686018427387904, 4611686018427387903]\n")}
#[test] fn int_min_and_max_no_opt() { run_unopt("[int.min, max(int)] . print", "[-4611686018427387904, 4611686018427387903]\n") }
#[test] fn int_min_negative_underflow() { run("min(int) - 1 . print", "4611686018427387903\n") }
#[test] fn int_max_positive_overflow() { run("max(int) + 1 . print", "-4611686018427387904\n") }
#[test] fn int_max_pow_by_negative() { run("123 ** -3", "Compile Error:\n\nValueError: Power of an integral value by '-3' which is negative\n  at: line 1 (<test>)\n\n1 | 123 ** -3\n2 |     ^^\n") }
#[test] fn complex_add() { run("(1 + 2i) + (3 + 4j) . print", "4 + 6i\n") }
#[test] fn complex_mul() { run("(1 + 2i) * (3 + 4j) . print", "-5 + 10i\n") }
#[test] fn complex_str() { run("1 + 1i . print", "1 + 1i\n") }
#[test] fn complex_str_negative() { run("-1 - 1i . print", "-1 - 1i\n") }
#[test] fn complex_str_no_real_part() { run("123i . print", "123i\n") }
#[test] fn complex_typeof() { run("123i . typeof . print", "complex\n") }
#[test] fn complex_no_real_part_is_int() { run("1i * 1i . typeof . print", "int\n") }
#[test] fn complex_to_vector() { run("1 + 3i . vector . print", "(1, 3)\n") }
#[test] fn complex_mod() { run("(15i + -9) % 4 . print", "3 + 3i\n") }
#[test] fn complex_mod_by_zero() { run("(15i + -9) % 0", "Compile Error:\n\nValueError: Modulo by zero\n  at: line 1 (<test>)\n\n1 | (15i + -9) % 0\n2 |            ^\n") }
#[test] fn complex_mod_by_negative() { run("(15i + -9) % -4 . print", "-1 - 1i\n") }
#[test] fn complex_pow_by_negative() { run("(15i + -9) ** -3 . print", "Compile Error:\n\nValueError: Power of an integral value by '-3' which is negative\n  at: line 1 (<test>)\n\n1 | (15i + -9) ** -3 . print\n2 |            ^^\n") }
#[test] fn complex_compare_with_integers() { run("[1, -1 - 1i, 1 + 1i, -1, 1i, 0, -1i, -1 + 1i, 1 - 1i] . sort . print", "[-1 - 1i, -1i, 1 - 1i, -1, 0, 1, -1 + 1i, 1i, 1 + 1i]\n") }
#[test] fn rational_zero() { run("rational 0 . print", "0 / 1\n") }
#[test] fn rational_zero_div_by_non_one() { run("rational(0, -5) . print", "0 / 1\n") }
#[test] fn rational_one() { run("rational 1 . print", "1 / 1\n") }
#[test] fn rational_one_always_in_lowest_form() { run("rational(25, -25) . print", "-1 / 1\n") }
#[test] fn rational_compare_with_integers() { run("[rational(3, 2), rational 0, 1, rational 2, 5, rational(4, 3), 2, rational(5, 2), 4, rational 4, 3, rational(1, 3), 0, rational 1, rational(7, 3)] . sort . print", "[0 / 1, 0, 1 / 3, 1, 1 / 1, 4 / 3, 3 / 2, 2 / 1, 2, 7 / 3, 5 / 2, 3, 4, 4 / 1, 5]\n") }
#[test] fn rational_compare_with_complex() { run("[0, rational(1, 3), -3i, rational 1, 1 + 1i, rational(-1, 2)] . sort . print", "[-3i, -1 / 2, 0, 1 / 3, 1 / 1, 1 + 1i]\n") }
#[test] fn rational_eq_with_integers() { run("print(rational 0 == 0, rational(1, 3) == 3, rational(5, 1) == 5)", "true false true\n") }
#[test] fn rational_converts_div_by_zero() { run("rational(123, 0) . print", "ValueError: Division by zero\n  at: line 1 (<test>)\n\n1 | rational(123, 0) . print\n2 |         ^^^^^^^^\n") }
#[test] fn rational_converts_div_by_false() { run("rational(123, false) . print", "ValueError: Division by zero\n  at: line 1 (<test>)\n\n1 | rational(123, false) . print\n2 |         ^^^^^^^^^^^^\n") }
#[test] fn rational_converts_int_like() { run("rational(false, true) . print", "0 / 1\n") }
#[test] fn rational_converts_rationals_1() { run("rational(rational 1, rational 3) . print", "1 / 3\n") }
#[test] fn rational_converts_rationals_2() { run("rational(rational(3, 5), rational(2, 7)) . print", "21 / 10\n") }
#[test] fn rational_parses_integers() { run("rational '1234' . print", "1234 / 1\n") }
#[test] fn rational_parses_big_integers() { run("rational '10000000000000000000000000000000000000000' . print", "10000000000000000000000000000000000000000 / 1\n") }
#[test] fn rational_parses_rationals() { run("rational '3/5' . print", "3 / 5\n") }
#[test] fn rational_parses_big_rationals() { run("rational '10000000000000000000000000000000000000000/20000000000000000000000000000000000000000' . print", "1 / 2\n") }
#[test] fn rational_parses_zero() { run("rational '0/123' . print", "0 / 1\n") }
#[test] fn rational_parses_divide_by_zero() { run("rational '123/0' . print", "ValueError: Cannot convert '123/0' of type 'str' to an rational: string has zero denominator\n  at: line 1 (<test>)\n\n1 | rational '123/0' . print\n2 |          ^^^^^^^\n") }
#[test] fn rational_typeof() { run("rational(3, 2) . typeof . print ; rational 1 . typeof . print", "rational\nrational\n") }
#[test] fn rational_unary_sub() { run("print( - rational(1, -3) )", "1 / 3\n") }
#[test] fn rational_binary_mul_with_int() { run("print( rational(5, 6) * 4 )", "10 / 3\n") }
#[test] fn rational_binary_mul_with_rational() { run("print( rational(2, 3) * rational(7, 2) )", "7 / 3\n") }
#[test] fn rational_binary_div_with_int() { run("print( rational(1, 2) / 6 )", "1 / 12\n") }
#[test] fn rational_binary_div_with_rational() { run("print( rational(2, 3) / rational(7, 2) )", "4 / 21\n") }
#[test] fn rational_binary_div_with_rational_zero() { run("print( rational(2, 3) / rational(0, 5) )", "ValueError: Division by zero\n  at: line 1 (<test>)\n\n1 | print( rational(2, 3) / rational(0, 5) )\n2 |                       ^\n") }
#[test] fn rational_binary_pow_with_int() { run("print( rational(2, 3) ** 5 )", "32 / 243\n") }
#[test] fn rational_binary_pow_with_bigger_int() { run("print( rational(1, 2) ** 20 )", "1 / 1048576\n") }
#[test] fn rational_binary_pow_with_even_bigger_int() { run("print( rational(10) ** 80 )", "100000000000000000000000000000000000000000000000000000000000000000000000000000000 / 1\n") }
#[test] fn rational_binary_pow_with_unreasonably_bigger_int() { run("print( rational(1, 2) ** (max int) )", "ValueError: value would be too large!\n  at: line 1 (<test>)\n\n1 | print( rational(1, 2) ** (max int) )\n2 |                       ^^\n") }
#[test] fn rational_binary_is() { run("[nil, true, 1, 1 + 1i, 1 / 1, 3 / 2] . map(is rational) . print", "[false, true, true, false, true, true]\n") }
#[test] fn rational_numer_of_rational() { run("print(numer(rational(1, 3)), numer(rational 5), rational(7, 3) . numer)", "1 / 1 5 / 1 7 / 1\n") }
#[test] fn rational_numer_of_non_rational() { run("print(numer 3, numer false)", "3 / 1 0 / 1\n") }
#[test] fn rational_numer_of_non_integer() { run("print(numer(1 + 1i))", "TypeError: Expected '1 + 1i' of type 'complex' to be a rational\n  at: line 1 (<test>)\n\n1 | print(numer(1 + 1i))\n2 |            ^^^^^^^^\n") }
#[test] fn rational_denom_of_rational() { run("[rational 3, rational(1, 6), rational(7, 3)] . map denom . print", "[1 / 1, 6 / 1, 3 / 1]\n") }
#[test] fn rational_denom_of_non_rational() { run("[rational true, rational 7] . map denom . print", "[1 / 1, 1 / 1]\n") }
#[test] fn rational_denom_of_non_integer() { run("denom('yes')", "TypeError: Expected 'yes' of type 'str' to be a rational\n  at: line 1 (<test>)\n\n1 | denom('yes')\n2 |      ^^^^^^^\n") }
#[test] fn rational_to_int() { run("rational(25, 1) . int . print", "25\n") }
#[test] fn rational_to_int_of_non_integral() { run("rational(7, 3) . int . print", "ValueError: Cannot convert rational ''7 / 3' of type 'rational'' to an integer as it is not an integral value.\n  at: line 1 (<test>)\n\n1 | rational(7, 3) . int . print\n2 |                ^^^^^\n") }
#[test] fn rational_to_int_of_too_large() { run("rational(10, 1) ** 30 . int . print", "ValueError: Cannot convert rational ''1000000000000000000000000000000 / 1' of type 'rational'' to an integer as it is too large for the target type\n  at: line 1 (<test>)\n\n1 | rational(10, 1) ** 30 . int . print\n2 |                       ^^^^^\n") }
#[test] fn rational_to_int_of_fits_in_i64_not_in_i63() { run("rational(min int) - 2 . int . print", "ValueError: Cannot convert rational ''-4611686018427387906 / 1' of type 'rational'' to an integer as it is too large for the target type\n  at: line 1 (<test>)\n\n1 | rational(min int) - 2 . int . print\n2 |                       ^^^^^\n") }
#[test] fn str_empty() { run("'' . print", "\n") }
#[test] fn str_add() { run("print(('a' + 'b') + (3 + 4) + (' hello' + 3) + (' and' + true + nil))", "ab7 hello3 andtruenil\n") }
#[test] fn str_partial_left_add() { run("'world ' . (+'hello') . print", "world hello\n") }
#[test] fn str_partial_right_add() { run("' world' . ('hello'+) . print", "hello world\n") }
#[test] fn str_mul() { run("print('abc' * 3)", "abcabcabc\n") }
#[test] fn str_index() { run("'hello'[1] . print", "e\n") }
#[test] fn str_slice_start() { run("'hello'[1:] . print", "ello\n") }
#[test] fn str_slice_stop() { run("'hello'[:3] . print", "hel\n") }
#[test] fn str_slice_start_stop() { run("'hello'[1:3] . print", "el\n") }
#[test] fn str_slice_large_step_positive() { run("'abc'[1:1000000000:1000000000] . print", "b\n") }
#[test] fn str_slice_large_step_negative() { run("'abc'[-1:-1000000000:-1000000000] . print", "c\n") }
#[test] fn str_operator_in_yes() { run("'hello' in 'hey now, hello world' . print", "true\n") }
#[test] fn str_operator_in_no() { run("'hello' in 'hey now, \\'ello world' . print", "false\n") }
#[test] fn str_format_with_percent_no_args() { run("'100 %%' % vector() . print", "100 %\n") }
#[test] fn str_format_with_one_int_arg() { run("'an int: %d' % (123,) . print", "an int: 123\n") }
#[test] fn str_format_with_one_neg_int_arg() { run("'an int: %d' % (-123,) . print", "an int: -123\n") }
#[test] fn str_format_with_one_zero_pad_int_arg() { run("'an int: %05d' % (123,) . print", "an int: 00123\n") }
#[test] fn str_format_with_one_zero_pad_neg_int_arg() { run("'an int: %05d' % (-123,) . print", "an int: -0123\n") }
#[test] fn str_format_with_one_space_pad_int_arg() { run("'an int: %5d' % (123,) . print", "an int:   123\n") }
#[test] fn str_format_with_one_space_pad_neg_int_arg() { run("'an int: %5d' % (-123,) . print", "an int:  -123\n") }
#[test] fn str_format_with_one_hex_arg() { run("'an int: %x' % (123,) . print", "an int: 7b\n") }
#[test] fn str_format_with_one_zero_pad_hex_arg() { run("'an int: %04x' % (123,) . print", "an int: 007b\n") }
#[test] fn str_format_with_one_space_pad_hex_arg() { run("'an int: %4x' % (123,) . print", "an int:   7b\n") }
#[test] fn str_format_with_one_bin_arg() { run("'an int: %b' % (123,) . print", "an int: 1111011\n") }
#[test] fn str_format_with_one_zero_pad_bin_arg() { run("'an int: %012b' % (123,) . print", "an int: 000001111011\n") }
#[test] fn str_format_with_one_space_pad_bin_arg() { run("'an int: %12b' % (123,) . print", "an int:      1111011\n") }
#[test] fn str_format_with_many_args() { run("'%d %s %x %b ALL THE THINGS %%!' % (10, 'fifteen', 0xff, 0b10101) . print", "10 fifteen ff 10101 ALL THE THINGS %!\n") }
#[test] fn str_format_with_solo_arg_nil() { run("'hello %s' % nil . print", "hello nil\n") }
#[test] fn str_format_with_solo_arg_int() { run("'hello %s' % 123 . print", "hello 123\n") }
#[test] fn str_format_with_solo_arg_str() { run("'hello %s' % 'world' . print", "hello world\n") }
#[test] fn str_format_nested_0() { run("'%s w%sld %s' % ('hello', 'or', '!') . print", "hello world !\n") }
#[test] fn str_format_nested_1() { run("'%%%s%%s%s %%s' % ('s w', 'ld') % ('hello', 'or', '!') . print", "hello world !\n") }
#[test] fn str_format_nested_2() { run("'%ss%%%%s%s%s%ss' % ('%'*3, '%s', ' ', '%'*2) % ('s w', 'ld') % ('hello', 'or', '!') . print", "hello world !\n") }
#[test] fn str_format_too_many_args() { run("'%d %d %d' % (1, 2)", "ValueError: Not enough arguments for format string\n  at: line 1 (<test>)\n\n1 | '%d %d %d' % (1, 2)\n2 |            ^\n") }
#[test] fn str_format_too_few_args() { run("'%d %d %d' % (1, 2, 3, 4)", "ValueError: Not all arguments consumed in format string, next: '4' of type 'int'\n  at: line 1 (<test>)\n\n1 | '%d %d %d' % (1, 2, 3, 4)\n2 |            ^\n") }
#[test] fn str_format_incorrect_character() { run("'%g' % (1,)", "ValueError: Invalid format character 'g' in format string\n  at: line 1 (<test>)\n\n1 | '%g' % (1,)\n2 |      ^\n") }
#[test] fn str_format_incorrect_width() { run("'%00' % (1,)", "ValueError: Invalid format character '0' in format string\n  at: line 1 (<test>)\n\n1 | '%00' % (1,)\n2 |       ^\n") }
#[test] fn list_empty_constructor() { run("list() . print", "[]\n") }
#[test] fn list_literal_empty() { run("[] . print", "[]\n") }
#[test] fn list_literal_len_1() { run("['hello'] . print", "['hello']\n") }
#[test] fn list_literal_len_2() { run("['hello', 'world'] . print", "['hello', 'world']\n") }
#[test] fn list_literal_unroll_at_start() { run("[...[1, 2, 3], 4, 5] . print", "[1, 2, 3, 4, 5]\n") }
#[test] fn list_literal_unroll_at_end() { run("[0, ...[1, 2, 3]] . print", "[0, 1, 2, 3]\n") }
#[test] fn list_literal_unroll_once() { run("[...[1, 2, 3]] . print", "[1, 2, 3]\n") }
#[test] fn list_literal_unroll_multiple() { run("[...[1, 2, 3], ...[4, 5]] . print", "[1, 2, 3, 4, 5]\n") }
#[test] fn list_literal_unroll_multiple_and_empty() { run("[...[], 0, ...[1, 2, 3], ...[4, 5], ...[], 6] . print", "[0, 1, 2, 3, 4, 5, 6]\n") }
#[test] fn list_from_str() { run("'funny beans' . list . print", "['f', 'u', 'n', 'n', 'y', ' ', 'b', 'e', 'a', 'n', 's']\n") }
#[test] fn list_add() { run("[1, 2, 3] + [4, 5, 6] . print", "[1, 2, 3, 4, 5, 6]\n") }
#[test] fn list_multiply_left() { run("[1, 2, 3] * 3 . print", "[1, 2, 3, 1, 2, 3, 1, 2, 3]\n") }
#[test] fn list_multiply_right() { run("3 * [1, 2, 3] . print", "[1, 2, 3, 1, 2, 3, 1, 2, 3]\n") }
#[test] fn list_multiply_nested() { run("let a = [[1]] * 3; a[0][0] = 2; a . print", "[[2], [2], [2]]\n") }
#[test] fn list_power_left() { run("[1, 2, []] ** 3 . print", "[1, 2, [], 1, 2, [], 1, 2, []]\n") }
#[test] fn list_power_left_is_deep_copy() { run("let a = [[1]] ** 3 ; a[0].push(2) ; print(a)", "[[1, 2], [1], [1]]\n") }
#[test] fn list_power_right() { run("3 ** [1, 2, []] . print", "[1, 2, [], 1, 2, [], 1, 2, []]\n") }
#[test] fn list_power_right_is_deep_copy() { run("let a = 3 ** [[1]] ; a[0].push(2) ; print(a)", "[[1, 2], [1], [1]]\n") }
#[test] fn list_operator_in_yes() { run("13 in [10, 11, 12, 13, 14, 15] . print", "true\n") }
#[test] fn list_operator_in_no() { run("3 in [10, 11, 12, 13, 14, 15] . print", "false\n") }
#[test] fn list_operator_not_in_yes() { run("3 not in [1, 2, 3] . print", "false\n") }
#[test] fn list_operator_not_in_no() { run("3 not in [1, 5, 8] . print", "true\n") }
#[test] fn list_index() { run("[1, 2, 3] [1] . print", "2\n") }
#[test] fn list_index_out_of_bounds() { run("[1, 2, 3] [3] . print", "Index '3' is out of bounds for list of length [0, 3)\n  at: line 1 (<test>)\n\n1 | [1, 2, 3] [3] . print\n2 |           ^^^\n") }
#[test] fn list_index_negative() { run("[1, 2, 3] [-1] . print", "3\n") }
#[test] fn list_slice_01() { run("[1, 2, 3, 4] [:] . print", "[1, 2, 3, 4]\n") }
#[test] fn list_slice_02() { run("[1, 2, 3, 4] [::] . print", "[1, 2, 3, 4]\n") }
#[test] fn list_slice_03() { run("[1, 2, 3, 4] [::1] . print", "[1, 2, 3, 4]\n") }
#[test] fn list_slice_04() { run("[1, 2, 3, 4] [1:] . print", "[2, 3, 4]\n") }
#[test] fn list_slice_05() { run("[1, 2, 3, 4] [:2] . print", "[1, 2]\n") }
#[test] fn list_slice_06() { run("[1, 2, 3, 4] [0:] . print", "[1, 2, 3, 4]\n") }
#[test] fn list_slice_07() { run("[1, 2, 3, 4] [:4] . print", "[1, 2, 3, 4]\n") }
#[test] fn list_slice_08() { run("[1, 2, 3, 4] [1:3] . print", "[2, 3]\n") }
#[test] fn list_slice_09() { run("[1, 2, 3, 4] [2:4] . print", "[3, 4]\n") }
#[test] fn list_slice_10() { run("[1, 2, 3, 4] [0:2] . print", "[1, 2]\n") }
#[test] fn list_slice_11() { run("[1, 2, 3, 4] [:-1] . print", "[1, 2, 3]\n") }
#[test] fn list_slice_12() { run("[1, 2, 3, 4] [:-2] . print", "[1, 2]\n") }
#[test] fn list_slice_13() { run("[1, 2, 3, 4] [-2:] . print", "[3, 4]\n") }
#[test] fn list_slice_14() { run("[1, 2, 3, 4] [-3:] . print", "[2, 3, 4]\n") }
#[test] fn list_slice_15() { run("[1, 2, 3, 4] [::2] . print", "[1, 3]\n") }
#[test] fn list_slice_16() { run("[1, 2, 3, 4] [::3] . print", "[1, 4]\n") }
#[test] fn list_slice_17() { run("[1, 2, 3, 4] [::4] . print", "[1]\n") }
#[test] fn list_slice_18() { run("[1, 2, 3, 4] [1::2] . print", "[2, 4]\n") }
#[test] fn list_slice_19() { run("[1, 2, 3, 4] [1:3:2] . print", "[2]\n") }
#[test] fn list_slice_20() { run("[1, 2, 3, 4] [:-1:2] . print", "[1, 3]\n") }
#[test] fn list_slice_21() { run("[1, 2, 3, 4] [1:-1:3] . print", "[2]\n") }
#[test] fn list_slice_22() { run("[1, 2, 3, 4] [::-1] . print", "[4, 3, 2, 1]\n") }
#[test] fn list_slice_23() { run("[1, 2, 3, 4] [1::-1] . print", "[2, 1]\n") }
#[test] fn list_slice_24() { run("[1, 2, 3, 4] [:2:-1] . print", "[4]\n") }
#[test] fn list_slice_25() { run("[1, 2, 3, 4] [3:1:-1] . print", "[4, 3]\n") }
#[test] fn list_slice_26() { run("[1, 2, 3, 4] [-1:-2:-1] . print", "[4]\n") }
#[test] fn list_slice_27() { run("[1, 2, 3, 4] [-2::-1] . print", "[3, 2, 1]\n") }
#[test] fn list_slice_28() { run("[1, 2, 3, 4] [:-3:-1] . print", "[4, 3]\n") }
#[test] fn list_slice_29() { run("[1, 2, 3, 4] [::-2] . print", "[4, 2]\n") }
#[test] fn list_slice_30() { run("[1, 2, 3, 4] [::-3] . print", "[4, 1]\n") }
#[test] fn list_slice_31() { run("[1, 2, 3, 4] [::-4] . print", "[4]\n") }
#[test] fn list_slice_32() { run("[1, 2, 3, 4] [-2::-2] . print", "[3, 1]\n") }
#[test] fn list_slice_33() { run("[1, 2, 3, 4] [-3::-2] . print", "[2]\n") }
#[test] fn list_slice_34() { run("[1, 2, 3, 4] [1:1] . print", "[]\n") }
#[test] fn list_slice_35() { run("[1, 2, 3, 4] [-1:-1] . print", "[]\n") }
#[test] fn list_slice_36() { run("[1, 2, 3, 4] [-1:1:] . print", "[]\n") }
#[test] fn list_slice_37() { run("[1, 2, 3, 4] [1:1:-1] . print", "[]\n") }
#[test] fn list_slice_38() { run("[1, 2, 3, 4] [-2:2:-3] . print", "[]\n") }
#[test] fn list_slice_39() { run("[1, 2, 3, 4] [-1:1:-1] . print", "[4, 3]\n") }
#[test] fn list_slice_40() { run("[1, 2, 3, 4] [1:-1:-1] . print", "[]\n") }
#[test] fn list_slice_41() { run("[1, 2, 3, 4] [1:10:1] . print", "[2, 3, 4]\n") }
#[test] fn list_slice_42() { run("[1, 2, 3, 4] [10:1:-1] . print", "[4, 3]\n") }
#[test] fn list_slice_43() { run("[1, 2, 3, 4] [-10:1] . print", "[1]\n") }
#[test] fn list_slice_44() { run("[1, 2, 3, 4] [1:-10:-1] . print", "[2, 1]\n") }
#[test] fn list_slice_45() { run("[1, 2, 3, 4] [::0]", "ValueError: 'step' argument cannot be zero\n  at: line 1 (<test>)\n\n1 | [1, 2, 3, 4] [::0]\n2 |              ^^^^^\n") }
#[test] fn list_slice_46() { run("[1, 2, 3, 4][:-1] . print", "[1, 2, 3]\n") }
#[test] fn list_slice_47() { run("[1, 2, 3, 4][:0] . print", "[]\n") }
#[test] fn list_slice_48() { run("[1, 2, 3, 4][:1] . print", "[1]\n") }
#[test] fn list_slice_49() { run("[1, 2, 3, 4][5:] . print", "[]\n") }
#[test] fn list_pop_empty() { run("let x = [] , y = x . pop ; (x, y) . print", "ValueError: Expected 'pop' argument to be a non-empty iterable\n  at: line 1 (<test>)\n\n1 | let x = [] , y = x . pop ; (x, y) . print\n2 |                    ^^^^^\n") }
#[test] fn list_pop() { run("let x = [1, 2, 3] , y = x . pop ; (x, y) . print", "([1, 2], 3)\n") }
#[test] fn list_pop_front_empty() { run("let x = [], y = x . pop_front ; (x, y) . print", "ValueError: Expected 'pop_front' argument to be a non-empty iterable\n  at: line 1 (<test>)\n\n1 | let x = [], y = x . pop_front ; (x, y) . print\n2 |                   ^^^^^^^^^^^\n") }
#[test] fn list_pop_front() { run("let x = [1, 2, 3], y = x . pop_front ; (x, y) . print", "([2, 3], 1)\n") }
#[test] fn list_push() { run("let x = [1, 2, 3] ; x . push(4) ; x . print", "[1, 2, 3, 4]\n") }
#[test] fn list_push_front() { run("let x = [1, 2, 3] ; x . push_front(4) ; x . print", "[4, 1, 2, 3]\n") }
#[test] fn list_insert_front() { run("let x = [1, 2, 3] ; x . insert(0, 4) ; x . print", "[4, 1, 2, 3]\n") }
#[test] fn list_insert_middle() { run("let x = [1, 2, 3] ; x . insert(1, 4) ; x . print", "[1, 4, 2, 3]\n") }
#[test] fn list_insert_end() { run("let x = [1, 2, 3] ; x . insert(2, 4) ; x . print", "[1, 2, 4, 3]\n") }
#[test] fn list_insert_out_of_bounds() { run("let x = [1, 2, 3] ; x . insert(4, 4) ; x . print", "Index '4' is out of bounds for list of length [0, 3)\n  at: line 1 (<test>)\n\n1 | let x = [1, 2, 3] ; x . insert(4, 4) ; x . print\n2 |                       ^^^^^^^^^^^^^^\n") }
#[test] fn list_remove_front() { run("let x = [1, 2, 3] , y = x . remove(0) ; (x, y) . print", "([2, 3], 1)\n") }
#[test] fn list_remove_middle() { run("let x = [1, 2, 3] , y = x . remove(1) ; (x, y) . print", "([1, 3], 2)\n") }
#[test] fn list_remove_end() { run("let x = [1, 2, 3] , y = x . remove(2) ; (x, y) . print", "([1, 2], 3)\n") }
#[test] fn list_clear() { run("let x = [1, 2, 3] ; x . clear ; x . print", "[]\n") }
#[test] fn list_peek() { run("let x = [1, 2, 3], y = x . peek ; (x, y) . print", "([1, 2, 3], 1)\n") }
#[test] fn list_str() { run("[1, 2, '3'] . print", "[1, 2, '3']\n") }
#[test] fn list_repr() { run("['1', 2, '3'] . repr . print", "['1', 2, '3']\n") }
#[test] fn list_recursive_repr() { run("let x = [] ; x.push(x) ; x.print", "[[...]]\n") }
#[test] fn list_recursive_knot_repr() { run("let x = [] ; let y = [x] ; x.push(y) ; x.print", "[[[...]]]\n") }
#[test] fn list_recursive_complex_repr() { run("struct S(x) ; let x = [S(nil)] ; x[0]->x = [S(x)] ; x.print", "[S(x=[S(x=[...])])]\n") }
#[test] fn vector_empty_constructor() { run("vector() . print", "()\n") }
#[test] fn vector_empty_iterable_constructor() { run("vector([]) . print", "()\n") }
#[test] fn vector_iterable_constructor() { run("vector([1, 2, 3]) . print", "(1, 2, 3)\n") }
#[test] fn vector_multiple_constructor() { run("vector(1, 2, 3) . print", "(1, 2, 3)\n") }
#[test] fn vector_literal_single() { run("(1,) . print", "(1)\n") }
#[test] fn vector_literal_multiple() { run("(1,2,3) . print", "(1, 2, 3)\n") }
#[test] fn vector_literal_multiple_trailing_comma() { run("(1,2,3,) . print", "(1, 2, 3)\n") }
#[test] fn vector_literal_unroll_at_start() { run("(...(1, 2, 3), 4, 5) . print", "(1, 2, 3, 4, 5)\n") }
#[test] fn vector_literal_unroll_at_end() { run("(0, ...(1, 2, 3)) . print", "(0, 1, 2, 3)\n") }
#[test] fn vector_literal_unroll_once() { run("(...(1, 2, 3)) . print", "(1, 2, 3)\n") }
#[test] fn vector_literal_unroll_multiple() { run("(...(1, 2, 3), ...(4, 5)) . print", "(1, 2, 3, 4, 5)\n") }
#[test] fn vector_literal_unroll_multiple_and_empty() { run("(...vector(), 0, ...(1, 2, 3), ...(4, 5), ...vector(), 6) . print", "(0, 1, 2, 3, 4, 5, 6)\n") }
#[test] fn vector() { run("vector(1, 2, 3) . print", "(1, 2, 3)\n") }
#[test] fn vector_add() { run("vector(1, 2, 3) + vector(6, 3, 2) . print", "(7, 5, 5)\n") }
#[test] fn vector_add_constant() { run("vector(1, 2, 3) + 3 . print", "(4, 5, 6)\n") }
#[test] fn set_empty_constructor() { run("set() . print", "{}\n") }
#[test] fn vector_array_assign() { run("let x = (1, 2, 3) ; x[0] = 3 ; x . print", "(3, 2, 3)\n") }
#[test] fn vector_recursive_repr() { run("let x = (nil,) ; x[0] = x ; x.print", "((...))\n") }
#[test] fn set_literal_empty() { run("{} is set . print ; {} . print", "true\n{}\n") }
#[test] fn set_literal_single() { run("{'hello'} . print", "{'hello'}\n") }
#[test] fn set_literal_multiple() { run("{1, 2, 3, 4} . print", "{1, 2, 3, 4}\n") }
#[test] fn set_literal_unroll_at_start() { run("{...{1, 2, 3}, 4, 5} . print", "{1, 2, 3, 4, 5}\n") }
#[test] fn set_literal_unroll_at_end() { run("{0, ...{1, 2, 3}} . print", "{0, 1, 2, 3}\n") }
#[test] fn set_literal_unroll_once() { run("{...{1, 2, 3}} . print", "{1, 2, 3}\n") }
#[test] fn set_literal_unroll_multiple() { run("{...{1, 2, 3}, ...{4, 5}} . print", "{1, 2, 3, 4, 5}\n") }
#[test] fn set_literal_unroll_multiple_and_empty() { run("{...{}, 0, ...{1, 2, 3}, ...{4, 5}, ...set(), 6} . print", "{0, 1, 2, 3, 4, 5, 6}\n") }
#[test] fn set_literal_unroll_from_dict_implicit() { run("{...{(1, 1), (2, 2)}} . print", "{(1, 1), (2, 2)}\n") }
#[test] fn set_literal_unroll_from_dict_explicit() { run("{...{(1, 1), (2, 2)}, 3} . print", "{(1, 1), (2, 2), 3}\n") }
#[test] fn set_from_str() { run("'funny beans' . set . print", "{'f', 'u', 'n', 'y', ' ', 'b', 'e', 'a', 's'}\n") }
#[test] fn set_pop_empty() { run("let x = set() , y = x . pop ; (x, y) . print", "ValueError: Expected 'pop' argument to be a non-empty iterable\n  at: line 1 (<test>)\n\n1 | let x = set() , y = x . pop ; (x, y) . print\n2 |                       ^^^^^\n") }
#[test] fn set_pop() { run("let x = {1, 2, 3} , y = x . pop ; (x, y) . print", "({1, 2}, 3)\n") }
#[test] fn set_push() { run("let x = {1, 2, 3} ; x . push(4) ; x . print", "{1, 2, 3, 4}\n") }
#[test] fn set_remove_yes() { run("let x = {1, 2, 3}, y = x . remove(2) ; (x, y) . print", "({1, 3}, true)\n") }
#[test] fn set_remove_no() { run("let x = {1, 2, 3}, y = x . remove(5) ; (x, y) . print", "({1, 2, 3}, false)\n") }
#[test] fn set_clear() { run("let x = {1, 2, 3} ; x . clear ; x . print", "{}\n") }
#[test] fn set_peek() { run("let x = {1, 2, 3}, y = x . peek ; (x, y) . print", "({1, 2, 3}, 1)\n") }
#[test] fn set_insert_self() { run("let x = set() ; x.push(x)", "ValueError: Cannot create recursive hash-based collection\n  at: line 1 (<test>)\n\n1 | let x = set() ; x.push(x)\n2 |                  ^^^^^^^^\n") }
#[test] fn set_indirect_insert_self() { run("let x = set() ; x.push([x])", "ValueError: Cannot create recursive hash-based collection\n  at: line 1 (<test>)\n\n1 | let x = set() ; x.push([x])\n2 |                  ^^^^^^^^^^\n") }
#[test] fn set_recursive_repr() { run("let x = set() ; x.push(x) ; x.print", "ValueError: Cannot create recursive hash-based collection\n  at: line 1 (<test>)\n\n1 | let x = set() ; x.push(x) ; x.print\n2 |                  ^^^^^^^^\n") }
#[test] fn set_union() { run("{1, 2, 3} . union({5, 6, 7}) . print", "{1, 2, 3, 5, 6, 7}\n") }
#[test] fn set_union_with_list() { run("{1, 2, 3} . union([5, 6, 7]) . print", "{1, 2, 3, 5, 6, 7}\n") }
#[test] fn set_union_mutates_self() { run("let x = {1, 2, 3} ; x . union([5, 6, 7]) ; x . print", "{1, 2, 3, 5, 6, 7}\n") }
#[test] fn set_intersect() { run("{1, 2, 3} . intersect({2, 3, 4, 5}) . print", "{2, 3}\n") }
#[test] fn set_intersect_with_list() { run("{1, 2, 3} . intersect([2, 3, 4, 5]) . print", "{2, 3}\n") }
#[test] fn set_intersect_mutates_self() { run("let x = {1, 2, 3} ; x . intersect([2, 3, 4, 5]) ; x . print", "{2, 3}\n") }
#[test] fn set_difference() { run("{1, 2, 3, 4, 5} . difference({4, 5, 6}) . print", "{1, 2, 3}\n") }
#[test] fn set_difference_with_list() { run("{1, 2, 3, 4, 5} . difference([4, 5, 6]) . print", "{1, 2, 3}\n") }
#[test] fn set_difference_mutates_self() { run("let x = {1, 2, 3, 4, 5} ; x . difference([4, 5, 6]) ; x . print", "{1, 2, 3}\n") }
#[test] fn dict_empty_constructor() { run("dict() . print", "{}\n") }
#[test] fn dict_literal_single() { run("{'hello': 'world'} . print", "{'hello': 'world'}\n") }
#[test] fn dict_literal_multiple() { run("{1: 'a', 2: 'b', 3: 'c'} . print", "{1: 'a', 2: 'b', 3: 'c'}\n") }
#[test] fn dict_literal_unroll_at_start() { run("{...{1: 1, 2: 2}, 3: 3} . print", "{1: 1, 2: 2, 3: 3}\n") }
#[test] fn dict_literal_unroll_at_end() { run("{0: 0, ...{1: 1, 2: 2}} . print", "{0: 0, 1: 1, 2: 2}\n") }
#[test] fn dict_literal_unroll_multiple() { run("{...{1: 1, 2: 2}, 3: 3, ...{4: 4}} . print", "{1: 1, 2: 2, 3: 3, 4: 4}\n") }
#[test] fn dict_literal_unroll_multiple_and_empty() { run("{...{}, 0: 0, ...{1: 1, 2: 2, 3: 3}, ...{4: 4, 5: 5}, ...set(), ...dict(), 6: 6} . print", "{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}\n") }
#[test] fn dict_literal_unroll_from_set() { run("{...{(1, 1), (2, 2)}, 3: 3} . print", "{1: 1, 2: 2, 3: 3}\n") }
#[test] fn dict_literal_unroll_from_not_pair() { run("{...{1, 2, 3}, 4: 4}", "ValueError: Cannot collect key-value pair '1' of type 'int' into a dict\n  at: line 1 (<test>)\n\n1 | {...{1, 2, 3}, 4: 4}\n2 |  ^^^\n") }
#[test] fn dict_get_and_set() { run("let d = dict() ; d['hi'] = 'yes' ; d['hi'] . print", "yes\n") }
#[test] fn dict_get_when_not_present() { run("let d = dict() ; d['hello']", "ValueError: Key 'hello' of type 'str' not found in dictionary\n  at: line 1 (<test>)\n\n1 | let d = dict() ; d['hello']\n2 |                   ^^^^^^^^^\n") }
#[test] fn dict_get_when_not_present_with_default() { run("let d = dict() . default('haha') ; d['hello'] . print", "haha\n") }
#[test] fn dict_keys() { run("[[1, 'a'], [2, 'b'], [3, 'c']] . dict . keys . print", "{1, 2, 3}\n") }
#[test] fn dict_values() { run("[[1, 'a'], [2, 'b'], [3, 'c']] . dict . values . print", "['a', 'b', 'c']\n") }
#[test] fn dict_pop_empty() { run("let x = dict() , y = x . pop ; (x, y) . print", "ValueError: Expected 'pop' argument to be a non-empty iterable\n  at: line 1 (<test>)\n\n1 | let x = dict() , y = x . pop ; (x, y) . print\n2 |                        ^^^^^\n") }
#[test] fn dict_pop() { run("let x = {1: 'a', 2: 'b', 3: 'c'} , y = x . pop ; (x, y) . print", "({1: 'a', 2: 'b'}, (3, 'c'))\n") }
#[test] fn dict_insert() { run("let x = {1: 'a', 2: 'b', 3: 'c'} ; x . insert(4, 'd') ; x . print", "{1: 'a', 2: 'b', 3: 'c', 4: 'd'}\n") }
#[test] fn dict_remove_yes() { run("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . remove(2) ; (x, y) . print", "({1: 'a', 3: 'c'}, true)\n") }
#[test] fn dict_remove_no() { run("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . remove(5) ; (x, y) . print", "({1: 'a', 2: 'b', 3: 'c'}, false)\n") }
#[test] fn dict_clear() { run("let x = {1: 'a', 2: 'b', 3: 'c'} ; x . clear ; x . print", "{}\n") }
#[test] fn dict_from_enumerate() { run("'hey' . enumerate . dict . print", "{0: 'h', 1: 'e', 2: 'y'}\n") }
#[test] fn dict_peek() { run("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . peek ; (x, y) . print", "({1: 'a', 2: 'b', 3: 'c'}, (1, 'a'))\n") }
#[test] fn dict_default_with_query() { run("let d = dict() . default(3) ; d[0] ; d.print", "{0: 3}\n") }
#[test] fn dict_default_with_function() { run("let d = dict() . default(list) ; d[0].push(2) ; d[1].push(3) ; d.print", "{0: [2], 1: [3]}\n") }
#[test] fn dict_default_with_mutable_default() { run("let d = dict() . default([]) ; d[0].push(2) ; d[1].push(3) ; d.print", "{0: [2, 3], 1: [2, 3]}\n") }
#[test] fn dict_default_with_self_entry() { run("let d ; d = dict() . default(fn() { d['count'] += 1 ; d['hello'] = 'special' ; 'otherwise' }) ; d['count'] = 0 ; d['hello'] ; d['world'] ; d.print", "{'count': 2, 'hello': 'special', 'world': 'otherwise'}\n") }
#[test] fn dict_increment() { run("let d = dict() . default(fn() -> 3) ; d[0] . print ; d[0] += 1 ; d . print ; d[0] += 1 ; d . print", "3\n{0: 4}\n{0: 5}\n") }
#[test] fn dict_insert_self_as_key() { run("let x = dict() ; x[x] = 'yes'", "ValueError: Cannot create recursive hash-based collection\n  at: line 1 (<test>)\n\n1 | let x = dict() ; x[x] = 'yes'\n2 |                       ^\n") }
#[test] fn dict_insert_self_as_value() { run("let x = dict() ; x['yes'] = x", "") }
#[test] fn dict_recursive_key_index() { run("let x = dict() ; x[x] = 'yes' ; x.print", "ValueError: Cannot create recursive hash-based collection\n  at: line 1 (<test>)\n\n1 | let x = dict() ; x[x] = 'yes' ; x.print\n2 |                       ^\n") }
#[test] fn dict_recursive_key_insert() { run("let x = dict() ; x.insert(x, 'yes') ; x.print", "ValueError: Cannot create recursive hash-based collection\n  at: line 1 (<test>)\n\n1 | let x = dict() ; x.insert(x, 'yes') ; x.print\n2 |                   ^^^^^^^^^^^^^^^^^\n") }
#[test] fn dict_recursive_value_repr() { run("let x = dict() ; x['yes'] = x ; x.print", "{'yes': {...}}\n") }
#[test] fn heap_empty_constructor() { run("heap() . print", "[]\n") }
#[test] fn heap_from_list() { run("let h = [1, 7, 3, 2, 7, 6] . heap; h . print", "[1, 2, 3, 7, 7, 6]\n") }
#[test] fn heap_pop() { run("let h = [1, 7, 3, 2, 7, 6] . heap; [h.pop, h.pop, h.pop] . print", "[1, 2, 3]\n") }
#[test] fn heap_push() { run("let h = [1, 7, 3, 2, 7, 6] . heap; h.push(3); h.push(-1); h.push(16); h . print", "[-1, 1, 3, 2, 7, 6, 3, 7, 16]\n") }
#[test] fn heap_recursive_repr() { run("let x = heap() ; x.push(x) ; x.print", "[[...]]\n") }
#[test] fn print_hello_world() { run("print('hello world!')", "hello world!\n") }
#[test] fn print_empty() { run("print()", "\n") }
#[test] fn print_strings() { run("print('first', 'second', 'third')", "first second third\n") }
#[test] fn print_other_things() { run("print(nil, -1, 1, true, false, 'test', print)", "nil -1 1 true false test print\n") }
#[test] fn print_unary_operators() { run("print(-1, --1, ---1, !3, !!3, !true, !!true)", "-1 1 -1 -4 3 false true\n") }
#[test] fn exit_in_expression() { run("'this will not print' + exit . print", "") }
#[test] fn exit_in_ternary() { run("print(if 3 > 2 then exit else 'hello')", "") }
#[test] fn assert_pass() { run("assert [1, 2] . len . (==2) ; print('yes!')", "yes!\n")}
#[test] fn assert_pass_with_no_message() { run("assert [1, 2] .len . (==2) : print('should not show') ; print('should show')", "should show\n") }
#[test] fn assert_fail_with_test() { run("assert false", "Assertion Failed:\n  at: line 1 (<test>)\n\n1 | assert false\n2 |        ^^^^^\n") }
#[test] fn assert_fail_with_test_and_message() { run("assert false : 'oh nose'", "Assertion Failed: oh nose\n  at: line 1 (<test>)\n\n1 | assert false : 'oh nose'\n2 |        ^^^^^\n") }
#[test] fn assert_fail_with_compare() { run("assert 1 + 2 != 3", "Assertion Failed: Expected 3 != 3\n  at: line 1 (<test>)\n\n1 | assert 1 + 2 != 3\n2 |        ^^^^^^^^^^\n") }
#[test] fn assert_fail_with_compare_and_message() { run("assert 2 * 4 > 3 * 3 : ['did', 'we', 'do', 'math', 'wrong?'] . join ' '", "Assertion Failed: did we do math wrong?\nExpected 8 > 9\n  at: line 1 (<test>)\n\n1 | assert 2 * 4 > 3 * 3 : ['did', 'we', 'do', 'math', 'wrong?'] . join ' '\n2 |        ^^^^^^^^^^^^^\n") }
#[test] fn assert_fail_with_unusual_compare() { run("assert 'here' in 'the goose is gone'", "Assertion Failed:\n  at: line 1 (<test>)\n\n1 | assert 'here' in 'the goose is gone'\n2 |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n") }
#[test] fn assert_fail_with_unusual_compare_and_message() { run("assert 'here' in 'the goose is gone' : 'goose issues are afoot'", "Assertion Failed: goose issues are afoot\n  at: line 1 (<test>)\n\n1 | assert 'here' in 'the goose is gone' : 'goose issues are afoot'\n2 |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n") }
#[test] fn assert_messages_are_lazy() { run("assert true : exit ; print('should reach here')", "should reach here\n") }
#[test] fn monitor_stack() { run("let x = 1, b = [], c ; monitor 'stack' . print", "[1, [], nil, fn print(...), fn monitor(cmd)]\n") }
#[test] fn monitor_stack_modification() { run("let x = [false] ; ( monitor 'stack' )[0][0] = true ; x . print", "[true]\n") }
#[test] fn monitor_call_stack() { run("fn foo() { bar() } fn bar() { monitor 'call-stack' . print } foo()", "[(0, 0), (3, 6), (4, 10)]\n") }
#[test] fn monitor_code() { run("monitor 'code' . print", "['Print', 'Monitor', 'Str('code')', 'Call(1)', 'Call(1)', 'Pop', 'Exit']\n") }
#[test] fn monitor_error() { run("monitor 'foobar'", "MonitorError: Illegal monitor command 'foobar'\n  at: line 1 (<test>)\n\n1 | monitor 'foobar'\n2 |         ^^^^^^^^\n") }
#[test] fn len_list() { run("[1, 2, 3] . len . print", "3\n") }
#[test] fn len_str() { run("'12345' . len . print", "5\n") }
#[test] fn sum_list() { run("[1, 2, 3, 4] . sum . print", "10\n") }
#[test] fn sum_values() { run("sum(1, 3, 5, 7) . print", "16\n") }
#[test] fn sum_no_arg() { run("sum()", "Incorrect number of arguments for fn sum(...), got 0\n  at: line 1 (<test>)\n\n1 | sum()\n2 |    ^^\n") }
#[test] fn sum_empty_list() { run("[] . sum . print", "0\n") }
#[test] fn abs_bool() { run("[true, false] . map abs . print", "[1, 0]\n") }
#[test] fn abs_int() { run("[int.max, int.min, 0, -1, 1, 25, -32] . map abs . print", "[4611686018427387903, -4611686018427387904, 0, 1, 1, 25, 32]\n") }
#[test] fn abs_complex() { run("[0, 1i, -2i, 3 + 4i, -5 + 6i, -7 - 8i, 9 - 10i] . map abs . print", "[0, 1i, 2i, 3 + 4i, 5 + 6i, 7 + 8i, 9 + 10i]\n") }
#[test] fn abs_vector_1() { run("(1, -2, 0, -12, 10) . abs . print", "(1, 2, 0, 12, 10)\n") }
#[test] fn abs_vector_2() { run("[(0, 0), (-1, -1), (2, -2), (-3, 3), (4, 4)] . map abs . print", "[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]\n") }
#[test] fn map() { run("[1, 2, 3] . map(str) . repr . print", "['1', '2', '3']\n") }
#[test] fn map_lambda() { run("[-1, 2, -3] . map(fn(x) -> x . abs) . print", "[1, 2, 3]\n") }
#[test] fn filter() { run("[2, 3, 4, 5, 6] . filter (>3) . print", "[4, 5, 6]\n") }
#[test] fn filter_lambda() { run("[2, 3, 4, 5, 6] . filter (fn(x) -> x % 2 == 0) . print", "[2, 4, 6]\n") }
#[test] fn reduce_with_operator() { run("[1, 2, 3, 4, 5, 6] . reduce (*) . print", "720\n") }
#[test] fn reduce_with_function() { run("[1, 2, 3, 4, 5, 6] . reduce (fn(a, b) -> a * b) . print", "720\n") }
#[test] fn reduce_with_unary_operator() { run("[1, 2, 3] . reduce (!) . print", "Incorrect number of arguments for fn (!)(x), got 2\n  at: line 1 (<test>)\n\n1 | [1, 2, 3] . reduce (!) . print\n2 |           ^^^^^^^^^^^^\n") }
#[test] fn reduce_with_sum() { run("[1, 2, 3, 4, 5, 6] . reduce (sum) . print", "21\n") }
#[test] fn reduce_with_empty() { run("[] . reduce(+) . print", "ValueError: Expected 'reduce' argument to be a non-empty iterable\n  at: line 1 (<test>)\n\n1 | [] . reduce(+) . print\n2 |    ^^^^^^^^^^^\n") }
#[test] fn sorted() { run("[6, 2, 3, 7, 2, 1] . sort . print", "[1, 2, 2, 3, 6, 7]\n") }
#[test] fn sorted_with_set_of_str() { run("'funny' . set . sort . print", "['f', 'n', 'u', 'y']\n") }
#[test] fn sorted_with_int_nil_and_bool() { run("[true, 0, nil, -2, false, 0, 2, true, 1, false, -1, nil] . sort . print", "[-2, -1, nil, nil, false, false, 0, 0, true, true, 1, 2]\n") }
#[test] fn sorted_with_long_and_short_str() { run("range(14) . map(fn(f) -> 'abcdefghijklmnop'[:1+f]) . reverse . sort . print", "['a', 'ab', 'abc', 'abcd', 'abcde', 'abcdef', 'abcdefg', 'abcdefgh', 'abcdefghi', 'abcdefghij', 'abcdefghijk', 'abcdefghijkl', 'abcdefghijklm', 'abcdefghijklmn']\n") }
#[test] fn group_by_int_negative() { run("group_by(-1, [1, 2, 3, 4]) . print", "ValueError: Expected value '-1: int' to be positive\n  at: line 1 (<test>)\n\n1 | group_by(-1, [1, 2, 3, 4]) . print\n2 |         ^^^^^^^^^^^^^^^^^^\n") }
#[test] fn group_by_int_zero() { run("group_by(0, [1, 2, 3, 4]) . print", "ValueError: Expected value '0: int' to be positive\n  at: line 1 (<test>)\n\n1 | group_by(0, [1, 2, 3, 4]) . print\n2 |         ^^^^^^^^^^^^^^^^^\n") }
#[test] fn group_by_int_by_one() { run("group_by(1, [1, 2, 3, 4]) . print", "[(1), (2), (3), (4)]\n") }
#[test] fn group_by_int_by_three() { run("[1, 2, 3, 4, 5, 6] . group_by(3) . print", "[(1, 2, 3), (4, 5, 6)]\n") }
#[test] fn group_by_int_by_one_empty_iterable() { run("[] . group_by(1) . print", "[]\n") }
#[test] fn group_by_int_by_three_empty_iterable() { run("[] . group_by(3) . print", "[]\n") }
#[test] fn group_by_int_by_three_with_remainder() { run("[1, 2, 3, 4] . group_by(3) . print", "[(1, 2, 3), (4)]\n") }
#[test] fn group_by_int_by_three_not_enough() { run("[1, 2] . group_by(3) . print", "[(1, 2)]\n") }
#[test] fn group_by_function_empty_iterable() { run("[] . group_by(fn(x) -> nil) . print", "{}\n") }
#[test] fn group_by_function_all_same_keys() { run("[1, 2, 3, 4] . group_by(fn(x) -> nil) . print", "{nil: (1, 2, 3, 4)}\n") }
#[test] fn group_by_function_all_different_keys() { run("[1, 2, 3, 4] . group_by(fn(x) -> x) . print", "{1: (1), 2: (2), 3: (3), 4: (4)}\n") }
#[test] fn group_by_function_remainder_by_three() { run("[1, 2, 3, 4, 5] . group_by(%3) . print", "{1: (1, 4), 2: (2, 5), 0: (3)}\n") }
#[test] fn reverse() { run("[8, 1, 2, 6, 3, 2, 3] . reverse . print", "[3, 2, 3, 6, 2, 1, 8]\n") }
#[test] fn range_1() { run("range(3) . list . print", "[0, 1, 2]\n") }
#[test] fn range_2() { run("range(3, 7) . list . print", "[3, 4, 5, 6]\n") }
#[test] fn range_3() { run("range(1, 9, 3) . list . print", "[1, 4, 7]\n") }
#[test] fn range_4() { run("range(6, 3) . list . print", "[]\n") }
#[test] fn range_5() { run("range(10, 4, -2) . list . print", "[10, 8, 6]\n") }
#[test] fn range_6() { run("range(0, 20, -1) . list . print", "[]\n") }
#[test] fn range_7() { run("range(10, 0, 3) . list . print", "[]\n") }
#[test] fn range_8() { run("range(1, 1, 1) . list . print", "[]\n") }
#[test] fn range_9() { run("range(1, 1, 0) . list . print", "ValueError: 'step' argument cannot be zero\n  at: line 1 (<test>)\n\n1 | range(1, 1, 0) . list . print\n2 |      ^^^^^^^^^\n") }
#[test] fn range_operator_in_yes() { run("13 in range(10, 15) . print", "true\n") }
#[test] fn range_operator_in_no() { run("3 in range(10, 15) . print", "false\n") }
#[test] fn range_used_twice() { run("let r = range(4) ; for a in r { a.print } for b in r { b.print }", "0\n1\n2\n3\n0\n1\n2\n3\n") }
#[test] fn enumerate_1() { run("[] . enumerate . list . print", "[]\n") }
#[test] fn enumerate_2() { run("[1, 2, 3] . enumerate . list . print", "[(0, 1), (1, 2), (2, 3)]\n") }
#[test] fn enumerate_3() { run("'foobar' . enumerate . list . print", "[(0, 'f'), (1, 'o'), (2, 'o'), (3, 'b'), (4, 'a'), (5, 'r')]\n") }
#[test] fn sqrt() { run("[0, 1, 4, 9, 25, 3, 6, 8, 13] . map(sqrt) . print", "[0, 1, 2, 3, 5, 1, 2, 2, 3]\n") }
#[test] fn sqrt_very_large() { run("[1 << 61, (1 << 61) + 1, (1 << 61) - 1] . map(sqrt) . print", "[1518500249, 1518500249, 1518500249]\n") }
#[test] fn gcd() { run("gcd(12, 8) . print", "4\n") }
#[test] fn gcd_iter() { run("[12, 18, 16] . gcd . print", "2\n") }
#[test] fn lcm() { run("lcm(9, 7) . print", "63\n") }
#[test] fn lcm_iter() { run("[12, 10, 18] . lcm . print", "180\n") }
#[test] fn flat_map_identity() { run("['hi', 'bob'] . flat_map(fn(i) -> i) . print", "['h', 'i', 'b', 'o', 'b']\n") }
#[test] fn flat_map_with_func() { run("['hello', 'bob'] . flat_map(fn(i) -> i[2:]) . print", "['l', 'l', 'o', 'b']\n") }
#[test] fn concat() { run("[[], [1], [2, 3], [4, 5, 6], [7, 8, 9, 0]] . concat . print", "[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]\n") }
#[test] fn zip() { run("zip([1, 2, 3, 4, 5], 'hello') . print", "[(1, 'h'), (2, 'e'), (3, 'l'), (4, 'l'), (5, 'o')]\n") }
#[test] fn zip_with_empty() { run("zip('hello', []) . print", "[]\n") }
#[test] fn zip_with_longer_last() { run("zip('hi', 'hello', 'hello the world!') . print", "[('h', 'h', 'h'), ('i', 'e', 'e')]\n") }
#[test] fn zip_with_longer_first() { run("zip('hello the world!', 'hello', 'hi') . print", "[('h', 'h', 'h'), ('e', 'e', 'i')]\n") }
#[test] fn zip_of_list() { run("[[1, 2, 3], [4, 5, 6], [7, 8, 9]] . zip . print", "[(1, 4, 7), (2, 5, 8), (3, 6, 9)]\n") }
#[test] fn permutations_empty() { run("[] . permutations(3) . print", "[]\n") }
#[test] fn permutations_n_larger_than_size() { run("[1, 2, 3] . permutations(5) . print", "[]\n") }
#[test] fn permutations() { run("[1, 2, 3] . permutations(2) . print", "[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]\n") }
#[test] fn combinations_empty() { run("[] . combinations(3) . print", "[]\n") }
#[test] fn combinations_n_larger_than_size() { run("[1, 2, 3] . combinations(5) . print", "[]\n") }
#[test] fn combinations() { run("[1, 2, 3] . combinations(2) . print", "[(1, 2), (1, 3), (2, 3)]\n") }
#[test] fn replace_regex_1() { run("'apples and bananas' . replace('[abe]+', 'o') . print", "opplos ond ononos\n") }
#[test] fn replace_regex_2() { run("'[a] [b] [c] [d]' . replace('[ac]', '$0$0') . print", "[aa] [b] [cc] [d]\n") }
#[test] fn replace_regex_with_function() { run("'apples and bananas' . replace('apples', to_upper) . print", "APPLES and bananas\n") }
#[test] fn replace_regex_with_wrong_function() { run("'apples and bananas' . replace('apples', argv) . print", "Incorrect number of arguments for fn argv(), got 1\n  at: line 1 (<test>)\n\n1 | 'apples and bananas' . replace('apples', argv) . print\n2 |                      ^^^^^^^^^^^^^^^^^^^^^^^^^\n") }
#[test] fn replace_regex_with_capture_group() { run("'apples and bananas' . replace('([a-z])([a-z]+)', 'yes') . print", "yes yes yes\n") }
#[test] fn replace_regex_with_capture_group_function() { run("'apples and bananas' . replace('([a-z])([a-z]+)', fn((_, a, b)) -> to_upper(a) + b) . print", "Apples And Bananas\n") }
#[test] fn replace_regex_implicit_newline() { run("'first\nsecond\nthird\nfourth' . replace('\\n', ', ') . print", "first, second, third, fourth\n") }
#[test] fn replace_regex_explicit_newline() { run("'first\nsecond\nthird\nfourth' . replace('\n', ', ') . print", "first, second, third, fourth\n") }
#[test] fn replace_regex_example_1() { run("'bob and alice' . replace('and', 'or') . repr . print", "'bob or alice'\n") }
#[test] fn replace_regex_example_2() { run("'bob and alice' . replace('\\sa', ' Ba') . repr . print", "'bob Band Balice'\n") }
#[test] fn replace_regex_example_3() { run("'bob and alice' . replace('[a-z]+', '$0!') . repr . print", "'bob! and! alice!'\n") }
#[test] fn replace_regex_example_4() { run("'bob and alice' . replace('[a-z]+', fn(g) -> g . reverse . reduce(+)) . repr . print", "'bob dna ecila'\n") }
#[test] fn replace_regex_example_5() { run("'bob and alice' . replace('([a-z])([a-z]+)', fn((_, g1, g2)) -> to_upper(g1) + g2) . repr . print", "'Bob And Alice'\n") }
#[test] fn search_regex_match_all_yes() { run("'test' . search('test') . print", "['test']\n") }
#[test] fn search_regex_match_all_no() { run("'test' . search('nope') . print", "[]\n") }
#[test] fn search_regex_match_partial_yes() { run("'any and nope and nothing' . search('nope') . print", "['nope']\n") }
#[test] fn search_regex_match_partial_no() { run("'any and nope and nothing' . search('some') . print", "[]\n") }
#[test] fn search_regex_match_partial_no_start() { run("'any and nope and nothing' . search('^some') . print", "[]\n") }
#[test] fn search_regex_match_partial_no_end() { run("'any and nope and nothing' . search('some$') . print", "[]\n") }
#[test] fn search_regex_match_partial_no_start_and_end() { run("'any and nope and nothing' . search('^some$') . print", "[]\n") }
#[test] fn search_regex_capture_group_match_none() { run("'some WORDS with CAPITAL letters' . search('[A-Z]([a-z]+)') . print", "[]\n") }
#[test] fn search_regex_capture_group_match_one() { run("'some WORDS with Capital letters' . search('[A-Z]([a-z]+)') . print", "[('Capital', 'apital')]\n") }
#[test] fn search_regex_capture_group_match_some() { run("'some Words With Capital letters' . search('[A-Z]([a-z]+)') . print", "[('Words', 'ords'), ('With', 'ith'), ('Capital', 'apital')]\n") }
#[test] fn search_regex_many_capture_groups_match_none() { run("'some WORDS with CAPITAL letters' . search('([A-Z])([a-z]+)') . print", "[]\n") }
#[test] fn search_regex_many_capture_groups_match_one() { run("'some WORDS with Capital letters' . search('([A-Z])[a-z]([a-z]+)') . print", "[('Capital', 'C', 'pital')]\n") }
#[test] fn search_regex_many_capture_groups_match_some() { run("'some Words With Capital letters' . search('([A-Z])[a-z]([a-z]+)') . print", "[('Words', 'W', 'rds'), ('With', 'W', 'th'), ('Capital', 'C', 'pital')]\n") }
#[test] fn search_regex_cannot_compile() { run("'test' . search('missing close bracket lol ( this one') . print", "ValueError: Cannot compile regex 'missing close bracket lol ( this one'\n            Parsing error at position 36: Opening parenthesis without closing parenthesis\n  at: line 1 (<test>)\n\n1 | 'test' . search('missing close bracket lol ( this one') . print\n2 |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n") }
#[test] fn search_regex_optional_match() { run("'123 |  456' . search '^ ([\\d ]+) | ([\\d ]+)' . print", "[('  456', nil, ' 456')]\n") }
#[test] fn search_regex_empty_str_match() { run("'abcd' . search '' . print", "[]\n") }
#[test] fn search_regex_empty_possible_match() { run("'abcd' . search '\\w*' . print", "['abcd']\n") }
#[test] fn search_regex_find_integers_with_no_groups() { run("'1 and 3 but not 7 and 21' . search '\\d+' . map int . print", "[1, 3, 7, 21]\n") }
#[test] fn search_regex_find_integers_with_groups() { run("'1 and 3 but not 7 and 21' . search '(\\d+)' . map(fn((_, g)) -> int g) . print", "[1, 3, 7, 21]\n") }
#[test] fn split_regex_empty_str() { run("'abc' . split('') . print", "['a', 'b', 'c']\n") }
#[test] fn split_regex_space() { run("'a b c' . split(' ') . print", "['a', 'b', 'c']\n") }
#[test] fn split_regex_space_duplicates() { run("' a  b   c' . split(' ') . print", "['', 'a', '', 'b', '', '', 'c']\n") }
#[test] fn split_regex_space_any_whitespace() { run("' a  b   c' . split(' +') . print", "['', 'a', 'b', 'c']\n") }
#[test] fn split_regex_space_any_with_trim() { run("' \nabc  \rabc \\r\\n  abc \\t  \t  \t' . trim . split('\\s+') . print", "['abc', 'abc', 'abc']\n") }
#[test] fn split_regex_on_substring() { run("'the horse escaped the barn' . split('the') . print", "['', ' horse escaped ', ' barn']\n") }
#[test] fn split_regex_on_substring_with_or() { run("'the horse escaped the barn' . split('(the| )') . print", "['', '', 'horse', 'escaped', '', '', 'barn']\n") }
#[test] fn split_regex_on_substring_with_wildcard() { run("'the horse escaped the barn' . split(' *e *') . print", "['th', 'hors', '', 'scap', 'd th', 'barn']\n") }
#[test] fn join_empty() { run("[] . join('test') . print", "\n") }
#[test] fn join_single() { run("['apples'] . join('test') . print", "apples\n") }
#[test] fn join_strings() { run("'test' . join(' ') . print", "t e s t\n") }
#[test] fn join_ints() { run("[1, 3, 5, 7, 9] . join('') . print", "13579\n") }
#[test] fn find_value_empty() { run("[] . find(1) . print", "nil\n") }
#[test] fn find_func_empty() { run("[] . find(==3) . print", "nil\n") }
#[test] fn find_value_not_found() { run("[1, 3, 5, 7] . find(6) . print", "nil\n") }
#[test] fn find_func_not_found() { run("[1, 3, 5, 7] . find(fn(i) -> i % 2 == 0) . print", "nil\n") }
#[test] fn find_value_found() { run("[1, 3, 5, 7] . find(5) . print", "5\n") }
#[test] fn find_func_found() { run("[1, 3, 5, 7] . find(>3) . print", "5\n") }
#[test] fn find_value_found_multiple() { run("[1, 3, 5, 5, 7, 5] . find(5) . print", "5\n") }
#[test] fn find_func_found_multiple() { run("[1, 3, 5, 5, 7, 5] . find(>3) . print", "5\n") }
#[test] fn rfind_value_empty() { run("[] . rfind(1) . print", "nil\n") }
#[test] fn rfind_func_empty() { run("[] . rfind(==3) . print", "nil\n") }
#[test] fn rfind_value_not_found() { run("[1, 3, 5, 7] . rfind(6) . print", "nil\n") }
#[test] fn rfind_func_not_found() { run("[1, 3, 5, 7] . rfind(fn(i) -> i % 2 == 0) . print", "nil\n") }
#[test] fn rfind_value_found() { run("[1, 3, 5, 7] . rfind(5) . print", "5\n") }
#[test] fn rfind_func_found() { run("[1, 3, 5, 7] . rfind(>3) . print", "7\n") }
#[test] fn rfind_value_found_multiple() { run("[1, 3, 5, 5, 7, 5, 3, 1] . rfind(5) . print", "5\n") }
#[test] fn rfind_func_found_multiple() { run("[1, 3, 5, 5, 7, 5, 3, 1] . rfind(>3) . print", "5\n") }
#[test] fn index_of_value_empty() { run("[] . index_of(1) . print", "-1\n") }
#[test] fn index_of_func_empty() { run("[] . index_of(==3) . print", "-1\n") }
#[test] fn index_of_value_not_found() { run("[1, 3, 5, 7] . index_of(6) . print", "-1\n") }
#[test] fn index_of_func_not_found() { run("[1, 3, 5, 7] . index_of(fn(i) -> i % 2 == 0) . print", "-1\n") }
#[test] fn index_of_value_found() { run("[1, 3, 5, 7] . index_of(5) . print", "2\n") }
#[test] fn index_of_func_found() { run("[1, 3, 5, 7] . index_of(>3) . print", "2\n") }
#[test] fn index_of_value_found_multiple() { run("[1, 3, 5, 5, 7, 5] . index_of(5) . print", "2\n") }
#[test] fn index_of_func_found_multiple() { run("[1, 3, 5, 5, 7, 5] . index_of(>3) . print", "2\n") }
#[test] fn rindex_of_value_empty() { run("[] . rindex_of(1) . print", "-1\n") }
#[test] fn rindex_of_func_empty() { run("[] . rindex_of(==3) . print", "-1\n") }
#[test] fn rindex_of_value_not_found() { run("[1, 3, 5, 7] . rindex_of(6) . print", "-1\n") }
#[test] fn rindex_of_func_not_found() { run("[1, 3, 5, 7] . rindex_of(fn(i) -> i % 2 == 0) . print", "-1\n") }
#[test] fn rindex_of_value_found() { run("[1, 3, 5, 7] . rindex_of(5) . print", "2\n") }
#[test] fn rindex_of_func_found() { run("[1, 3, 5, 7] . rindex_of(>3) . print", "3\n") }
#[test] fn rindex_of_value_found_multiple() { run("[1, 3, 5, 5, 7, 5, 3, 1] . rindex_of(5) . print", "5\n") }
#[test] fn rindex_of_func_found_multiple() { run("[1, 3, 5, 5, 7, 5, 3, 1] . rindex_of(>3) . print", "5\n") }
#[test] fn min_by_key() { run("[[1, 5], [2, 3], [6, 4]] . min_by(fn(i) -> i[1]) . print", "[2, 3]\n") }
#[test] fn min_by_cmp() { run("[[1, 5], [2, 3], [6, 4]] . min_by(fn(a, b) -> a[1] - b[1]) . print", "[2, 3]\n") }
#[test] fn min_by_wrong_fn() { run("[[1, 5], [2, 3], [6, 4]] . min_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n  at: line 1 (<test>)\n\n1 | [[1, 5], [2, 3], [6, 4]] . min_by(fn() -> 1) . print\n2 |                          ^^^^^^^^^^^^^^^^^^^\n") }
#[test] fn max_by_key() { run("[[1, 5], [2, 3], [6, 4]] . max_by(fn(i) -> i[1]) . print", "[1, 5]\n") }
#[test] fn max_by_cmp() { run("[[1, 5], [2, 3], [6, 4]] . max_by(fn(a, b) -> a[1] - b[1]) . print", "[1, 5]\n") }
#[test] fn max_by_wrong_fn() { run("[[1, 5], [2, 3], [6, 4]] . max_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n  at: line 1 (<test>)\n\n1 | [[1, 5], [2, 3], [6, 4]] . max_by(fn() -> 1) . print\n2 |                          ^^^^^^^^^^^^^^^^^^^\n") }
#[test] fn sort_by_key() { run("[[1, 5], [2, 3], [6, 4]] . sort_by(fn(i) -> i[1]) . print", "[[2, 3], [6, 4], [1, 5]]\n") }
#[test] fn sort_by_cmp() { run("[[1, 5], [2, 3], [6, 4]] . sort_by(fn(a, b) -> a[1] - b[1]) . print", "[[2, 3], [6, 4], [1, 5]]\n") }
#[test] fn sort_by_wrong_fn() { run("[[1, 5], [2, 3], [6, 4]] . sort_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n  at: line 1 (<test>)\n\n1 | [[1, 5], [2, 3], [6, 4]] . sort_by(fn() -> 1) . print\n2 |                          ^^^^^^^^^^^^^^^^^^^^\n") }
#[test] fn ord() { run("'a' . ord . print", "97\n") }
#[test] fn char() { run("97 . char . repr . print", "'a'\n") }
#[test] fn eval_nil() { run("'nil' . eval . print", "nil\n") }
#[test] fn eval_bool() { run("'true' . eval . print", "true\n") }
#[test] fn eval_int_expression() { run("'3 + 4' . eval . print", "7\n") }
#[test] fn eval_zero_equals_zero() { run("'0==0' . eval . print", "true\n") }
#[test] fn eval_create_new_function() { run("eval('fn() { print . print }')()", "print\n") }
#[test] fn eval_overwrite_function() { run("fn foo() {} ; foo = eval('fn() { print . print }') ; foo()", "print\n") }
#[test] fn eval_with_runtime_error_in_different_source() { run("eval('%sprint + 1' % (' ' * 100))", "TypeError: Operator '+' is not supported for arguments of type native function and int\n  at: line 1 (<eval>)\n  at: `<script>` (line 1)\n\n1 |                                                                                                     print + 1\n2 |                                                                                                           ^\n") }
#[test] fn eval_function_with_runtime_error_in_different_source() { run("eval('%sfn() -> print + 1' % (' ' * 100))()", "TypeError: Operator '+' is not supported for arguments of type native function and int\n  at: line 1 (<eval>)\n  at: `fn _()` (line 1)\n\n1 |                                                                                                     fn() -> print + 1\n2 |                                                                                                                   ^\n") }
#[test] fn all_yes_all() { run("[1, 3, 4, 5] . all(>0) . print", "true\n") }
#[test] fn all_yes_some() { run("[1, 3, 4, 5] . all(>3) . print", "false\n") }
#[test] fn all_yes_none() { run("[1, 3, 4, 5] . all(<0) . print", "false\n") }
#[test] fn any_yes_all() { run("[1, 3, 4, 5] . any(>0) . print", "true\n") }
#[test] fn any_yes_some() { run("[1, 3, 4, 5] . any(>3) . print", "true\n") }
#[test] fn any_yes_none() { run("[1, 3, 4, 5] . any(<0) . print", "false\n") }
#[test] fn typeof_of_basic_types() { run("[nil, 0, false, 'test', [], {1}, {1: 2}, heap(), (1, 2), range(30), enumerate([])] . map(typeof) . map(print)", "nil\nint\nbool\nstr\nlist\nset\ndict\nheap\nvector\nrange\nenumerate\n") }
#[test] fn typeof_functions() { run("[range, fn() -> nil, push(3), ((fn(a, b) -> nil)(1))] . map(typeof) . all(==function) . print", "true\n") }
#[test] fn typeof_struct_constructor() { run("struct Foo(a, b) Foo . typeof . print", "function\n") }
#[test] fn typeof_struct_instance() { run("struct Foo(a, b) Foo(1, 2) . typeof . print", "struct Foo(a, b)\n") }
#[test] fn typeof_slice() { run("[:] . typeof . print", "function\n") }
#[test] fn count_ones() { run("0b11011011 . count_ones . print", "6\n") }
#[test] fn count_zeros() { run("0 . count_zeros . print", "64\n") }
#[test] fn env_exists() { run("env . repr . print", "fn env(...)\n") }
#[test] fn argv_exists() { run("argv . repr . print", "fn argv()\n") }
#[test] fn real_of_bool() { run("true . real . print", "1\n") }
#[test] fn real_of_int() { run("123 . real . print", "123\n") }
#[test] fn real_of_imag() { run("123i . real . print", "0\n") }
#[test] fn real_of_complex() { run("1 + 2j . real . print", "1\n") }
#[test] fn real_of_str() { run("'hello' . real . print", "TypeError: Expected 'hello' of type 'str' to be a complex\n  at: line 1 (<test>)\n\n1 | 'hello' . real . print\n2 |         ^^^^^^\n") }
#[test] fn imag_of_bool() { run("true . imag . print", "0\n") }
#[test] fn imag_of_int() { run("123 . imag . print", "0\n") }
#[test] fn imag_of_imag() { run("123j . imag . print", "123\n") }
#[test] fn imag_of_complex() { run("4i + 6 . imag . print", "4\n") }
#[test] fn imag_of_str() { run("'4i + 6' . imag . print", "TypeError: Expected '4i + 6' of type 'str' to be a complex\n  at: line 1 (<test>)\n\n1 | '4i + 6' . imag . print\n2 |          ^^^^^^\n") }
#[test] fn counter_of_str_1() { run("'hello the world!!!!' . counter . print", "{'h': 2, 'e': 2, 'l': 3, 'o': 2, ' ': 2, 't': 1, 'w': 1, 'r': 1, 'd': 1, '!': 4}\n") }
#[test] fn counter_of_str_2() { run("'hello world' . counter . print", "{'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}\n") }
#[test] fn counter_of_empty() { run("[] . counter . print", "{}\n") }
#[test] fn copy_of_primitives() { run("[nil, 123, true, 'abc', 'finally this works'] . map copy . print", "[nil, 123, true, 'abc', 'finally this works']\n") }
#[test] fn copy_of_list_is_unique() { run("let a = [1, 2, 3] ; let b = copy a ; a.push(4) ; print(a, b)", "[1, 2, 3, 4] [1, 2, 3]\n") }
#[test] fn copy_of_nested_lists_are_unique() { run("let a = [[1], [2], [3]] ; let b = copy a ; a[1].push(2) ; print(a, b)", "[[1], [2, 2], [3]] [[1], [2], [3]]\n") }
#[test] fn copy_of_same_identity_lists_preserve_identity() { run("let a = [[]] * 4 ; let b = copy a ; a[0].push(1) ; b[1].push(2) ; print(a, b)", "[[1], [1], [1], [1]] [[2], [2], [2], [2]]\n") }
#[test] fn copy_of_recursive_lists_works() { run("let a = [1, 2] ; a.push(a) ; let b = copy a ; a.push(3) ; b.push(4) ; print(a, b)", "[1, 2, [...], 3] [1, 2, [...], 4]\n") }
#[test] fn copy_of_nested_collections_works() { run("let x = set(), y = dict(), z = (1, 2) ; let a = [x, {y: [x, y], z: z}, x] ; let b = copy a ; x.push(3) ; y[4] = 5 ; z[0] = 6 ; print(a, b)", "[{3}, {{4: 5}: [{3}, {4: 5}], (6, 2): (6, 2)}, {3}] [{}, {{}: [{}, {}], (1, 2): (1, 2)}, {}]\n") }
#[test] fn copy_heap_keeps_same_order() { run("let a = heap(6, 5, 2, 7, 3, 1, 9, 0, -1, -6, -4, -7, -2, -8) ; let b = copy a ; print(b, a == b)", "[-8, -6, -7, -1, -4, -2, 2, 0, 7, 3, 5, 1, 6, 9] true\n") }
#[test] fn copy_enumerate_preserve_identity() { run("let x = [1, 2, 3] ; let a = [enumerate x, x, enumerate x] ; let b = copy a ; x.push(4) ; b[1].push(5) ; print(a, b)", "[enumerate([1, 2, 3, 4]), [1, 2, 3, 4], enumerate([1, 2, 3, 4])] [enumerate([1, 2, 3, 5]), [1, 2, 3, 5], enumerate([1, 2, 3, 5])]\n") }
#[test] fn memoize() { run!("memoize") }
#[test] fn memoize_recursive() { run!("memoize_recursive") }
#[test] fn memoize_recursive_as_annotation() { run!("memoize_recursive_as_annotation") }

#[test] fn fibonacci() { run!("fibonacci") }
#[test] fn quine() { run!("quine") }

#[test] fn function_capture_from_inner_scope() { run!("function_capture_from_inner_scope", "'world'\n'world'\n") }
#[test] fn function_capture_from_outer_scope() { run!("function_capture_from_outer_scope", "'world'\n") }
#[test] fn late_bound_global() { run!("late_bound_global", "hello\n") }
#[test] fn late_bound_global_assignment() { run!("late_bound_global_assignment", "hello\n") }
#[test] fn late_bound_global_in_pattern() { run!("late_bound_global_in_pattern", "1 2\n") }
#[test] fn late_bound_global_in_pattern_invalid() { run!("late_bound_global_in_pattern_invalid", "ValueError: 'y' was referenced but has not been declared yet\n  at: line 2 (<test>)\n  at: `fn foo()` (line 5)\n\n2 |     x, y = (1, 2)\n3 |                 ^\n") }
#[test] fn late_bound_global_invalid() { run!("late_bound_global_invalid", "ValueError: 'bar' was referenced but has not been declared yet\n  at: line 3 (<test>)\n  at: `fn foo()` (line 6)\n\n3 |     bar . print\n4 |     ^^^\n") }
#[test] fn map_loop_with_multiple_references() { run!("map_loop_with_multiple_references", "[1, 2, 3, 4]\n") }
#[test] fn runtime_error_with_trace() { run!("runtime_error_with_trace") }
