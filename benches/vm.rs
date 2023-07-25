extern crate core;

use criterion::{criterion_group, criterion_main, Criterion, BatchSize};

use cordy::{compiler, SourceView};
use cordy::compiler::CompileResult;
use cordy::vm::{ExitType, VirtualMachine};


fn bench_add_1_loop_setup(c: &mut Criterion) { run("+1 loop setup", "let x = range(40).list", c) }
fn bench_add_1_loop_for(c: &mut Criterion) { run("+1 loop with for", "let x = range(40).list ; for i in range(len(x)) { x[i] += 1 }", c) }
fn bench_add_1_loop_map_fn(c: &mut Criterion) { run("+1 loop with map(fn)", "let x = range(40).list ; x .= map(fn(i) -> i + 1)", c) }
fn bench_add_1_loop_map_partial_fn(c: &mut Criterion) { run("+1 loop with map(partial fn)", "fn add(i,j) -> i+j ; let x = range(40).list ; x .= map(add(1))", c) }
fn bench_add_1_loop_map_op(c: &mut Criterion) { run("+1 loop with map(+1)", "let x = range(40).list ; x .= map(+1)", c) }
fn bench_sum_list_setup(c: &mut Criterion) { run("sum list setup", "let x = range(40).list, y = 0", c) }
fn bench_sum_list_for(c: &mut Criterion) { run("sum list with for", "let x = range(40).list, y = 0; for i in x { y += i }", c) }
fn bench_sum_list_for_range(c: &mut Criterion) { run("sum list with for range()", "let x = range(40).list, y = 0; for i in range(len(x)) { y += x[i] }", c) }
fn bench_sum_list_sum(c: &mut Criterion) { run("sum list with sum()", "let x = range(40).list, y = 0 ; y = x.sum", c) }
fn bench_sum_list_reduce_fn(c: &mut Criterion) { run("sum list with reduce(fn)", "let x = range(40).list, y = 0 ; y = x.reduce(fn(a, b) -> a + b)", c) }
fn bench_sum_list_reduce_op(c: &mut Criterion) { run("sum list with reduce(+)", "let x = range(40).list, y = 0 ; y = x.reduce(+)", c) }
fn bench_unpack_setup(c: &mut Criterion) { run("unpacking setup", "let x = range(20) . enumerate . list, y = x . map(fn(i) -> i)", c) }
fn bench_unpack_index(c: &mut Criterion) { run("unpacking with [1]", "let x = range(20) . enumerate . list , y = x . map(fn(i) -> i[1])", c) }
fn bench_unpack_pattern(c: &mut Criterion) { run("unpacking with (_, a)", "let x = range(20) . enumerate . list, y = x . map(fn((_, i)) -> i)", c) }
fn bench_unpack_var_pattern(c: &mut Criterion) { run("unpacking with (*_, a)", "let x = range(20) . enumerate . list, y = x . map(fn((*_, i)) -> i)", c) }
fn bench_fib_recursive(c: &mut Criterion) { run("fn fib() recursive", "fn fib(x) -> if x <= 2 then 1 else fib(x - 1) + fib(x - 2) ; fib(12)", c); }
fn bench_fib_recursive_memoize(c: &mut Criterion) { run("fn fib() memoized recursive", "@memoize fn fib(x) -> if x <= 2 then 1 else fib(x - 1) + fib(x - 2) ; fib(12)", c); }
fn bench_fib_loop(c: &mut Criterion) { run("fn fib() loop", "fn fib(x) { let a = 1, b = 1, i = 0 ; loop { a += b ; b = a - b; i += 1 ; if i >= x { break } } a } ; fib(10)", c) }
fn bench_fib_while(c: &mut Criterion) { run("fn fib() while", "fn fib(x) { let a = 1, b = 1, i = 0 ; while i < 10 { a += b ; b = a - b; i += 1 } a } ; fib(10)", c) }
fn bench_fib_for_range(c: &mut Criterion) { run("fn fib() for i in range()", "fn fib(x) { let a = 1, b = 1 ; for i in range(10) { a += b ; b = a - b } a } ; fib(10)", c) }
fn bench_fib_for_range_discard(c: &mut Criterion) { run("fn fib() for _ in range()", "fn fib(x) { let a = 1, b = 1 ; for _ in range(10) { a += b ; b = a - b } a } ; fib(10)", c) }
fn bench_is_prime_any(c: &mut Criterion) { run("fn is_prime() with any()", "fn is_prime(n) -> range(2, 1 + sqrt(n)) . any(fn(p) -> n % p == 0) ; is_prime(53)", c); }
fn bench_is_prime_for(c: &mut Criterion) { run("fn is_prime() with for", "fn is_prime(n) { for p in range(2, 1 + sqrt(n)) { if n % p == 0 { return true } } false } ; is_prime(53)", c); }


criterion_group!(benches,
    bench_add_1_loop_setup,
    bench_add_1_loop_for,
    bench_add_1_loop_map_fn,
    bench_add_1_loop_map_partial_fn,
    bench_add_1_loop_map_op,
    bench_sum_list_setup,
    bench_sum_list_for,
    bench_sum_list_for_range,
    bench_sum_list_sum,
    bench_sum_list_reduce_fn,
    bench_sum_list_reduce_op,
    bench_unpack_setup,
    bench_unpack_index,
    bench_unpack_pattern,
    bench_unpack_var_pattern,
    bench_fib_recursive,
    bench_fib_recursive_memoize,
    bench_fib_loop,
    bench_fib_while,
    bench_fib_for_range,
    bench_fib_for_range_discard,
    bench_is_prime_any,
    bench_is_prime_for
);
criterion_main!(benches);


fn run(name: &'static str, text: &'static str, criterion: &mut Criterion) {
    let view: SourceView = SourceView::new(String::from("<benchmark>"), String::from(text));
    let compile: CompileResult = match compiler::compile(true, &view) {
        Ok(c) => c,
        Err(e) => panic!("Compile Error:\n\n{}", e.join("\n")),
    };

    // Run once initially and ensure that we don't error
    let mut vm = VirtualMachine::new(compile.clone(), view, &b""[..], vec![], vec![]);
    match vm.run_until_completion() {
        ExitType::Exit => {},
        ExitType::Error(e) => panic!("{}", vm.view().format(&e)),
        e => panic!("Abnormal exit: {:?}", e)
    };

    // Then run the benchmark
    criterion.bench_function(name, |b| b.iter_batched(
        || {
            // For benchmarks, clone the compile result but use a new empty `SourceView`, since the runtime should not throw an error
            VirtualMachine::new(compile.clone(), SourceView::empty(), &b""[..], vec![], vec![])
        },
        |mut vm| {
            vm.run_until_completion()
        },
        BatchSize::SmallInput));
}