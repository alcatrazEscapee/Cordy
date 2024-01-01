## Cordy

Cordy is a dynamically typed, interpreted, semi-functional / semi-procedural language. It is designed as a quick-to-write, simple-yet-feature-full, scripting language for solving puzzles and other fun things as an alternative to Python.

An online Cordy REPL can be found [here](https://alcatrazescapee.com/cordy/), using the `cordy-web` subproject targeting Web Assembly. Language support and syntax highlighting is available via a [VS Code Extension](https://github.com/alcatrazEscapee/CordyLanguageSupport).

### Quick Introduction

This language is inspired by parts from Python, Rust, Haskell, Java, and JavaScript. It is also heavily inspired by the [Crafting Interpreters](https://craftinginterpreters.com/) book. A basic rundown of the syntax:

- The basic structure of the language is C-style, with `{` and `}` to separate code blocks, and imperative constructs such as `if`, `else`, `while`, etc.
- `let x` is used to declare a variable. `fn foo(x, y, z)` declares a function. Functions can either be followed be expressions (like `fn add1(x) -> x + 1`) or blocks (like `fn my_print(x) { print(x) }`).
  - Functions can be used in expressions, too, as anonymous functions by omitting the name, i.e. `let x = fn() -> 3`. They can be followed by either `->` or `{` when used in expressions.
- A few basic types:
  - `nil`: The absence of a value, and the default value for all declared uninitialized variables
  - `bool`: A boolean, which can be either `true` or `false
  - `int`: A 63-bit signed integer
  - `complex`: A 64-bit signed integral complex number. Declared with the `i` or `j` suffix, i.e. `1 + 3i`
  - `str`: A UTF-8 string
  - `function`: The type of all functions
- Along with some basic library collections:
  - `list`: A ring buffer with O(1) index, pop/push front and back.
  - `set`: A collection with unique elements and O(1) `contains` checks, along with insertion-order iteration.
  - `dict`: A mapping from keys to values with O(1) lookups, along with insertion-order iteration.
  - `heap`: A min-heap.
  - `vector`: A `list` variant which behaves elementwise with all basic operators.
- And user definable named tuple types, with the `struct` keyword.
- Expressions should be familiar from most imperative programming languages, as should be operator precedence.
  - Operators on their own are functions, so `(+)` is a two argument function which adds values.
  - `/` is floor division, and `%` a modulo operator (same as Python)
  - The `.` operator is actually a low precedence function composition operator: `a . b . c` is equivalent to `c(b(a))`, and it can be chained in a functional style.
  - Short-circuiting `and` and `or` use the keywords from Python.
- Most functions (that aren't variadic) can be partially evaluated (like Haskell): `(+ 3)` is a function which takes one argument and adds three.
- The language is almost completely newline independent, and whitespace only used to delimit tokens. There are a few edge cases where whitespace (or semicolons) is required between expressions to reduce ambiguity.

### Examples

Below is a solution to [Advent of Code 2022 Day 1 Part 1](https://adventofcode.com/2022/day/1), written in a functional style:

```rust
read_text "input.txt"
    . split "\n\n"
    . map(fn(g) -> g . split "\n" . map int . sum)
    . max
    . print
```

Or the same solution, written in a different style, with the same language:

```rust
let inp = read_text("input.txt")
let answer = 0
for group in inp.split("\n\n") {
    let elf = 0
    for weight in group.split("\n") {
        elf += int(weight)
    }
    answer max= elf
}
print(answer)

```

For a more comprehensive documentation, see the [language documentation](https://alcatrazescapee.com/cordy/language/) or the [standard library](https://alcatrazescapee.com/cordy/library/).

### Usage

Build with Rust (nightly), `cargo build --release`, and invoke the `cordy` executable at `/target/release/cordy`. With `--help`, this prints the following message:

```
$ cordy [options] <file> [program arguments...]
When invoked with no arguments, this will open a REPL for the Cordy language (exit with 'exit' or Ctrl-C)
Options:
  -h --help         : Show this message, then exit.
  -v --version      : Print the version, then exit.
  -d --disassembly  : Dump the disassembly view. Does nothing in REPL mode.
  -o --optimize     : Enables compiler optimizations and transformations.
  --no-line-numbers : In disassembly view, omits the leading '0001' style line numbers
```

### Building

This project requires a rust nightly toolchain due to use of two unstable features:

- `try_trait_v2` : Used for the implementation of `?` for `ValueResult`, and is fairly critical to code clarity and avoiding overhead of `Result<ValuePtr, ErrorPtr>`
- `variant_count` : Used for the automatic implementation of `NativeFunction::total()`, rather than supplying a constant.

The `gmp-mpfr-sys` crate requires the `GMP` library, built from source. This requires [additional setup (Windows)](https://docs.rs/gmp-mpfr-sys/1.6.1/gmp_mpfr_sys/index.html#building-on-windows), and also seems to require a `nightly-gnu` toolchain (as opposed to the default `msvc`):

```bash
$ rustup install nightly-gnu
$ rustup default nightly-gnu
```

Debug features can be enabled, which provide trace output, primarily intended for use during debugging tests:

- `trace_parser` traces the parser execution, logging tokens accepted, pushed, and rules entered.
- `trace_interpreter` traces the virtual machine execution, logging instructions, and key events such as function invocations.
- `trace_interpreter_stack` traces the virtual machine's stack, including a full view of the stack after every `pop` and `push`.
  - Note that this will cause some tests to fail due to excessive output, only use on small, targeted tests.

Verification features can be enabled to run a much wider suite of possible test cases:

- `verify_parser` causes every compilation to compile every prefix of the input first. This does not check the output (and the vast majority of these will result in expected compile errors anyway), but instead this is intended to catch pathological cases where the parser error handling does not function correctly and hangs, or crashes.