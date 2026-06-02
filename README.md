## Cordy

Cordy is a dynamically typed, interpreted, semi-functional / semi-procedural language. It is designed as a quick-to-write, simple-yet-feature-full, scripting language for solving puzzles and other fun things as an alternative to Python.

An online Cordy REPL can be found [here](https://alcatrazescapee.com/cordy/), using the `cordy-web` subproject targeting Web Assembly. Language support and syntax highlighting is available via a [VS Code Extension](https://github.com/alcatrazEscapee/CordyLanguageSupport).

### Quick Introduction

This language is inspired by parts from Python, Rust, Haskell, Java, Go, and JavaScript. It is also heavily inspired by the [Crafting Interpreters](https://craftinginterpreters.com/) book, and by the existence of [Noulith](https://github.com/betaveros/noulith). The language is named after the professor of my fourth-year elective compilers course, which I loved.

A basic rundown of the syntax:

- The basic structure of the language is C-style, with `{` and `}` to separate code blocks, and imperative constructs such as `if`, `else`, `while`, etc.
- `let x` is used to declare a variable. `fn foo(x, y, z)` declares a function. Functions can either be followed be expressions (like `fn add1(x) -> x + 1`) or blocks (like `fn my_print(x) { print(x) }`).
  - Functions can be used in expressions, too, as anonymous functions by omitting the name, i.e. `let x = fn() -> 3`. They can be followed by either `->` or `{` when used in expressions.
- A few basic types:
  - `nil`: The absence of a value, and the default value for all declared uninitialized variables
  - `bool`: A boolean, which can be either `true` or `false`
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
  - `/` is floor division rounding to negative infinity, and `%` a modulo operator, not remainder (same as Python)
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

Cordy has several optional features, which provide functionality, debugging, or verification:

- (Default) `rational`: This enables usage of the `rational`, `numer`, and `denom` native functions, and the creation and usage of `rational` values in Cordy.

This feature requires the `rug` and `gmp-mpfr-sys` crates, which rely on the `GMP` library, built from source. This, notably, cannot be cross-compiled to WASM, and requires [additional setup (Windows)](https://docs.rs/gmp-mpfr-sys/1.6.1/gmp_mpfr_sys/index.html#building-on-windows). It also requires a `nightly-gnu` toolchain (as opposed to the default `msvc` on Windows):

```bash
$ rustup install nightly-gnu
$ rustup default nightly-gnu
```

Attempting to use any of the above functions on a version of Cordy built without this feature will raise a `PlatformError`. 

Debug features can be available, which provide trace output, primarily intended for use during debugging tests:

- `trace_parser` traces the parser execution, logging tokens accepted, pushed, and rules entered.
- `trace_interpreter` traces the virtual machine execution, logging instructions, and key events such as function invocations.
- `trace_stack` traces the virtual machine's stack after every `pop` and `push`.

Verification features can be enabled to run a much wider suite of possible test cases:

- `verify_parser` causes every compilation to compile every prefix of the input first. This does not check the output (and the vast majority of these will result in expected compile errors anyway), but instead this is intended to catch pathological cases where the parser error handling does not function correctly and hangs, or crashes.

### Future Work

Ultimately, this is a toy language - it is missing functionality that would make it more useful as more than that, and also makes additional implementation of an expanded standard library prohibitively difficult. There are, however, still features I would like to one day explore in Cordy, as I think the core language has some interesting ideas. In no particular order:

- Various language features related to the standard library usability, mechanisms for builtins, the C FFI, and native modules:
  - A better, cleaner way to implement stuff like `native module { ... }` or even `native struct { ... }` to be able to provide things like file pointer manipulation, file opening, closing, write modes, etc.
  - A mechanism for cordy-sys to be used as a dependency (somehow) to write Rust modules, that interface with the Cordy runtime. And then implement the cordy builtins within that system.
  - Imports! Cordy is a one-file-language and that is a problem, especially for kind of expanded standard library project that doesn't rely on making new keywords all the time (which is another real problem).
  - With that in mind, a better separation of builtins and a standard library. I think my ideal scenario is one where builtins don't exist, and are simply automatically-imported-without-a-namespace `native fn` functions, that exist in some library file somewhere. However, that would represent performance tradeoffs, and Cordy is actually fairly fast (for what it does).
  - Some kind of 'try' functionality, for instance to be able to write `if let value = my_dict[key_maybe_not_in_dict] { ... }`. Modeled after similar features in Rust and Go
- Static typing, or static type hints. Type inference would be very cool, but also very difficult, hence why I have not attempted it.
- More functionality on the cordy VS Code extension. It is currently very barebones, mostly because javascript VS Code extensions are not my area of expertise.
- *Potentially* revisit the 63-bit integer decision. This was an excellent example of optimizations from the backend leaking out into the front end. It is really weird, and is a result of the internal tagged pointer optimization (which was responsible for a ~30% across-the-board speedup!) but it is still weird.
- An actual binary generator. Either one that generates a binary of Cordy bytecode, or one that compiles to a target architecture (perhaps with LLVM).
