## Cordy

Cordy is a dynamically typed, interpreted, semi-functional / semi-procedural language I made up. It is meant to be a quick scripting language for solving puzzles and other fun things as an alternative to Python, and as a personal challenge to myself to use. It is implemented here in Rust, with a custom recursive descent parser, and a stack based interpreter.

### Usage

It is a standard Rust project, and thus built with `cargo build --release`. The `cordy` executable serves both as the compiler and runtime. Running `cordy` without any arguments opens a (primitive) REPL.

```
cordy [options] <file>
    -d : Show the disassembly output, then exit.

Compiles and executes the file <file>
```

For additional debugging information, compile with the environment variable `RUSTFLAGS` set to

```
--cfg trace_parser="off" --cfg trace_interpreter="off" --cfg trace_interpreter_stack="off"
```

And replace the values for the traces you want to enable with `on`:

- `trace_parser` traces the parser execution, logging tokens accepted, pushed, and rules entered.
- `trace_interpreter` traces the virtual machine execution, logging every instruction executed.
- `trace_interpreter_stack` traces the virtual machine's stack, including a view of the full stack after every `pop`, and `push`


### Quick Introduction

This language is inspired by parts from Python, Rust, Haskell, Java, and JavaScript. It is also heavily inspired by the [Crafting Interpreters](https://craftinginterpreters.com/) book. A basic rundown of the syntax:

- The basic structure of the language is C-style, with `{` and `}` to separate code blocks, and imperative constructs such as `if`, `else`, `while`, etc.
- `let x` is used to declare a variable. `fn foo(x, y, z)` declares a function. Functions can either be followed be expressions (like `fn add1(x) -> x + 1`) or blocks (like `fn my_print(x) { print(x) }`).
    - Functions can be used in expressions, too, as anonymous functions by omitting the name, i.e. `let x = fn() -> 3`. They can be followed by either `->` or `{` when used in expressions.
- A few basic types:
    - `nil`: The absence of a value, and the default value for all declared uninitialized variables
    - `bool`: A boolean
    - `int`: A 64-bit signed integer
    - `str`: A UTF-8 string
    - `function`: The type of all functions
- Along with some basic library collections:
    - `list`: A ring buffer with O(1) index, pop/push front and back.
    - `set`: A collection with unique elements and O(1) `contains` checks, along with insertion-order iteration.
    - `dict`: A mapping from keys to values with O(1) lookups, along with insertion-order iteration.
    - `heap`: A min-heap.
    - `vector`: A `list` variant which behaves elementwise with all basic operators.
- Expressions should be familiar from most imperative programming languages, as should be operator precedence.
    - Operators on their own are functions, so `(+)` is a two argument function which adds values.
    - `%` is mathematical modulo, and `/` rounds to negative infinity, and `-(a / b) == -a / b == a / -b` (similar to Python)
    - The `.` operator is actually a low precedence function composition operator: `a . b . c` is equivalent to `c(b(a))`, and it can be chained in a functional style.
    - Short-circuiting `and` and `or` use the keywords from Python.
- Most functions (that aren't variadic) can be partially evaluated (like Haskell): `(+ 3)` is a function which takes one argument and adds three.
- The language is almost completely newline independent, and whitespace only used to delimit tokens. There are a few edge cases where whitespace (or semicolons) is required between expressions to reduce ambiguity.


### Examples

Below is a solution to [Advent of Code 2022 Day 1 Part 1](https://adventofcode.com/2022/day/1), written in a functional style:

```rust
'input.txt' . read_text . split ('\n\n')
    . map(fn(g) -> g . split('\n') . map(int) . sum )
    . max
    . print
```

Or the same solution, written in a different style, with the same language:

```rust
let inp = read_text('input.txt')
let answer = 0
for group in inp.split('\n\n') {
    let elf = 0
    for weight in group.split('\n') {
        elf += int(weight)
    }
    answer max= elf
}
print(answer)

```

For a more comprehensive documentation, see the [language documentation](./docs/language.md) or the [standard library](./docs/stdlib.md).


### To-Do

- implement basic structs: compile time verified access with `->`, type-dispatched at runtime.
- `continue` and `break` in `for` statements
- `else` on `while` and `for` statements, like Python
- Let top level `if`-`then`-`else` get parsed in `parse_if_statement()`
- some kind of `switch` expression statement?
- `assert` statement?
- Even MORE standard library functions
  - regex match, find, and replace -> just on normal strings. Make `\` less required for escaping (only escape if needed)
- Investigate making a separate `Queue` type and making `List` a `Vec`? Benchmarks...
- Benchmarks + bytecode optimizer