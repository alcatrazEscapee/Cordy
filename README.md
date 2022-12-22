## Cordy

**Note** This is still very unfinished and rough around the edges.

Cordy is a dynamically typed, interpreted, semi-functional / semi-procedural language I made up. It is meant to be a quick scripting language for solving puzzles and other fun things as an alternative to Python, and as a personal challenge to myself to use. It is implemented here in Rust, with a custom recursive descent parser, and a stack based interpreter.

### Usage

It is a standard Rust project, and thus built with `cargo build --release`. The `cordy` executable serves both as the compiler and runtime.

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
- `let x` is used to declare a variable. `fn(x, y, z)` declares a function. Functions can either be followed be expressions (like `fn(x) -> x + 1`) or blocks (like `fn (x) { print(x) }`).
- The basic types are `nil` (the absence of a value, and the default value for all declared uninitialized variables), `bool` (a boolean), `int` (a 64-bit unsigned integer), `str` (a UTF-8 string), `function` (the type of all functions), and `list` (A basic array backed list).
- Expressions should be familiar from most imperative programming languages, as should be operator precedence.
    - Operators on their own are functions, so `(+)` is a two argument function which adds values.
    - `%` is mathematical modulo, and `/` rounds to negative infinity, and `-(a / b) == -a / b == a / -b` (similar to Python)
    - The `.` operator is actually a low precedence function composition operator: `a . b . c` is equivalent to `c(b(a))`, and it can be chained in a functional style.
- Most functions (that aren't variadic) can be partially evaluated (like Haskell): `(+ 3)` is a function which takes one argument and adds three.
- The language is almost completely newline independent, and whitespace only used to delimit tokens. There are a few edge cases where whitespace (or semicolons) is required between expressions to reduce ambiguity.


### Examples

Below is a solution to [Advent of Code Day 1 Part 1](https://adventofcode.com/2022/day/1), written in a functional style:

```java
'input.txt' . read_text . split ('\n\n')
    . map(fn(g) -> g . split('\n') . map(int) . sum )
    . max
    . print
```

Or the same solution, written in a different style, with the same language:

```java
let inp = read_text('input.txt')
let lines = inp.split('\n\n')
let elves = lines.map(fn(g) {
    let elf = g.split('\n')
    elf.map(int).sum
})
print(max(elves))
```


### To-Do

- REPL, functional CLI and arguments.
- Implement `->` operator, basic named tuple types (structs), or binding resolution on macros
- Sets (HashSet)
- Maps (HashMap) and dictionary literals (or just use list syntax?)
- Polymorphic 'List' types that can be used as arbitrary iterators, etc. using rust traits rather than dispatch at every call site
- Deque / Queue type (VecDeque)
- Format strings (just normal strings with the python `%` operator, but with rust formatting? does that even work?)
- Regex (some sort of impl)
- Pattern matching / deconstruction
- Even MORE standard library functions
- Closures - capture value not variable (like in Java)?
- [VS Code](https://code.visualstudio.com/api/language-extensions/overview)?