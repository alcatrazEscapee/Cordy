# Cordy Language

Cordy is a dynamically typed, interpreted, semi-functional / semi-procedural language, designed to be fast to write for scripting and solving puzzles. Its design is inspired by parts of Python, Rust, Haskell, Java, Haskell, and JavaScript.

This document should give a comprehensive overview of how the language of Cordy is defined, from a perspective of some prior programming experience.


### Types

The primitive types in Cordy are:

- `nil` (The absence of a value)
- A boolean (`bool`), which can take the values `true` and `false`.
- `int`, which is a 64-bit integer. It can be expressed as decimal numbers (`5`), binary (`0b101`), or hexidecimal (`0x5`).
- `str`, which is a UTF-8 string. Like Python, there is no separate `char` data type, instead a string is a sequence of single element strings.

All primitive types are **immutable**. In addition to these, Cordy has a number of [Collection Types](#collection-types)

### Expressions

Expressions in Cordy are similar to C style languages. Cordy has a number of mathematical operators:

- `+`, `-`, `*`, `/`: Addition, Subtraction, Multiplication, and Division.
  - Note: multiplying a `str` and an `int` repeats the string by the int number of times, as in Python.
  - Note: addition with `str` will convert other arguments to a string and concatenate them.
  - Division for integers with a remainder rounds to negative infinity, and obeys the relation `-(a / b) == -a / b == a / -b` when it comes to negative operands.
- `a ** b` computes a raised to the power of b.
- `a % b` computes the mathematical modulo `a mod b`, and will always return a value in `[0, b)`.
- `&`, `|`, and `^` are bitwise AND, OR, and XOR, respectively. `<<` and `>>` are left and right shifts.
  - Shifts be negative values shift in reverse, so `1 >> -3` is `8`.
- `!` computes a logical not of boolean inputs, or a bitwise not of integer inputs.
- `and` and `or` are short-circuiting, logical operators.
- `<`, `>`, `>=`, `<=`, `==`, and `!=` compare values. Any values, regardless of types, can be compared for equality or ordering.
  - Note: different types will always compare as equal ordering.
- `if condition then value_if_true else value_if_false` is a short-circuiting ternary operator.
  - Note that all boolean comparisons will take the truthy value of it's argument. `nil`, `0`, `false`, `''`, and empty collections are the only falsey values, everything else is truthy.
- `is` is an operator used to check the type of a value.
- `in` (along with `not in`) is a special operator used for checking membership in collections, or substrings.
- `max=` and `min=` are special cases of the builtin functions `max` and `min`, expressed as an assignment operator. `a max= b` is semantically equivalent to `a = if b > a then b else a`, similar for `min=`.


All the above binary operators come in operator-equals variants: `+=`, `-=`, `*=`, `/=`, etc.

All operators are left associative (except `=` for assigning variables). Their precedence is noted as below, where higher entries are higher precedence:

| Precedence | Operators                                                                                      | Description                                                                            |
|------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| 1          | `[]`, `()`, `if then else`                                                                     | Array Access, Function Evaluation, Ternary `if`                                        |
| 2          | `-`, `!`, `~`                                                                                  | Unary Negation, Logical Not, and Bitwise Not                                           |
| 3          | `*`, `/`, `%`, `**`, `is`, `in`, `not in`                                                      | Multiplication, Division, Modulo, Power, Is In, Not In                                 |
| 4          | `+`, `-`                                                                                       | Addition, Subtraction                                                                  |
| 5          | `<<`, `>>`                                                                                     | Left Shift, Right Shift                                                                |
| 6          | `&`, `∣`, `^`                                                                                  | Bitwise AND, Bitwise OR, Bitwise XOR                                                   |
| 7          | `.`                                                                                            | [Function Composition](#function-evaluation)                                           |
| 8          | `<`, `<=`, `>`, `>=`, `==`, `!=`                                                               | Less Than, Less Than or Equal, Greater Than, Greater Than or Equal, Equals, Not Equals |
| 9          | `and`, `or`                                                                                    | Logical And, Logical Or                                                                |
| 10         | `=`, `+=`, `-=`, `*=`, `/=`, `&=`, `∣=`, `^=`, `<<=`, `>>=`, `%=`, `**=`, `.=`, `max=`, `min=` | Assignment, and Operator Assignment                                                    |

### Variables

Variables must be declared with `let`. They can optionally be followed by a initialization.

```rust
// A variable declaration, it is initialized to `nil`
let x

// A variable declaration and assignment
let y = 'hello'
```

Variable declarations can be chained:

```rust
let x, y, z
```

Even with assignments:

```rust
let x = 1, y = 2, z = 3
```

### Functions

Cordy has functions as a first class type, so they can be declared anywhere. It also supports anonymous (lambda) functions which can be used as part of expressions. A function begins with `fn`, followed by a name (if it is not an anonymous function), and then the function body:

```rust
// the most basic function, `foo`, which does nothing
fn foo() {}

// this is an anonymous function
fn() {}

// Like JavaScript, anonymous functions can be wrapped in an IIFE:
(fn() { ... })()
```

The body of a function can either be a block statement (a series of statements wrapped in `{` curly `}` brackets), or it can be an arrow `->` followed by a single expression.

A function will return the last expression present in the function, or whenever a `return` keyword is reached.

```rust
// these functions are semantically equivilant
fn three() { 
    3
}

fn three() { 
    return 3
}

fn three() -> 3
```

#### Native and Operator Functions

Cordy has a number of [Native Functions](./stdlib.md), which can be used as normal functions. In addition to these, each operator is also a function, which can be referenced by placing it in `(` parentheses `)`.

```rust
let addition = (+)
```

Note in some cases the additional parenthesis can be omitted, for instance if passing an operator to a function:

```
>>> print(+)
(+)
```

One such important native function is `print`, which prints all arguments, space seperated, to standard out, followed by a newline:

```rust
// rite of passage!
print('hello world!')
```

#### Function Evaluation

Functions can be invoked in two ways. First, in a C-style function invocation, the function name and passing arguments within brackets:

```rust
// calls the function foo with the arguments 1, 2
foo(1, 2)
```

Alternatively, single-argument functions can be invoked with the `.` operator, which reverses the order of function and argument:

```rust
// same as writing foo('hello')
'hello' . foo
```

#### Partial Functions

Functions can be *partially evaluated*, that is, if a function is evaluated with less than the required number of arguments, it returns a new function which can be evaluated with the remaining arguments:

```rust
// foo takes three arguments
fn foo(a, b, c) {}

// this creates a partial function...
let partial_foo = foo(1, 2)

// ...which only needs the third argument
partial_foo(3)
```

Operators can also be partially evaluated, and like above, the parenthesis can be omitted around the operator:

```rust
let add3 = (+3)
add3(4) // returns 7
```

Note that function evaluation with `()` is high precedence, whereas function evaluation with `.` is low precedence. This can be used alongside partial functions to great effect:

```rust
input . map(int) . filter(>0) . reduce(+) . print
```

#### Closures

Functions can reference local and global variables outside of themselves, mutate them, and assign to them. Closures capture the *value* of variables at the time of the closure being closed.

```rust
fn foo() {
    let x = 'hello'
    fn bar() -> x = 'goodbye'
    fn baz() -> x . print
    baz() // prints 'hello'
    bar()
    baz() // prints 'goodbye'
}
```

Variables declared in loops are captured each iteration of the loop. So the following code:

```rust
let values = []
for i in range(5) {
    values.push(fn() -> i)
}
values . map(fn(f) -> f()) . print
```

Will print the sequence `[1, 2, 3, 4, 5]`, as intuitively expected.


### Control Structures

Cordy has a number of procedural style control structures, some familiar from C, Python, or Rust.

`loop` is a simple infinite loop, which must be terminated via the use of `break` (which exits the loop), or `return`:

```rust
loop {
    // statements
}
```

`while` is a conditional loop which evaluates so long as the expression returns truthy:

```rust
while condition {
    // statements
}
```

Like in Python, it can have an optional `else`, which will only be entered if a `break` statement was **not** encountered.

```rust
while condition {
    // statements
} else {
    // only if no `break`
}
```

`for-in` is a loop which iterates through a collection or string, yielding elements from the collection each iteration.

```rust
// declares `x` local to the loop
for x in collection {
    // statements
}
```

Note: `break` and `continue` can be used in all loop-like structures, which will exit the loop (`break`), or jump to the top of the next iteration of the loop (`continue`)

Finally, `if`, `else`, `elif` perform control flow not in expressions:

```rust
if condition1 {
    // statements
} elif condition2 {
    // statements
} else {
    // statements
}
```

Note that `if`, `else` blocks will return the last expression in the block that was executed, meaning they can be used as return values:

```rust
// These are semantically equivilant

fn foo(x) {
    if x {
        'yes'
    } else {
        'no'
    }
}

fn foo(x) -> if x then 'yes' else 'no'
```

### Collection Types

In addition to primitive types, Cordy supports a number of mutable collection types. These are:

- `list`: A list of arbitrary type objects. It is implemented with a ring buffer, and so supports O(1) push front and back.
  - List literals can be declared with `[` square `]` brackets.
  - Lists can be accessed using array syntax, i.e. `my_list[1]`
  - Negative indexes wrap to the end of the list, i.e. `my_list[-1]` is the last element in the list
  - Lists (and strings) can also be sliced via `list[start:stop:step]`, using Python-like slicing mechanics, i.e. `my_list[2:]` takes everything after the first two elements.

```rust
let my_list = [1, 2, 3, 4]
```

- `set`: A hash set with unique elements, and O(1) `in` checks.
- `dict`: A hash map, with O(1) element lookup, and optional support for default values.
  - Accessing and mutating the `dict` can be done with array-like syntax:

```rust
let my_dict = dict()
my_dict['hello'] = 'world'
my_dict['hello'] . print
```

- `heap`: A min-heap, implemented as a binary heap, with O(log n) access to the minimum element.
- `vector`: A `list` like data type, but with a fixed length, and where all operations behave in an elementwise fashion.
  - Operating on a vector and a constant will apply the constant to each element of the vector:
  - Vectors can be declared in literals like lists, but with `(` parenthesis `)`.
  - Single argument vectors require a trailing comma (i.e. `(1,)`, not `(1)`)

```
>>> (1,)
(1)
>>> (1, 2, 3)
(1, 2, 3)
>>> (1, 2, 3) * (2, 4, 6)
(2, 8, 18)
>>> (1, 2, 3) * 3
(3, 6, 9)
```

### Pattern Matching

Variable declarations, both in `let` statements, and in the declaration of a `for-in` loop, support pattern matching / destructuring. This takes the form of mirroring the iterable-like structure, like Python:

```rust
// multiple variables 'unpack' the list
let x, y, z = [1, 2, 3]

// using '*' indicates it should collect the remaining items
let first, *rest = [1, 2, 3, 4]

// '*' can be used in the middle too!
let front, *middle, back = [1, 2, 3, 4]

// Patterns can be nested using `(` parenthesis `)`
let a, (b, c), d = [[1, 2], [3, 4], [5, 6]]

// Any values which wish to be ignored can be replaced with `_`
let _, _, x, _ = [1, 2, 3, 4]
```

Pattern matching is also supported within function arguments:

```rust
// This function:
fn flip((x, y)) -> (y, x)

// Is semantically identical to this one
fn flip(pair) {
    let x, y = pair
    return (y, x)
}
```

And within expressions, if all the variables are declared:

```rust
let a, b, c

a = b, c = (1, 2) // assigns a = (1, 2), b = 1, c = 2
```

### Decorators

Functions can optionally be *decorated*, which is a way to modify the function in-place, without having to reassign to it. A decorator consists of a `@` followed by an expression, before the function is declared:

```rust
@memoize
fn fib(n) -> if n <= 1 then 1 else fib(n - 1) + fib(n - 2)
```

This can be understood as the following:

```rust
fn fib(n) -> if n <= 1 then 1 else fib(n - 1) + fib(n - 2)
fib = memoize(fib)
```

Decorators can be chained - the innermost decorators will apply first, then the outermost ones. They can also be attached to anonymous functions. The expression of a decorator must resolve to a function, which takes in the original function as an argument, and outputs a new value - most typically a function, but it doesn't need to. For example:

```rust
fn iife(f) -> f() // 'Immediately Invoked Function Expression', from JavaScript

@iffe
fn do_stuff() { print('hello world') } // prints 'hello world' immediately, and assigns `do_stuff` to `nil`
```