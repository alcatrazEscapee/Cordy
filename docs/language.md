# Cordy Language

Cordy is a dynamically typed, interpreted, semi-functional / semi-procedural language, designed to be fast to write for scripting and solving puzzles. Its design is inspired primarily by Python, Rust, and Haskell. This document assumes some prior programming experience.

## Contents

1. [Introduction](#introduction)
2. [Expressions](#expressions)
    1. [Collections](#collections)
    2. [Operators](#operators)
    3. [Precedence and Associativity](#precedence-and-associativity)
3. [Variables](#variables)
4. [Functions](#functions)
    1. [Partial Functions](#partial-functions)
    2. [Operator Functions](#operator-functions)
    3. [User Functions](#user-functions)
    4. [Closures](#closures)
5. [Control Flow](#control-flow)
6. [Advanced Cordy](#advanced-cordy)
    1. [Slicing](#slicing)
    2. [Pattern Matching](#pattern-matching)
    3. [Decorators](#decorators)
    4. [Assertions](#assertions)
    5. [Structs and Modules](#structs-and-modules)
    6. [Native Modules and FFI](#native-modules-and-ffi)

### Introduction

```
>>> print 'hello world'
hello world
```

Cordy is a dynamically typed, interpreted language. A cordy program is compiled into a _bytecode_, and then the bytecode is _interpreted_. Cordy programs are made up of statements and expressions, similar to most procedural languages.

Cordy can be used in two basic modes:

1. REPL (Read-Evaluate-Print Loop) Mode. When the `cordy` executable is invoked with no arguments, it opens a interactive session where individual expressions (or statements) may be entered, immediately executed, and the result printed back to the console. The symbol `>>>` indicates this line is an input in REPL mode.
2. Compile Mode. When `cordy my_first_program.cor` is invoked, it will attempt to compile, then immediately execute the file `my_first_program.cor`.

A web-based REPL can be found [here](https://alcatrazescapee.com/cordy/).

With that basic introduction, it's time to explore the basics of expressions...

### Expressions

Expressions in Cordy are similar to expressions in many other procedural languages. They are a composition of _values_ and _operators_. The basic types of values are:

- Nil: `nil`. This is the null/None/empty value, and is the default value of any uninitialized variable.
- Boolean: `bool`. This is a type which can take the values `true` or `false`.
- Integers: `int`. Integers in Cordy are 63-bit, 2's compliment signed integers. They can be expressed as decimal numbers (e.g. `13`), binary with a `0b` prefix (e.g. `0b1101`), or hexadecimal with a `0x` prefix (e.g. `0xD`).
- Complex Integers: `complex`. These are a pair of a 64-bit, 2's compliment signed integer, with one real and one imaginary component.
- Strings: `str`. These are a sequence of characters encoded in UTF-8. Like in Python, there is no separate single-character data type. Strings can be declared using either `'single quotes'` or `"double quotes"` Strings can also include newlines:

```
'this is a string
with a newline in the middle'
```

- Strings can also contain the escape sequences `\'`, `\"`, `\\`, `\r`, `\n` and `\t`.

#### Collections

Cordy also contains a number of _collection types_. These are mutable data structures which can contain other values:

- `list` is a dequeue. It supports O(1) insertion on both ends, and O(1) access by index.
  - Lists can be declared with comma-separated values within `[` square brackets `]`.
  - Accessing a list can also be done with `[` square brackets `]`.
- `vector` is a fixed-size sequence of values. It supports O(1) access by index, but more importantly all operators on `vector` act element-wise.
  - Expressions involving a vector and a scalar will apply the operator to each element of the vector, i.e. `(1, 2, 3) * 4` produces `(4, 8, 12)`
  - Vectors can be declared with comma-separated values within `(` parenthesis `)`.
  - Accessing a vector can be done with `[` square brackets `]`.
- `set` is a hash-backed set of elements. It supports O(1) contains checks, and enforces strict uniqueness of its elements. It also maintains insertion order for iteration over its elements.
  - Sets can be declared with comma-separated values within `{` curly brackets `}`.
  - Checking if a value is in a set can be done with the `in` operator.
- `dict` is a hash-backed key-value mapping. It supports O(1) contains and value access, and enforces strict uniqueness of its keys. It also maintains insertion order for iteration over its elements.
  - Dictionaries can be declared with colon-delimited key value pairs, comma-separated, within `{` curly brackets `}`
  - Accessing a value by key can be done with `[` square brackets `]`.
  - Note that `{}` declares an empty set, **not** an empty dictionary.

**Examples:**

```rust
>>> [1, 2, 3] // a list
[1, 2, 3]
>>> (4, 5, 6) // a vector
(4, 5, 6)
>>> {1, 3, 5} // a set
{1, 3, 5}
>>> {1: 10, 2: 20, 3: 30} // a dictionary
{1: 10, 2: 20, 3: 30}
```

Cordy also allows the creation of user-defined structs and modules. But more on them [later](#structs-and-modules).

#### Operators

Operators in Cordy are similar to procedural languages, with all operators being **infix** operators by default. Cordy supports the following basic operators:

- `+`, `-`, `*`, `/`: Addition, Subtraction, Multiplication, and Division.
  - Multiplying a `str` and an `int` repeats the string by the int number of times, as in Python.
  - Multiplying a `list` and an `int` repeats the elements in the list by the int number of times, also as in Python. Note that this does not deep copy the elements, be careful with mutable values (see `**` instead).
  - Addition with `str` will convert other arguments to a string and concatenate them.
  - `/` for integers and complex numbers is floor division, rounding to negative infinity.
- `**`: Exponentiation
  - With integral arguments, `a ** b` computes `a` raised to the power of `b`.
  - With a `list` and an `int`, this performs a deep copy on the elements of the list by the int number of times.
- `%`: Modulo / String formatting.
  - With integer arguments, `a % b` computes the mathematical modulo `a mod b` and will always return a value in `[0, b)`.
  - When `a` is a string, this behaves like Python's string formatting `%` operator.
- `&`, `|`, and `^` are bitwise AND, OR, and XOR, respectively. `<<` and `>>` are bitwise arithmetic left and right shifts.
  - Shifts of negative values are shifts in reverse, so `(a << b) == (a >> (-b))` for all a, b.
- `!` computes a logical not of boolean inputs, or a bitwise not of integer inputs.
- `and` and `or` are short-circuiting, logical operators.
- `<`, `>`, `>=`, `<=`, `==`, and `!=`: Comparison operators.
  - Every value in Cordy can be compared for equality or ordering. However, objects of different type will always order as equal.
  - Chained comparison operators behave intuitively like in Python: `a < b < c` is equivalent to `(a < b) and (b < c)` (without evaluating `b` twice).
- `if condition then value_if_true else value_if_false` is a short-circuiting ternary operator.
- `is` (along with `is not`) is an operator used to check if a value (left hand operand) is of a given type (right hand operand).
- `in` (along with `not in`) is used for checking if the left hand side is _contained in_ the right hand side.
  - When the right hand side is a string, this performs a substring check.
  - When the right hand side is a collection, this checks if the value is contained in the collection.
- Almost all operators can be used as assignment operators by suffixing them with an `=`, i.e. `+=`, `-=`, `/=`, etc.
  - In addition, `max=` and `min=` can be used as assignment operators, which assign if the right hand side is larger or smaller, respectively.

#### Precedence and Associativity

All operators are left associative (except `=` for assigning variables). Their precedence is noted as below, where higher entries are higher precedence:

| Precedence | Operators                                                                                      | Description                                                                            |
|------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| 1          | `[]`, `()`, `if then else`                                                                     | Array Access, Function Evaluation, Ternary `if`                                        |
| 2          | `-`, `!`, `~`, `->`                                                                            | Unary Negation, Logical Not, Bitwise Not, Struct Access                                |
| 3          | `*`, `/`, `%`, `**`, `is`, `is not`, `in`, `not in`                                            | Multiplication, Division, Modulo, Power, Is, Is Not, In, Not In                        |
| 4          | `+`, `-`                                                                                       | Addition, Subtraction                                                                  |
| 5          | `<<`, `>>`                                                                                     | Left Shift, Right Shift                                                                |
| 6          | `&`, `∣`, `^`                                                                                  | Bitwise AND, Bitwise OR, Bitwise XOR                                                   |
| 7          | `.`                                                                                            | Function Composition                                                                   |
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

Variables with the same name may not be re-declared within the same scope:

```rust
let x = 1
let x = 2 // Compile Error!
```

However, variables in outer scopes may be shadowed by variables in inner scopes:

```rust
let x = 1
if x > 0 {
    let x = 2
    print x // prints 2
}
print x // prints 1
```

Variable names are mostly unrestricted - they may be any alphanumeric identifier that starts with a alphabetic character. However they may _not_ share the name with a native function, of which the full list is found in the [library](https://alcatrazescapee.com/cordy/library/) documentation.

```rust
let map // Compile Error!
```

### Functions

Functions in Cordy come in many different types. First, Cordy has a number of _native functions_ which are provided by the Cordy standard library. These take the form of a few reserved keywords, like `print`, `map` and `int`.

Functions can be _called_ in three different ways:

- C-style, with the function arguments in `(` parenthesis `)` following the function name.
- Haskell-style, with the `.` operator, where the function name comes _after_ the function.
- C-style-with-less-typing, where if the arguments are simply space separated after the function name, they are treated (in most cases) as a function call.

```java
>>> print('hello') // calls 'print' with the argument 'hello'
hello
>>> 'world' . print // calls 'print' with the argument 'world'
world
>>> print 'goodbye' // calls 'print' with the argument 'goodbye'
goodbye
```

Note that the last type of evaluation is not always possible, and if it would be ambiguous with some other syntax, that is universally preferred. For example:

```rust
foo [1] // Evaluates to `foo[1]`, not `foo([1])`
```

Also note that when placed this way, arguments are evaluated _one by one_, meaning `foo 1 2 3` will be interpreted as `foo(1)(2)(3)`. This may be not an issue however, due to the presence of _partial functions_...

#### Partial Functions

Some native functions, and all user-defined functions, can be _partially evaluated_. This means that a function can be invoked with less than the required number of arguments, and it will return a new function which only needs to be invoked with the remaining arguments. One such example from the Cordy standard library is `map`:

```rust
let my_list = [1, -2, 3, -4, 5]

map(abs, my_list) // returns [1, 2, 3, 4, 5]

// We can partially evaluate `map(abs)` and then invoke that as a function
let f = map(abs)
f(my_list) // returns [1, 2, 3, 4, 5]

// Due to ( ) evaluation having higher precedence than . evaluation, we can also chain these two:
my_list . map(abs) // returns [1, 2, 3, 4, 5]

// This is also equivalent to the above
my_list . map abs // returns [1, 2, 3, 4, 5]
```

This mixing of the function composition operator (`.`) and regular function calls means that long sequential statements in Cordy can be written in a very functional style:

```java
'123456789' . map(int) . filter(>3) . sum . print // Prints 39
```

#### Operator Functions

In addition to the Cordy standard library, which provides a number of _native functions_, every operator in Cordy can also be used as a function by surrounding it in `(` parenthesis `)`. For example:

```rust
let add = (+)

add(2, 3) // returns 5
```

Note in some cases the additional parenthesis can be omitted, for instance if passing an operator to a function:

```python
>>> print(+)
(+)
```

Operators can be partially evaluated by placing the partial argument either to the left, or right, of the operator. The placement affects which side of the operator is treated as partially evaluated:

```rust
let divide_3_left = (/6)
let divide_3_right = (6/)

divide_3_left(18) // same as 18/6 = 3
divide_3_right(3) // same as 6/3 = 2
```

Note that the parenthesis can also be omitted in the partial-evaluated operator when passing to a function:

```rust
[1, 2, 3, 4, 5] . filter(>3) // returns [4, 5]
```

#### User Functions

In addition to native and operator functions, Cordy allows the user to declare their own functions. These are variables, declared with the `fn` keyword.

- Functions can either be _named_, or _anonymous_. Named functions declare themselves as a variable and are statements, while anonymous functions do not declare a variable, and are part of expressions.
- Functions can define _parameters_, including both default parameters and variadic parameters, which affect the arguments passed to the function.
- The body of a function can either be an _expression_ (indicated by a `->`) or a series of statements (indicated by `{ }`).
- Functions always return a value: either the last expression in the function, or `nil` if no such expression was present, or via an explicit `return` keyword.
- All user functions are _partial_ when evaluated with less than their required number of arguments.

**Examples:**

```rust
// A function named 'foo' which is followed by a statement body. It prints 'hello' and then returns 'world'
fn foo() {
    print('hello')
    return 'world'
}

// An anonymous function, which is stored in the variable 'f'. It takes one argument, and returns that argument plus three
let f = fn(x) -> x + 3

// A function named 'norm1' which takes two arguments, and returns the 1-norm of (x, y).
fn norm1(x, y) -> abs(x) + abs(y)

// A function named 'goodbye' which prints 'goodbye' and then returns nil.
fn goodbye() {
    print('goodbye')
}
```

As above, functions will return the last expression present in the function, or one specified via a `return` keyword. If no expression is given, they will return `nil`:

```rust
// these functions are semantically equivalent
fn three() { 
    3
}

fn three() { 
    return 3
}

fn three() -> 3
```

User functions can be _partially evaluated_:

```rust
// foo takes three arguments
fn foo(a, b, c) {
    a + ' and ' + b + ' or ' + c
}

let partial_foo = foo(1, 2)
partial_foo(3) // returns '1 and 2 or 3'
```

Functions can define optional and default arguments. All optional and default arguments must come after all other arguments in the function. Functions can be invoked with or without their optional or default arguments, which will take the default value `nil` (for optional arguments), or the default value (for default arguments).

Optional arguments are declared by appending a `?` to the argument in the function declaration:

```rust
fn optional_argument(a, b?) -> print(a, b)

optional_argument('hello') // prints 'hello nil'
optional_argument('hello', 'world') // prints 'hello world'
```

Default arguments are declared by appending an `=` followed by an expression to the argument in the function declaration:

```rust
fn default_argument(a, b = 'world') -> print(a, b)

default_argument('hello') // prints 'hello world'
default_argument('hello', 'earth') // prints 'hello earth'
```

Note that default argument values are constructed each time, and are not mutable across function calls, unlike in Python:

```rust
fn foo(a = []) {
    a.push('yes')
    print(a)
}

foo() // prints ['yes']
foo() // prints ['yes']
```

Functions can be called with **unrolled arguments**. This unrolls an iterable into a sequence of function arguments, by prepending a `...` to the argument in question. This is like the unary `*` operator in Python:

```rust
fn foo(a, b, c) -> print(a, b, c)

// All of the below call `foo(1, 2, 3)`, and print '1 2 3'
foo(...[1, 2, 3]) // unrolls each list element as an argument
foo(...[1, 2], 3) // they can be used with normal arguments, in any order
foo(...[], 1, ...[2], 3, ...[]) // An empty iterable is treated as adding no new arguments
```

In addition to this, they support variadic arguments (via a `*` like in Python). These must be the last argument in the function, and they collect all arguments into a vector when called:

```rust
// Note that `b` will be a vector of all arguments excluding the first. It may be empty.
fn foo(a, *b) -> print(b)

foo('hello') // prints ()
foo('hello', 'world') // prints ('world')
foo('hello', 'world', 'and', 'others') // prints ('world', 'and', 'others')
```

#### Closures

Functions can reference local and global variables outside themselves, mutate them, and assign to them. Closures are able to reference and mutate captured variables, even after they have fallen out of scope of the original declaration.

```rust
fn make_box() {
    let x = 'hello'
    fn foo() -> x = 'goodbye'
    fn bar() -> x.print
    (foo, bar)
}
let foo, bar = make_box()

bar() // prints 'hello'
foo()
bar() // prints 'goodbye'
```

Variables declared in loops are captured each iteration of the loop. So the following code:

```rust
let numbers = []
for i in range(5) {
    numbers.push(fn() -> i)
}
numbers . map(fn(f) -> f()) . print
```

Will print the sequence `[1, 2, 3, 4, 5]`, as intuitively expected.


### Control Flow

Cordy has a number of procedural style control structures, some familiar from C, Python, or Rust.

`loop` is a simple infinite loop, which must be terminated via the use of `break` (which exits the loop), or `return`:

```rust
loop {
    // statements
}
```

`while` is a conditional loop which evaluates so long as the expression returns a truthy value (N.B. `false`, `0`, `nil`, `''`, and empty collections are all falsey values - everything else is truthy.):

```rust
while condition {
    // statements
}
```

Like in Python, it can have an optional `else`, which will only be entered if a `break` statement was **not** encountered.

```java
while condition {
    // statements
} else {
    // only if no `break`
}
```

`do-while` is a variant of the above which runs statements first, then evaluates the loop

```java
do {
    // statements
} while condition
```

Like a typical `while` loop, it can also have an optional `else` block, which will only be entered if a `break` statement was **not** encountered.

```java
do {
    // statements
} while condition else {
    // only if no `break`    
}
```

Note that the `while condition` can be omitted entirely, in which case the `do { }` statement functions as a single scoped block, that also supports `break` (jump to the end of the block) and `continue` (jump to the top of the block) semantics.

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

Note that `if`, `else` with `{` curly brackets `}` are **not** expressions, and thus don't produce a value, however the `if`, `then`, `else` block is, and so **does** produce a value.

### Advanced Cordy

The below series of features are arbitrarily deemed advanced. 

#### Slicing

List, vector, and string indexing works identically to Python:

- All indexing is zero-based, with `0` being the first index, and `len(x) - 1` being the last.
- Indexing can be _negative_, counting backwards from the last index, so `-1` is the last index, `-2` is the second last index, and `-len(x)` is the first.

```rust
>>> let my_list = [1, 2, 3, 4, 5]
>>> my_list[0]
1
>>> my_list[len(my_list) - 1]
5
>>> my_list[-1]
5
>>> my_list[-3]
3
```

Lists, vectors, and strings can also be _sliced_ like in Python. A slice takes the form `[ <start> : <stop> : <step> ]` or `[ <start> : <stop> ]`.

- Any argument can be omitted (or `nil`), in which case it will be treated as slicing to the start or end, inclusive.
- `<step>` indicates what direction the slice will take, and by how much. If it is not present, the slice will step by 1.
- Slices support negative indices in the same manner as above.

**Examples**

```python
>>> [1, 2, 3, 4] [:]
[1, 2, 3, 4]
>>> [1, 2, 3, 4] [1:]
[2, 3, 4]
>>> [1, 2, 3, 4] [:2]
[1, 2]
>>> [1, 2, 3, 4] [:-2]
[1, 2]
>>> [1, 2, 3, 4] [1::2]
[2, 4]
>>> [1, 2, 3, 4] [1:-1:3]
[2]
>>> [1, 2, 3, 4] [3:1:-1]
[4, 3]
```

#### Pattern Matching

Variable declarations support pattern matching / destructuring. This takes the form of mirroring the iterable-like structure, like Python.

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

Pattern matching is supported in:

- Variable declarations (`let` statements)
- Expressions, if all the variables are already declared
- The variable declaration in a `for` loop
- Function arguments (when surrounded with parenthesis)


```rust
// In expressions
let a, b, c

a = b, c = (1, 2) // assigns a = (1, 2), b = 1, c = 2

// In 'for' loops
for x, y in 'hello' . enumerate {}

// In function arguments
fn flip((x, y)) -> (y, x)
```

#### Decorators

Named functions can optionally be *decorated*, which is a way to modify the function in-place, without having to reassign to it. A decorator consists of a `@` followed by an expression, before the function is declared:

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
fn run(f) -> f()

@run
fn do_stuff() { print('hello world') } // prints 'hello world' immediately, and assigns `do_stuff` to `nil`
```

#### Assertions

The `assert` keyword can be used to raise an error, or assert a condition is true. Note that runtime errors in cordy are **unrecoverable**, meaning if this assertion fails, the program will effectively call `exit`. An assert statement consists of `assert <expression>`, optionally followed by `: <expression>`, where the second expression will be used in the error message.

```java
assert false // Errors with 'Assertion Failed: nil'

assert false : 'Oh no!' // Errors with 'Assertion Failed: Oh no!
```

Assertions will point to the expression in question being asserted:

```
Assertion Failed: message goes here
  at: line 1 (<test>)
  
1 | assert false : 'message goes here'
2 |        ^^^^^
```

#### Structs and Modules

Cordy can allow the user to define their own type via _structs_. These are a type with a fixed set of values that can be accessed with the field access operator `->`. They can be used to program in an object-oriented style within Cordy.

A struct is declared with the `struct` keyword, a struct name, and a list of field names. These names - unlike variable declarations - may shadow native function names or other variable names:

```rust
>>> struct Point(x, y)
```

New instances of a struct can be created by invoking the struct name as a function:

```rust
>>> let p = Point(2, 3)
>>> p
Point(x=2, y=3)
```

Fields can be individually accessed or mutated with the `->` operator:

```rust
>>> p->x
2
>>> p->x = 5
5
>>> p
Point(x=2, y=3)
```

When declaring a struct, it can be optionally followed by an _implementation_ block. This block of code can define functions which are local to the struct - called _struct methods_.

```rust
struct Point(x, y) {
    fn zero() -> Point(0, 0)
}
```

These functions can be invoked as fields on the _struct name_ itself:

```rust
>>> Point->zero()
Point(x=0, y=0)
```

Note that struct methods can optionally take a `self` keyword as their first parameter. This creates _instance methods_, which can be invoked on instances of the struct. These have a few important properties:

- The `self` keyword is a local variable which accesses the instance itself. It can be used to access fields, or call other instance methods on the struct.
- Within an instance method, fields and other instance methods can be referenced _without_ a field access - these will be bound at compile time to the correct method.
- When calling an instance method, the `self` parameter is automatically filled from the instance - it does not need to be provided explicitly.
- Instance methods can be accessed from the struct type itself, in which case they are treated as normal methods that take the instance as their first parameter.

**Examples:**

```rust
struct Point(x, y) {
    fn norm1(self) {
        abs(self->x) + abs(y) // Fields can be accessed with or without 'self->'
    }

    fn is_norm1_smaller_than(self, d) {
        norm1() < d // Methods can be called, where 'self->' is implicit.
    }
}

let p = Point(3, 5)

p->norm1() // Returns 8
Point->norm1(p) // Returns 8

p->is_norm1_smaller_than(10) // Returns 'true'
```

**Modules** are like structs, but with a few key differences:

- They do not have fields.
- They are not able to be constructed into instances.
- They cannot have `self` methods.

```rust
module Powers {
    fn square(x) -> x ** 2
    fn cube(x) -> x ** 3
}

Powers->square(3) // Returns 9
Powers->cube(4) // Returns 64
```

#### Native Modules and FFI

Cordy has a basic FFI (Foreign Function Interface) that can be used to interface with external C-compatible libraries. In order to do this, _native modules_ are required. A native module is like a normal module with a few differences:

- The keyword `native` comes before `module`
- Any functions may **not** have implementations - either block statements or expressions
- Function parameters are restricted - pattern matching is not supported (although optional, default, and variadic arguments are).

```rust
// In a file my_ffi_test.cor
native module my_ffi_module {
    fn hello_world()
}

my_ffi_module->hello_world()
```

When compiling `cordy`, for each native module present in the Cordy code, it will need to provide a `--link` (or `-l`) argument in order to link the native module, with a compile shared object library:

```bash
$ cordy --link my_ffi_module=my_ffi_lib.so 
```

This native module must then export the symbol `hello_world` as a function - compatible with C. In order to do this, and facilitate the passing of values to and from native code, there is a [header](https://github.com/alcatrazEscapee/Cordy/blob/main/cordy/cordy.h) which can be used to create a C-compatible function. This provides a number of types and macros to make writing Cordy FFI functions easier.

**Example:**

```c
// In a file my_ffi_lib.c
#include <stdio.h>
#include "cordy.h"

CORDY_EXTERN(hello_world) {
    printf("Hello World!\n");
    return NIL()
}
```

This can then be compiled and invoked with the `--link` parameter mentioned above:

```bash
$ gcc -shared -o my_ffi_lib.so my_ffi_lib.c
$ cordy --link my_ffi_module=my_ffi_lib.so my_ffi_test.cor
Hello World!
```
