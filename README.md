### Cordy

Cordy is a dynamically typed, interpreted, stack-based, semi-functional / semi-procedural language I made up. It is meant to be a quick scripting language for solving puzzles and other fun things as an alternative to Python.

### Grammar

```

<program> := <statement> *

<statement>
    | <block-statement>
    | <function-statement>
    | <let-statement>
    | <if-statement>
    | <loop-statement>
    | <for-statement>
    | <assignment-statement>
    | <expression-statement>
    | `break`
    | `continue`

<block-statement> := `{` <statement> * `}`

<function-statement> := `fn` <name> <function-parameters> ? <function-type> ? <block-statement>

<function-parameters> := `(` <function-parameter> [ `,` <function-parameter> ] * `)`
<function-parameter> := <name> [ `:` <type> ] ?
<function-type> := `:` <type>

<let-statement> := `let` <pattern> <let-type> ? <let-assignment> ?

<let-type> := `:` <type>
<let-assignment> := `=` <expression>

<if-statement> := `if` <expression> <block-statement> <if-elif-statement> * <if-else-statement> ?

<if-elif-statement> := `elif` <block-statement>
<if-else-statement> := `else` <block-statement>

<loop-statement> := `loop` <block-statement>

<for-statement> := `for` <pattern> `in` <expression> <block-statement>

<assignment-statement> := `<pattern> <assignment-operator> <expression>

<expression-statement> := <expression>


The complicated expression-like recursive guts

<type>
    | `[` <type> `]`
    | `(` <type> [ `,` <type> ] * `)`
    | `int`
    | `str`

<pattern>
    | <name>
    | `(` <pattern> [ `,` <pattern> ] * `)`
    | `_`

<expression>
    | <expression> <expression-suffix>
    | <expression> <binary-operator> <expression>
    | <unary-operator> <expression>

<expression-suffix>
    | `(` <expression> [ `,` <expression> ] * `)`
    | `[` <expression> [ `:` <expression> [ `:` <expression> ] ? ] ? `]`


Detailed View of <expression>

Precedence Table:

1  | `()` `[]`                   | Function application or array access
2  | `!` `~` `-`                 | Logical and Bitwise NOT, Negation
3  | `*` `/` `%`                 | Multiplication, Division, Modulo
4  | `+` `-`                     | Addition, Subtraction
5  | `>>` `<<`                   | Bitwise Shifts
7  | `&` `|` `^`                 | Bitwise AND, OR, XOR
8  | `.`                         | Function composition
6  | `<` `>` `<=` `>=` `==` `!=` | Comparison operators
9  | `and` `or`                  | Logical (Short Circuiting) AND, OR

<expression> := <expr-9>

<expr-1>
    | <number>
    | <name>
    | <string>
    | `true`
    | `false`
    | `nil`
    | `(` <expression> `)`

<expr-2> := [ `!` | `~` | `-` ] <expr-2> <expr-suffix>
<expr-3> := <expr-2> ( [ `*` | `/` | `%` ] <expr-2> ) *
<expr-4> := <expr-3> ( [ `+` | `-` ] <expr-3> ) *
<expr-5> := <expr-4> ( [ `>>` | `<<` ] <expr-4> ) *
<expr-6> := <expr-5> ( [ `<` | `>` | `<=` | `>=` | `==` | `!=` ] <expr-5> ) *
<expr-7> := <expr-6> ( [ `&` | `|` | `^` ] <expr-6> ) *
<expr-8> := <expr-7> ( `.` <expr-7> ) *
<expr-9> := <expr-8> ( [ `&&` | `||` ] <expr-8> )

<expr-suffix>
    | `(` <expression> [ `,` <expression> ] * `)`
    | `[` <expression> [ `:` <expression> [ `:` <expression> ] ? ] ?

```



### Enabling Debug Traces

Compile with the environment variable `RUSTFLAGS` set to

```
--cfg trace_parser="off" --cfg trace_interpreter="off" --cfg trace_interpreter_stack="off"
```

And replace the values for the traces you want to enable with `on`:

- `trace_parser` traces the parser execution, logging tokens accepted, pushed, and rules entered.
- `trace_interpreter` traces the virtual machine execution, logging every instruction executed.

### To-Do

- Implement `->` binding resolution
- ~~Global~~ and Local (block scoped) variables
- ~~List types~~ (and list literals)
- Tuple types (and tuple literals), if we need them
- Set types
- Map types and map literals
- Function types
- Polymorphic 'List' types that can be used as arbitrary iterators, etc. using rust traits rather than dispatch at every call site
- Primitive operator partial evaluation `map (+ 3)` kinda deal
- Functional wankery (map, filter, max, min, reduce) that operate on polymorphic lists
- Deque / Queue type
- Structs (fancy named tuples)
- Format strings
- Regex (some sort of impl)
- Pattern matching / deconstruction
- Even MORE standard library functions
- User defined functions
- Lambdas
- Closures
- VS Code: https://code.visualstudio.com/api/language-extensions/overview