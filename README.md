### Cordy

Cordy is a dynamically typed, interpreted, stack-based, semi-functional / semi-procedural language I made up. It is meant to be a quick scripting language for solving puzzles and other fun things as an alternative to Python.

### Enabling Debug Traces

Compile with the environment variable `RUSTFLAGS` set to

```
--cfg trace_parser="off" --cfg trace_interpreter="off" --cfg trace_interpreter_stack="off"
```

And replace the values for the traces you want to enable with `on`:

- `trace_parser` traces the parser execution, logging tokens accepted, pushed, and rules entered.
- `trace_interpreter` traces the virtual machine execution, logging every instruction executed.

### To-Do

- Finish implementing binary operators
- More stdlib functions (int, etc)
- Implement `->` binding resolution
- Global and Local (block scoped) variables
- Partial function evaluation
- List types (and list literals)
- Tuple types (and tuple literals)
- Set types
- Map types
- Format strings
- Regex (some sort of impl)
- Pattern matching / deconstruction
- Even MORE standard library functions
- User defined functions
- Lambdas
- Closures