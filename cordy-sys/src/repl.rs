use std::io::Write;

use crate::{compiler, SourceView};
use crate::compiler::{IncrementalCompileResult, Locals};
use crate::vm::{ExitType, VirtualInterface, VirtualMachine};


pub trait Repl {
    /// Reads a line from input.
    /// On the target implementation this also flushes any previous buffered output.
    ///
    /// - `Some(Ok(String))` indicates a string was read
    /// - `Some(Err(String))` indicates an error was raised, and should be raised
    /// - `None` indicates the interface was closed.
    fn read(&mut self, prompt: &'static str) -> Option<Result<String, String>>;
}

/// If `repeat_input` is true, everything written to input will be written directly back to output via the VM's `println` functions
/// This is used for testing purposes, as the `writer` must be given solely to the VM for output purposes.
pub fn run<R : Repl, W: Write>(mut reader: R, writer: W, repeat_input: bool) -> Result<(), String> {
    let mut continuation: bool = false;

    // Retain local variables through the entire lifetime of the REPL
    let view: SourceView = SourceView::new(String::from("<stdin>"), String::new());
    let mut locals = Locals::empty();
    let mut vm = VirtualMachine::new(compiler::default(), view, &b""[..], writer, vec![]);

    loop {
        let prompt: &'static str = if continuation { "... " } else { ">>> " };
        let line: String = match reader.read(prompt) {
            Some(Ok(line)) => {
                if repeat_input {
                    vm.println(format!("{}{}", prompt, line))
                }
                line
            },
            Some(Err(e)) => return Err(e),
            None => return Ok(()),
        };

        match line.as_str() {
            "" => continue,
            "#stack" => {
                vm.println(vm.debug_stack());
                continue
            },
            "#call-stack" => {
                vm.println(vm.debug_call_stack());
                continue
            },
            _ => {},
        }

        let buffer = vm.view_mut().text_mut();

        buffer.push_str(line.as_str());
        buffer.push('\n');
        continuation = false;


        match vm.incremental_compile(&mut locals) {
            IncrementalCompileResult::Success => {},
            IncrementalCompileResult::Errors(errors) => {
                for e in errors {
                    vm.println(e);
                }
                vm.view_mut().push(String::from("<stdin>"), String::new());
                continue
            },
            IncrementalCompileResult::Aborted => {
                continuation = true;
                continue
            }
        }

        match vm.run_until_completion() {
            ExitType::Exit | ExitType::Return => return Ok(()),
            ExitType::Yield => {},
            ExitType::Error(error) => vm.println(vm.view().format(&error)),
        }

        vm.view_mut().push(String::from("<stdin>"), String::new());
        vm.run_recovery(locals[0].len());
    }
}


#[cfg(test)]
mod tests {
    use crate::repl;
    use crate::repl::Repl;

    impl Repl for Vec<String> {
        fn read(self: &mut Self, _: &'static str) -> Option<Result<String, String>> {
            self.pop().map(|u| Ok(u))
        }
    }

    #[test] fn test_hello_world() { run("\
let text = 'hello world'
print(text)", "\
>>> let text = 'hello world'
>>> print(text)
hello world
nil
")}

    #[test] fn test_simple_continuation() { run("\
if 2 > 1 {
    print('that is to be expected')
}", "\
>>> if 2 > 1 {
...     print('that is to be expected')
... }
that is to be expected
")}

    #[test] fn test_debug_stack() { run("\
let x = nil, y = 123, z = 'hello world'
#stack", "\
>>> let x = nil, y = 123, z = 'hello world'
>>> #stack
: ['hello world': str, 123: int, nil: nil]
")}

    #[test] fn test_declare_and_exec_function() { run("\
fn foo(what) {
    print('yes', what)
}
foo
foo . repr
foo('bob')
", "\
>>> fn foo(what) {
...     print('yes', what)
... }
>>> foo
foo
>>> foo . repr
fn foo(what)
>>> foo('bob')
yes bob
nil
")}

    #[test] fn test_eval() { run("\
eval('1 + 2')
eval('fn() { let x = 1, y = 2 ; x + y }()')
", "\
>>> eval('1 + 2')
3
>>> eval('fn() { let x = 1, y = 2 ; x + y }()')
3
")}

    #[test] fn test_error_and_continue() { run("\
print + 1
print(1)
#stack
", "\
>>> print + 1
TypeError: Cannot add 'print' of type 'native function' and '1' of type 'int'
  at: line 1 (<stdin>)

1 | print + 1
2 |       ^

>>> print(1)
1
nil
>>> #stack
: []
")}

    #[test] fn test_error_in_declared_function() { run("\
fn foo() -> print + 1
foo()
", "\
>>> fn foo() -> print + 1
>>> foo()
TypeError: Cannot add 'print' of type 'native function' and '1' of type 'int'
  at: line 1 (<stdin>)
  at: `fn foo()` (line 1)

1 | fn foo() -> print + 1
2 |                   ^

")}

    #[test] fn test_locals_are_retained_between_incremental_compiles() { run("\
let x = 2, y = 4
#stack
x
x + y
y += x
let z = y * 2
z
#stack
", "\
>>> let x = 2, y = 4
>>> #stack
: [4: int, 2: int]
>>> x
2
>>> x + y
6
>>> y += x
6
>>> let z = y * 2
>>> z
12
>>> #stack
: [12: int, 6: int, 2: int]
")}

    fn run(inputs: &'static str, outputs: &'static str) {
        let repl: Vec<String> = inputs.lines()
            .rev() // rev() because we pop from the end, but list them sequentially.
            .map(String::from)
            .collect();
        let mut buf: Vec<u8> = Vec::new();
        let result = repl::run(repl, &mut buf, true);

        assert!(result.is_ok());
        assert_eq!(String::from_utf8(buf).unwrap(), String::from(outputs));
    }
}