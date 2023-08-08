use std::io;
use std::io::{BufRead, Read, Write};

use crate::{compiler, SourceView};
use crate::compiler::{IncrementalCompileResult, Locals};
use crate::vm::{ExitType, VirtualInterface, VirtualMachine};


/// A trait implementing a predictable, callback-based reader. This is the implementation used by the executable REPL
pub trait Reader {

    /// Returns the next line from the input.
    ///
    /// - `ReadResult::Exit` indicates the input is closed
    /// - `ReadResult::Error(e)` indicates an error occurred during reading, which should be propagated to the caller of `run()`
    /// - `ReadResult::Ok(line)` returns the next line from the input.
    fn read(&mut self, prompt: &'static str) -> ReadResult;
}

pub struct Repl<W: Write> {
    /// If `repeat_input` is true, everything written to input will be written directly back to output via the VM's `println` functions
    /// This is used for testing purposes, as the `writer` must be given solely to the VM for output purposes.
    repeat_input: bool,
    continuation: bool,
    locals: Vec<Locals>,
    vm: VirtualMachine<Empty, W>
}

struct Empty;

impl Read for Empty {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Ok(0)
    }
}

impl BufRead for Empty {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(&[])
    }

    fn consume(&mut self, _: usize) {}
}

/// A result returned from a read line operation.
pub enum ReadResult {
    Exit,
    Error(String),
    Ok(String),
}

/// A result returned from a single invocation of `Repl::run`
///
/// - `Exit` indicates the VM has exited normally
/// - `Error(e)` indicates the VM has encountered an unrecoverable error from input reading
/// - `Ok` indicates the VM ran successfully and is ready for further reads
pub enum RunResult {
    Exit,
    Error(String),
    Ok,
}

/// Create a new REPL, and invoke it in a loop with the given `Reader` until it is exhausted.
pub fn run<R : Reader, W: Write>(mut reader: R, writer: W, repeat_input: bool) -> Result<(), String> {
    let mut repl: Repl<W> = Repl::new(writer, repeat_input);
    loop {
        let read = reader.read(repl.prompt());
        match repl.run(read) {
            RunResult::Exit => break Ok(()),
            RunResult::Error(e) => break Err(e),
            RunResult::Ok => {},
        }
    }
}

impl<W: Write> Repl<W> {

    pub fn new(writer: W, repeat_input: bool) -> Repl<W> {
        let compile = compiler::default();
        let view = SourceView::new(String::from("<stdin>"), String::new());

        Repl {
            repeat_input,
            continuation: false,
            locals: Locals::empty(),
            vm: VirtualMachine::new(compile, view, Empty, writer, vec![])
        }
    }

    pub fn prompt(&self) -> &'static str {
        if self.continuation { "... " } else { ">>> " }
    }

    pub fn run(&mut self, input: ReadResult) -> RunResult {
        let line: String = match input {
            ReadResult::Ok(line) => {
                if self.repeat_input {
                    self.vm.println(format!("{}{}", self.prompt(), line))
                }
                line
            },
            ReadResult::Error(e) => return RunResult::Error(e),
            ReadResult::Exit => return RunResult::Exit,
        };

        match line.as_str() {
            "" => return RunResult::Ok,
            "#stack" => {
                self.vm.println(self.vm.debug_stack());
                return RunResult::Ok
            },
            "#call-stack" => {
                self.vm.println(self.vm.debug_call_stack());
                return RunResult::Ok
            },
            _ => {},
        }

        let buffer = self.vm.view_mut().text_mut();

        buffer.push_str(line.as_str());
        buffer.push('\n');
        self.continuation = false;

        match self.vm.incremental_compile(&mut self.locals) {
            IncrementalCompileResult::Success => {},
            IncrementalCompileResult::Errors(errors) => {
                for e in errors {
                    self.vm.println(e);
                }
                self.vm.view_mut().push(String::from("<stdin>"), String::new());
                return RunResult::Ok
            },
            IncrementalCompileResult::Aborted => {
                self.continuation = true;
                return RunResult::Ok
            }
        }

        match self.vm.run_until_completion() {
            ExitType::Exit | ExitType::Return => return RunResult::Exit,
            ExitType::Yield => {},
            ExitType::Error(error) => self.vm.println(self.vm.view().format(&error)),
        }

        self.vm.view_mut().push(String::from("<stdin>"), String::new());
        self.vm.run_recovery(self.locals[0].len());
        RunResult::Ok
    }
}


#[cfg(test)]
mod tests {
    use crate::repl;
    use crate::repl::{Reader, ReadResult};

    impl Reader for Vec<String> {
        fn read(self: &mut Self, _: &'static str) -> ReadResult {
            match self.pop() {
                None => ReadResult::Exit,
                Some(line) => ReadResult::Ok(line)
            }
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