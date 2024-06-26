use std::io::{Write};

use crate::{compiler, SourceView};
use crate::compiler::{IncrementalCompileResult, Locals};
use crate::util::Noop;
use crate::vm::{ExitType, FunctionInterface, VirtualInterface, VirtualMachine};


/// A trait implementing a predictable, callback-based reader. This is the implementation used by the executable REPL
pub trait Reader {

    /// Returns the next line from the input.
    ///
    /// - `ReadResult::Exit` indicates the input is closed
    /// - `ReadResult::Error(e)` indicates an error occurred during reading, which should be propagated to the caller of `run()`
    /// - `ReadResult::Ok(line)` returns the next line from the input.
    fn read(&mut self, prompt: &'static str) -> ReadResult;
}

pub struct Repl<W: Write, F : FunctionInterface> {
    continuation: bool,
    locals: Vec<Locals>,
    vm: VirtualMachine<Noop, W, F>
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
pub fn run<R : Reader, W: Write, F: FunctionInterface>(mut reader: R, writer: W, ffi: F) -> Result<(), String> {
    let mut repl: Repl<W, F> = Repl::new(writer, ffi);
    loop {
        let read = reader.read(repl.prompt());
        match repl.run(read) {
            RunResult::Exit => break Ok(()),
            RunResult::Error(e) => break Err(e),
            RunResult::Ok => {},
        }
    }
}

impl<W: Write, F : FunctionInterface> Repl<W, F> {

    pub fn new(writer: W, ffi: F) -> Repl<W, F> {
        let compile = compiler::default();
        let view = SourceView::new(String::from("<stdin>"), String::new());

        Repl {
            continuation: false,
            locals: Locals::empty(),
            vm: VirtualMachine::new(compile, view, Noop, writer, ffi)
        }
    }

    pub fn view(&self) -> &SourceView {
        self.vm.view()
    }

    pub fn prompt(&self) -> &'static str {
        if self.continuation { "... " } else { ">>> " }
    }

    pub fn run(&mut self, input: ReadResult) -> RunResult {
        let line: String = match input {
            ReadResult::Ok(line) => line,
            ReadResult::Error(e) => return RunResult::Error(e),
            ReadResult::Exit => return RunResult::Exit,
        };

        if line.as_str() == "" && !self.continuation {
            return RunResult::Ok;
        }

        let buffer = self.vm.view_mut().text_mut();

        buffer.push_str(line.as_str());
        buffer.push('\n');
        self.continuation = false;

        match self.vm.incremental_compile(&mut self.locals) {
            IncrementalCompileResult::Success => {},
            IncrementalCompileResult::Errors(errors) => {
                for e in errors {
                    self.vm.println(e.as_str());
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
            ExitType::Error(error) => self.vm.println(self.vm.view().format(&error).as_str()),
        }

        self.vm.view_mut().push(String::from("<stdin>"), String::new());
        self.vm.run_recovery(self.locals[0].len());
        RunResult::Ok
    }
}


#[cfg(test)]
mod tests {
    use std::io::{Write};
    use crate::{repl, util};
    use crate::repl::{Reader, ReadResult};
    use crate::util::{Noop, SharedWrite};

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
monitor('stack')", "\
>>> let x = nil, y = 123, z = 'hello world'
>>> monitor('stack')
[nil, 123, 'hello world', fn monitor(cmd)]
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
'stack'.monitor
", "\
>>> print + 1
TypeError: Operator '+' is not supported for arguments of type native function and int
  at: line 1 (<stdin>)

1 | print + 1
2 |       ^

>>> print(1)
1
nil
>>> 'stack'.monitor
[fn monitor(cmd)]
")}

    #[test] fn test_error_in_declared_function() { run("\
fn foo() -> print + 1
foo()
", "\
>>> fn foo() -> print + 1
>>> foo()
TypeError: Operator '+' is not supported for arguments of type native function and int
  at: line 1 (<stdin>)
  at: `fn foo()` (line 1)

1 | fn foo() -> print + 1
2 |                   ^

")}

    #[test] fn test_locals_are_retained_between_incremental_compiles() { run("\
let x = 2, y = 4
monitor('stack')
x
x + y
y += x
let z = y * 2
z
monitor('stack')
", "\
>>> let x = 2, y = 4
>>> monitor('stack')
[2, 4, fn monitor(cmd)]
>>> x
2
>>> x + y
6
>>> y += x
6
>>> let z = y * 2
>>> z
12
>>> monitor('stack')
[2, 6, 12, fn monitor(cmd)]
")}

    #[test] fn test_unterminated_strings_and_block_comments_cause_continuations() { run("\
/* block
comment */
'long
string'
", "\
>>> /* block
... comment */
>>> 'long
... string'
long
string
")}

    // Note that \n\ is used to indicate a newline, with trailing space, so IDE doesn't nuke the trailing spaces
    #[test] fn test_continuation_of_empty_lines_in_string_literal() { run("\
'


' . repr
", "\
>>> '
... \n\
... \n\
... ' . repr
'\\n\\n\\n'
")}

    struct MockReader(Vec<String>, SharedWrite);

    impl Reader for MockReader {
        fn read(self: &mut Self, prompt: &'static str) -> ReadResult {
            match self.0.pop() {
                None => ReadResult::Exit,
                Some(line) => {
                    writeln!(self.1, "{}{}", prompt, line.as_str()).unwrap();
                    ReadResult::Ok(line)
                }
            }
        }
    }

    fn run(inputs: &'static str, outputs: &'static str) {
        let writer: SharedWrite = SharedWrite::new();
        let reader: MockReader = MockReader(
            inputs.lines()
                .rev() // rev() because we pop from the end, but list them sequentially.
                .map(String::from)
                .collect(),
            writer.clone()
        );
        let result = repl::run(reader, writer.clone(), Noop);

        assert!(result.is_ok());
        util::assert_eq(String::from_utf8(writer.inner().borrow().clone()).unwrap(), String::from(outputs));
    }
}