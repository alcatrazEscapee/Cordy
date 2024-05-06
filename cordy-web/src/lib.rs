use std::cell::RefCell;
use std::io;
use std::io::Write;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use cordy_sys::{ScanTokenType, syntax, SYS_VERSION};
use cordy_sys::compiler::FunctionLibrary;
use cordy_sys::repl::{ReadResult, Repl, RunResult};
use cordy_sys::syntax::{BlockingFormatter, Formatter};
use cordy_sys::util::SharedWrite;
use cordy_sys::vm::{FunctionInterface, IntoValue, ValuePtr, ValueResult};


#[allow(dead_code)]
#[wasm_bindgen(getter_with_clone)]
pub struct RunResultJs {
    pub exit: bool,
    pub lines: String,
}

#[wasm_bindgen]
pub fn version() -> String {
    String::from(SYS_VERSION)
}

/// Initializes the console panic hook (the first time this is called), and initially creates the `Manager`
///
/// N.B. It is important that this function is invoked before any call to `prompt()` or `exec()` as otherwise it will panic.
#[wasm_bindgen]
pub fn load() {
    console_error_panic_hook::set_once();
    Manager::set();
}

/// Retrieves the current console prompt. This will either be `>>> ` or `... ` based on if we are currently in a continuation.
#[wasm_bindgen]
pub fn prompt() -> String {
    let lock = Lock;
    Manager::get(&lock).repl.prompt().to_string()
}

/// With a given input string, runs just the compiler's `scan` phase, and uses that to syntax-highlight the input text, and return it.
#[wasm_bindgen]
pub fn scan(input: String) -> String {
    let lock = Lock;
    let prefix: &String = Manager::get(&lock).repl.view().text();
    let mut fmt = BlockingFormatter::new(TerminalFormatter(String::with_capacity(input.len())));
    syntax::scan(input, prefix, &mut fmt);
    fmt.fmt.0
}


/// Executes the provided code string, and returns the results.
///
/// - `exit` will be `true` if the REPL was exited, via encountering a `exit` opcode.
/// - `lines` will contain the output printed to the console.
#[wasm_bindgen]
pub fn exec(input: String) -> RunResultJs {
    let lock = Lock;
    let manager = Manager::get(&lock);
    let mut exit: bool = false;

    match manager.repl.run(ReadResult::Ok(input)) {
        RunResult::Ok => {},
        RunResult::Exit => exit = true,
        RunResult::Error(e) => panic!("No read error, so should never receive a run error, {}", e)
    }

    let mut lines = unsafe {
        // Safe, because we know that the only thing that can be written via the VM is UTF-8
        String::from_utf8_unchecked(manager.writer.inner().borrow_mut().drain(..).collect())
    };

    // Strip the trailing newline from the output, because of how the terminal renders, it will always expect to be there
    cordy_sys::util::strip_line_ending(&mut lines);

    RunResultJs { exit, lines }
}


/// `static mut` is safe here because we're only accessing through JS which is inherently single threaded
static mut INSTANCE: Option<Manager> = None;

/// Since we borrow from a single, static mut in `Manager::get()`, we need a lifetime to control the lifetime of the borrow.
/// This is the simplest way around that, by creating a arbitrary `Lock` which we then borrow.
///
/// Note that this is **not** safe, since we could just create multiple locks, but it's sane enough for our use cases.
struct Lock;

/// A `Write` implementor that can be shared between the VM, and then drained/cleared when we want to pass on the written output to JS
/// The VM needs to own an instance, therefor we need `Rc<RefCell<>>` for interior mutability. The `Manager` owns a second instance it uses to move
/// the written text to the output.
#[derive(Debug, Clone)]
struct SharedBufWriter(Rc<RefCell<Vec<u8>>>);

impl Write for SharedBufWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(())
    }
}


struct TerminalFormatter(String);

impl Formatter for TerminalFormatter {
    fn begin(&mut self, ty: ScanTokenType) {
        self.0.push_str(match ty {
            ScanTokenType::Keyword => "[[b;#b5f;]",
            ScanTokenType::Constant => "[[b;#27f;]",
            ScanTokenType::Native => "[[;#b80;]",
            ScanTokenType::Type => "[[;#2aa;]",
            ScanTokenType::Number => "[[;#385;]",
            ScanTokenType::String => "[[;#b10;]",
            ScanTokenType::Comment => "[[;#aaa;]",
            _ => "",
        });
    }

    fn end(&mut self, ty: ScanTokenType) {
        self.0.push_str(match ty {
            ScanTokenType::Syntax |
            ScanTokenType::Blank => "",
            _ => "]"
        });
    }

    fn push(&mut self, c: char) {
        match c {
            '[' => self.0.push_str("&#91;"),
            '\\' => self.0.push_str("&#92;"),
            ']' => self.0.push_str("&#93;"),
            _ => self.0.push(c),
        }
    }
}


struct JsInterface;

impl FunctionInterface for JsInterface {
    fn handle(&mut self, library: &FunctionLibrary, handle_id: u32, args: Vec<ValuePtr>) -> ValueResult {
        let function = library.lookup(handle_id);
        let _ = Manager::get(&Lock).writer.write(format!("called native function '{}->{}{}'\n", function.module_name, function.method_name, args.to_value().to_repr_str()).as_bytes());
        ValuePtr::nil().ok()
    }
}


struct Manager {
    repl: Repl<SharedWrite, JsInterface>,
    writer: SharedWrite
}

impl Manager {
    fn new() -> Manager {
        let writer: SharedWrite = SharedWrite::new();
        Manager {
            repl: Repl::new(writer.clone(), JsInterface),
            writer,
        }
    }

    fn set() {
        unsafe {
            INSTANCE = Some(Manager::new());
        }
    }

    fn get(_: &Lock) -> &mut Manager {
        unsafe {
            INSTANCE.as_mut().unwrap()
        }
    }
}