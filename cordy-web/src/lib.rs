use std::cell::RefCell;
use std::io;
use std::io::Write;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use cordy_sys::{compiler, SourceView};
use cordy_sys::compiler::ScanTokenType;

use cordy_sys::repl::{ReadResult, Repl, RunResult};


#[allow(dead_code)]
#[wasm_bindgen(getter_with_clone)]
pub struct RunResultJs {
    pub exit: bool,
    pub lines: String,
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
    let view = SourceView::new(String::new(), input);
    let scan = compiler::scan(&view);
    let mut output: String = String::with_capacity(view.text().len());
    let mut chars = view.text().chars();
    let mut index = 0; // The next character index to be consumed

    for (loc, token) in scan {
        if index < loc.start() {
            output.push_str("[[;#aaa;]"); // Anything between tokens must be whitespace, or a comment. So style them all as comments
            while index < loc.start() { // Consume up until this token
                output.push(chars.next().unwrap());
                index += 1;
            }
            output.push(']');
        }

        let prefix: &'static str = match token {
            ScanTokenType::Keyword => "[[b;#b5f;]",
            ScanTokenType::Constant => "[[b;#27f;]",
            ScanTokenType::Native => "[[;#b80;]",
            ScanTokenType::Number => "[[;#385;]",
            ScanTokenType::String => "[[;#b10;]",
            ScanTokenType::Syntax => "",
        };

        output.push_str(prefix);

        while index <= loc.end() { // Consume the token itself
            output.push(chars.next().unwrap());
            index += 1;
        }

        let suffix: &'static str = match token {
            ScanTokenType::Syntax => "",
            _ => "]",
        };

        output.push_str(suffix);
    }

    // Consume any remaining characters
    output.push_str("[[;#777;]");
    for c in chars {
        output.push(c);
    }
    output.push(']');

    output
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

    let mut buffer = manager.writer.0.borrow_mut();
    let lines = unsafe {
        // Safe, because we know that the only thing that can be written via the VM is UTF-8
        String::from_utf8_unchecked(buffer.drain(..).collect())
    };

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


struct Manager {
    repl: Repl<SharedBufWriter>,
    writer: SharedBufWriter
}

impl Manager {
    fn new() -> Manager {
        let writer: SharedBufWriter = SharedBufWriter(Rc::new(RefCell::new(Vec::new())));
        Manager {
            repl: Repl::new(writer.clone(), false),
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