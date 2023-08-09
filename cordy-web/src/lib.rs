use std::cell::RefCell;
use std::io;
use std::io::Write;
use std::iter::Peekable;
use std::rc::Rc;
use std::str::Chars;
use wasm_bindgen::prelude::*;

use cordy_sys::{compiler, SourceView, SYS_VERSION};
use cordy_sys::compiler::ScanTokenType;
use cordy_sys::repl::{ReadResult, Repl, RunResult};


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

    // For multiline syntax (i.e. strings, comments), we need to scan the entire view thus far - which is the current REPL text, plus the input here
    // But, we don't want to *output* any of that - only output the new input
    let prefix: &String = Manager::get(&lock).repl.view().text();
    let mut full_input: String = prefix.clone();
    full_input.push_str(input.as_str());

    let view = SourceView::new(String::new(), full_input);
    let scan = compiler::scan(&view);
    let mut output: EscapedString = EscapedString::new(String::with_capacity(view.text().len()));
    let mut chars = input.chars().peekable();

    // The index of the next character of `chars` to be consumed
    // We start at the length of the `prefix`, so we don't output anything that occurred before the prefix
    let mut index = prefix.len();

    for (loc, token) in scan {
        if index > loc.end() {
            continue; // In the beginning, skip any tokens that end before the start of the current text
        }

        // Consume any leading whitespace and/or comments, leading up to the token
        consume_whitespace_and_comment(&mut chars, &mut output, &mut index, |_, index| *index < loc.start());

        // Don't escape the prefix
        output.inner.push_str(match token {
            ScanTokenType::Keyword => "[[b;#b5f;]",
            ScanTokenType::Constant => "[[b;#27f;]",
            ScanTokenType::Native => "[[;#b80;]",
            ScanTokenType::Type => "[[;#2aa;]",
            ScanTokenType::Number => "[[;#385;]",
            ScanTokenType::String => "[[;#b10;]",
            ScanTokenType::Syntax => "",
        });

        while index <= loc.end() { // Consume the token itself
            output.push(chars.next().unwrap());
            index += 1;
        }

        // Don't escape the suffix - all except a syntax generate a color and need to close it
        if !matches!(token, ScanTokenType::Syntax) {
            output.inner.push(']');
        }
    }

    // Consume any trailing whitespace or comment characters, after all tokens have been parsed
    consume_whitespace_and_comment(&mut chars, &mut output, &mut index, |c, _| c.peek().is_some());

    output.inner
}


fn consume_whitespace_and_comment<F>(chars: &mut Peekable<Chars>, output: &mut EscapedString, index: &mut usize, mut end: F)
    where F : FnMut(&mut Peekable<Chars>, &mut usize) -> bool {
    while let Some(' ' | '\t' | '\r' | '\n') = chars.peek() {
        output.push(chars.next().unwrap());
        *index += 1;
    }

    if end(chars, index) {
        output.inner.push_str("[[;#aaa;]");
        while end(chars, index) {
            output.push(chars.next().unwrap());
            *index += 1;
        }
        output.inner.push(']');
    }
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
    let mut lines = unsafe {
        // Safe, because we know that the only thing that can be written via the VM is UTF-8
        String::from_utf8_unchecked(buffer.drain(..).collect())
    };

    // Strip the trailing newline from the output, because of how the terminal renders, it will always expect to be there
    cordy_sys::util::strip_line_ending(&mut lines);

    RunResultJs { exit, lines }
}

/// Handles escaping control (formatting) characters in the output
struct EscapedString {
    inner: String
}

impl EscapedString {
    fn new(inner: String) -> EscapedString {
        EscapedString { inner }
    }

    fn push_str(&mut self, s: &str) {
        for c in s.chars() {
            self.push(c)
        }
    }

    fn push(&mut self, c: char) {
        match c {
            '[' => self.inner.push_str("&#91;"),
            '\\' => self.inner.push_str("&#92;"),
            ']' => self.inner.push_str("&#93;"),
            _ => self.inner.push(c),
        }
    }
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