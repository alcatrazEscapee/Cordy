use std::{fs, io};
use std::collections::HashSet;
use std::io::Write;
use rustyline::{DefaultEditor, Editor};
use rustyline::error::ReadlineError;
use mimalloc::MiMalloc;

use cordy_sys::{compiler, repl, ScanTokenType, SourceView, syntax, SYS_VERSION};
use cordy_sys::compiler::CompileResult;
use cordy_sys::repl::{Reader, ReadResult};
use cordy_sys::syntax::{BlockingFormatter, Formatter};
use cordy_sys::vm::{ExitType, VirtualMachine};


#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;


const HELP_MESSAGE: &'static str = "\
cordy [options] <file> [program arguments...]
When invoked with no arguments, this will open a REPL for the Cordy language (exit with 'exit' or Ctrl-C)
Options:
  -h --help             : Show this message, then exit.
  -v --version          : Print the version, then exit.
  -d --disassembly      : Dump the disassembly view. Does nothing in REPL mode.
     --no-line-numbers  : In disassembly view, omits the leading '0001' style line numbers
  -o --optimize         : Enables compiler optimizations and transformations.
  -f --format           : Outputs a formatted view HTML view of the code
     --format-no-style  : Omits the <style> tag containing color definitions for the formatted code
     --format-no=...    : Omits any of the <span> tags for the given categories (comma seperated) of token.
                          Categories are any of [keyword, constant, native, type, number, string, syntax, comment]
";

const FORMAT_COLORS: &'static str = "\
<style>
    p.cordy { font-family: monospace; }
    span.cordy-keyword { color: #b5f; font-weight: bold; }
    span.cordy-constant { color: #27f; font-weight: bold; }
    span.cordy-native { color: #b80; }
    span.cordy-type { color: #2aa; }
    span.cordy-number { color: #385; }
    span.cordy-string { color: #b10; }
    span.cordy-syntax { }
    span.cordy-comment { color: #aaa; }
</style>
";


fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut options: Options = match parse_args(args) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{}", err);
            return;
        }
    };
    match options.mode {
        Mode::Help => {
            print_help();
            return;
        },
        Mode::Version => {
            print_version();
            return;
        },
        _ => {}
    }
    let result = match options.file.take() {
        Some(name) => run_main(name, options),
        None => run_repl()
    };
    match result {
        Ok(()) => {},
        Err(e) => eprintln!("{}", e)
    }
}

fn parse_args(args: Vec<String>) -> Result<Options, String> {
    let mut iter = args.into_iter();
    let mut options: Options = Options {
        file: None,
        args: Vec::new(),
        mode: Mode::Default,
        optimize: false,
        no_line_numbers: false,
        format_no_style: false,
        format_no: HashSet::new(),
    };

    if iter.next().is_none() {
        panic!("Unexpected first argument");
    }

    for arg in iter.by_ref() {
        match arg.as_str() {
            "-h" | "--help" => options.mode.set(Mode::Help)?,
            "-v" | "--version" => options.mode.set(Mode::Version)?,
            "-d" | "--disassembly" => options.mode.set(Mode::Disassembly)?,
            "-f" | "--format" => options.mode.set(Mode::Format)?,
            "-o" | "--optimize" => options.optimize = true,
            "--no-line-numbers" => options.no_line_numbers = true,
            "--format-no-style" => options.format_no_style = true,
            a if a.starts_with("--format-no=") => {
                for key in a.strip_prefix("--format-no=")
                    .unwrap()
                    .split(',') {
                    options.format_no.insert(match key {
                        "keyword" => ScanTokenType::Keyword,
                        "constant" => ScanTokenType::Constant,
                        "native" => ScanTokenType::Native,
                        "type" => ScanTokenType::Type,
                        "number" => ScanTokenType::Number,
                        "string" => ScanTokenType::String,
                        "syntax" => ScanTokenType::Syntax,
                        "comment" => ScanTokenType::Comment,
                        key => return Err(format!("Unrecognized argument to --format-no={}", key)),
                    });
                }
            },
            a => {
                options.file = Some(String::from(a));
                break
            },
        }
    }

    options.args.extend(iter);
    Ok(options)
}

fn print_help() {
    println!("{}", HELP_MESSAGE);
}

fn print_version() {
    println!("Cordy v{}", SYS_VERSION);
}

fn run_main(name: String, options: Options) -> Result<(), String> {
    let text: String = fs::read_to_string(&name).map_err(|_| format!("Unable to read file '{}'", name))?;

    if options.mode == Mode::Format {
        let mut fmt = BlockingFormatter::new(RenderedFormatter { inner: String::new(), skip: options.format_no });
        syntax::scan(text, &String::new(), &mut fmt);
        if !options.format_no_style {
            println!("{}", FORMAT_COLORS);
        }
        println!("<code class=\"cordy\"><pre>{}</pre></code>", fmt.fmt.inner);
        return Ok(())
    }

    let view: SourceView = SourceView::new(name, text);
    let compiled: CompileResult = compiler::compile(options.optimize, &view).map_err(|e| e.join("\n"))?;

    match options.mode {
        Mode::Disassembly => {
            for line in compiled.disassemble(&view, !options.no_line_numbers) {
                println!("{}", line);
            }
            Ok(())
        },
        Mode::Default => run_vm(compiled, options.args, view),
        _ => panic!("Unsupported mode"),
    }
}

fn run_vm(compiled: CompileResult, program_args: Vec<String>, view: SourceView) -> Result<(), String> {

    let stdin = io::stdin().lock();
    let stdout = io::stdout();
    let mut vm = VirtualMachine::new(compiled, view, stdin, stdout, program_args);

    match vm.run_until_completion() {
        ExitType::Error(error) => Err(vm.view().format(&error)),
        _ => Ok(())
    }
}

pub fn run_repl() -> Result<(), String> {
    println!("Welcome to Cordy v{}! (exit with 'exit' or Ctrl-C)", SYS_VERSION);
    repl::run(EditorRepl { editor: Editor::new().unwrap() }, io::stdout(), false)
}


struct EditorRepl {
    editor: DefaultEditor
}

impl Reader for EditorRepl {
    fn read(&mut self, prompt: &'static str) -> ReadResult {
        io::stdout().flush().unwrap();
        match self.editor.readline(prompt) {
            Ok(line) => {
                self.editor.add_history_entry(line.as_str()).unwrap();
                ReadResult::Ok(line)
            },
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => ReadResult::Exit,
            Err(e) => ReadResult::Error(format!("Error: {}", e)),
        }
    }
}

struct Options {
    file: Option<String>,
    args: Vec<String>,
    mode: Mode,
    optimize: bool,
    no_line_numbers: bool,
    format_no_style: bool,
    format_no: HashSet<ScanTokenType>,
}

#[derive(Eq, PartialEq)]
enum Mode { Default, Help, Version, Disassembly, Format }

impl Mode {
    fn set(&mut self, new: Mode) -> Result<(), String> {
        if *self != Mode::Default {
            Err(String::from("Must only specify one of --help, --version, --disassembly, or --format"))
        } else {
            *self = new;
            Ok(())
        }
    }
}

struct RenderedFormatter {
    inner: String,
    skip: HashSet<ScanTokenType>
}

impl Formatter for RenderedFormatter {
    fn begin(&mut self, ty: ScanTokenType) {
        if ty != ScanTokenType::Blank {
            if !self.skip.contains(&ty) {
                self.inner.push_str("<span class=\"cordy-");
                self.inner.push_str(match ty {
                    ScanTokenType::Keyword => "keyword",
                    ScanTokenType::Constant => "constant",
                    ScanTokenType::Native => "native",
                    ScanTokenType::Type => "type",
                    ScanTokenType::Number => "number",
                    ScanTokenType::String => "string",
                    ScanTokenType::Syntax => "syntax",
                    ScanTokenType::Blank => "",
                    ScanTokenType::Comment => "comment",
                });
                self.inner.push_str("\">");
            }
        }
    }

    fn end(&mut self, ty: ScanTokenType) {
        if !self.skip.contains(&ty) {
            self.inner.push_str("</span>")
        }
    }

    fn push(&mut self, c: char) {
        match c {
            '\r' => {},
            '\t' => for _ in 0..4 { self.push(' ') },
            '<' => self.inner.push_str("&lt;"),
            '>' => self.inner.push_str("&gt;"),
            '"' => self.inner.push_str("&quot;"),
            '\'' => self.inner.push_str("&#39;"),
            '&' => self.inner.push_str("&amp;"),
            _ => self.inner.push(c),
        }
    }
}
