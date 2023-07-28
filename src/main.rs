use std::{fs, io};

use cordy::{compiler, repl};
use cordy::compiler::CompileResult;
use cordy::SourceView;
use cordy::vm::{ExitType, VirtualMachine};


fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (file, mode, enable_optimization, program_args) = match parse_args(args) {
        Some(args) => args,
        None => return
    };
    let result = match file {
        Some(name) => run_main(name, mode, enable_optimization, program_args),
        None => repl::run_repl()
    };
    match result {
        Ok(()) => {},
        Err(e) => eprintln!("{}", e)
    }
}

fn run_main(name: String, mode: Mode, enable_optimization: bool, program_args: Vec<String>) -> Result<(), String> {
    let text: String = fs::read_to_string(&name).map_err(|_| format!("Unable to read file '{}'", name))?;
    let view: SourceView = SourceView::new(name, text);
    let compiled: CompileResult = compiler::compile(enable_optimization, &view).map_err(|e| e.join("\n"))?;

    match mode {
        Mode::Disassembly => {
            for line in compiled.disassemble(&view) {
                println!("{}", line);
            }
            Ok(())
        },
        Mode::Default => run_vm(compiled, program_args, view)
    }
}

fn parse_args(args: Vec<String>) -> Option<(Option<String>, Mode, bool, Vec<String>)> {
    let mut iter = args.into_iter();
    let mut file: Option<String> = None;
    let mut enable_optimization: bool = false;
    let mut mode: Mode = Mode::Default;

    if iter.next().is_none() {
        panic!("Unexpected first argument");
    }

    for arg in iter.by_ref() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return None;
            },
            "-d" | "--disassembly" => mode.set(Mode::Disassembly).ok()?,
            "-o" | "--optimize" => {
                enable_optimization = true;
            },
            a => {
                file = Some(String::from(a));
                break
            },
        }
    }

    Some((file, mode, enable_optimization, iter.collect()))
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

fn print_help() {
    println!("cordy [options] <file> [program arguments...]");
    println!("When invoked with no arguments, this will open a REPL for the Cordy language (exit with 'exit' or Ctrl-C)");
    println!("Options:");
    println!("  -h --help        : Show this message and then exit.");
    println!("  -d --disassembly : Dump the disassembly view. Does nothing in REPL mode.");
    println!("  -o --optimize    : Enables compiler optimizations.");
}


#[derive(Eq, PartialEq)]
enum Mode { Default, Disassembly }

impl Mode {
    fn set(&mut self, new: Mode) -> Result<(), String> {
        if *self != Mode::Default {
            Err(String::from("Must only specify one of --disassembly"))
        } else {
            *self = new;
            Ok(())
        }
    }
}
