use std::{fs, io};

use cordy::{compiler, repl};
use cordy::compiler::CompileResult;
use cordy::SourceView;
use cordy::vm::{ExitType, VirtualMachine};


fn main() {
    let args: Vec<String> = std::env::args().collect();
    match if args.len() == 1 {
        repl::run_repl()
    } else {
        run_main(args)
    } {
        Ok(()) => {},
        Err(e) => eprintln!("{}", e)
    }
}

fn run_main(args: Vec<String>) -> Result<(), String> {
    let (name, mode, enable_optimization, program_args) = match parse_args(args) {
        Ok(args) => args,
        Err(None) => return Ok(()),
        Err(Some(err)) => return Err(err),
    };


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
        Mode::Default => run_vm(compiled, program_args, Some(view))
    }
}

fn parse_args(args: Vec<String>) -> Result<(String, Mode, bool, Vec<String>), Option<String>> {
    let mut iter = args.into_iter();
    let mut file: Option<String> = None;
    let mut enable_optimization: bool = false;
    let mut mode: Mode = Mode::Default;

    if let None = iter.next() {
        return Err(Some(String::from("Unexpected first argument")));
    }

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Err(None);
            },
            "-d" | "--disassembly" => mode.set(Mode::Disassembly)?,
            "-o" | "--optimize" => {
                enable_optimization = true;
            },
            a => {
                file = Some(String::from(a));
                break
            },
        }
    }

    Ok((file.ok_or_else(|| String::from("No file specified"))?, mode, enable_optimization, iter.collect()))
}

fn run_vm(compiled: CompileResult, program_args: Vec<String>, view: Option<SourceView>) -> Result<(), String> {

    let stdin = io::stdin().lock();
    let stdout = io::stdout();
    let mut vm = VirtualMachine::new(compiled, view.unwrap(), stdin, stdout, program_args);

    match vm.run_until_completion() {
        ExitType::Error(error) => Err(format!("{}", vm.view().format(&error))),
        _ => Ok(())
    }
}

fn print_help() {
    println!("cordy [options] <file> [program arguments...]");
    println!("When invoked with no arguments, this will open a REPL for the Cordy language (exit with 'exit' or Ctrl-C)");
    println!("Options:");
    println!("  -h --help        : Show this message and then exit");
    println!("  -d --disassembly : Dump the disassembly view. Use -o to dump to a file.");
    println!("  -o --optimize    : Enables compiler optimizations");
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
