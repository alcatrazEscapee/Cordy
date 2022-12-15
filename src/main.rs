use std::fs;
use crate::error_reporter::ErrorReporter;

use crate::vm::VirtualMachine;


mod compiler;
mod stdlib;
mod vm;

pub mod trace;
pub mod error_reporter;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.is_empty() {
        // REPL Mode
    } else {
        // CLI Mode

        let mut iter = args.into_iter();
        let mut file: Option<String> = None;

        if let None = iter.next() {
            eprintln!("Unexpected first argument");
            return;
        }

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" => {
                    print_help();
                    return;
                },
                a => {
                    file = Some(String::from(a));
                    break
                },
            }
        }

        if let Some(arg) = iter.next() {
            eprintln!("Unrecognized argument: {}", arg);
            return;
        }

        if file.is_none() {
            eprintln!("No file specified.");
            return;
        }

        let file_ref: &String = file.as_ref().unwrap();
        let text: String = match fs::read_to_string(file_ref) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("Unable to read file '{}'", file_ref);
                return;
            }
        };

        let compiled = match compiler::compile(file_ref, &text) {
            Some(t) => t,
            None => return
        };

        let mut vm: VirtualMachine = VirtualMachine::new(compiled.code);
        match vm.run() {
            Err(e) => {
                eprintln!("Error!\n\n{}", ErrorReporter::new(&text, file_ref).format_runtime_error(&e))
            },
            Ok(_) => {}
        }
    }
}

fn print_help() {
    println!("aocli [options] <file>");
    println!("Command line interface, compiler and interpreter for the AoCL language.");
    println!("Options:");
    println!("  --help : Show this message and then exit");
}
