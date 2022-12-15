use std::{fs, io};

use crate::reporting::ErrorReporter;
use crate::vm::VirtualMachine;

pub mod compiler;
pub mod stdlib;
pub mod vm;

pub mod trace;
pub mod reporting;


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
            Ok(c) => c,
            Err(errors) => {
                for e in errors {
                    eprintln!("{}", e);
                }
                return
            }
        };

        let result = {
            let stdin = io::stdin().lock();
            let stdout = io::stdout();
            let mut vm = VirtualMachine::new(compiled, stdin, stdout);
            vm.run()
        };
        match result {
            Err(e) => {
                eprintln!("{}", ErrorReporter::new(&text, file_ref).format_runtime_error(&e))
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
