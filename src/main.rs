use std::{fs, io};
use std::io::{BufRead, Write};

use crate::reporting::ErrorReporter;
use crate::vm::VirtualMachine;

pub mod compiler;
pub mod stdlib;
pub mod vm;

pub mod trace;
pub mod reporting;


fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        // REPL Mode
        println!("Welcome to Cordy! (exit with 'exit' or Ctrl-C)");

        let source = &String::from("<stdin>");
        let mut buffer: String = String::new();
        let mut continuation: bool = false;

        loop {
            {
                if continuation { print!(". ") } else { print!("> ") }
                continuation = false;

                io::stdout().flush().unwrap();
                io::stdin().lock().read_line(&mut buffer).unwrap();

                if buffer.ends_with('\n') {
                    buffer.pop();
                    if buffer.ends_with('\r') {
                        buffer.pop();
                    }
                }
                if buffer.ends_with("\\") {
                    continuation = true;
                    continue
                }
                if buffer.ends_with("exit") {
                    break
                }
            }

            // This is the hackiest REPL you have ever seen
            buffer = format!("({}) . print", buffer);

            let compiled = match compiler::compile(&source, &buffer) {
                Ok(c) => c,
                Err(errors) => {
                    for e in errors {
                        println!("{}", e);
                    }
                    buffer.clear();
                    continue
                }
            };

            let result = {
                let stdin = io::stdin().lock();
                let stdout = io::stdout();
                let mut vm = VirtualMachine::new(compiled, stdin, stdout);
                vm.run_until_completion()
            };
            match result {
                Err(e) => {
                    println!("{}", ErrorReporter::new(&buffer, source).format_runtime_error(&e))
                },
                Ok(_) => {}
            }

            buffer.clear();
            io::stdout().flush().unwrap();
        }
    } else {
        // CLI Mode

        let mut iter = args.into_iter();
        let mut file: Option<String> = None;
        let mut disassembly: bool = false;

        if let None = iter.next() {
            eprintln!("Unexpected first argument");
            return;
        }

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_help();
                    return;
                },
                "-d" | "--disassembly" => {
                    disassembly = true;
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

        if disassembly {
            for line in compiled.disassemble() {
                println!("{}", line);
            }
            return;
        }

        let result = {
            let stdin = io::stdin().lock();
            let stdout = io::stdout();
            let mut vm = VirtualMachine::new(compiled, stdin, stdout);
            vm.run_until_completion()
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
    println!("cordy [options] <file>");
    println!("When invoked with no arguments, this will open a REPL for the Cordy language (exit with 'exit' or Ctrl-C)");
    println!("Options:");
    println!("  -h --help        : Show this message and then exit");
    //println!("  -c --compile     : Compile only. Outputs the compiled program to <file>.o by default, or what the -o option specifies");
    println!("  -d --disassembly : Dump the disassembly view. Use -o to dump to a file.");
    //println!("  -e --execute     : Execute the provided file as a compiled binary (with -c), rather than compling a text file");
    //println!("  -o --output      : Output file to write to. Used by -d")
}
