use std::{fs, io};
use std::io::Write;
use rustyline::Editor;
use rustyline::error::ReadlineError;

use cordy::compiler;
use cordy::compiler::{IncrementalCompileResult, Locals};
use cordy::reporting::SourceView;
use cordy::vm::{ExitType, VirtualMachine};


fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        // REPL Mode
        println!("Welcome to Cordy! (exit with 'exit' or Ctrl-C)");

        let name = &String::from("<stdin>");
        let mut editor = Editor::<()>::new().unwrap();
        let mut buffer: String = String::new();
        let mut continuation: bool = false;

        // Retain local variables through the entire lifetime of the REPL
        let mut locals = Locals::empty();
        let mut vm = VirtualMachine::new(compiler::default(), &b""[..], io::stdout());

        loop {
            io::stdout().flush().unwrap();
            let line: String = match editor.readline(if continuation { "... " } else { ">>> " }) {
                Ok(line) => {
                    editor.add_history_entry(line.as_str());
                    line
                },
                Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    break
                }
            };

            match line.as_str() {
                "" => continue,
                "#stack" => {
                    println!("{}", vm.debug_stack());
                    continue
                },
                "#call-stack" => {
                    println!("{}", vm.debug_call_stack());
                    continue
                },
                _ => {},
            }

            buffer.push_str(line.as_str());
            buffer.push('\n');
            continuation = false;

            let view: SourceView = SourceView::new(&name, &buffer);
            match vm.incremental_compile(&view, &mut locals) {
                IncrementalCompileResult::Success => {},
                IncrementalCompileResult::Errors(errors) => {
                    for e in errors {
                        println!("{}", e);
                    }
                    buffer.clear();
                    continue
                },
                IncrementalCompileResult::Aborted => {
                    continuation = true;
                    continue
                }
            }

            match vm.run_until_completion() {
                ExitType::Exit | ExitType::Return => return,
                ExitType::Yield => {},
                ExitType::Error(error) => println!("{}", view.format(&error)),
            }

            buffer.clear();
            vm.run_recovery(locals[0].len());
        }
    } else {
        // CLI Mode

        let mut iter = args.into_iter();
        let mut file: Option<String> = None;
        let mut disassembly: bool = false;
        let mut enable_optimization: bool = false;

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
                "-o" | "--optimize" => {
                    enable_optimization = true;
                }
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

        let name: &String = file.as_ref().unwrap();
        let text: String = match fs::read_to_string(name) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("Unable to read file '{}'", name);
                return;
            }
        };

        let view: SourceView = SourceView::new(name, &text);
        let compiled = match compiler::compile(enable_optimization, &view) {
            Ok(c) => c,
            Err(errors) => {
                for e in errors {
                    eprintln!("{}", e);
                }
                return
            }
        };

        if disassembly {
            for line in compiled.disassemble(&view) {
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

        if let ExitType::Error(error) = result {
            eprintln!("{}", view.format(&error));
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
