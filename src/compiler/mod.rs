use std::rc::Rc;

use crate::compiler::parser::{Locals, ParserError, ParseRule};
use crate::compiler::scanner::ScanResult;
use crate::reporting::{ErrorReporter, ProvidesLineNumber};
use crate::vm::opcode::Opcode;
use crate::vm::value::FunctionImpl;
use crate::vm::error::RuntimeError;

use Opcode::{*};

pub mod scanner;
pub mod parser;

pub fn default() -> CompileResult {
    parser::default()
}

pub fn compile(source: &String, text: &String) -> Result<CompileResult, Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(text);
    if !scan_result.errors.is_empty() {
        let rpt: ErrorReporter = ErrorReporter::new(text, source);
        for error in &scan_result.errors {
            errors.push(rpt.format_scan_error(&error));
        }
        return Err(errors);
    }

    // Parse
    let compile_result: CompileResult = parser::parse(scan_result);
    if !compile_result.errors.is_empty() {
        let rpt: ErrorReporter = ErrorReporter::new(text, source);
        for error in &compile_result.errors {
            errors.push(rpt.format_parse_error(&error));
        }
        return Err(errors);
    }

    // Compilation Successful
    Ok(compile_result)
}

/// Performs an incremental compile, given the following input parameters.
///
/// This is used for incremental REPL structure. The result will have a `Print` instead of a delayed pop (if needed), and end with a `Yield` instruction instead of `Exit`.
pub fn incremental_compile(source: &String, text: &String, code: &mut Vec<Opcode>, locals: &mut Vec<Locals>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, line_numbers: &mut Vec<u16>, globals: &mut Vec<String>) -> IncrementalCompileResult {
    // Stage changes, so an error or aborted compile doesn't overwrite the current valid compile state
    // Doing this for code and line numbers (since those are 1-1 ordering) is sufficient
    let code_len: usize = code.len();
    let line_numbers_len: usize = line_numbers.len();

    let ret: IncrementalCompileResult = try_incremental_compile(source, text, code, locals, strings, constants, functions, line_numbers, globals, parser::RULE_INCREMENTAL, true);

    if ret.is_success() {
        // Replace the final `Exit` with `Yield`, to yield control back to the REPL loop
        assert_eq!(Some(Exit), code.pop());
        code.push(Yield);
    } else {
        // Revert staged changes
        code.truncate(code_len);
        line_numbers.truncate(line_numbers_len);
    }

    ret
}

/// Performs an incremental compile, given the following input parameters.
///
/// The top level rule is `<expression>`, and the code will be appended to the end of the output.
/// Note that this does not insert a terminal `Pop` or `Exit`, and instead pushes a `Return` which exits `eval`'s special call frame.
///
/// This is the API used to run an `eval()` statement.
pub fn eval_compile(text: &String, code: &mut Vec<Opcode>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, line_numbers: &mut Vec<u16>, globals: &mut Vec<String>) -> Result<(), Box<RuntimeError>> {

    let mut locals: Vec<Locals> = Locals::empty();
    let ret: IncrementalCompileResult = try_incremental_compile(&String::from("<eval>"), text, code, &mut locals, strings, constants, functions, line_numbers, globals, parser::RULE_EXPRESSION, false);

    if ret.is_success() {
        // Insert a `Return` at the end, to return out of `eval`'s frame
        code.push(Return);
        Ok(())
    } else {
        RuntimeError::RuntimeCompilationError(ret.errors()).err()
    }
}



fn try_incremental_compile(source: &String, text: &String, code: &mut Vec<Opcode>, locals: &mut Vec<Locals>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, line_numbers: &mut Vec<u16>, globals: &mut Vec<String>, rule: ParseRule, abort_in_eof: bool) -> IncrementalCompileResult {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(text);
    if !scan_result.errors.is_empty() {
        let rpt: ErrorReporter = ErrorReporter::new(text, source);
        for error in &scan_result.errors {
            if error.is_eof() && abort_in_eof && errors.is_empty() {
                return IncrementalCompileResult::Aborted;
            }
            errors.push(rpt.format_scan_error(&error));
        }
        return IncrementalCompileResult::Errors(errors);
    }

    // Parse
    let parse_errors: Vec<ParserError> = parser::parse_incremental(scan_result, code, locals, strings, constants, functions, line_numbers, globals, rule);
    if !parse_errors.is_empty() {
        let rpt: ErrorReporter = ErrorReporter::new(text, source);
        for error in &parse_errors {
            if error.is_eof() && abort_in_eof && errors.is_empty() {
                return IncrementalCompileResult::Aborted;
            }
            errors.push(rpt.format_parse_error(&error));
        }
        return IncrementalCompileResult::Errors(errors);
    }

    IncrementalCompileResult::Success
}


pub struct CompileResult {
    pub code: Vec<Opcode>,
    pub errors: Vec<ParserError>,

    pub strings: Vec<String>,
    pub constants: Vec<i64>,
    pub functions: Vec<Rc<FunctionImpl>>,

    pub line_numbers: Vec<u16>,
    pub locals: Vec<String>,
    pub globals: Vec<String>,
}

impl CompileResult {

    pub fn disassemble(self: &Self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut width: usize = 0;
        let mut longest: usize = (self.line_numbers.last().unwrap_or(&0) + 1) as usize;
        while longest > 0 {
            width += 1;
            longest /= 10;
        }

        let mut last_line_no: u16 = 0;
        let mut locals = self.locals.iter();
        for (ip, token) in self.code.iter().enumerate() {
            let line_no = self.line_number(ip);
            let label: String = if line_no + 1 != last_line_no {
                last_line_no = line_no + 1;
                format!("L{:0>width$}: ", line_no + 1, width = width)
            } else {
                " ".repeat(width + 3)
            };
            let asm: String = match token {
                Int(cid) => format!("Int({}) -> {}", cid, self.constants[*cid as usize]),
                Str(sid) => format!("Str({}) -> {:?}", sid, self.strings[*sid as usize]),
                List(cid) => format!("List({}) -> {}", cid, self.constants[*cid as usize]),
                Function(fid) => format!("Function({}) -> {:?}", fid, self.functions[*fid as usize]),
                t @ (PushGlobal(_, _) | StoreGlobal(_, _) | PushLocal(_) | StoreLocal(_)) => {
                    if let Some(local) = locals.next() {
                        format!("{:?} -> {}", t, local)
                    } else {
                        format!("{:?}", t)
                    }
                },
                t => format!("{:?}", t),
            };
            lines.push(format!("{}{:0>4} {}", label, ip % 10_000, asm));
        }
        lines
    }
}


#[derive(Debug, Clone)]
pub enum IncrementalCompileResult {
    Aborted,
    Errors(Vec<String>),
    Success,
}

impl IncrementalCompileResult {
    fn is_success(self: &Self) -> bool {
        match self {
            IncrementalCompileResult::Success => true,
            _ => false
        }
    }

    fn errors(self: Self) -> Vec<String> {
        match self {
            IncrementalCompileResult::Errors(e) => e,
            _ => panic!("Tried to unwrap {:?}", self),
        }
    }
}

