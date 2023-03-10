use std::rc::Rc;

use crate::compiler::parser::ParseRule;
use crate::compiler::scanner::ScanResult;
use crate::reporting::{Locations, SourceView};
use crate::vm::{FunctionImpl, Opcode, RuntimeError};

pub use crate::compiler::parser::{Locals, ParserError, ParserErrorType};
pub use crate::compiler::scanner::{ScanError, ScanErrorType, ScanToken};

use Opcode::{*};

mod scanner;
mod parser;

pub fn default() -> CompileResult {
    parser::default()
}

pub fn compile(enable_optimization: bool, view: &SourceView) -> Result<CompileResult, Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(view.text);
    if !scan_result.errors.is_empty() {
        for error in &scan_result.errors {
            errors.push(view.format(error));
        }
        return Err(errors);
    }

    // Parse
    let compile_result: CompileResult = parser::parse(enable_optimization, scan_result);
    if !compile_result.errors.is_empty() {
        for error in &compile_result.errors {
            errors.push(view.format(error));
        }
        return Err(errors);
    }

    // Compilation Successful
    Ok(compile_result)
}

/// Performs an incremental compile, given the following input parameters.
///
/// This is used for incremental REPL structure. The result will have a `Print` instead of a delayed pop (if needed), and end with a `Yield` instruction instead of `Exit`.
pub fn incremental_compile(view: &SourceView, code: &mut Vec<Opcode>, locals: &mut Vec<Locals>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, locations: &mut Locations
                           , globals: &mut Vec<String>) -> IncrementalCompileResult {
    // Stage changes, so an error or aborted compile doesn't overwrite the current valid compile state
    // Doing this for code and locations (since those are 1-1 ordering) is sufficient
    let code_len: usize = code.len();
    let locations_len: usize = locations.len();

    let ret: IncrementalCompileResult = try_incremental_compile(false, view, code, locals, strings, constants, functions, locations, globals, parser::RULE_REPL, true);

    if !ret.is_success() {
        // Revert staged changes
        code.truncate(code_len);
        locations.truncate(locations_len);
    }

    ret
}

/// Performs an incremental compile, given the following input parameters.
///
/// The top level rule is `<expression>`, and the code will be appended to the end of the output.
/// Note that this does not insert a terminal `Pop` or `Exit`, and instead pushes a `Return` which exits `eval`'s special call frame.
///
/// This is the API used to run an `eval()` statement.
pub fn eval_compile(text: &String, code: &mut Vec<Opcode>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, locations: &mut Locations, globals: &mut Vec<String>) -> Result<(), Box<RuntimeError>> {

    let mut locals: Vec<Locals> = Locals::empty();
    let name: String = String::from("<eval>");
    let view: SourceView = SourceView::new(&name, text);
    let ret: IncrementalCompileResult = try_incremental_compile(true, &view, code, &mut locals, strings, constants, functions, locations, globals, parser::RULE_EVAL, false);

    if ret.is_success() {
        Ok(())
    } else {
        RuntimeError::RuntimeCompilationError(ret.errors()).err()
    }
}



fn try_incremental_compile(enable_optimization: bool, view: &SourceView, code: &mut Vec<Opcode>, locals: &mut Vec<Locals>, strings: &mut Vec<String>, constants: &mut Vec<i64>, functions: &mut Vec<Rc<FunctionImpl>>, locations: &mut Locations, globals: &mut Vec<String>, rule: ParseRule, abort_in_eof: bool) -> IncrementalCompileResult {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(view.text);
    if !scan_result.errors.is_empty() {
        for error in &scan_result.errors {
            if error.is_eof() && abort_in_eof && errors.is_empty() {
                return IncrementalCompileResult::Aborted;
            }
            errors.push(view.format(error));
        }
        return IncrementalCompileResult::Errors(errors);
    }

    // Parse
    let parse_errors: Vec<ParserError> = parser::parse_incremental(enable_optimization, scan_result, code, locals, strings, constants, functions, locations, globals, rule);
    if !parse_errors.is_empty() {
        for error in &parse_errors {
            if error.is_eof() && abort_in_eof && errors.is_empty() {
                return IncrementalCompileResult::Aborted;
            }
            errors.push(view.format(error));
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

    pub locations: Locations,
    pub locals: Vec<String>,
    pub globals: Vec<String>,
}

impl CompileResult {

    pub fn disassemble(self: &Self, view: &SourceView) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut width: usize = 0;
        let mut longest: usize = view.len();
        while longest > 0 {
            width += 1;
            longest /= 10;
        }

        let mut last_line_no: usize = 0;
        let mut locals = self.locals.iter();
        for (ip, token) in self.code.iter().enumerate() {
            let loc = self.locations[ip];
            let line_no = view.lineno(loc);
            let label: String = if line_no + 1 != last_line_no {
                last_line_no = line_no + 1;
                format!("L{:0>width$}: ", line_no + 1, width = width)
            } else {
                " ".repeat(width + 3)
            };
            let asm: String = match *token {
                Int(cid) | CheckLengthEqualTo(cid) | CheckLengthGreaterThan(cid) => format!("{:?} -> {}", token, self.constants[cid as usize]),
                Str(sid) => format!("Str({}) -> {:?}", sid, self.strings[sid as usize]),
                Function(fid) => format!("Function({}) -> {:?}", fid, self.functions[fid as usize]),
                PushGlobal(_) | StoreGlobal(_) | PushLocal(_) | StoreLocal(_) => match locals.next() {
                    Some(local) => format!("{:?} -> {}", token, local),
                    None => format!("{:?}", token),
                },
                _ => format!("{:?}", token.to_absolute_jump(ip)),
            };
            lines.push(format!("{}{:0>4} {}", label, ip % 10_000, asm));
            // Debug for locations
            //lines.push(format!("{}{:0>4}[{:>4}{:>4}] {}", label, ip % 10_000, loc.start(), loc.end(), asm));
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

