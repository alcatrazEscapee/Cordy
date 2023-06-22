use std::rc::Rc;

use crate::compiler::parser::ParseRule;
use crate::compiler::scanner::ScanResult;
use crate::reporting::{Locations, SourceView};
use crate::vm::{FunctionImpl, Opcode, RuntimeError, StructTypeImpl};

pub use crate::compiler::parser::{Locals, Fields, ParserError, ParserErrorType, default};
pub use crate::compiler::scanner::{ScanError, ScanErrorType, ScanToken};

use Opcode::{*};

mod scanner;
mod parser;


pub fn compile(enable_optimization: bool, view: &SourceView) -> Result<CompileResult, Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(view.text());
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
pub fn incremental_compile(view: &SourceView, mut params: CompileParameters) -> IncrementalCompileResult {
    // Stage changes, so an error or aborted compile doesn't overwrite the current valid compile state
    let state: CompileState = params.save();
    let ret: IncrementalCompileResult = try_incremental_compile(view, &mut params, parser::RULE_REPL, true);

    if !ret.is_success() { // Revert staged changes
        params.restore(state);
    }

    ret
}

/// Performs an incremental compile, given the following input parameters.
///
/// The top level rule is `<expression>`, and the code will be appended to the end of the output.
/// Note that this does not insert a terminal `Pop` or `Exit`, and instead pushes a `Return` which exits `eval`'s special call frame.
///
/// This is the API used to run an `eval()` statement.
pub fn eval_compile(text: &String, mut params: CompileParameters) -> Result<(), Box<RuntimeError>> {

    let name: String = String::from("<eval>");
    let view: SourceView = SourceView::new(&name, text);
    let ret: IncrementalCompileResult = try_incremental_compile(&view, &mut params, parser::RULE_EVAL, false);

    ret.as_ok_or_runtime_error()
}



fn try_incremental_compile(view: &SourceView, params: &mut CompileParameters, rule: ParseRule, abort_in_eof: bool) -> IncrementalCompileResult {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(view.text());
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
    let parse_errors: Vec<ParserError> = parser::parse_incremental(scan_result, params, rule);
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

pub struct CompileParameters<'a> {
    enable_optimization: bool,

    code: &'a mut Vec<Opcode>,
    locals: &'a mut Vec<Locals>,
    fields: &'a mut Fields,
    strings: &'a mut Vec<String>,
    constants: &'a mut Vec<i64>,
    functions: &'a mut Vec<Rc<FunctionImpl>>,
    structs: &'a mut Vec<Rc<StructTypeImpl>>,
    locations: &'a mut Locations,
    globals: &'a mut Vec<String>,
}

/// This is a cloned, static version of `CompileParameters`. The sole purpose is to create a save state, and restore after.
/// This save-restore is used during incremental compiles that get aborted.
///
/// - `locals`, `fields` are mutable and require the full state to be saved.
/// - `code`, `strings`, `constants`, `structs`, `locations`, `globals` are append-only, and thus we can optimize by only saving the length, and restoring by truncating.
/// - `functions` are only modified during teardown, during conversion from un-baked functions, to baked functions, and so no save/restore state is needed
struct CompileState {
    code: usize,
    locals: Vec<Locals>,
    fields: Fields,
    strings: usize,
    constants: usize,
    structs: usize,
    locations: usize,
    globals: usize,
}

impl<'a> CompileParameters<'a> {

    pub fn new(
        enable_optimization: bool,
        code: &'a mut Vec<Opcode>,
        locals: &'a mut Vec<Locals>,
        fields: &'a mut Fields,
        strings: &'a mut Vec<String>,
        constants: &'a mut Vec<i64>,
        functions: &'a mut Vec<Rc<FunctionImpl>>,
        structs: &'a mut Vec<Rc<StructTypeImpl>>,
        locations: &'a mut Locations,
        globals: &'a mut Vec<String>
    ) -> CompileParameters<'a> {
        CompileParameters { enable_optimization, code, locals, fields, strings, constants, functions, structs, locations, globals }
    }

    fn save(self: &Self) -> CompileState {
        CompileState {
            code: self.code.len(),
            locals: self.locals.clone(),
            fields: self.fields.clone(),
            strings: self.strings.len(),
            constants: self.constants.len(),
            structs: self.structs.len(),
            locations: self.locations.len(),
            globals: self.globals.len(),
        }
    }

    fn restore(self: &mut Self, state: CompileState) {
        self.code.truncate(state.code);
        *self.locals = state.locals;
        *self.fields = state.fields;
        self.strings.truncate(state.strings);
        self.constants.truncate(state.constants);
        self.structs.truncate(state.structs);
        self.locations.truncate(state.locations);
        self.globals.truncate(state.globals);
    }
}


pub struct CompileResult {
    pub code: Vec<Opcode>,
    pub errors: Vec<ParserError>,

    pub strings: Vec<String>,
    pub constants: Vec<i64>,
    pub functions: Vec<Rc<FunctionImpl>>,
    pub structs: Vec<Rc<StructTypeImpl>>,

    pub locations: Locations,
    pub locals: Vec<String>,
    pub globals: Vec<String>,
    pub fields: Fields,
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
                GetField(fid) | SetField(fid) | GetFieldFunction(fid) => format!("{:?} -> {}", token, self.fields.get_field_name(fid)),
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
    fn as_ok_or_runtime_error(self: Self) -> Result<(), Box<RuntimeError>> {
        match self {
            IncrementalCompileResult::Success => Ok(()),
            IncrementalCompileResult::Errors(e) => RuntimeError::RuntimeCompilationError(e).err(),
            _ => panic!("{:?} should not be unboxed as a Result<(), Box<RuntimeError>>", self),
        }
    }

    fn is_success(self: &Self) -> bool {
        match self {
            IncrementalCompileResult::Success => true,
            _ => false
        }
    }
}

