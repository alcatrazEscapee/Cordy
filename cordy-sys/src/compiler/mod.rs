use indexmap::IndexSet;
use crate::compiler::parser::ParseRule;
use crate::compiler::scanner::ScanResult;
use crate::reporting::{Location, SourceView};
use crate::vm::{AnyResult, Opcode, RuntimeError, StoreOp, ValuePtr};
use crate::core::Pattern;

pub use crate::compiler::parser::{default, Fields, FunctionLibrary, Locals, ParserError, ParserErrorType};
pub use crate::compiler::scanner::{ScanError, ScanErrorType, ScanToken, ScanTokenType};


mod scanner;
mod parser;

pub fn scan(view: &SourceView) -> Vec<(Location, ScanTokenType)> {
    scanner::scan(view).tokens
        .into_iter()
        .map(|(loc, t)| (loc, t.ty()))
        .collect()
}

#[cfg(feature = "test_parser_verification")]
pub fn compile(enable_optimization: bool, view: &SourceView) -> Result<CompileResult, Vec<String>> {
    let mut buffer = String::new();
    for c in view.text().chars() {
        buffer.push(c); // Insert one character
        let view = SourceView::new(view.name().clone(), buffer.clone());
        let _ = compile_internal(enable_optimization, &view); // And then just prove that the compile passes
    }
    compile_internal(enable_optimization, view)
}

#[cfg(not(feature = "test_parser_verification"))]
pub fn compile(enable_optimization: bool, view: &SourceView) -> Result<CompileResult, Vec<String>> {
    compile_internal(enable_optimization, view)
}

fn compile_internal(enable_optimization: bool, view: &SourceView) -> Result<CompileResult, Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(view);
    if !scan_result.errors.is_empty() {
        for error in &scan_result.errors {
            errors.push(view.format(error));
        }
        return Err(errors);
    }

    // Parse
    let compile: CompileResult = parser::parse(enable_optimization, scan_result);
    if !compile.errors.is_empty() {
        for error in &compile.errors {
            errors.push(view.format(error));
        }
        return Err(errors);
    }

    // Compilation Successful
    Ok(compile)
}

/// Performs an incremental compile, given the following input parameters.
///
/// This is used for incremental REPL structure. The result will have a `Print` instead of a delayed pop (if needed), and end with a `Yield` instruction instead of `Exit`.
pub fn incremental_compile(mut params: CompileParameters) -> IncrementalCompileResult {
    // Stage changes, so an error or aborted compile doesn't overwrite the current valid compile state
    let state: CompileState = params.save();
    let ret: IncrementalCompileResult = try_incremental_compile(&mut params, |parser| parser.parse_incremental_repl(), true);

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
pub fn eval_compile(text: String, mut params: CompileParameters) -> AnyResult {
    params.view.push(String::from("<eval>"), text);
    try_incremental_compile(&mut params, |parser| parser.parse_incremental_eval(), false)
        .ok_or_runtime_error()
}



fn try_incremental_compile(params: &mut CompileParameters, rule: ParseRule, abort_in_eof: bool) -> IncrementalCompileResult {
    let mut errors: Vec<String> = Vec::new();

    // Scan
    let scan_result: ScanResult = scanner::scan(params.view);
    if !scan_result.errors.is_empty() {
        for error in &scan_result.errors {
            if error.is_eof() && abort_in_eof && errors.is_empty() {
                return IncrementalCompileResult::Aborted;
            }
            errors.push(params.view.format(error));
        }
        return IncrementalCompileResult::Errors(errors);
    }

    // Parse
    let parse_errors: IndexSet<ParserError> = parser::parse_incremental(scan_result, params, rule);
    if !parse_errors.is_empty() {
        for error in &parse_errors {
            if error.is_eof() && abort_in_eof && errors.is_empty() {
                return IncrementalCompileResult::Aborted;
            }
            errors.push(params.view.format(error));
        }
        return IncrementalCompileResult::Errors(errors);
    }

    IncrementalCompileResult::Success
}

pub struct CompileParameters<'a> {
    enable_optimization: bool,

    code: &'a mut Vec<Opcode>,

    constants: &'a mut Vec<ValuePtr>,
    patterns: &'a mut Vec<Pattern<StoreOp>>,
    globals: &'a mut Vec<String>,
    locations: &'a mut Vec<Location>,
    fields: &'a mut Fields,
    functions: &'a mut FunctionLibrary,

    locals: &'a mut Vec<Locals>,
    view: &'a mut SourceView,
}

/// This is a cloned, static version of `CompileParameters`. The sole purpose is to create a save state, and restore after.
/// This save-restore is used during incremental compiles that get aborted.
///
/// - `locals`, `fields` are mutable and require the full state to be saved.
/// - `code`, `constants`, `patterns`, `locations`, `globals` are append-only, and thus we can optimize by only saving the length, and restoring by truncating.
struct CompileState {
    code: usize,

    constants: usize,
    patterns: usize,
    globals: usize,
    locations: usize,
    fields: Fields,
    functions: (usize, usize),

    locals: Vec<Locals>,
}

impl<'a> CompileParameters<'a> {

    pub fn new(
        enable_optimization: bool,
        code: &'a mut Vec<Opcode>,
        constants: &'a mut Vec<ValuePtr>,
        patterns: &'a mut Vec<Pattern<StoreOp>>,
        globals: &'a mut Vec<String>,
        locations: &'a mut Vec<Location>,
        fields: &'a mut Fields,
        functions: &'a mut FunctionLibrary,
        locals: &'a mut Vec<Locals>,
        view: &'a mut SourceView,
    ) -> CompileParameters<'a> {
        CompileParameters { enable_optimization, code, constants, patterns, globals, locations, fields, functions, locals, view }
    }

    fn save(&self) -> CompileState {
        CompileState {
            code: self.code.len(),
            constants: self.constants.len(),
            patterns: self.patterns.len(),
            globals: self.globals.len(),
            locations: self.locations.len(),
            fields: self.fields.clone(),
            functions: self.functions.len(),
            locals: self.locals.clone(),
        }
    }

    fn restore(&mut self, state: CompileState) {
        self.code.truncate(state.code);
        self.constants.truncate(state.constants);
        self.patterns.truncate(state.patterns);
        self.globals.truncate(state.globals);
        self.locations.truncate(state.locations);
        *self.fields = state.fields;
        self.functions.truncate(state.functions);
        *self.locals = state.locals;
    }
}


#[derive(Debug, Clone)]
pub struct CompileResult {
    pub code: Vec<Opcode>,

    /// Errors returned by the parser/semantic/codegen stage of the compiler.
    /// Since `parser::parse()` returns a `CompileResult`, these errors are checked in the various public interface methods on `compiler`.
    /// Incremental compiles will return a `IndexSet<ParserError>` instead as they don't own the structures to create a `CompileResult`.
    errors: IndexSet<ParserError>,

    pub constants: Vec<ValuePtr>,
    pub patterns: Vec<Pattern<StoreOp>>,
    pub globals: Vec<String>,
    pub locations: Vec<Location>,
    pub fields: Fields,
    pub functions: FunctionLibrary,

    /// Local variable names, by order of access (either `Push` or `Store` local/global opcodes) in the output code.
    /// This is only used for the decompiler to report local variable names. Otherwise these are discarded before passing to the VM
    locals: Vec<String>,
}

impl CompileResult {

    /// `line_numbers` : If true, then the `0001` style line numbers will be included in the output. Turning this off is useful when diffing two outputs.
    pub fn disassemble(&self, view: &SourceView, line_numbers: bool) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut width: usize = 0;
        let mut longest: usize = view.len();
        while longest > 0 {
            width += 1;
            longest /= 10;
        }

        let mut last_line_no: usize = usize::MAX;
        let mut locals = self.locals.iter().cloned();
        for (ip, opcode) in self.code.iter().enumerate() {
            let loc = self.locations[ip];
            let line_no = view.lineno(loc).unwrap_or(last_line_no);
            let label: String = if line_no != last_line_no {
                last_line_no = line_no;
                format!("L{:0>width$}: ", line_no + 1, width = width)
            } else {
                " ".repeat(width + 3)
            };
            let asm: String = opcode.disassembly(ip, &mut locals, &self.fields, &self.constants);
            lines.push(format!("{}{} {}", label, if line_numbers { format!("{:0>4}", ip % 10_000) } else { String::new() }, asm));
        }
        lines
    }

    /// Outputs the raw disassembly view, used for testing
    /// This would emit a sequence of `\n` seperated opcodes, i.e. `Int(1)\nInt(2)\nAdd`
    #[cfg(test)]
    pub fn raw_disassembly(self: Self) -> String {
        use itertools::Itertools;

        let mut locals = self.locals.iter().cloned();
        self.code
            .iter()
            .enumerate()
            .map(|(ip, op)| op.disassembly(ip, &mut locals, &self.fields, &self.constants)
                .replace(" ", "")) // This replacement is the easiest solution to a test DSL problem where we split instructions by " "
            .join("\n")
    }
}


#[derive(Debug, Clone)]
pub enum IncrementalCompileResult {
    Aborted,
    Errors(Vec<String>),
    Success,
}

impl IncrementalCompileResult {
    fn ok_or_runtime_error(self) -> AnyResult {
        match self {
            IncrementalCompileResult::Success => Ok(()),
            IncrementalCompileResult::Errors(e) => RuntimeError::RuntimeCompilationError(e).err(),
            _ => panic!("{:?} should not be unboxed", self),
        }
    }

    fn is_success(&self) -> bool {
        matches!(self, IncrementalCompileResult::Success)
    }
}

