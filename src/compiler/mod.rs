use crate::compiler::parser::ParserError;
use crate::compiler::scanner::ScanResult;
use crate::reporting::{ErrorReporter, ProvidesLineNumber};
use crate::stdlib;
use crate::vm::opcode::Opcode;
use crate::vm::value::FunctionImpl;

pub mod scanner;
pub mod parser;


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

    // Load Native Bindings
    let bindings = stdlib::bindings();

    // Parse
    let parse_result: CompileResult = parser::parse(bindings, scan_result);
    if !parse_result.errors.is_empty() {
        let rpt: ErrorReporter = ErrorReporter::new(text, source);
        for error in &parse_result.errors {
            errors.push(rpt.format_parse_error(&error));
        }
        return Err(errors);
    }

    // Compilation Successful
    Ok(parse_result)
}


pub struct CompileResult {
    pub code: Vec<Opcode>,
    pub errors: Vec<ParserError>,

    pub strings: Vec<String>,
    pub constants: Vec<i64>,
    pub functions: Vec<FunctionImpl>,

    pub line_numbers: Vec<u16>,
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
        for (ip, token) in self.code.iter().enumerate() {
            let line_no = self.line_number(ip);
            let label: String = if line_no + 1 != last_line_no {
                last_line_no = line_no + 1;
                format!("L{:0>width$}: ", line_no + 1, width = width)
            } else {
                " ".repeat(width + 3)
            };
            let asm: String = match token {
                Opcode::Int(cid) => format!("Int({} -> {})", cid, self.constants[*cid as usize]),
                Opcode::Str(sid) => format!("Str({} -> {:?})", sid, self.strings[*sid as usize]),
                Opcode::Function(fid) => format!("Function({} -> {:?})", fid, self.functions[*fid as usize]),
                t => format!("{:?}", t),
            };
            lines.push(format!("{}{:0>4} {}", label, ip % 10_000, asm));
        }
        lines
    }
}
