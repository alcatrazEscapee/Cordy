use crate::compiler::parser::ParserResult;
use crate::compiler::scanner::ScanResult;
use crate::reporting::ErrorReporter;
use crate::stdlib;

pub mod scanner;
pub mod parser;


pub fn compile(source: &String, text: &String) -> Result<ParserResult, Vec<String>> {
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
    let parse_result: ParserResult = parser::parse(bindings, scan_result);
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