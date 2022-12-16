use crate::compiler::parser::{ParserError, ParserErrorType};
use crate::compiler::scanner::{ScanError, ScanErrorType, ScanToken};
use crate::stdlib;
use crate::stdlib::StdBinding;
use crate::vm::{Opcode, RuntimeError, RuntimeErrorType};

const FORMAT_RESET: &'static str = ""; //\x1B[0m";
const FORMAT_BOLD: &'static str = ""; //\x1B[1m";
const FORMAT_RED: &'static str = ""; // \x1B[31m";


pub struct ErrorReporter<'a> {
    lines: Vec<&'a str>,
    src: &'a String,
}

impl<'a> ErrorReporter<'a> {
    pub fn new(text: &'a String, src: &'a String) -> ErrorReporter<'a> {
        ErrorReporter {
            lines: text.lines().collect(),
            src
        }
    }

    pub fn format_scan_error(self: &Self, error: &ScanError) -> String {
        format_scan_error(&self.lines, self.src, error)
    }

    pub fn format_parse_error(self: &Self, error: &ParserError) -> String {
        format_parse_error(&self.lines, self.src, error)
    }

    pub fn format_runtime_error(self: &Self, error: &RuntimeError) -> String {
        format_runtime_error(&self.lines, self.src, error)
    }
}

pub fn format_scan_error(source_lines: &Vec<&str>, source_file: &String, error: &ScanError) -> String {
    let mut text: String = format!("{}{}{}{}", FORMAT_RED, FORMAT_BOLD, error.format_error(), FORMAT_RESET);
    text.push_str(format!("\n  at: line {} ({})\n  at:\n\n", error.lineno + 1, source_file).as_str());
    text.push_str(source_lines.get(error.lineno).map(|t| *t).unwrap_or(""));
    text.push('\n');
    text
}

pub fn format_parse_error(source_lines: &Vec<&str>, source_file: &String, error: &ParserError) -> String {
    let mut text: String = format!("{}{}{}{}", FORMAT_RED, FORMAT_BOLD, error.format_error(), FORMAT_RESET);
    text.push_str(format!("\n  at: line {} ({})\n  at:\n\n", error.lineno + 1, &source_file).as_str());
    text.push_str(source_lines.get(error.lineno).map(|t| *t).unwrap_or(""));
    text.push('\n');
    text
}

pub fn format_runtime_error(source_lines: &Vec<&str>, source_file: &String, error: &RuntimeError) -> String {
    let mut text: String = format!("{}{}{}{}", FORMAT_RED, FORMAT_BOLD, error.format_error(), FORMAT_RESET);
    text.push_str(format!("\n  at: line {} ({})\n  at:\n\n", error.lineno + 1, &source_file).as_str());
    text.push_str(source_lines.get(error.lineno).map(|t| *t).unwrap_or(""));
    text.push('\n');
    text
}


trait AsError {
    fn format_error(self: &Self) -> String;
}


impl AsError for RuntimeError {
    fn format_error(self: &Self) -> String {
        match &self.error {
            RuntimeErrorType::ValueIsNotFunctionEvaluable(v) => format!("Tried to evaluate '{}' of type '{}' but it is not a function.", v.as_type_str(), v.as_str()),
            RuntimeErrorType::BindingIsNotFunctionEvaluable(b) => format!("Tried to evaluate '{}' but it is not an evaluable function.", b.format_error()),
            RuntimeErrorType::IncorrectNumberOfArguments(b, e, a) => format!("Function '{}' requires {} parameters but {} were present.", b.format_error(), e, a),
            RuntimeErrorType::TypeErrorUnaryOp(op, v) => format!("TypeError: Argument to unary '{}' must be an int, got '{}' of type '{}'", op.format_error(), v.as_str(), v.as_type_str()),
            RuntimeErrorType::TypeErrorBinaryOp(op, l, r) => format!("TypeError: Cannot {} '{}' of type '{}' and '{}' of type '{}'", op.format_error(), l.as_str(), l.as_type_str(), r.as_str(), r.as_type_str()),
            RuntimeErrorType::TypeErrorBinaryIs(l, r) => format!("TypeError: Object '{}' of type '{}' is not a type and cannot be used with binary 'is' on '{}' of type '{}'", r.as_str(), r.as_type_str(), l.as_str(), l.as_type_str()),
            RuntimeErrorType::TypeErrorCannotConvertToInt(v) => format!("TypeError: Cannot convert '{}' of type '{}' to an int", v.as_str(), v.as_type_str()),
            RuntimeErrorType::TypeErrorCannotCompare(l, r) => format!("TypeError: Cannot compare '{}' of type '{}' to and '{}' of type '{}'", l.as_str(), l.as_type_str(), r.as_str(), r.as_type_str()),
            RuntimeErrorType::TypeErrorFunc1(e, v1) => format!("TypeError: incorrect arguments for {}, got '{}' of type '{}' instead", e, v1.as_str(), v1.as_type_str()),
            RuntimeErrorType::TypeErrorFunc2(e, v1, v2) => format!("TypeError: incorrect arguments for {}, got '{}' of type '{}', '{}' of type '{}' instead", e, v1.as_str(), v1.as_type_str(), v2.as_str(), v2.as_type_str()),
            RuntimeErrorType::TypeErrorFunc3(e, v1, v2, v3) => format!("TypeError: incorrect arguments for {}, got '{}' of type '{}', '{}' of type '{}', '{}' of type '{}' instead", e, v1.as_str(), v1.as_type_str(), v2.as_str(), v2.as_type_str(), v3.as_str(), v3.as_type_str()),
        }
    }
}

impl AsError for Opcode {
    fn format_error(self: &Self) -> String {
        String::from(match self {
            Opcode::UnarySub => "-",
            Opcode::UnaryLogicalNot => "!",
            Opcode::UnaryBitwiseNot => "~",
            Opcode::OpDiv => "divide",
            Opcode::OpMul => "multiply",
            Opcode::OpMod => "modulo",
            Opcode::OpAdd => "add",
            Opcode::OpSub => "subtract",
            Opcode::OpLeftShift => "left shift",
            Opcode::OpRightShift => "right shift",
            op => panic!("AsError not implemented for opcode {:?}", op)
        })
    }
}

impl AsError for StdBinding {
    fn format_error(self: &Self) -> String {
        String::from(stdlib::lookup_binding(self))
    }
}

impl AsError for ParserError {
    fn format_error(self: &Self) -> String {
        match &self.error {
            ParserErrorType::UnexpectedEoF => String::from("Unexpected end of file."),
            ParserErrorType::UnexpectedEoFExpecting(e) => format!("Unexpected end of file, was expecting {}.", e.format_error()),
            ParserErrorType::UnexpectedTokenAfterEoF(e) => format!("Unexpected {} after parsing finished", e.format_error()),
            ParserErrorType::Expecting(e, a) => format!("Expected a {}, got {} instead", e.format_error(), a.format_error()),
            ParserErrorType::ExpectedExpressionTerminal(e) => format!("Expected an expression terminal, got {} instead", e.format_error()),
            ParserErrorType::ExpectedCommaOrEndOfArguments(e) => format!("Expected a ',' or ')' after function invocation, got {} instead", e.format_error()),
            ParserErrorType::ExpectedStatement(e) => format!("Expecting a statement, got {} instead", e.format_error()),
            ParserErrorType::ExpectedVariableNameAfterLet(e) => format!("Expecting a variable name after 'let' keyword, got {} instead", e.format_error()),
            ParserErrorType::GlobalVariableConflictWithBinding(e) => format!("Declaration for 'let {}' conflicts with builtin function of the same name", e),
            ParserErrorType::GlobalVariableConflictWithGlobal(e) => format!("Declaration for 'let {}' conflicts with existing global variable of the same name", e),
            ParserErrorType::AssignmentToNotVariable(e) => format!("Cannot to assign to '{}' as it is not a global or local variable", e),
            ParserErrorType::BreakOutsideOfLoop => String::from("Invalid 'break' statement outside of an enclosing loop"),
            ParserErrorType::ContinueOutsideOfLoop => String::from("Invalid 'continue' statement outside of an enclosing loop"),
        }
    }
}

impl AsError for ScanError {
    fn format_error(self: &Self) -> String {
        match &self.error {
            ScanErrorType::InvalidNumericPrefix(c) => format!("Invalid numeric prefix: '0{}'", c),
            ScanErrorType::InvalidNumericValue(e) => format!("Invalid numeric value: {}", e),
            ScanErrorType::InvalidCharacter(c) => format!("Invalid character: '{}'", c),
            ScanErrorType::UnterminatedStringLiteral => String::from("Unterminated string literal (missing a closing single quote)")
        }
    }
}

impl AsError for ScanToken {
    fn format_error(self: &Self) -> String {
        match &self {
            ScanToken::Identifier(s) => format!("identifier \'{}\'", s),
            ScanToken::StringLiteral(s) => format!("string '{}'", s),
            ScanToken::Int(i) => format!("integer '{}'", i),

            ScanToken::KeywordLet => String::from("'let' keyword"),
            ScanToken::KeywordFn => String::from("'fn' keyword"),
            ScanToken::KeywordIf => String::from("'if' keyword"),
            ScanToken::KeywordElif => String::from("'elif' keyword"),
            ScanToken::KeywordElse => String::from("'else' keyword"),
            ScanToken::KeywordLoop => String::from("'loop' keyword"),
            ScanToken::KeywordFor => String::from("'for' keyword"),
            ScanToken::KeywordIn => String::from("'in' keyword"),
            ScanToken::KeywordIs => String::from("'is' keyword"),
            ScanToken::KeywordBreak => String::from("'break' keyword"),
            ScanToken::KeywordContinue => String::from("'continue' keyword"),
            ScanToken::KeywordTrue => String::from("'true' keyword"),
            ScanToken::KeywordFalse => String::from("'false' keyword"),
            ScanToken::KeywordNil => String::from("'nil' keyword"),
            ScanToken::KeywordStruct => String::from("'struct' keyword"),
            ScanToken::KeywordExit => String::from("'exit' keyword"),

            ScanToken::Equals => String::from("'=' token"),
            ScanToken::PlusEquals => String::from("'+=' token"),
            ScanToken::MinusEquals => String::from("'-=' token"),
            ScanToken::MulEquals => String::from("'*=' token"),
            ScanToken::DivEquals => String::from("'/=' token"),
            ScanToken::AndEquals => String::from("'&=' token"),
            ScanToken::OrEquals => String::from("'|=' token"),
            ScanToken::XorEquals => String::from("'^=' token"),
            ScanToken::LeftShiftEquals => String::from("'<<=' token"),
            ScanToken::RightShiftEquals => String::from("'>>=' token"),
            ScanToken::ModEquals => String::from("'%=' token"),
            ScanToken::PowEquals => String::from("'**=' token"),

            ScanToken::Plus => String::from("'+' token"),
            ScanToken::Minus => String::from("'-' token"),
            ScanToken::Mul => String::from("'*' token"),
            ScanToken::Div => String::from("'/' token"),
            ScanToken::BitwiseAnd => String::from("'&' token"),
            ScanToken::BitwiseOr => String::from("'|' token"),
            ScanToken::BitwiseXor => String::from("'^' token"),
            ScanToken::Mod => String::from("'%' token"),
            ScanToken::Pow => String::from("'**' token"),
            ScanToken::LeftShift => String::from("'<<' token"),
            ScanToken::RightShift => String::from("'>>' token"),

            ScanToken::LogicalNot => String::from("'!' token"),
            ScanToken::BitwiseNot => String::from("'~' token"),

            ScanToken::LogicalAnd => String::from("'&&' token"),
            ScanToken::LogicalOr => String::from("'||' token"),

            ScanToken::NotEquals => String::from("'!=' token"),
            ScanToken::DoubleEquals => String::from("'==' token"),
            ScanToken::LessThan => String::from("'<' token"),
            ScanToken::LessThanEquals => String::from("'<=' token"),
            ScanToken::GreaterThan => String::from("'>' token"),
            ScanToken::GreaterThanEquals => String::from("'>=' token"),

            ScanToken::OpenParen => String::from("'(' token"), // ( )
            ScanToken::CloseParen => String::from("')' token"),
            ScanToken::OpenSquareBracket => String::from("'[' token"), // [ ]
            ScanToken::CloseSquareBracket => String::from("']' token"),
            ScanToken::OpenBrace => String::from("'{' token"), // { }
            ScanToken::CloseBrace => String::from("'}' token"),

            ScanToken::Comma => String::from("',' token"),
            ScanToken::Dot => String::from("'.' token"),
            ScanToken::Colon => String::from("':' token"),
            ScanToken::Arrow => String::from("'->' token"),
            ScanToken::Underscore => String::from("'_' token"),
            ScanToken::Semicolon => String::from("';' token"),

            ScanToken::NewLine => String::from("new line"),
        }
    }
}