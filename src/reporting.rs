use std::rc::Rc;

use crate::compiler::parser::{ParserError, ParserErrorType};
use crate::compiler::CompileResult;
use crate::compiler::scanner::{ScanError, ScanErrorType, ScanToken};
use crate::stdlib;
use crate::stdlib::StdBinding;
use crate::vm::error::{RuntimeError, DetailRuntimeError, StackTraceFrame};
use crate::vm::opcode::Opcode;
use crate::vm::value::{FunctionImpl, Value};
use crate::misc;

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

    pub fn format_runtime_error(self: &Self, error: DetailRuntimeError) -> String {
        format_runtime_error(&self.lines, self.src, error)
    }
}

pub fn format_scan_error(source_lines: &Vec<&str>, source_file: &String, error: &ScanError) -> String {
    let mut text: String = error.format_error();
    text.push_str(format!("\n  at: line {} ({})\n  at:\n\n", error.lineno + 1, source_file).as_str());
    text.push_str(source_lines.get(error.lineno).map(|t| *t).unwrap_or(""));
    text.push('\n');
    text
}

pub fn format_parse_error(source_lines: &Vec<&str>, source_file: &String, error: &ParserError) -> String {
    let mut text: String = error.format_error();
    text.push_str(format!("\n  at: line {} ({})\n  at:\n\n", error.lineno + 1, &source_file).as_str());
    text.push_str(source_lines.get(error.lineno).map(|t| *t).unwrap_or(""));
    text.push('\n');
    text
}

pub fn format_runtime_error(source_lines: &Vec<&str>, source_file: &String, mut error: DetailRuntimeError) -> String {
    let mut text: String = error.error.format_error();
    let root: &mut StackTraceFrame = &mut error.stack[0];
    let mut src: String = String::from(source_lines.get(root.lineno as usize).map(|t| *t).unwrap_or(""));
    misc::strip_line_ending(&mut src);
    root.src = Some(src);
    for frame in error.stack {
        text.push_str(format!("\n    at: `{}` (line {})", frame.src.unwrap(), frame.lineno + 1).as_str())
    }
    text.push_str(format!("\n    at: execution of script '{}'\n", source_file).as_str());
    text
}

pub trait ProvidesLineNumber {
    fn line_number(self: &Self, ip: usize) -> u16;
}

impl ProvidesLineNumber for Vec<u16> {
    fn line_number(self: &Self, ip: usize) -> u16 {
        *self.get(ip).unwrap_or_else(|| self.last().unwrap())
    }
}

impl ProvidesLineNumber for CompileResult {
    fn line_number(self: &Self, ip: usize) -> u16 {
        self.line_numbers.line_number(ip)
    }
}


trait AsError {
    fn format_error(self: &Self) -> String;
}


impl AsError for RuntimeError {
    fn format_error(self: &Self) -> String {
        match self {
            RuntimeError::RuntimeExit => panic!("Not a real error"),
            RuntimeError::ValueIsNotFunctionEvaluable(v) => format!("Tried to evaluate {} but it is not a function.", v.format_error()),
            RuntimeError::BindingIsNotFunctionEvaluable(b) => format!("Tried to evaluate '{}' but it is not an evaluable function.", b.format_error()),
            RuntimeError::IncorrectNumberOfFunctionArguments(f, a) => format!("Function {} requires {} parameters but {} were present.", f.format_error(), f.nargs, a),
            RuntimeError::IncorrectNumberOfArguments(b, e, a) => format!("Function '{}' requires {} parameters but {} were present.", b.format_error(), e, a),
            RuntimeError::IncorrectNumberOfArgumentsVariadicAtLeastOne(b) => format!("Function '{}' requires at least 1 parameter but none were present.", b.format_error()),
            RuntimeError::ValueErrorIndexOutOfBounds(i, ln) => format!("Index '{}' is out of bounds for list of length [0, {})", i, ln),
            RuntimeError::ValueErrorStepCannotBeZero => String::from("ValueError: 'step' argument cannot be zero"),
            RuntimeError::ValueErrorVariableNotDeclaredYet(x) => format!("ValueError: '{}' was referenced but has not been declared yet", x),
            RuntimeError::ValueErrorValueMustBeNonEmpty => format!("ValueError: Expected value to be a non empty iterable"),
            RuntimeError::ValueErrorValueMustBeNonNegative(v) => format!("ValueError: Expected value '{}: int' to be non-negative", v),

            RuntimeError::TypeErrorUnaryOp(op, v) => format!("TypeError: Argument to unary '{}' must be an int, got {}", op.format_error(), v.format_error()),
            RuntimeError::TypeErrorBinaryOp(op, l, r) => format!("TypeError: Cannot {} {} and {}", op.format_error(), l.format_error(), r.format_error()),
            RuntimeError::TypeErrorBinaryIs(l, r) => format!("TypeError: {} is not a type and cannot be used with binary 'is' on {}", r.format_error(), l.format_error()),
            RuntimeError::TypeErrorCannotConvertToInt(v) => format!("TypeError: Cannot convert {} to an int", v.format_error()),
            RuntimeError::TypeErrorCannotSlice(ls) => format!("TypeError: Cannot slice {}", ls.format_error()),
            RuntimeError::TypeErrorArgMustBeInt(v) => format!("TypeError: Expected {} to be a int", v.format_error()),
            RuntimeError::TypeErrorArgMustBeStr(v) => format!("TypeError: Expected {} to be a string", v.format_error()),
            RuntimeError::TypeErrorArgMustBeIterable(v) => format!("TypeError: Expected {} to be an iterable", v.format_error()),
            RuntimeError::TypeErrorFunc1(e, v1) => format!("TypeError: incorrect arguments for {}, got {} instead", e, v1.format_error()),
            RuntimeError::TypeErrorFunc2(e, v1, v2) => format!("TypeError: incorrect arguments for {}, got '{}, {} instead", e, v1.format_error(), v2.format_error()),
            RuntimeError::TypeErrorFunc3(e, v1, v2, v3) => format!("TypeError: incorrect arguments for {}, got {}, {}, {} instead", e, v1.format_error(), v2.format_error(), v3.format_error()),
        }
    }
}

impl AsError for Value {
    fn format_error(self: &Self) -> String {
        format!("'{}' of type '{}'", self.as_str(), self.as_type_str())
    }
}

impl AsError for FunctionImpl {
    fn format_error(self: &Self) -> String {
        Value::Function(Rc::new(self.clone())).format_error()
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
            Opcode::OpPow => "**",
            Opcode::OpIs => "is",
            Opcode::OpBitwiseAnd => "&",
            Opcode::OpBitwiseOr => "|",
            Opcode::OpBitwiseXor => "^",
            Opcode::OpLessThan => "<",
            Opcode::OpGreaterThan => ">",
            Opcode::OpLessThanEqual => "<=",
            Opcode::OpGreaterThanEqual => ">=",
            Opcode::OpEqual => "==",
            Opcode::OpNotEqual => "!=",
            Opcode::OpIndex => "array index",
            Opcode::OpFuncEval(_) => "()",
            Opcode::OpSlice => "list slice",
            Opcode::OpSliceWithStep => "list slice",
            op => panic!("AsError not implemented for opcode {:?}", op)
        })
    }
}

impl AsError for StdBinding {
    fn format_error(self: &Self) -> String {
        String::from(stdlib::lookup_name(*self))
    }
}

impl AsError for ParserError {
    fn format_error(self: &Self) -> String {
        match &self.error {
            ParserErrorType::UnexpectedTokenAfterEoF(e) => format!("Unexpected {} after parsing finished", e.format_error()),
            ParserErrorType::ExpectedToken(e, a) => format!("Expected a {}, got {} instead", e.format_error(), a.format_error()),
            ParserErrorType::ExpectedExpressionTerminal(e) => format!("Expected an expression terminal, got {} instead", e.format_error()),
            ParserErrorType::ExpectedCommaOrEndOfArguments(e) => format!("Expected a ',' or ')' after function invocation, got {} instead", e.format_error()),
            ParserErrorType::ExpectedCommaOrEndOfList(e) => format!("Expected a ',' or ']' after list literal, got {} instead", e.format_error()),
            ParserErrorType::ExpectedColonOrEndOfSlice(e) => format!("Expected a ':' or ']' in slice, got {} instead", e.format_error()),
            ParserErrorType::ExpectedStatement(e) => format!("Expecting a statement, got {} instead", e.format_error()),
            ParserErrorType::ExpectedVariableNameAfterLet(e) => format!("Expecting a variable name after 'let' keyword, got {} instead", e.format_error()),
            ParserErrorType::ExpectedVariableNameAfterFor(e) => format!("Expecting a variable name after 'for' keyword, got {} instead", e.format_error()),
            ParserErrorType::ExpectedFunctionNameAfterFn(e) => format!("Expecting a function name after 'fn' keyword, got {} instead", e.format_error()),
            ParserErrorType::ExpectedFunctionBlockOrArrowAfterFn(e) => format!("Expecting a function body starting with '{{' or `->` after 'fn', got {} instead", e.format_error()),
            ParserErrorType::ExpectedParameterOrEndOfList(e) => format!("Expected a function parameter or ')' after function declaration, got {} instead", e.format_error()),
            ParserErrorType::ExpectedCommaOrEndOfParameters(e) => format!("Expected a ',' or ')' after function parameter, got {} instead", e.format_error()),
            ParserErrorType::LocalVariableConflict(e) => format!("Multiple declarations for 'let {}' in the same scope", e),
            ParserErrorType::LocalVariableConflictWithNativeFunction(e) => format!("Name for variable '{}' conflicts with the native function by the same name", e),
            ParserErrorType::UndeclaredIdentifier(e) => format!("Undeclared identifier: '{}'", e),
            ParserErrorType::InvalidAssignmentTarget => format!("The left hand side of an assignment expression must be a variable, array access, or property access"),
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

impl AsError for Option<ScanToken> {
    fn format_error(self: &Self) -> String {
        match self {
            Some(t) => t.format_error(),
            None => String::from("end of input"),
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
            ScanToken::KeywordReturn => String::from("'return' keyword"),
            ScanToken::KeywordIf => String::from("'if' keyword"),
            ScanToken::KeywordElif => String::from("'elif' keyword"),
            ScanToken::KeywordElse => String::from("'else' keyword"),
            ScanToken::KeywordLoop => String::from("'loop' keyword"),
            ScanToken::KeywordWhile => String::from("'while' keyword"),
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