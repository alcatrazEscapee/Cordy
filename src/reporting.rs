use std::ops::{BitOr, BitOrAssign};
use std::rc::Rc;

use crate::compiler::{ParserError, ParserErrorType, ScanError, ScanErrorType, ScanToken};
use crate::stdlib::NativeFunction;
use crate::vm::{FunctionImpl, RuntimeError, Value};
use crate::vm::operator::{BinaryOp, UnaryOp};

pub type Locations = Vec<Location>;


/// A closed interval of a source code location.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Location {
    /// Start character index, inclusive
    start: usize,
    /// Total width of the location. A width of `0` indicates this is an empty location.
    width: usize,
}

impl Location {
    pub fn new(start: usize, width: usize) -> Location {
        Location { start, width }
    }

    pub fn empty() -> Location {
        Location::new(0, 0)
    }

    pub fn as_opt(self: Self) -> Option<Location> {
        if self.width > 0 { Some(self) } else { None }
    }
    pub fn start(self: &Self) -> usize { self.start }
    pub fn end(self: &Self) -> usize { self.start + self.width - 1 }
}

impl BitOr for Location {
    type Output = Location;

    fn bitor(self, rhs: Self) -> Self::Output {
        let start = self.start.min(rhs.start);
        let end = self.end().max(rhs.end());
        Location::new(start, end - start + 1)
    }
}

impl BitOrAssign for Location {
    fn bitor_assign(&mut self, rhs: Self) {
        let start = self.start.min(rhs.start);
        let end = self.end().max(rhs.end());

        self.start = start;
        self.width = end - start + 1;
    }
}



/// Indexed source code information.
/// This is used to report errors in a readable fashion, by resolving `Location` references to a line and column number.
#[derive(Debug, Clone)]
pub struct SourceView<'a> {
    name: &'a String,

    pub text: &'a String,

    lines: Vec<&'a str>,
    starts: Vec<usize>,
}

/// A simple common trait for converting arbitrary objects to human-readable errors
/// This could be implemented directly on the types, but using the trait allows all end-user-exposed text to be concentrated in this module.
pub trait AsError {
    fn as_error(self: &Self) -> String;
}

/// An extension of `AsError` which is used for actual error types (`ScanError`, `ParserError`, `DetailRuntimeError`)
/// This is intentionally polymorphic, as it's used in `SourceView::format()`
pub trait AsErrorWithContext: AsError {
    fn location(self: &Self) -> Location;

    /// When formatting a `RuntimeError`, allows inserting additional stack trace elements.
    /// This is appended *after* the initial `at: line X (source file)` line is appended.
    fn add_stack_trace_elements(self: &Self, _: &SourceView, _: &mut String) {}
}



impl<'a> SourceView<'a> {

    pub fn new(name: &'a String, text: &'a String) -> SourceView<'a> {
        let mut lines: Vec<&'a str> = Vec::new();
        let mut starts: Vec<usize> = Vec::new();
        let mut start = 0;

        for line in text.split('\n') {
            lines.push(line.strip_suffix('\r').unwrap_or(line));
            starts.push(start);
            start += line.len() + 1;
        }

        starts.push(start);

        SourceView {
            name,
            text,
            lines,
            starts,
        }
    }

    pub fn len(self: &Self) -> usize {
        self.lines.len()
    }

    pub fn lineno(self: &Self, loc: Location) -> usize {
        match loc.as_opt() {
            Some(loc) => self.starts.partition_point(|u| u <= &loc.start) - 1,
            None => self.len() - 1
        }
    }

    pub fn format<E : AsErrorWithContext>(self: &Self, error: &E) -> String {
        let mut text = error.as_error();
        let loc = error.location();
        let start_lineno = self.lineno(loc);
        let mut end_lineno = start_lineno;

        if let Some(loc) = loc.as_opt() {
            while self.starts[end_lineno + 1] < loc.end() {
                end_lineno += 1;
            }
        }

        debug_assert!(start_lineno < self.lines.len());
        debug_assert!(end_lineno >= start_lineno && end_lineno < self.lines.len());

        let lineno = if start_lineno == end_lineno {
            format!("{}", start_lineno + 1)
        } else {
            format!("{} - {}", start_lineno + 1, end_lineno + 1)
        };

        let width: usize = format!("{}", end_lineno + 2).len();

        text.push_str(format!("\n  at: line {} ({})\n", lineno, self.name).as_str());
        error.add_stack_trace_elements(self, &mut text);
        text.push('\n');

        for i in start_lineno..=end_lineno {
            let line = self.lines[i];
            text.push_str(format!("{:width$} |", i + 1, width = width).as_str());
            if !line.is_empty() {
                text.push(' ');
                text.push_str(line);
            }
            text.push('\n');
        }

        let (start_col, end_col) = match loc.as_opt() {
            Some(loc) => (loc.start - self.starts[start_lineno], loc.end() - self.starts[end_lineno]),
            None => {
                let last_col = self.lines[end_lineno].len();
                (last_col + 1, last_col + 3)
            }
        };

        text.push_str(format!("{:width$} |", end_lineno + 2, width = width).as_str());

        if start_lineno == end_lineno {
            // Single line error, so point to the exact column start + end
            for _ in 0..=start_col { text.push(' '); }
            for _ in start_col..=end_col { text.push('^'); }
        } else {
            // For multi-line errors, just point directly up across the entire bottom line
            let longest_line_len: usize = (start_lineno..=end_lineno)
                .map(|u| self.lines[u].len())
                .max()
                .unwrap();
            if longest_line_len > 0 {
                text.push(' ');
                for _ in 0..longest_line_len { text.push('^'); }
            }
        }
        text.push('\n');
        text
    }
}


impl AsError for RuntimeError {
    fn as_error(self: &Self) -> String {
        match self {
            RuntimeError::RuntimeExit | RuntimeError::RuntimeYield => panic!("Not a real error"),
            RuntimeError::RuntimeCompilationError(vec) => format!("Encountered compilation error(s) within 'eval':\n\n{}", vec.join("\n")),
            RuntimeError::ValueIsNotFunctionEvaluable(v) => format!("Tried to evaluate {} but it is not a function.", v.as_error()),
            RuntimeError::IncorrectNumberOfFunctionArguments(f, a) => format!("Function {} requires {} parameters but {} were present.", f.as_error(), f.nargs, a),
            RuntimeError::IncorrectNumberOfArguments(b, a, e) => format!("Function '{}' requires {} parameters but {} were present.", b.as_error(), e, a),
            RuntimeError::IncorrectNumberOfArgumentsVariadicAtLeastOne(b) => format!("Function '{}' requires at least 1 parameter but none were present.", b.as_error()),
            RuntimeError::ValueErrorIndexOutOfBounds(i, ln) => format!("Index '{}' is out of bounds for list of length [0, {})", i, ln),
            RuntimeError::ValueErrorStepCannotBeZero => String::from("ValueError: 'step' argument cannot be zero"),
            RuntimeError::ValueErrorVariableNotDeclaredYet(x) => format!("ValueError: '{}' was referenced but has not been declared yet", x),
            RuntimeError::ValueErrorValueMustBeNonEmpty => format!("ValueError: Expected value to be a non empty iterable"),
            RuntimeError::ValueErrorCannotUnpackLengthMustBeGreaterThan(e, a, v) => format!("ValueError: Cannot unpack {} with length {}, expected at least {} elements", v.as_error(), a, e),
            RuntimeError::ValueErrorCannotUnpackLengthMustBeEqual(e, a, v) => format!("ValueError: Cannot unpack {} with length {}, expected exactly {} elements", v.as_error(), a, e),
            RuntimeError::ValueErrorValueMustBeNonNegative(v) => format!("ValueError: Expected value '{}: int' to be non-negative", v),
            RuntimeError::ValueErrorValueMustBePositive(v) => format!("ValueError: Expected value '{}: int' to be positive", v),
            RuntimeError::ValueErrorValueMustBeNonZero => format!("ValueError: Expected value to be non-zero"),
            RuntimeError::ValueErrorCannotCollectIntoDict(v) => format!("ValueError: Cannot collect key-value pair {} into a dict", v.as_error()),
            RuntimeError::ValueErrorKeyNotPresent(v) => format!("ValueError: Key {} not found in dictionary", v.as_error()),
            RuntimeError::ValueErrorInvalidCharacterOrdinal(i) => format!("ValueError: Cannot convert int {} to a character", i),
            RuntimeError::ValueErrorInvalidFormatCharacter(c) => format!("ValueError: Invalid format character '{}' in format string", c.as_error()),
            RuntimeError::ValueErrorNotAllArgumentsUsedInStringFormatting(v) => format!("ValueError: Not all arguments consumed in format string, next: {}", v.as_error()),
            RuntimeError::ValueErrorMissingRequiredArgumentInStringFormatting => format!("ValueError: Not enough arguments for format string"),
            RuntimeError::ValueErrorEvalListMustHaveUnitLength(len) => format!("ValueError: Evaluating an index must have len = 1, got len = {}", len),

            RuntimeError::TypeErrorUnaryOp(op, v) => format!("TypeError: Argument to unary '{}' must be an int, got {}", op.as_error(), v.as_error()),
            RuntimeError::TypeErrorBinaryOp(op, l, r) => format!("TypeError: Cannot {} {} and {}", op.as_error(), l.as_error(), r.as_error()),
            RuntimeError::TypeErrorBinaryIs(l, r) => format!("TypeError: {} is not a type and cannot be used with binary 'is' on {}", r.as_error(), l.as_error()),
            RuntimeError::TypeErrorCannotConvertToInt(v) => format!("TypeError: Cannot convert {} to an int", v.as_error()),
            RuntimeError::TypeErrorArgMustBeInt(v) => format!("TypeError: Expected {} to be a int", v.as_error()),
            RuntimeError::TypeErrorArgMustBeStr(v) => format!("TypeError: Expected {} to be a string", v.as_error()),
            RuntimeError::TypeErrorArgMustBeChar(v) => format!("TypeError: Expected {} to be a single character string", v.as_error()),
            RuntimeError::TypeErrorArgMustBeIterable(v) => format!("TypeError: Expected {} to be an iterable", v.as_error()),
            RuntimeError::TypeErrorArgMustBeIndexable(ls) => format!("TypeError: Cannot index {}", ls.as_error()),
            RuntimeError::TypeErrorArgMustBeSliceable(ls) => format!("TypeError: Cannot slice {}", ls.as_error()),
            RuntimeError::TypeErrorArgMustBeDict(v) => format!("TypeError: Expected {} to be a dict", v.as_error()),
            RuntimeError::TypeErrorArgMustBeFunction(v) => format!("TypeError: Expected {} to be a function", v.as_error()),
            RuntimeError::TypeErrorArgMustBeCmpOrKeyFunction(v) => format!("TypeError: Expected {} to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function", v.as_error()),
        }
    }
}

impl AsError for Option<char> {
    fn as_error(self: &Self) -> String {
        match self {
            None => String::from("end of format string"),
            Some(c) => String::from(*c),
        }
    }
}

impl AsError for Value {
    fn as_error(self: &Self) -> String {
        format!("'{}' of type '{}'", self.to_str(), self.as_type_str())
    }
}

impl AsError for FunctionImpl {
    fn as_error(self: &Self) -> String {
        Value::Function(Rc::new(self.clone())).as_error()
    }
}

impl AsError for UnaryOp {
    fn as_error(self: &Self) -> String {
        String::from(match self {
            UnaryOp::Minus => "-",
            UnaryOp::Not => "!",
        })
    }
}

impl AsError for BinaryOp {
    fn as_error(self: &Self) -> String {
        String::from(match self {
            BinaryOp::Div => "divide",
            BinaryOp::Mul => "multiply",
            BinaryOp::Mod => "modulo",
            BinaryOp::Add => "add",
            BinaryOp::Sub => "subtract",
            BinaryOp::LeftShift => "left shift",
            BinaryOp::RightShift => "right shift",
            BinaryOp::Pow => "**",
            BinaryOp::Is => "is",
            BinaryOp::And => "&",
            BinaryOp::Or => "|",
            BinaryOp::Xor => "^",
            BinaryOp::In => "in",
            BinaryOp::LessThan => "<",
            BinaryOp::GreaterThan => ">",
            BinaryOp::LessThanEqual => "<=",
            BinaryOp::GreaterThanEqual => ">=",
            BinaryOp::Equal => "==",
            BinaryOp::NotEqual => "!=",
            BinaryOp::Max => "min",
            BinaryOp::Min => "max",
        })
    }
}

impl AsError for NativeFunction {
    fn as_error(self: &Self) -> String {
        String::from(self.name())
    }
}

impl AsError for ParserError {
    fn as_error(self: &Self) -> String {
        match &self.error {
            ParserErrorType::UnexpectedTokenAfterEoF(e) => format!("Unexpected {} after parsing finished", e.as_error()),

            ParserErrorType::ExpectedToken(e, a) => format!("Expected a {}, got {} instead", e.as_error(), a.as_error()),
            ParserErrorType::ExpectedExpressionTerminal(e) => format!("Expected an expression terminal, got {} instead", e.as_error()),
            ParserErrorType::ExpectedCommaOrEndOfArguments(e) => format!("Expected a ',' or ')' after function invocation, got {} instead", e.as_error()),
            ParserErrorType::ExpectedCommaOrEndOfList(e) => format!("Expected a ',' or ']' after list literal, got {} instead", e.as_error()),
            ParserErrorType::ExpectedCommaOrEndOfVector(e) => format!("Expected a ',' or ')' after vector literal, got {} instead", e.as_error()),
            ParserErrorType::ExpectedCommaOrEndOfDict(e) => format!("Expected a ',' or '}}' after dict literal, got {} instead", e.as_error()),
            ParserErrorType::ExpectedCommaOrEndOfSet(e) => format!("Expected a ',' or '}}' after set literal, got {} instead", e.as_error()),
            ParserErrorType::ExpectedColonOrEndOfSlice(e) => format!("Expected a ':' or ']' in slice, got {} instead", e.as_error()),
            ParserErrorType::ExpectedStatement(e) => format!("Expecting a statement, got {} instead", e.as_error()),
            ParserErrorType::ExpectedVariableNameAfterLet(e) => format!("Expecting a variable name after 'let' keyword, got {} instead", e.as_error()),
            ParserErrorType::ExpectedVariableNameAfterFor(e) => format!("Expecting a variable name after 'for' keyword, got {} instead", e.as_error()),
            ParserErrorType::ExpectedFunctionNameAfterFn(e) => format!("Expecting a function name after 'fn' keyword, got {} instead", e.as_error()),
            ParserErrorType::ExpectedFunctionBlockOrArrowAfterFn(e) => format!("Expecting a function body starting with '{{' or `->` after 'fn', got {} instead", e.as_error()),
            ParserErrorType::ExpectedParameterOrEndOfList(e) => format!("Expected a function parameter or ')' after function declaration, got {} instead", e.as_error()),
            ParserErrorType::ExpectedCommaOrEndOfParameters(e) => format!("Expected a ',' or ')' after function parameter, got {} instead", e.as_error()),
            ParserErrorType::ExpectedPatternTerm(e) => format!("Expected a name, '_', or variadic term in a pattern variable, got {} instead", e.as_error()),
            ParserErrorType::ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(e) => format!("Expected a name or '_' after '*' in a pattern variable, got {} instead", e.as_error()),
            ParserErrorType::ExpectedUnderscoreOrVariableNameOrPattern(e) => format!("Expected a variable binding, either a name, or '_', or pattern (i.e. 'x, (_, y), *z'), got {} instead", e.as_error()),
            ParserErrorType::ExpectedAnnotationOrNamedFunction(e) => format!("Expected another decorator, or a named function after decorator, got {} instead", e.as_error()),
            ParserErrorType::ExpectedAnnotationOrAnonymousFunction(e) => format!("Expected another decorator, or an expression function after decorator, got {} instead", e.as_error()),

            ParserErrorType::LocalVariableConflict(e) => format!("Multiple declarations for 'let {}' in the same scope", e),
            ParserErrorType::LocalVariableConflictWithNativeFunction(e) => format!("Name for variable '{}' conflicts with the native function by the same name", e),
            ParserErrorType::UndeclaredIdentifier(e) => format!("Undeclared identifier: '{}'", e),

            ParserErrorType::InvalidAssignmentTarget => format!("The left hand side of an assignment expression must be a variable, array access, or property access"),
            ParserErrorType::MultipleVariadicTermsInPattern => format!("Pattern is not allowed to have more than one variadic (i.e. '*') term."),
            ParserErrorType::LetWithPatternBindingNoExpression => format!("'let' with a pattern variable must be followed by an expression if the pattern contains non-simple elements such as variadic (i.e. '*'), empty (i.e. '_'), or nested (i.e. 'x, (_, y)) terms."),
            ParserErrorType::BreakOutsideOfLoop => String::from("Invalid 'break' statement outside of an enclosing loop"),
            ParserErrorType::ContinueOutsideOfLoop => String::from("Invalid 'continue' statement outside of an enclosing loop"),

            ParserErrorType::Runtime(e) => e.as_error(),
        }
    }
}

impl AsError for ScanError {
    fn as_error(self: &Self) -> String {
        match &self.error {
            ScanErrorType::InvalidNumericPrefix(c) => format!("Invalid numeric prefix: '0{}'", c),
            ScanErrorType::InvalidNumericValue(e) => format!("Invalid numeric value: {}", e),
            ScanErrorType::InvalidCharacter(c) => format!("Invalid character: '{}'", c),
            ScanErrorType::UnterminatedStringLiteral => String::from("Unterminated string literal (missing a closing single quote)")
        }
    }
}

impl AsError for Option<ScanToken> {
    fn as_error(self: &Self) -> String {
        match self {
            Some(t) => t.as_error(),
            None => String::from("end of input"),
        }
    }
}

impl AsError for ScanToken {
    fn as_error(self: &Self) -> String {
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
            ScanToken::KeywordThen => String::from("'then' keyword"),
            ScanToken::KeywordLoop => String::from("'loop' keyword"),
            ScanToken::KeywordWhile => String::from("'while' keyword"),
            ScanToken::KeywordFor => String::from("'for' keyword"),
            ScanToken::KeywordIn => String::from("'in' keyword"),
            ScanToken::KeywordIs => String::from("'is' keyword"),
            ScanToken::KeywordNot => String::from("'not' keyword"),
            ScanToken::KeywordBreak => String::from("'break' keyword"),
            ScanToken::KeywordContinue => String::from("'continue' keyword"),
            ScanToken::KeywordDo => String::from("'do' keyword"),
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
            ScanToken::DotEquals => String::from("'.=' token"),

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

            ScanToken::Not => String::from("'!' token"),

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
            ScanToken::At => String::from("'@' token"),

            ScanToken::NewLine => String::from("new line"),
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::reporting::{AsError, AsErrorWithContext, Location, SourceView};

    #[test]
    fn test_or_location() {
        let l1 = Location::new(0, 5);
        let l2 = Location::new(8, 2);

        assert_eq!(l1 | l2, Location::new(0, 10));

        let mut l3 = Location::new(2, 4);
        l3 |= l1;

        assert_eq!(l3, Location::new(0, 6));
    }

    #[test]
    fn test_error_first_word_first_line() {
        run(0, 4, "Error
  at: line 1 (<test>)

1 | first += line
2 | ^^^^^
")
    }

    #[test]
    fn test_error_second_word_first_line() {
        run(6, 7, "Error
  at: line 1 (<test>)

1 | first += line
2 |       ^^
")
    }

    #[test]
    fn test_error_third_word_first_line() {
        run(9, 12, "Error
  at: line 1 (<test>)

1 | first += line
2 |          ^^^^
")
    }

    #[test]
    fn test_error_first_word_second_line() {
        run(14, 19, "Error
  at: line 2 (<test>)

2 | second line?
3 | ^^^^^^
")
    }

    #[test]
    fn test_error_across_first_and_second_line() {
        run(9, 19, "Error
  at: line 1 - 2 (<test>)

1 | first += line
2 | second line?
3 | ^^^^^^^^^^^^^
")
    }

    #[test]
    fn test_error_after_windows_line() {
        run(39, 45, "Error
  at: line 4 (<test>)

4 | windows line
5 | ^^^^^^^
")
    }

    #[test]
    fn test_error_after_empty_lines() {
        run(53, 57, "Error
  at: line 6 (<test>)

6 | empty
7 | ^^^^^
")
    }

    #[test]
    fn test_error_after_empty_windows_lines() {
        run(62, 65, "Error
  at: line 8 (<test>)

8 | more empty
9 | ^^^^
")
    }

    #[test]
    fn test_error_last_word_last_line() {
        run(67, 71, "Error
  at: line 8 (<test>)

8 | more empty
9 |      ^^^^^
")
    }

    struct MockError(&'static str, Location);

    impl AsError for MockError { fn as_error(self: &Self) -> String { String::from(self.0) } }
    impl AsErrorWithContext for MockError { fn location(self: &Self) -> Location { self.1 } }

    fn run(start: usize, end: usize, expected: &'static str) {
        let name = String::from("<test>");
        let text = String::from("first += line\nsecond line?\nthird line\r\nwindows line\n\nempty\r\n\r\nmore empty");
        let src = SourceView::new(&name, &text);
        let error = src.format(&MockError("Error", Location::new(start, end - start + 1)));

        assert_eq!(error.as_str(), expected);
    }
}
