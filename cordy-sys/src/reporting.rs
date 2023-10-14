use std::cell::{Ref, RefCell};
use std::ops::{BitOr, BitOrAssign};

use crate::compiler::{ParserError, ParserErrorType, ScanError, ScanErrorType, ScanToken};
use crate::core::NativeFunction;
use crate::vm::{FunctionImpl, RuntimeError, ValuePtr};
use crate::vm::operator::{BinaryOp, UnaryOp};


/// `Location` represents a position in the source code.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct Location {
    /// Start character index, inclusive
    start: usize,
    /// Total width of the location. A width of `0` indicates this is an empty location.
    width: u32,
    /// Index of this location.
    /// The index refers to which source view the location refers to, which may matter if there are more than one (for example, in `eval` or incremental compiles)
    index: u32,
}

impl Location {
    /// Creates a new location from a given start cursor and width
    pub fn new(start: usize, width: u32, index: u32) -> Location {
        Location { start, width, index }
    }

    /// Returns a sentinel empty location
    pub fn empty() -> Location {
        Location::new(0, 0, 0)
    }

    /// Returns the start pointer of the location, inclusive
    pub fn start(&self) -> usize { self.start }

    /// Returns the end pointer of the location, inclusive
    pub fn end(&self) -> usize { self.start + self.width as usize - 1 }

    // Returns `true` if the location is empty, i.e. zero width
    pub fn is_empty(&self) -> bool { self.width == 0 }
}

impl BitOr for Location {
    type Output = Location;

    fn bitor(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.index, rhs.index);

        let start = self.start.min(rhs.start);
        let end = self.end().max(rhs.end());
        Location::new(start, (end - start + 1) as u32, self.index)
    }
}

impl BitOrAssign for Location {
    fn bitor_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.index, rhs.index);

        let start = self.start.min(rhs.start);
        let end = self.end().max(rhs.end());

        self.start = start;
        self.width = (end - start + 1) as u32;
    }
}


/// An indexed view of source code, in the form of a vector of all source code entries.
///
/// Entries are indexed according to the `index` field in a `Location`.
/// New locations are always created at the highest index.
#[derive(Debug)]
pub struct SourceView(Vec<SourceEntry>);

#[derive(Debug)]
struct SourceEntry {
    /// The name of the entry.
    /// For external inputs this will be the name of the file, for incremental compiles this can be `<eval>`, `<stdin>`, etc.
    name: String,

    /// The raw text source code of the entry.
    /// This is the sequence which `Location`s are indexes into.
    text: String,

    /// Lazily populated index data
    index: RefCell<Option<SourceIndex>>,
}

#[derive(Debug)]
struct SourceIndex {
    /// The raw text, split into lines, with `\r` and `\n` characters removed.
    lines: Vec<String>,

    /// A vector of starting character indices.
    /// `starts[i]` represents the location index of the first character in this line.
    starts: Vec<usize>,
}

impl SourceView {

    pub fn empty() -> SourceView { SourceView(Vec::new()) }

    pub fn new(name: String, text: String) -> SourceView {
        let mut view = SourceView(Vec::new());
        view.push(name, text);
        view
    }

    /// Returns the name of the currently active entry.
    pub fn name(&self) -> &String {
        &self.0.last().unwrap().name
    }

    /// Returns the source code of the currently active entry.
    pub fn text(&self) -> &String {
        &self.0.last().unwrap().text
    }

    /// Returns a mutable reference to the source code buffer of the currently active entry.
    pub fn text_mut(&mut self) -> &mut String {
        &mut self.0.last_mut().unwrap().text
    }

    /// Returns the currently active entry index.
    pub fn index(&self) -> u32 {
        self.0.len() as u32 - 1
    }

    /// Returns the length (number of lines) of the currently active entry.
    pub fn len(&self) -> usize {
        self.0.last().unwrap().index().lines.len()
    }

    pub fn lineno(&self, loc: Location) -> Option<usize> {
        self.0[loc.index as usize].lineno(loc)
    }

    pub fn push(&mut self, name: String, text: String) {
        self.0.push(SourceEntry { name, text, index: RefCell::new(None) });
    }

    pub fn format<E : AsErrorWithContext>(&self, error: &E) -> String {
        self.0[error.location().index as usize].format(self, error)
    }
}


impl SourceEntry {

    pub fn lineno(&self, loc: Location) -> Option<usize> {
        if loc.is_empty() {
            None
        } else {
            Some(self.index().starts.partition_point(|u| u <= &loc.start) - 1)
        }
    }

    fn format<E : AsErrorWithContext>(&self, view: &SourceView, error: &E) -> String {
        let mut text = error.as_error();
        let index: Ref<'_, SourceIndex> = self.index();
        let loc = error.location();
        let start_lineno = self.lineno(loc).unwrap_or(0);
        let mut end_lineno = start_lineno;

        if !loc.is_empty() {
            while index.starts[end_lineno + 1] < loc.end() {
                end_lineno += 1;
            }
        }

        debug_assert!(start_lineno < index.lines.len());
        debug_assert!(end_lineno >= start_lineno && end_lineno < index.lines.len());

        let lineno = if start_lineno == end_lineno {
            format!("{}", start_lineno + 1)
        } else {
            format!("{} - {}", start_lineno + 1, end_lineno + 1)
        };

        let width: usize = format!("{}", end_lineno + 2).len();

        text.push_str(format!("\n  at: line {} ({})\n", lineno, self.name).as_str());
        error.add_stack_trace_elements(view, &mut text);
        text.push('\n');

        for i in start_lineno..=end_lineno {
            let line = &index.lines[i];
            text.push_str(format!("{:width$} |", i + 1, width = width).as_str());
            if !line.is_empty() {
                text.push(' ');
                text.push_str(line.as_str());
            }
            text.push('\n');
        }

        let (start_col, end_col) = if loc.is_empty() {
            let last_col: usize = index.lines[end_lineno].len();
            (last_col + 1, last_col + 3)
        } else {
            (loc.start - index.starts[start_lineno], loc.end() - index.starts[end_lineno])
        };

        text.push_str(format!("{:width$} |", end_lineno + 2, width = width).as_str());

        if start_lineno == end_lineno {
            // Single line error, so point to the exact column start + end
            // N.B. We match the indentation of the original line so that despite any weird tab/space mixing, at least both the '^' and original line match up
            let line = &index.lines[start_lineno];
            if !line.is_empty() {
                text.push(' ');
                for c in line.chars().take(start_col) {
                    match c {
                        '\t' => text.push('\t'),
                        _ => text.push(' '),
                    }
                }
            }

            for _ in start_col..=end_col {
                text.push('^');
            }
        } else {
            // For multi-line errors, just point directly up across the entire bottom line
            let longest_line_len: usize = (start_lineno..=end_lineno)
                .map(|u| index.lines[u].len())
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

    fn index(&self) -> Ref<'_, SourceIndex> {
        if self.index.borrow().is_none() {
            let mut lines: Vec<String> = Vec::new();
            let mut starts: Vec<usize> = Vec::new();
            let mut start = 0;

            for line in self.text.split('\n') {
                lines.push(String::from(line.strip_suffix('\r').unwrap_or(line)));
                starts.push(start);
                start += line.len() + 1;
            }

            starts.push(start);

            *self.index.borrow_mut() = Some(SourceIndex { starts, lines })
        }
        Ref::map(self.index.borrow(), |u| u.as_ref().unwrap())
    }
}




/// A simple common trait for converting arbitrary objects to human-readable errors
/// This could be implemented directly on the types, but using the trait allows all end-user-exposed text to be concentrated in this module.
pub trait AsError {
    fn as_error(&self) -> String;
}

/// An extension of `AsError` which is used for actual error types (`ScanError`, `ParserError`, `DetailRuntimeError`)
/// This is intentionally polymorphic, as it's used in `SourceView::format()`
pub trait AsErrorWithContext: AsError {
    fn location(&self) -> Location;

    /// When formatting a `RuntimeError`, allows inserting additional stack trace elements.
    /// This is appended *after* the initial `at: line X (source file)` line is appended.
    fn add_stack_trace_elements(&self, _: &SourceView, _: &mut String) {}
}


impl AsError for RuntimeError {
    fn as_error(&self) -> String {
        match self {
            RuntimeError::RuntimeExit | RuntimeError::RuntimeYield => panic!("Not a real error"),
            RuntimeError::RuntimeAssertFailed(reason) => format!("Assertion Failed: {}", reason),
            RuntimeError::RuntimeCompilationError(vec) => format!("Encountered compilation error(s) within 'eval':\n\n{}", vec.join("\n")),

            RuntimeError::ValueIsNotFunctionEvaluable(v) => format!("Tried to evaluate {} but it is not a function.", v.as_error()),
            RuntimeError::IncorrectArgumentsUserFunction(f, n) => format!("Incorrect number of arguments for {}, got {}", f.as_error(), n),
            RuntimeError::IncorrectArgumentsNativeFunction(f, n) => format!("Incorrect number of arguments for {}, got {}", f.as_error(), n),
            RuntimeError::IncorrectArgumentsGetField(s, n) => format!("Incorrect number of arguments for native (->'{}'), got {}", s, n),
            RuntimeError::IncorrectArgumentsStruct(s, n) => format!("Incorrect number of arguments for {}, got {}", s, n),

            RuntimeError::IOError(e) => format!("IOError: {}", e),
            RuntimeError::MonitorError(e) => format!("MonitorError: Illegal monitor command '{}'", e),

            RuntimeError::ValueErrorIndexOutOfBounds(i, ln) => format!("Index '{}' is out of bounds for list of length [0, {})", i, ln),
            RuntimeError::ValueErrorStepCannotBeZero => String::from("ValueError: 'step' argument cannot be zero"),
            RuntimeError::ValueErrorVariableNotDeclaredYet(x) => format!("ValueError: '{}' was referenced but has not been declared yet", x),
            RuntimeError::ValueErrorValueMustBeNonEmpty => String::from("ValueError: Expected value to be a non empty iterable"),
            RuntimeError::ValueErrorCannotUnpackLengthMustBeGreaterThan(e, a, v) => format!("ValueError: Cannot unpack {} with length {}, expected at least {} elements", v.as_error(), a, e),
            RuntimeError::ValueErrorCannotUnpackLengthMustBeEqual(e, a, v) => format!("ValueError: Cannot unpack {} with length {}, expected exactly {} elements", v.as_error(), a, e),
            RuntimeError::ValueErrorValueMustBeNonNegative(v) => format!("ValueError: Expected value '{}: int' to be non-negative", v),
            RuntimeError::ValueErrorValueMustBePositive(v) => format!("ValueError: Expected value '{}: int' to be positive", v),
            RuntimeError::ValueErrorValueMustBeNonZero => String::from("ValueError: Expected value to be non-zero"),
            RuntimeError::ValueErrorCannotCollectIntoDict(v) => format!("ValueError: Cannot collect key-value pair {} into a dict", v.as_error()),
            RuntimeError::ValueErrorKeyNotPresent(v) => format!("ValueError: Key {} not found in dictionary", v.as_error()),
            RuntimeError::ValueErrorInvalidCharacterOrdinal(i) => format!("ValueError: Cannot convert int {} to a character", i),
            RuntimeError::ValueErrorInvalidFormatCharacter(c) => format!("ValueError: Invalid format character '{}' in format string", c.as_error()),
            RuntimeError::ValueErrorNotAllArgumentsUsedInStringFormatting(v) => format!("ValueError: Not all arguments consumed in format string, next: {}", v.as_error()),
            RuntimeError::ValueErrorMissingRequiredArgumentInStringFormatting => String::from("ValueError: Not enough arguments for format string"),
            RuntimeError::ValueErrorEvalListMustHaveUnitLength(len) => format!("ValueError: Evaluating an index must have len = 1, got len = {}", len),
            RuntimeError::ValueErrorCannotCompileRegex(raw, err) => format!("ValueError: Cannot compile regex '{}'\n            {}", raw, err),
            RuntimeError::ValueErrorRecursiveHash(value) => format!("ValueError: Cannot create recursive hash based collection from {}", value.as_error()),

            RuntimeError::TypeErrorUnaryOp(op, v) => format!("TypeError: Argument to unary '{}' must be an int, got {}", op.as_error(), v.as_error()),
            RuntimeError::TypeErrorBinaryOp(op, l, r) => format!("TypeError: Cannot {} {} and {}", op.as_error(), l.as_error(), r.as_error()),
            RuntimeError::TypeErrorBinaryIs(l, r) => format!("TypeError: {} is not a type and cannot be used with binary 'is' on {}", r.as_error(), l.as_error()),
            RuntimeError::TypeErrorCannotConvertToInt(v) => format!("TypeError: Cannot convert {} to an int", v.as_error()),
            RuntimeError::TypeErrorFieldNotPresentOnValue(v, f, b) => match *b {
                true => format!("TypeError: Cannot get field '{}' on {}", f, v.to_repr_str().as_slice()),
                false => format!("TypeError: Cannot get field '{}' on {}", f, v.as_error().as_str()),
            },
            RuntimeError::TypeErrorArgMustBeInt(v) => format!("TypeError: Expected {} to be a int", v.as_error()),
            RuntimeError::TypeErrorArgMustBeComplex(v) => format!("TypeError: Expected {} to be a complex", v.as_error()),
            RuntimeError::TypeErrorArgMustBeStr(v) => format!("TypeError: Expected {} to be a string", v.as_error()),
            RuntimeError::TypeErrorArgMustBeChar(v) => format!("TypeError: Expected {} to be a single character string", v.as_error()),
            RuntimeError::TypeErrorArgMustBeIterable(v) => format!("TypeError: Expected {} to be an iterable", v.as_error()),
            RuntimeError::TypeErrorArgMustBeIndexable(ls) => format!("TypeError: Cannot index {}", ls.as_error()),
            RuntimeError::TypeErrorArgMustBeSliceable(ls) => format!("TypeError: Cannot slice {}", ls.as_error()),
            RuntimeError::TypeErrorArgMustBeList(v) => format!("TypeError: Expected {} to be a list", v.as_error()),
            RuntimeError::TypeErrorArgMustBeSet(v) => format!("TypeError: Expected {} to be a set", v.as_error()),
            RuntimeError::TypeErrorArgMustBeDict(v) => format!("TypeError: Expected {} to be a dict", v.as_error()),
            RuntimeError::TypeErrorArgMustBeFunction(v) => format!("TypeError: Expected {} to be a function", v.as_error()),
            RuntimeError::TypeErrorArgMustBeCmpOrKeyFunction(v) => format!("TypeError: Expected {} to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function", v.as_error()),
            RuntimeError::TypeErrorArgMustBeReplaceFunction(v) => format!("TypeError: Expected {} to be a 'fn replace(vector<str>) -> str' function", v.as_error()),
        }
    }
}

impl AsError for Option<char> {
    fn as_error(&self) -> String {
        match self {
            None => String::from("end of format string"),
            Some(c) => String::from(*c),
        }
    }
}

impl AsError for ValuePtr {
    fn as_error(&self) -> String {
        format!("'{}' of type '{}'", self.to_str().as_slice(), self.as_type_str())
    }
}

impl AsError for FunctionImpl {
    fn as_error(&self) -> String {
        self.repr()
    }
}

impl AsError for UnaryOp {
    fn as_error(&self) -> String {
        String::from(match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
        })
    }
}

impl AsError for BinaryOp {
    fn as_error(&self) -> String {
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
            BinaryOp::IsNot => "is not",
            BinaryOp::And => "&",
            BinaryOp::Or => "|",
            BinaryOp::Xor => "^",
            BinaryOp::In => "in",
            BinaryOp::NotIn => "in",
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
    fn as_error(&self) -> String {
        self.repr()
    }
}

impl AsError for ParserError {
    fn as_error(&self) -> String {
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
            ParserErrorType::ExpectedUnderscoreOrVariableNameOrPattern(e) => format!("Expected a variable binding, either a name, '_', or pattern, got {} instead", e.as_error()),
            ParserErrorType::ExpectedAnnotationOrNamedFunction(e) => format!("Expected another decorator, or a named function after decorator, got {} instead", e.as_error()),
            ParserErrorType::ExpectedStructNameAfterStruct(e) => format!("Expected a struct name after 'struct' keyword, got {} instead", e.as_error()),
            ParserErrorType::ExpectedFieldNameAfterArrow(e) => format!("Expected a field name after '->', got {} instead", e.as_error()),

            ParserErrorType::LocalVariableConflict(e) => format!("Multiple declarations for 'let {}' in the same scope", e),
            ParserErrorType::LocalVariableConflictWithNativeFunction(e) => format!("Name for variable '{}' conflicts with the native function by the same name", e),
            ParserErrorType::UndeclaredIdentifier(e) => format!("Undeclared identifier: '{}'", e),
            ParserErrorType::DuplicateFieldName(e) => format!("Duplicate field name: '{}'", e),
            ParserErrorType::InvalidFieldName(e) => format!("Invalid or unknown field name: '{}'", e),
            ParserErrorType::InvalidLValue(e) => format!("Invalid value used as a function parameter: '{}'", e),

            ParserErrorType::InvalidAssignmentTarget => String::from("The left hand side of an assignment expression must be a variable, array access, or property access"),
            ParserErrorType::MultipleVariadicTermsInPattern => String::from("Pattern is not allowed to have more than one variadic ('*') term."),
            ParserErrorType::LetWithPatternBindingNoExpression => String::from("'let' with a pattern variable must be followed by an expression if the pattern contains nontrivial pattern elements."),
            ParserErrorType::BreakOutsideOfLoop => String::from("Invalid 'break' statement outside of an enclosing loop"),
            ParserErrorType::ContinueOutsideOfLoop => String::from("Invalid 'continue' statement outside of an enclosing loop"),
            ParserErrorType::StructNotInGlobalScope => String::from("'struct' statements can only be present in global scope."),
            ParserErrorType::NonDefaultParameterAfterDefaultParameter => String::from("Non-default argument cannot follow default argument."),
            ParserErrorType::ParameterAfterVarParameter => String::from("Variadic parameter must be the last one in the function."),
            ParserErrorType::UnrollNotAllowedInSlice => String::from("Unrolled expression with '...' not allowed in slice literal."),
            ParserErrorType::UnrollNotAllowedInPartialOperator => String::from("Unrolled expression with '...' not allowed to be attached to a implicit partially-evaluated operator"),

            ParserErrorType::Runtime(e) => e.as_error(),
        }
    }
}

impl AsError for ScanError {
    fn as_error(&self) -> String {
        match &self.error {
            ScanErrorType::InvalidNumericPrefix(c) => format!("Invalid numeric prefix: '0{}'", c),
            ScanErrorType::InvalidNumericValue(e) => format!("Invalid numeric value: {}", e),
            ScanErrorType::InvalidCharacter(c) => format!("Invalid character: '{}'", c),
            ScanErrorType::UnterminatedStringLiteral => String::from("Unterminated string literal (missing a closing quote)"),
            ScanErrorType::UnterminatedBlockComment => String::from("Unterminated block comment (missing a closing '*/')"),
        }
    }
}

impl AsError for Option<ScanToken> {
    fn as_error(&self) -> String {
        match self {
            Some(t) => t.as_error(),
            None => String::from("end of input"),
        }
    }
}

impl AsError for ScanToken {
    fn as_error(&self) -> String {
        match &self {
            ScanToken::Identifier(s) => format!("identifier \'{}\'", s),
            ScanToken::StringLiteral(s) => format!("string '{}'", s),
            ScanToken::IntLiteral(i) => format!("integer '{}'", i),
            ScanToken::ComplexLiteral(i) => format!("complex integer '{}i'", i),

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
            ScanToken::KeywordAssert => String::from("'assert' keyword"),
            ScanToken::KeywordMod => String::from("'mod' keyword"),
            ScanToken::KeywordSelf => String::from("'self' keyword"),
            ScanToken::KeywordNative => String::from("'native' keyword"),

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

            ScanToken::LogicalAnd => String::from("'and' keyword"),
            ScanToken::LogicalOr => String::from("'or' keyword"),

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
            ScanToken::Ellipsis => String::from("'...' token"),
            ScanToken::QuestionMark => String::from("'?' token"),

            ScanToken::NewLine => String::from("new line"),
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::reporting::{AsError, AsErrorWithContext, Location, SourceView};

    #[test]
    fn test_or_location() {
        let l1 = Location::new(0, 5, 0);
        let l2 = Location::new(8, 2, 0);

        assert_eq!(l1 | l2, Location::new(0, 10, 0));

        let mut l3 = Location::new(2, 4, 0);
        l3 |= l1;

        assert_eq!(l3, Location::new(0, 6, 0));
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
        let text = String::from("first += line\nsecond line?\nthird line\r\nwindows line\n\nempty\r\n\r\nmore empty");
        let src = SourceView::new(String::from("<test>"), text);
        let error = src.format(&MockError("Error", Location::new(start, (end - start + 1) as u32, 0)));

        assert_eq!(error.as_str(), expected);
    }

    #[test] fn test_layout() { assert_eq!(16, std::mem::size_of::<Location>()); }
}
