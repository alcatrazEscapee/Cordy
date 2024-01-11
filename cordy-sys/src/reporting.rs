use std::borrow::Cow;
use std::cell::{Ref, RefCell};
use std::ops::{BitOr, BitOrAssign};

use crate::compiler::{ParserError, ParserErrorType, ScanError, ScanErrorType, ScanToken};
use crate::vm::{RuntimeError, ValuePtr};
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
        use RuntimeError::{*};
        match self {
            RuntimeExit | RuntimeYield => panic!("Not a real error"),
            RuntimeAssertFailed(reason, message) => {
                let mut error = String::from("Assertion Failed:");
                let mut any = false;
                if !reason.is_nil() {
                    error.push(' ');
                    error.push_str(reason.as_str_slice());
                    any = true;
                }
                if !message.is_nil() {
                    error.push(if any { '\n' } else { ' '});
                    error.push_str(message.as_str_slice())
                }
                error
            },
            RuntimeCompilationError(vec) => format!("Encountered compilation error(s) within 'eval':\n\n{}", vec.join("\n")),

            ValueIsNotFunctionEvaluable(v) => format!("Tried to evaluate {} but it is not a function.", v.to_repr_str()),
            IncorrectArgumentsUserFunction(f, n) => format!("Incorrect number of arguments for {}, got {}", f.to_repr_str(), n),
            IncorrectArgumentsNativeFunction(f, n) => format!("Incorrect number of arguments for {}, got {}", f.to_repr_str(), n),
            IncorrectArgumentsGetField(s, n) => format!("Incorrect number of arguments for native (->'{}'), got {}", s, n),
            IncorrectArgumentsStruct(s, n) => format!("Incorrect number of arguments for {}, got {}", s, n),

            IOError(e) => format!("IOError: {}", e),
            OSError(e) => format!("OSError: {}", e),
            MonitorError(e) => format!("MonitorError: Illegal monitor command '{}'", e),
            RationalNotImplementedError => String::from("PlatformError: Rationals are not implemented on this platform"),

            ValueErrorIndexOutOfBounds(i, ln) => format!("Index '{}' is out of bounds for list of length [0, {})", i, ln),
            ValueErrorStepCannotBeZero => String::from("ValueError: 'step' argument cannot be zero"),
            ValueErrorValueWouldBeTooLarge => String::from("ValueError: value would be too large!"),
            ValueErrorVariableNotDeclaredYet(x) => format!("ValueError: '{}' was referenced but has not been declared yet", x),
            ValueErrorArgMustBeNonEmpty(f) => format!("ValueError: Expected '{}' argument to be a non-empty iterable", f.name()),
            ValueErrorCannotConvertStringToIntBadParse(v, err) => format!("ValueError: Cannot convert {} to an integer: {}", v.as_error(), err),
            ValueErrorCannotConvertStringToRationalBadParse(v, err) => format!("ValueError: Cannot convert {} to an rational: {}", v.as_error(), err),
            ValueErrorCannotConvertRationalToIntTooLarge(v) => format!("ValueError: Cannot convert rational '{}' to an integer as it is too large for the target type", v.as_error()),
            ValueErrorCannotConvertRationalToIntNotAnInteger(v) => format!("ValueError: Cannot convert rational '{}' to an integer as it is not an integral value.", v.as_error()),
            ValueErrorCannotUnpackLengthMustBeGreaterThan(e, a, v) => format!("ValueError: Cannot unpack {} with length {}, expected at least {} elements", v.as_error(), a, e),
            ValueErrorCannotUnpackLengthMustBeEqual(e, a, v) => format!("ValueError: Cannot unpack {} with length {}, expected exactly {} elements", v.as_error(), a, e),
            ValueErrorValueMustBeNonNegative(v) => format!("ValueError: Expected {} to be non-negative", v.as_error()),
            ValueErrorArgMustBeNonNegativeInt(f) => format!("ValueError: Expected {} arg to be a non-negative integer", f.name()),
            ValueErrorBinaryOpMustNotBeNegative(lhs, op, n) => format!("ValueError: Operator '{}' is not supported for arguments of type {} and negative integer {}", op.as_error(), lhs.as_type_str(), n),
            ValueErrorPowerByNegative(v) => format!("ValueError: Power of an integral value by '{}' which is negative", v),
            ValueErrorValueMustBePositive(v) => format!("ValueError: Expected value '{}: int' to be positive", v),
            ValueErrorDivideByZero => String::from("ValueError: Division by zero"),
            ValueErrorModuloByZero => String::from("ValueError: Modulo by zero"),
            ValueErrorCannotCollectIntoDict(v) => format!("ValueError: Cannot collect key-value pair {} into a dict", v.as_error()),
            ValueErrorKeyNotPresent(v) => format!("ValueError: Key {} not found in dictionary", v.as_error()),
            ValueErrorInvalidCharacterOrdinal(i) => format!("ValueError: Cannot convert int {} to a character", i),
            ValueErrorInvalidFormatCharacter(c) => match c {
                Some(c) => format!("ValueError: Invalid format character '{}' in format string", c),
                None => format!("ValueError: Expected format character after '%', got end of string")
            }
            ValueErrorNotAllArgumentsUsedInStringFormatting(v) => format!("ValueError: Not all arguments consumed in format string, next: {}", v.as_error()),
            ValueErrorMissingRequiredArgumentInStringFormatting => String::from("ValueError: Not enough arguments for format string"),
            ValueErrorEvalListMustHaveUnitLength(len) => format!("ValueError: Evaluating an index must have len = 1, got len = {}", len),
            ValueErrorCannotCompileRegex(raw, err) => format!("ValueError: Cannot compile regex '{}'\n            {}", raw, err),
            ValueErrorRecursiveHash => String::from("ValueError: Cannot create recursive hash-based collection"),

            TypeErrorUnaryOp(op, v) => format!("TypeError: Operator unary '{}' is not supported for type {}", op.as_error(), v.as_type_str()),
            TypeErrorBinaryOp(op, l, r) => format!("TypeError: Operator '{}' is not supported for arguments of type {} and {}", op.as_error(), l.as_type_str(), r.as_type_str()),
            TypeErrorBinaryIs(l, r) => format!("TypeError: {} is not a type and cannot be used with binary 'is' on {}", r.as_error(), l.as_error()),
            TypeErrorCannotConvertToInt(v) => format!("TypeError: Cannot convert {} to an int", v.as_error()),
            TypeErrorCannotConvertToRational(v) => format!("TypeError: Cannot convert {} to an rational", v.as_error()),
            TypeErrorFieldNotPresentOnValue { value, field, repr, access} => format!(
                "TypeError: Cannot {} field '{}' on {}",
                if *access { "get" } else { "set" },
                field,
                if *repr { value.to_repr_str() } else { Cow::from(value.as_error()) }
            ),
            TypeErrorArgMustBeInt(v) => format!("TypeError: Expected {} to be a int", v.as_error()),
            TypeErrorArgMustBeIntOrStr(v) => format!("TypeError: Expected {} to be a int or string", v.as_error()),
            TypeErrorArgMustBeComplex(v) => format!("TypeError: Expected {} to be a complex", v.as_error()),
            TypeErrorArgMustBeRational(v) => format!("TypeError: Expected {} to be a rational", v.as_error()),
            TypeErrorArgMustBeStr(v) => format!("TypeError: Expected {} to be a string", v.as_error()),
            TypeErrorArgMustBeChar(v) => format!("TypeError: Expected {} to be a single character string", v.as_error()),
            TypeErrorArgMustBeIterable(v) => format!("TypeError: Expected {} to be an iterable", v.as_error()),
            TypeErrorArgMustBeIndexable(ls) => format!("TypeError: Cannot index {}", ls.as_error()),
            TypeErrorArgMustBeSliceable(ls) => format!("TypeError: Cannot slice {}", ls.as_error()),
            TypeErrorArgMustBeList(v) => format!("TypeError: Expected {} to be a list", v.as_error()),
            TypeErrorArgMustBeSet(v) => format!("TypeError: Expected {} to be a set", v.as_error()),
            TypeErrorArgMustBeDict(v) => format!("TypeError: Expected {} to be a dict", v.as_error()),
            TypeErrorArgMustBeFunction(v) => format!("TypeError: Expected {} to be a function", v.as_error()),
            TypeErrorArgMustBeCmpOrKeyFunction(v) => format!("TypeError: Expected {} to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function", v.as_error()),
            TypeErrorArgMustBeReplaceFunction(v) => format!("TypeError: Expected {} to be a 'fn replace(vector<str>) -> str' function", v.as_error()),
        }
    }
}

impl AsError for ValuePtr {
    fn as_error(&self) -> String {
        format!("'{}' of type '{}'", self.to_str(), self.as_type_str())
    }
}

impl AsError for UnaryOp {
    fn as_error(&self) -> String {
        use UnaryOp::{*};
        String::from(match self {
            Neg => "-",
            Not => "!",
            LogicalNot => "not",
        })
    }
}

impl AsError for BinaryOp {
    fn as_error(&self) -> String {
        use BinaryOp::{*};
        String::from(match self {
            Div => "/",
            Mul => "*",
            Mod => "%",
            Add => "+",
            Sub => "-",
            LeftShift => "<<",
            RightShift => ">>",
            Pow => "**",
            Is => "is",
            IsNot => "is not",
            And => "&",
            Or => "|",
            Xor => "^",
            In => "in",
            NotIn => "in",
            LessThan => "<",
            GreaterThan => ">",
            LessThanEqual => "<=",
            GreaterThanEqual => ">=",
            Equal => "==",
            NotEqual => "!=",
            Max => "min",
            Min => "max",
        })
    }
}

impl AsError for ParserError {
    fn as_error(&self) -> String {
        use ParserErrorType::{*};
        match &self.error {
            UnexpectedTokenAfterEoF(e) => format!("Unexpected {} after parsing finished", e.as_error()),

            ExpectedToken(e, a) => format!("Expected a {}, got {} instead", e.as_error(), a.as_error()),
            ExpectedExpressionTerminal(e) => format!("Expected an expression terminal, got {} instead", e.as_error()),
            ExpectedCommaOrEndOfArguments(e) => format!("Expected a ',' or ')' after function invocation, got {} instead", e.as_error()),
            ExpectedCommaOrEndOfList(e) => format!("Expected a ',' or ']' after list literal, got {} instead", e.as_error()),
            ExpectedCommaOrEndOfVector(e) => format!("Expected a ',' or ')' after vector literal, got {} instead", e.as_error()),
            ExpectedCommaOrEndOfDict(e) => format!("Expected a ',' or '}}' after dict literal, got {} instead", e.as_error()),
            ExpectedCommaOrEndOfSet(e) => format!("Expected a ',' or '}}' after set literal, got {} instead", e.as_error()),
            ExpectedColonOrEndOfSlice(e) => format!("Expected a ':' or ']' in slice, got {} instead", e.as_error()),
            ExpectedStatement(e) => format!("Expected a statement, got {} instead", e.as_error()),
            ExpectedVariableNameAfterLet(e) => format!("Expected a variable name after 'let' keyword, got {} instead", e.as_error()),
            ExpectedVariableNameAfterFor(e) => format!("Expected a variable name after 'for' keyword, got {} instead", e.as_error()),
            ExpectedFunctionNameAfterFn(e) => format!("Expected a function name after 'fn' keyword, got {} instead", e.as_error()),
            ExpectedFunctionBlockOrArrowAfterFn(e) => format!("Expected a function body starting with '{{' or `->` after 'fn', got {} instead", e.as_error()),
            ExpectedParameterOrEndOfList(e) => format!("Expected a function parameter or ')' after function declaration, got {} instead", e.as_error()),
            ExpectedCommaOrEndOfParameters(e) => format!("Expected a ',' or ')' after function parameter, got {} instead", e.as_error()),
            ExpectedPatternTerm(e) => format!("Expected a name, '_', or variadic term in a pattern variable, got {} instead", e.as_error()),
            ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(e) => format!("Expected a name or '_' after '*' in a pattern variable, got {} instead", e.as_error()),
            ExpectedUnderscoreOrVariableNameOrPattern(e) => format!("Expected a variable binding, either a name, '_', or pattern, got {} instead", e.as_error()),
            ExpectedAnnotationOrNamedFunction(e) => format!("Expected another decorator, or a named function after decorator, got {} instead", e.as_error()),
            ExpectedStructNameAfterStruct(e, is_module) => format!("Expected a name after '{}' keyword, got {} instead", if *is_module { "module" } else { "struct" }, e.as_error()),
            ExpectedFieldNameAfterArrow(e) => format!("Expected a field name after '->', got {} instead", e.as_error()),
            ExpectedFieldNameInStruct(e) => format!("Expected a field name in struct declaration, got {} instead", e.as_error()),
            ExpectedFunctionInStruct(e, is_module) => format!("Expected a function within {} body, got {} instead", if *is_module { "module" } else { "struct" }, e.as_error()),

            LocalVariableConflict(e) => format!("Duplicate definition of variable '{}' in the same scope", e),
            LocalVariableConflictWithNativeFunction(e) => format!("Name for variable '{}' conflicts with the native function by the same name", e),
            UndeclaredIdentifier(e) => format!("Undeclared identifier: '{}'", e),
            DuplicateFieldName(e) => format!("Duplicate field name: '{}'", e),
            InvalidFieldName(e) => format!("Invalid or unknown field name: '{}'", e),
            InvalidLValue(e, native) => format!("Invalid value used as a {}function parameter: '{}'", if *native { "native " } else { "" }, e),

            InvalidAssignmentTarget => String::from("The left hand side is not a valid assignment target"),
            MultipleVariadicTermsInPattern => String::from("Pattern is not allowed to have more than one variadic ('*') term"),
            LetWithPatternBindingNoExpression => String::from("'let' with a pattern variable must be followed by an expression if the pattern contains nontrivial pattern elements"),
            BreakOutsideOfLoop => String::from("Invalid 'break' statement outside of an enclosing loop"),
            ContinueOutsideOfLoop => String::from("Invalid 'continue' statement outside of an enclosing loop"),
            StructNotInGlobalScope => String::from("'struct' statements can only be present in global scope"),
            NonDefaultParameterAfterDefaultParameter => String::from("Non-default argument cannot follow default argument"),
            ParameterAfterVarParameter => String::from("Variadic parameter must be the last one in the function"),
            UnrollNotAllowedInSlice => String::from("Unrolled expression with '...' not allowed in slice literal"),
            UnrollNotAllowedInPartialOperator => String::from("Unrolled expression with '...' not allowed to be attached to a implicit partially-evaluated operator"),

            Runtime(e) => e.as_err().as_error(),
        }
    }
}

impl AsError for ScanError {
    fn as_error(&self) -> String {
        use ScanErrorType::{*};
        match &self.error {
            InvalidNumericPrefix(c) => format!("Invalid numeric prefix: '0{}'", c),
            InvalidNumericValue(e) => format!("Invalid numeric value: {}", e),
            InvalidCharacter(c) => format!("Invalid character: '{}'", c),
            UnterminatedStringLiteral => String::from("Unterminated string literal (missing a closing quote)"),
            UnterminatedBlockComment => String::from("Unterminated block comment (missing a closing '*/')"),
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
        use ScanToken::{*};
        match &self {
            Identifier(s) => format!("identifier \'{}\'", s),
            StringLiteral(s) => format!("string '{}'", s),
            IntLiteral(i) => format!("integer '{}'", i),
            ComplexLiteral(i) => format!("complex integer '{}i'", i),

            KeywordLet => String::from("'let' keyword"),
            KeywordFn => String::from("'fn' keyword"),
            KeywordReturn => String::from("'return' keyword"),
            KeywordIf => String::from("'if' keyword"),
            KeywordElif => String::from("'elif' keyword"),
            KeywordElse => String::from("'else' keyword"),
            KeywordThen => String::from("'then' keyword"),
            KeywordLoop => String::from("'loop' keyword"),
            KeywordWhile => String::from("'while' keyword"),
            KeywordFor => String::from("'for' keyword"),
            KeywordIn => String::from("'in' keyword"),
            KeywordIs => String::from("'is' keyword"),
            KeywordNot => String::from("'not' keyword"),
            KeywordBreak => String::from("'break' keyword"),
            KeywordContinue => String::from("'continue' keyword"),
            KeywordDo => String::from("'do' keyword"),
            KeywordTrue => String::from("'true' keyword"),
            KeywordFalse => String::from("'false' keyword"),
            KeywordNil => String::from("'nil' keyword"),
            KeywordStruct => String::from("'struct' keyword"),
            KeywordExit => String::from("'exit' keyword"),
            KeywordAssert => String::from("'assert' keyword"),
            KeywordModule => String::from("'mod' keyword"),
            KeywordSelf => String::from("'self' keyword"),
            KeywordNative => String::from("'native' keyword"),

            Equals => String::from("'=' token"),
            PlusEquals => String::from("'+=' token"),
            MinusEquals => String::from("'-=' token"),
            MulEquals => String::from("'*=' token"),
            DivEquals => String::from("'/=' token"),
            AndEquals => String::from("'&=' token"),
            OrEquals => String::from("'|=' token"),
            XorEquals => String::from("'^=' token"),
            LeftShiftEquals => String::from("'<<=' token"),
            RightShiftEquals => String::from("'>>=' token"),
            ModEquals => String::from("'%=' token"),
            PowEquals => String::from("'**=' token"),
            DotEquals => String::from("'.=' token"),

            Plus => String::from("'+' token"),
            Minus => String::from("'-' token"),
            Mul => String::from("'*' token"),
            Div => String::from("'/' token"),
            BitwiseAnd => String::from("'&' token"),
            BitwiseOr => String::from("'|' token"),
            BitwiseXor => String::from("'^' token"),
            Mod => String::from("'%' token"),
            Pow => String::from("'**' token"),
            LeftShift => String::from("'<<' token"),
            RightShift => String::from("'>>' token"),

            Not => String::from("'!' token"),

            LogicalAnd => String::from("'and' keyword"),
            LogicalOr => String::from("'or' keyword"),

            NotEquals => String::from("'!=' token"),
            DoubleEquals => String::from("'==' token"),
            LessThan => String::from("'<' token"),
            LessThanEquals => String::from("'<=' token"),
            GreaterThan => String::from("'>' token"),
            GreaterThanEquals => String::from("'>=' token"),

            OpenParen => String::from("'(' token"), // ( )
            CloseParen => String::from("')' token"),
            OpenSquareBracket => String::from("'[' token"), // [ ]
            CloseSquareBracket => String::from("']' token"),
            OpenBrace => String::from("'{' token"), // { }
            CloseBrace => String::from("'}' token"),

            Comma => String::from("',' token"),
            Dot => String::from("'.' token"),
            Colon => String::from("':' token"),
            Arrow => String::from("'->' token"),
            Underscore => String::from("'_' token"),
            Semicolon => String::from("';' token"),
            At => String::from("'@' token"),
            Ellipsis => String::from("'...' token"),
            QuestionMark => String::from("'?' token"),

            NewLine => String::from("new line"),
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
