use crate::compiler::scanner::ScanToken;
use crate::reporting::{AsErrorWithContext, Location};

use ParserErrorType::{*};


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ParserError {
    pub error: ParserErrorType,
    pub loc: Option<Location>,
}

impl ParserError {
    pub fn new(error: ParserErrorType, loc: Option<Location>) -> ParserError {
        ParserError { error, loc }
    }

    /// Returns `true` if the error is due to encountering an EoF (end of input) while expecting another token.
    /// Used for detecting if we need to let the user continue entering input in REPL mode.
    pub fn is_eof(self: &Self) -> bool {
        match &self.error {
            UnexpectedTokenAfterEoF(_) => false,

            ExpectedToken(_, it) => it.is_none(),
            ExpectedExpressionTerminal(it) |
            ExpectedCommaOrEndOfArguments(it) |
            ExpectedCommaOrEndOfList(it) |
            ExpectedCommaOrEndOfVector(it) |
            ExpectedCommaOrEndOfDict(it) |
            ExpectedCommaOrEndOfSet(it) |
            ExpectedColonOrEndOfSlice(it) |
            ExpectedStatement(it) |
            ExpectedVariableNameAfterLet(it) |
            ExpectedVariableNameAfterFor(it) |
            ExpectedFunctionNameAfterFn(it) |
            ExpectedFunctionBlockOrArrowAfterFn(it) |
            ExpectedParameterOrEndOfList(it) |
            ExpectedCommaOrEndOfParameters(it) |
            ExpectedPatternTerm(it) |
            ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(it) |
            ExpectedUnderscoreOrVariableNameOrPattern(it) |
            ExpectedAnnotationOrNamedFunction(it) |
            ExpectedAnnotationOrAnonymousFunction(it) => it.is_none(),

            LocalVariableConflict(_) |
            LocalVariableConflictWithNativeFunction(_) |
            UndeclaredIdentifier(_) => false,

            InvalidAssignmentTarget |
            MultipleVariadicTermsInPattern |
            LetWithPatternBindingNoExpression |
            BreakOutsideOfLoop |
            ContinueOutsideOfLoop => false,
        }
    }
}

impl AsErrorWithContext for ParserError {
    fn location(self: &Self) -> &Option<Location> {
        &self.loc
    }
}


#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ParserErrorType {
    UnexpectedTokenAfterEoF(ScanToken),

    ExpectedToken(ScanToken, Option<ScanToken>),
    ExpectedExpressionTerminal(Option<ScanToken>),
    ExpectedCommaOrEndOfArguments(Option<ScanToken>),
    ExpectedCommaOrEndOfList(Option<ScanToken>),
    ExpectedCommaOrEndOfVector(Option<ScanToken>),
    ExpectedCommaOrEndOfDict(Option<ScanToken>),
    ExpectedCommaOrEndOfSet(Option<ScanToken>),
    ExpectedColonOrEndOfSlice(Option<ScanToken>),
    ExpectedStatement(Option<ScanToken>),
    ExpectedVariableNameAfterLet(Option<ScanToken>),
    ExpectedVariableNameAfterFor(Option<ScanToken>),
    ExpectedFunctionNameAfterFn(Option<ScanToken>),
    ExpectedFunctionBlockOrArrowAfterFn(Option<ScanToken>),
    ExpectedParameterOrEndOfList(Option<ScanToken>),
    ExpectedCommaOrEndOfParameters(Option<ScanToken>),
    ExpectedPatternTerm(Option<ScanToken>),
    ExpectedUnderscoreOrVariableNameAfterVariadicInPattern(Option<ScanToken>),
    ExpectedUnderscoreOrVariableNameOrPattern(Option<ScanToken>),
    ExpectedAnnotationOrNamedFunction(Option<ScanToken>),
    ExpectedAnnotationOrAnonymousFunction(Option<ScanToken>),

    LocalVariableConflict(String),
    LocalVariableConflictWithNativeFunction(String),
    UndeclaredIdentifier(String),

    InvalidAssignmentTarget,
    MultipleVariadicTermsInPattern,
    LetWithPatternBindingNoExpression,
    BreakOutsideOfLoop,
    ContinueOutsideOfLoop,
}