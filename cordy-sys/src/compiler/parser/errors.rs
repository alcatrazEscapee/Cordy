use crate::compiler::scanner::ScanToken;
use crate::reporting::{AsErrorWithContext, Location};
use crate::vm::ErrorPtr;

use ParserErrorType::{*};


#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ParserError {
    pub error: ParserErrorType,
    pub loc: Location,
}

impl ParserError {
    pub fn new(error: ParserErrorType, loc: Location) -> ParserError {
        ParserError { error, loc }
    }

    /// Returns `true` if the error is due to encountering an EoF (end of input) while expecting another token.
    /// Used for detecting if we need to let the user continue entering input in REPL mode.
    pub fn is_eof(&self) -> bool {
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
            ExpectedStructNameAfterStruct(it, _) |
            ExpectedFieldNameAfterArrow(it) |
            ExpectedFieldNameInStruct(it) |
            ExpectedFunctionInStruct(it, _) => it.is_none(),

            LocalVariableConflict(_) |
            LocalVariableConflictWithNativeFunction(_) |
            UndeclaredIdentifier(_) |
            DuplicateFieldName(_) |
            InvalidFieldName(_) |
            InvalidLValue(_, _) => false,

            InvalidAssignmentTarget |
            MultipleVariadicTermsInPattern |
            LetWithNonTrivialPattern |
            LetWithTrivialEmptyPattern |
            LetWithTrivialVarNamed |
            BreakOutsideOfLoop |
            ContinueOutsideOfLoop |
            StructNotInGlobalScope |
            NonDefaultParameterAfterDefaultParameter |
            ParameterAfterVarParameter |
            UnrollNotAllowedInSlice |
            UnrollNotAllowedInPartialOperator => false,

            Runtime(_) => false,
        }
    }
}

impl AsErrorWithContext for ParserError {
    fn location(&self) -> Location {
        self.loc
    }
}


#[derive(Debug, Clone, Hash, Eq, PartialEq)]
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
    ExpectedStructNameAfterStruct(Option<ScanToken>, bool), // is_module
    ExpectedFieldNameAfterArrow(Option<ScanToken>),
    ExpectedFieldNameInStruct(Option<ScanToken>),
    ExpectedFunctionInStruct(Option<ScanToken>, bool), // is_module

    LocalVariableConflict(String),
    LocalVariableConflictWithNativeFunction(String),
    UndeclaredIdentifier(String),
    DuplicateFieldName(String),
    InvalidFieldName(String),
    InvalidLValue(String, bool), // parameter, native

    InvalidAssignmentTarget,
    MultipleVariadicTermsInPattern,
    LetWithNonTrivialPattern,
    LetWithTrivialEmptyPattern,
    LetWithTrivialVarNamed,
    BreakOutsideOfLoop,
    ContinueOutsideOfLoop,
    StructNotInGlobalScope,
    NonDefaultParameterAfterDefaultParameter,
    ParameterAfterVarParameter,
    UnrollNotAllowedInSlice,
    UnrollNotAllowedInPartialOperator,

    Runtime(ErrorPtr),
}