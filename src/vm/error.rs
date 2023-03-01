use std::rc::Rc;
use crate::reporting::{AsError, AsErrorWithContext, Location, Locations, SourceView};

use crate::stdlib::NativeFunction;
use crate::vm::CallFrame;
use crate::vm::operator::{BinaryOp, UnaryOp};
use crate::vm::value::{FunctionImpl, Value};


#[derive(Debug)]
pub enum RuntimeError {
    RuntimeExit,
    RuntimeYield,

    RuntimeCompilationError(Vec<String>),

    ValueIsNotFunctionEvaluable(Value),

    IncorrectNumberOfFunctionArguments(FunctionImpl, u8),
    IncorrectNumberOfArguments(NativeFunction, u8, u8),
    IncorrectNumberOfArgumentsVariadicAtLeastOne(NativeFunction),

    ValueErrorIndexOutOfBounds(i64, usize),
    ValueErrorStepCannotBeZero,
    ValueErrorVariableNotDeclaredYet(String),
    ValueErrorValueMustBeNonNegative(i64),
    ValueErrorValueMustBePositive(i64),
    ValueErrorValueMustBeNonZero,
    ValueErrorValueMustBeNonEmpty,
    ValueErrorCannotUnpackLengthMustBeGreaterThan(usize, usize, Value), // expected, actual
    ValueErrorCannotUnpackLengthMustBeEqual(usize, usize, Value), // expected, actual
    ValueErrorCannotCollectIntoDict(Value),
    ValueErrorKeyNotPresent(Value),
    ValueErrorInvalidCharacterOrdinal(i64),
    ValueErrorInvalidFormatCharacter(Option<char>),
    ValueErrorNotAllArgumentsUsedInStringFormatting(Value),
    ValueErrorMissingRequiredArgumentInStringFormatting,
    ValueErrorEvalListMustHaveUnitLength(usize),

    TypeErrorUnaryOp(UnaryOp, Value),
    TypeErrorBinaryOp(BinaryOp, Value, Value),
    TypeErrorBinaryIs(Value, Value),
    TypeErrorCannotConvertToInt(Value),

    TypeErrorArgMustBeInt(Value),
    TypeErrorArgMustBeStr(Value),
    TypeErrorArgMustBeChar(Value),
    TypeErrorArgMustBeIterable(Value),
    TypeErrorArgMustBeIndexable(Value),
    TypeErrorArgMustBeSliceable(Value),
    TypeErrorArgMustBeDict(Value),
    TypeErrorArgMustBeFunction(Value),
    TypeErrorArgMustBeCmpOrKeyFunction(Value),
}

impl RuntimeError {
    #[cold]
    pub fn err<T>(self: Self) -> Result<T, Box<RuntimeError>> {
        Err(Box::new(self))
    }

    pub fn with_stacktrace(self: Self, ip: usize, call_stack: &Vec<CallFrame>, functions: &Vec<Rc<FunctionImpl>>, locations: &Locations) -> DetailRuntimeError {
        // Top level stack frame refers to the code being executed
        let target: Location = locations[ip];
        let mut stack: Vec<StackFrame> = Vec::new();
        let mut prev_ip: usize = ip;

        for frame in call_stack.iter().rev() {
            if frame.return_ip > 0 {
                // Each frame from the call stack refers to the owning function of the previous frame
                let frame_ip: usize = frame.return_ip - 1;
                stack.push(StackFrame(frame_ip, locations[frame_ip], find_owning_function(prev_ip, functions)));
                prev_ip = frame_ip;
            }
        }

        DetailRuntimeError { error: self, target, stack }
    }
}

/// A `RuntimeError` with a filled-in stack trace, and source location which caused the error.
#[derive(Debug)]
pub struct DetailRuntimeError {
    error: RuntimeError,
    target: Location,

    /// The stack trace elements, including a location (typically of the function call), and the function name itself
    stack: Vec<StackFrame>,
}

#[derive(Debug)]
pub struct StackFrame(usize, Location, String);

impl AsError for DetailRuntimeError {
    fn as_error(self: &Self) -> String {
        self.error.as_error()
    }
}

impl AsErrorWithContext for DetailRuntimeError {
    fn location(self: &Self) -> Location {
        self.target
    }

    fn add_stack_trace_elements(self: &Self, view: &SourceView, text: &mut String) {
        for StackFrame(_, loc, site) in &self.stack {
            text.push_str(format!("  at: `{}` (line {})\n", site, view.lineno(*loc) + 1).as_str());
        }
    }
}


/// The owning function for a given IP can be defined as the closest function which encloses the desired instruction
/// We annotate both head and tail of `FunctionImpl` to make this search easy
fn find_owning_function(ip: usize, functions: &Vec<Rc<FunctionImpl>>) -> String {
    functions.iter()
        .filter(|f| f.head <= ip && ip <= f.tail)
        .min_by_key(|f| f.tail - f.head)
        .map(|f| f.as_str())
        .unwrap_or_else(|| String::from("<script>"))
}
