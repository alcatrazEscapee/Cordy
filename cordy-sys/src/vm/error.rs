use crate::core::NativeFunction;
use crate::reporting::{AsError, AsErrorWithContext, Location, SourceView};
use crate::vm::{CallFrame, IntoValue, Type, ValueResult};
use crate::vm::operator::{BinaryOp, UnaryOp};
use crate::vm::value::{FunctionImpl, Prefix, ValuePtr};


#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RuntimeError {
    RuntimeExit,
    RuntimeYield,
    RuntimeAssertFailed(String),
    RuntimeCompilationError(Vec<String>),

    ValueIsNotFunctionEvaluable(ValuePtr),

    IncorrectArgumentsUserFunction(FunctionImpl, u32),
    IncorrectArgumentsNativeFunction(NativeFunction, u32),
    IncorrectArgumentsGetField(String, u32),
    IncorrectArgumentsStruct(String, u32),

    IOError(String),
    MonitorError(String),

    ValueErrorIndexOutOfBounds(i64, usize),
    ValueErrorStepCannotBeZero,
    ValueErrorVariableNotDeclaredYet(String),
    ValueErrorValueMustBeNonNegative(i64),
    ValueErrorValueMustBePositive(i64),
    ValueErrorValueMustBeNonZero,
    ValueErrorValueMustBeNonEmpty,
    ValueErrorCannotUnpackLengthMustBeGreaterThan(u32, usize, ValuePtr), // expected, actual
    ValueErrorCannotUnpackLengthMustBeEqual(u32, usize, ValuePtr), // expected, actual
    ValueErrorCannotCollectIntoDict(ValuePtr),
    ValueErrorKeyNotPresent(ValuePtr),
    ValueErrorInvalidCharacterOrdinal(i64),
    ValueErrorInvalidFormatCharacter(Option<char>),
    ValueErrorNotAllArgumentsUsedInStringFormatting(ValuePtr),
    ValueErrorMissingRequiredArgumentInStringFormatting,
    ValueErrorEvalListMustHaveUnitLength(usize),
    ValueErrorCannotCompileRegex(String, String),
    ValueErrorRecursiveHash(ValuePtr),

    TypeErrorUnaryOp(UnaryOp, ValuePtr),
    TypeErrorBinaryOp(BinaryOp, ValuePtr, ValuePtr),
    TypeErrorBinaryIs(ValuePtr, ValuePtr),
    TypeErrorCannotConvertToInt(ValuePtr),
    TypeErrorFieldNotPresentOnValue {
        value: ValuePtr,
        field: String,
        repr: bool, // If true, print the value using it's repr() instead. Used for structs and modules
        access: bool, // Was the field reference an access (get) or mutation (set)
    },

    TypeErrorArgMustBeInt(ValuePtr),
    TypeErrorArgMustBeComplex(ValuePtr),
    TypeErrorArgMustBeStr(ValuePtr),
    TypeErrorArgMustBeChar(ValuePtr),
    TypeErrorArgMustBeIterable(ValuePtr),
    TypeErrorArgMustBeIndexable(ValuePtr),
    TypeErrorArgMustBeSliceable(ValuePtr),
    TypeErrorArgMustBeList(ValuePtr),
    TypeErrorArgMustBeSet(ValuePtr),
    TypeErrorArgMustBeDict(ValuePtr),
    TypeErrorArgMustBeFunction(ValuePtr),
    TypeErrorArgMustBeCmpOrKeyFunction(ValuePtr),
    TypeErrorArgMustBeReplaceFunction(ValuePtr),
}

impl<T> From<RuntimeError> for Result<T, Box<Prefix<RuntimeError>>> {
    fn from(value: RuntimeError) -> Self {
        Err(Box::new(Prefix::new(Type::Error, value)))
    }
}

impl From<RuntimeError> for ValueResult {
    fn from(value: RuntimeError) -> Self {
        ValueResult::err(value.to_value())
    }
}

impl RuntimeError {
    #[cold]
    pub fn err<E : From<RuntimeError>>(self) -> E {
        E::from(self)
    }

    pub fn with_stacktrace(self, ip: usize, call_stack: &[CallFrame], functions: &[ValuePtr], locations: &[Location]) -> DetailRuntimeError {
        const REPEAT_LIMIT: usize = 3;

        // Top level stack frame refers to the code being executed
        let target: Location = locations.get(ip).copied().unwrap_or(Location::empty());
        let mut stack: Vec<StackFrame> = Vec::new();
        let mut prev_ip: usize = ip;
        let mut prev_frame: Option<(usize, usize)> = None;
        let mut prev_count: usize = 0;

        for frame in call_stack.iter().rev() {
            if frame.return_ip > 0 {
                // Each frame from the call stack refers to the owning function of the previous frame
                let frame_ip: usize = frame.return_ip - 1;

                if prev_frame == Some((frame_ip, prev_ip)) {
                    prev_count += 1;
                } else {
                    if prev_count > REPEAT_LIMIT {
                        // Push a 'repeat' element
                        stack.push(StackFrame::Repeat(prev_count - REPEAT_LIMIT))
                    }
                    prev_count = 0;
                }

                if prev_count <= REPEAT_LIMIT {
                    stack.push(StackFrame::Simple(frame_ip, locations[frame_ip], find_owning_function(prev_ip, functions)));
                }

                prev_frame = Some((frame_ip, prev_ip));
                prev_ip = frame_ip;
            }
        }

        if prev_count > REPEAT_LIMIT {
            stack.push(StackFrame::Repeat(prev_count - REPEAT_LIMIT))
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
enum StackFrame {
    Simple(usize, Location, String),
    Repeat(usize),
}

impl AsError for DetailRuntimeError {
    fn as_error(&self) -> String {
        self.error.as_error()
    }
}

impl AsErrorWithContext for DetailRuntimeError {
    fn location(&self) -> Location {
        self.target
    }

    fn add_stack_trace_elements(&self, view: &SourceView, text: &mut String) {
        for frame in &self.stack {
            text.push_str(match frame {
                StackFrame::Simple(_, loc, site) => format!("  at: `{}` (line {})\n", site, view.lineno(*loc).unwrap_or(0) + 1),
                StackFrame::Repeat(n) => format!("  ... above line repeated {} more time(s) ...\n", n),
            }.as_str());
        }
    }
}


/// The owning function for a given IP can be defined as the closest function which encloses the desired instruction
/// We annotate both head and tail of `FunctionImpl` to make this search easy
fn find_owning_function(ip: usize, functions: &[ValuePtr]) -> String {
    functions.iter()
        .filter(|f| f.is_function())
        .map(|f| f.as_function().borrow_const())
        .filter(|f| f.head <= ip && ip <= f.tail)
        .min_by_key(|f| f.tail - f.head)
        .map(|f| f.repr())
        .unwrap_or_else(|| String::from("<script>"))
}
