use std::rc::Rc;

use crate::reporting::ProvidesLineNumber;
use crate::stdlib::StdBinding;
use crate::vm::CallFrame;
use crate::vm::opcode::Opcode;
use crate::vm::value::{FunctionImpl, Value};


#[derive(Eq, PartialEq, Debug, Clone)]
pub enum RuntimeError {
    RuntimeExit,

    ValueIsNotFunctionEvaluable(Value),
    BindingIsNotFunctionEvaluable(StdBinding),

    IncorrectNumberOfFunctionArguments(FunctionImpl, u8),
    IncorrectNumberOfArguments(StdBinding, u8, u8),
    IncorrectNumberOfArgumentsVariadicAtLeastOne(StdBinding),

    ValueErrorIndexOutOfBounds(i64, usize),
    ValueErrorStepCannotBeZero,
    ValueErrorVariableNotDeclaredYet(String),
    ValueErrorValueMustBeNonNegative(i64),
    ValueErrorValueMustBeNonEmpty,
    ValueErrorCannotUnpackLengthMustBeGreaterThan(usize, usize, Value), // expected, actual
    ValueErrorCannotUnpackLengthMustBeEqual(usize, usize, Value), // expected, actual
    ValueErrorCannotCollectIntoDict(Value),
    ValueErrorKeyNotPresent(Value),

    TypeErrorUnaryOp(Opcode, Value),
    TypeErrorBinaryOp(Opcode, Value, Value),
    TypeErrorBinaryIs(Value, Value),
    TypeErrorCannotConvertToInt(Value),

    TypeErrorArgMustBeInt(Value),
    TypeErrorArgMustBeStr(Value),
    TypeErrorArgMustBeIterable(Value),
    TypeErrorArgMustBeIndexable(Value),
    TypeErrorArgMustBeSliceable(Value),
    TypeErrorArgMustBeDict(Value),

    // Deprecated - find a better generic way to do this
    TypeErrorFunc1(&'static str, Value),
    TypeErrorFunc2(&'static str, Value, Value),
    TypeErrorFunc3(&'static str, Value, Value, Value),
}

impl RuntimeError {
    #[cold]
    pub fn err<T>(self: Self) -> Result<T, Box<RuntimeError>> {
        Err(Box::new(self))
    }
}

#[derive(Debug, Clone)]
pub struct DetailRuntimeError {
    pub error: RuntimeError,
    pub stack: Vec<StackTraceFrame>,
}

impl DetailRuntimeError {
    fn new(error: RuntimeError, stack: Vec<StackTraceFrame>) -> DetailRuntimeError {
        DetailRuntimeError { error, stack }
    }
}

#[derive(Debug, Clone)]
pub struct StackTraceFrame {
    ip: usize,
    pub lineno: usize,
    pub src: Option<String>,
}

impl StackTraceFrame {
    fn new(ip: usize, lineno: u16) -> StackTraceFrame {
        StackTraceFrame { ip, lineno: lineno as usize, src: None }
    }
}

pub fn detail_runtime_error(error: RuntimeError, ip: usize, call_stack: &Vec<CallFrame>, functions: &Vec<Rc<FunctionImpl>>, line_numbers: &Vec<u16>) -> DetailRuntimeError {

    // Top level stack frame refers to the code being executed
    let mut stack: Vec<StackTraceFrame> = vec![StackTraceFrame::new(ip, line_numbers.line_number(ip))];

    for frame in call_stack.iter().rev() {
        if frame.return_ip > 0 {
            let frame_ip: usize = frame.return_ip - 1;
            let mut stack_frame: StackTraceFrame = StackTraceFrame::new(frame_ip, line_numbers.line_number(frame_ip));

            // Each frame from the call stack refers to the owning function of the previous frame
            stack_frame.src = Some(find_owning_function(stack.last().unwrap().ip, functions));
            stack.push(stack_frame);
        }
    }

    DetailRuntimeError::new(error, stack)
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
