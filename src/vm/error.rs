use crate::stdlib::StdBinding;
use crate::vm::opcode::Opcode;
use crate::vm::value::{FunctionImpl, Value};

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct RuntimeErrorWithLineNumber {
    pub error: RuntimeError,
    pub lineno: u16,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum RuntimeError {
    RuntimeExit,

    ValueIsNotFunctionEvaluable(Value),
    BindingIsNotFunctionEvaluable(StdBinding),

    IncorrectNumberOfFunctionArguments(FunctionImpl, u8),
    IncorrectNumberOfArguments(StdBinding, u8, u8),
    IncorrectNumberOfArgumentsVariadicAtLeastOne(StdBinding),
    IndexOutOfBounds(i64, usize),
    SliceStepZero,

    ValueErrorMaxArgMustBeNonEmptySequence,
    ValueErrorMinArgMustBeNonEmptySequence,

    TypeErrorUnaryOp(Opcode, Value),
    TypeErrorBinaryOp(Opcode, Value, Value),
    TypeErrorBinaryIs(Value, Value),
    TypeErrorCannotConvertToInt(Value),
    TypeErrorCannotCompare(Value, Value),
    TypeErrorCannotSlice(Value),
    TypeErrorSliceArgMustBeInt(&'static str, Value),
    TypeErrorArgMustBeInt(Value),
    TypeErrorArgMustBeIterable(Value),
    TypeErrorFunc1(&'static str, Value),
    TypeErrorFunc2(&'static str, Value, Value),
    TypeErrorFunc3(&'static str, Value, Value, Value),
}

impl RuntimeError {

    pub fn err<T>(self: Self) -> Result<T, Box<RuntimeError>> {
        Err(Box::new(self))
    }

    pub fn with(self: Self, lineno: u16) -> RuntimeErrorWithLineNumber {
        RuntimeErrorWithLineNumber {
            error: self,
            lineno
        }
    }
}
