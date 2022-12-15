use RuntimeErrorType::ValueIsNotFunctionEvaluable;
use crate::{stdlib, trace};
use crate::stdlib::StdBinding;

use crate::vm::Opcode::{*};


#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Str(String),
    Binding(StdBinding)
}


impl Value {
    pub fn as_str(self: &Self) -> String {
        match self {
            Value::Nil => String::from("nil"),
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Str(s) => s.clone(),
            Value::Binding(b) => String::from(stdlib::lookup_binding(*b))
        }
    }

    pub fn as_type_str(self: &Self) -> String {
        String::from(match self {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Str(_) => "str",
            Value::Binding(_) => "{binding}"
        })
    }
}


#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Opcode {

    // Stack Operations
    StoreValue, // ... name, value] -> set name = value; -> ...]
    Dupe, // ... x, y, z] -> ... x, y, z, z]
    Pop,

    // Push
    Nil,
    True,
    False,
    Int(i64),
    Str(String),
    Bound(StdBinding),

    // todo: remove / replace with other things
    Identifier(String),

    // Unary Operators
    UnarySub,
    UnaryLogicalNot,
    UnaryBitwiseNot,

    // Binary Operators
    // Ordered by precedence, highest to lowest
    OpFuncEval(u8),
    OpArrayEval(u8),

    OpMul,
    OpDiv,
    OpMod,
    OpPow,

    OpAdd,
    OpSub,

    OpLeftShift,
    OpRightShift,

    OpLessThan,
    OpGreaterThan,
    OpLessThanEqual,
    OpGreaterThanEqual,
    OpEqual,
    OpNotEqual,

    OpBitwiseAnd,
    OpBitwiseOr,
    OpBitwiseXor,

    OpFuncCompose,

    OpLogicalAnd,
    OpLogicalOr,

    // Debug
    LineNumber(usize),

    // Flow Control
    Exit,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct RuntimeError {
    pub error: RuntimeErrorType,
    pub lineno: usize,
    ip: usize,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum RuntimeErrorType {
    ValueIsNotFunctionEvaluable(Value),

    TypeErrorExpectedInt(Value),
    TypeErrorExpectedBool(Value),
}



pub struct VirtualMachine {
    ip: usize,
    code: Vec<Opcode>,
    stack: Vec<Value>,

    lineno: usize,
}


impl VirtualMachine {

    pub fn new(code: Vec<Opcode>) -> VirtualMachine {
        VirtualMachine {
            ip: 0,
            code,
            stack: Vec::new(),
            lineno: 0,
        }
    }

    pub fn run(self: &mut Self) -> Result<(), RuntimeError> {
        loop {
            let op: &Opcode = self.code.get(self.ip).unwrap();
            self.ip += 1;
            match op {
                // Push Operations
                Nil => {
                    trace::trace_interpreter!("push nil");
                    self.push(Value::Nil);
                },
                True => {
                    trace::trace_interpreter!("push true");
                    self.push(Value::Bool(true));
                },
                False => {
                    trace::trace_interpreter!("push false");
                    self.push(Value::Bool(false));
                },
                Int(i) => {
                    trace::trace_interpreter!("push int {}", i);
                    self.push(Value::Int(*i));
                },
                Str(s) => {
                    trace::trace_interpreter!("push str {}", s);
                    self.push(Value::Str(s.clone()))
                }
                Bound(b) => {
                    trace::trace_interpreter!("push binding for {:?}", b);
                    self.push(Value::Binding(*b));
                },

                // Unary Operators
                UnarySub => {
                    trace::trace_interpreter!("op unary -");
                    let a1: i64 = self.pop_int()?;
                    self.push(Value::Int(-a1));
                },
                UnaryLogicalNot => {
                    trace::trace_interpreter!("op unary !");
                    let a1: bool = self.pop_bool()?;
                    self.push(Value::Bool(!a1));
                },
                UnaryBitwiseNot => {
                    trace::trace_interpreter!("op unary ~");
                    let a1: i64 = self.pop_int()?;
                    self.push(Value::Int(!a1));
                },

                OpFuncEval(a) => {
                    trace::trace_interpreter!("op function evaluate n = {}", a);
                    match a {
                        0 => {
                            let f: Value = self.stack.pop().unwrap();
                            let ret: Value = match f {
                                Value::Binding(b) => stdlib::invoke_binding_0(b),
                                _ => return self.error(ValueIsNotFunctionEvaluable(f)),
                            };
                            self.stack.push(ret);
                        },
                        _ => {
                            let mut args: Vec<Value> = Vec::with_capacity(*a as usize);
                            for _ in 0..*a {
                                args.push(self.stack.pop().unwrap());
                            }
                            let f: Value = self.stack.pop().unwrap();
                            let ret: Value = match f {
                                Value::Binding(b) => stdlib::invoke_binding_n(b, args),
                                _ => return self.error(ValueIsNotFunctionEvaluable(f)),
                            };
                            self.stack.push(ret);
                        }
                    }
                },

                Pop => {
                    trace::trace_interpreter!("stack pop {}", self.stack.last().unwrap().as_str());
                    self.stack.pop().unwrap();
                },

                LineNumber(lineno) => self.lineno = *lineno,
                Exit => break,

                _ => panic!("Unimplemented {:?}", op)
            }
        }
        Ok(())
    }

    // ===== Stack Manipulations ===== //

    /// Pops the top of the stack, and type checks that it is a `bool`
    fn pop_bool(self: &mut Self) -> Result<bool, RuntimeError> {
        match self.pop() {
            Value::Bool(b) => Ok(b),
            v => return self.error(RuntimeErrorType::TypeErrorExpectedBool(v))
        }
    }


    /// Pops the top of the stack, and type checks that it is an `int`
    fn pop_int(self: &mut Self) -> Result<i64, RuntimeError> {
        match self.pop() {
            Value::Int(i) => Ok(i),
            v => return self.error(RuntimeErrorType::TypeErrorExpectedInt(v))
        }
    }

    /// Pops the top of the stack
    fn pop(self: &mut Self) -> Value {
        match self.stack.pop() {
            Some(v) => v,
            None => panic!("Stack underflow!")
        }
    }

    fn push(self: &mut Self, value: Value) {
        assert!(self.stack.len() < 10_000, "Stack overflow!");
        self.stack.push(value);
    }

    /// Constructs a `RuntimeError` to be returned from the main VM loop
    fn error<T>(self: &Self, error: RuntimeErrorType) -> Result<T, RuntimeError> {
        Err(RuntimeError {
            error,
            lineno: self.lineno,
            ip: self.ip
        })
    }
}
