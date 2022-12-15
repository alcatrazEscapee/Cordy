use std::io::{BufRead, Write};
use crate::{stdlib, trace};
use crate::stdlib::StdBinding;
use crate::vm::value::Value;

use crate::vm::Opcode::{*};
use crate::vm::RuntimeErrorType::{*};

pub(crate) mod value;


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
    OpIs,

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
    BindingIsNotFunctionEvaluable(StdBinding),

    IncorrectNumberOfArguments(StdBinding, u8, u8),

    TypeErrorUnaryOp(Opcode, Value),
    TypeErrorBinaryOp(Opcode, Value, Value),
    TypeErrorBinaryIs(Value, Value),
    TypeErrorCannotConvertToInt(Value)
}



pub struct VirtualMachine<R, W> {
    ip: usize,
    code: Vec<Opcode>,
    stack: Vec<Value>,

    lineno: usize,

    read: R,
    write: W,
}


impl<R, W> VirtualMachine<R, W> where
    R: BufRead,
    W: Write {

    pub fn new(code: Vec<Opcode>, read: R, write: W) -> VirtualMachine<R, W> {
        VirtualMachine {
            ip: 0,
            code,
            stack: Vec::new(),
            lineno: 0,
            read,
            write,
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
                    let a1: Value = self.pop();
                    match a1 {
                        Value::Int(i1) => self.push(Value::Int(-i1)),
                        v => return self.error(TypeErrorUnaryOp(UnarySub, v)),
                    }
                },
                UnaryLogicalNot => {
                    trace::trace_interpreter!("op unary !");
                    let a1: Value = self.pop();
                    match a1 {
                        Value::Bool(b1) => self.push(Value::Bool(!b1)),
                        v => return self.error(TypeErrorUnaryOp(UnaryLogicalNot, v)),
                    }
                },
                UnaryBitwiseNot => {
                    trace::trace_interpreter!("op unary ~");
                    let a1: Value = self.pop();
                    match a1 {
                        Value::Int(i1) => self.push(Value::Int(!i1)),
                        v => return self.error(TypeErrorUnaryOp(UnaryBitwiseNot, v)),
                    }
                },

                // Binary Operators
                OpMul => {
                    trace::trace_interpreter!("op binary *");
                    let a2: Value = self.pop();
                    let a1: Value = self.pop();
                    match (a1, a2) {
                        (Value::Int(i1), Value::Int(i2)) => self.push(Value::Int(i1 * i2)),
                        (Value::Str(s1), Value::Int(i2)) if i2 > 0 => self.push(Value::Str(s1.repeat(i2 as usize))),
                        (Value::Int(i1), Value::Str(s2)) if i1 > 0 => self.push(Value::Str(s2.repeat(i1 as usize))),
                        (l, r) => return self.error(TypeErrorBinaryOp(OpMul, l, r))
                    }
                },
                OpDiv => {
                    trace::trace_interpreter!("op binary /");
                    let a2: Value = self.pop();
                    let a1: Value = self.pop();
                    match (a1, a2) {
                        (Value::Int(i1), Value::Int(i2)) if i2 != 0 => self.push(Value::Int(i1 / i2)),
                        (l, r) => return self.error(TypeErrorBinaryOp(OpDiv, l, r))
                    }
                },
                OpMod => {
                    trace::trace_interpreter!("op binary %");
                    let a2: Value = self.pop();
                    let a1: Value = self.pop();
                    match (a1, a2) {
                        (Value::Int(i1), Value::Int(i2)) if i2 != 0 => self.push(Value::Int(i1 % i2)),
                        (l, r) => return self.error(TypeErrorBinaryOp(OpMod, l, r))
                    }
                },
                OpPow => {
                    trace::trace_interpreter!("op binary **");
                    let a2: Value = self.pop();
                    let a1: Value = self.pop();
                    match (a1, a2) {
                        (Value::Int(i1), Value::Int(i2)) if i2 > 0 => self.push(Value::Int(i1.pow(i2 as u32))),
                        (l, r) => return self.error(TypeErrorBinaryOp(OpMod, l, r))
                    }
                },
                OpIs => {
                    trace::trace_interpreter!("op binary 'is'");
                    let a2: Value = self.pop();
                    let a1: Value = self.pop();
                    match a2 {
                        Value::Binding(b) => {
                            match stdlib::invoke_type_binding(b, a1) {
                                Ok(ret) => self.push(ret),
                                Err(e) => return self.error(e),
                            }
                        },
                        _ => return self.error(TypeErrorBinaryIs(a1, a2))
                    }
                }

                OpAdd => {
                    trace::trace_interpreter!("op binary +");
                    let a2: Value = self.pop();
                    let a1: Value = self.pop();
                    match (a1, a2) {
                        (Value::Int(i1), Value::Int(i2)) => self.push(Value::Int(i1 + i2)),
                        (Value::Str(s1), r) => self.push(Value::Str(format!("{}{}", s1, r.as_str()))),
                        (l, Value::Str(s2)) => self.push(Value::Str(format!("{}{}", l.as_str(), s2))),
                        (l, r) => return self.error(TypeErrorBinaryOp(OpAdd, l, r)),
                    }
                },
                OpSub => {
                    trace::trace_interpreter!("op binary -");
                    let a2: Value = self.pop();
                    let a1: Value = self.pop();
                    match (a1, a2) {
                        (Value::Int(i1), Value::Int(i2)) if i2 > 0 => self.push(Value::Int(i1 - i2)),
                        (l, r) => return self.error(TypeErrorBinaryOp(OpSub, l, r))
                    }
                },


                OpFuncEval(nargs) => {
                    trace::trace_interpreter!("op function evaluate n = {}", a);
                    let f: &Value = self.peek(*nargs as usize);
                    match f {
                        Value::Binding(b) => {
                            match stdlib::invoke_func_binding(*b, *nargs, self) {
                                Ok(ret) => {
                                    // invoke_func_binding() will pop `nargs` arguments off the stack and pass them to the provided function
                                    self.pop(); // Pop the binding
                                    self.push(ret); // Then push the return value
                                },
                                Err(e) => return self.error(e),
                            }
                        }
                        _ => return self.error(ValueIsNotFunctionEvaluable(f.clone())),
                    }
                }

                Pop => {
                    trace::trace_interpreter!("stack pop {}", self.peek().as_str());
                    self.pop();
                },

                LineNumber(lineno) => self.lineno = *lineno,
                Exit => break,

                _ => panic!("Unimplemented {:?}", op)
            }
        }
        Ok(())
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

pub trait IO {
    fn println0(self: &mut Self);
    fn println(self: &mut Self, str: String);
    fn print(self: &mut Self, str: String);
}

impl <R, W> IO for VirtualMachine<R, W> where
    R: BufRead,
    W: Write {
    fn println0(self: &mut Self) { writeln!(&mut self.write, "").unwrap(); }
    fn println(self: &mut Self, str: String) { writeln!(&mut self.write, "{}", str).unwrap(); }
    fn print(self: &mut Self, str: String) { write!(&mut self.write, "{}", str).unwrap(); }
}

pub trait Stack {
    fn peek(self: &Self, offset: usize) -> &Value;
    fn pop(self: &mut Self) -> Value;
    fn push(self: &mut Self, value: Value);
}

impl<R, W> Stack for VirtualMachine<R, W> {
    // ===== Stack Manipulations ===== //

    /// Peeks at the top element of the stack, or an element `offset` down from the top
    fn peek(self: &Self, offset: usize) -> &Value {
        trace::trace_interpreter_stack!("stack peek({}) {} : [{}]", offset, self.stack.len(), self.stack.iter().rev().map(|t| format!("{}: {}", t.as_str(), t.as_type_str())).collect::<Vec<String>>().join(", "));
        self.stack.get(self.stack.len() - 1 - offset).unwrap()
    }

    /// Pops the top of the stack
    fn pop(self: &mut Self) -> Value {
        trace::trace_interpreter_stack!("stack pop() {} : [{}]", self.stack.len(), self.stack.iter().rev().map(|t| format!("{}: {}", t.as_str(), t.as_type_str())).collect::<Vec<String>>().join(", "));
        match self.stack.pop() {
            Some(v) => v,
            None => panic!("Stack underflow!")
        }
    }

    /// Push a value onto the stack
    fn push(self: &mut Self, value: Value) {
        trace::trace_interpreter_stack!("stack push({}) {} : [{}]", format!("{}: {}", value.as_str(), value.as_type_str()), self.stack.len(), self.stack.iter().rev().map(|t| format!("{}: {}", t.as_str(), t.as_type_str())).collect::<Vec<String>>().join(", "));
        self.stack.push(value);
    }
}


#[cfg(test)]
mod test {
    use crate::{compiler, ErrorReporter, VirtualMachine};
    use crate::vm::{Opcode, RuntimeError};

    #[test] fn test_empty() { run_str("", ""); }
    #[test] fn test_hello_world() { run_str("print('hello world!')", "hello world!\n"); }
    #[test] fn test_empty_print() { run_str("print()", "\n"); }
    #[test] fn test_print_strings() { run_str("print('first', 'second', 'third')", "first second third\n"); }
    #[test] fn test_print_other_args() { run_str("print(nil, -1, 1, true, false, 'test', print)", "nil -1 1 true false test print->out\n"); }
    #[test] fn test_print_unary_ops() { run_str("print(-1, --1, ---1, ~3, ~~3, !true, !!true)", "-1 1 -1 -4 3 false true\n"); }
    #[test] fn test_print_add_str() { run_str("print(('a' + 'b') + (3 + 4) + (' hello' + 3) + (' and' + true + nil))", "ab7 hello3 andtruenil\n"); }
    #[test] fn test_print_mul_str() { run_str("print('abc' * 3)", "abcabcabc\n"); }
    #[test] fn test_print_add_sub_mul_div_int() { run_str("print(5 - 3, 12 + 5, 3 * 9, 16 / 3)", "2 17 27 5\n"); }
    #[test] fn test_print_div_mod_int() { run_str("print(3 / 2, 3 / 3, -3 / 2, 10 % 3, 11 % 3, 12 % 3)", "1 1 -1 1 2 0\n"); }
    #[test] fn test_print_div_by_zero() { run_str("print(15 / 0)", "TypeError: Cannot divide '15' of type 'int' and '0' of type 'int'\n  at: line 1 (<test>)\n  at:\n\nprint(15 / 0)\n"); }
    #[test] fn test_print_mod_by_zero() { run_str("print(15 % 0)", "TypeError: Cannot modulo '15' of type 'int' and '0' of type 'int'\n  at: line 1 (<test>)\n  at:\n\nprint(15 % 0)\n"); }



    fn run_str(code: &'static str, expected: &'static str) {
        let text: &String = &String::from(code);
        let source: &String = &String::from("<test>");
        let compile: Vec<Opcode> = compiler::compile(source, text).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        let mut vm = VirtualMachine::new(compile, &b""[..], &mut buf);

        let result: Result<(), RuntimeError> = vm.run();
        let mut output: String = String::from_utf8(buf).unwrap();

        match result {
            Ok(_) => {},
            Err(err) => output.push_str(ErrorReporter::new(text, source).format_runtime_error(&err).as_str()),
        }

        assert_eq!(expected, output.as_str());
    }
}
