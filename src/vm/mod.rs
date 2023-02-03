use std::cell::Cell;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::rc::Rc;

use itertools::Itertools;

use crate::{compiler, stdlib, trace};
use crate::compiler::{CompileResult, IncrementalCompileResult, Locals};
use crate::stdlib::NativeFunction;
use crate::vm::value::{PartialFunctionImpl, UpValue};

pub use crate::vm::error::{DetailRuntimeError, RuntimeError, StackTraceFrame};
pub use crate::vm::opcode::Opcode;
pub use crate::vm::value::{FunctionImpl, IntoDictValue, IntoIterableValue, IntoValue, Iterable, Value};

use Opcode::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;
type AnyResult = Result<(), Box<RuntimeError>>;

pub mod operator;

mod value;
mod opcode;
mod error;


pub struct VirtualMachine<R, W> {
    ip: usize,
    code: Vec<Opcode>,
    pub stack: Vec<Value>,
    pub call_stack: Vec<CallFrame>,
    global_count: usize,
    open_upvalues: HashMap<usize, Rc<Cell<UpValue>>>,

    strings: Vec<String>,
    globals: Vec<String>,
    constants: Vec<i64>,
    functions: Vec<Rc<FunctionImpl>>,
    line_numbers: Vec<u32>,

    read: R,
    write: W,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum FunctionType {
    Native, User
}

#[derive(Debug, Clone)]
pub enum ExitType {
    Exit, Return, Yield, Error(DetailRuntimeError)
}

impl ExitType {
    pub fn is_early_exit(self: &Self) -> bool {
        match self {
            ExitType::Exit | ExitType::Error(_) => true,
            _ => false,
        }
    }
}


pub trait VirtualInterface {
    // Invoking Functions

    fn invoke_func1(self: &mut Self, f: Value, a1: Value) -> ValueResult;
    fn invoke_func2(self: &mut Self, f: Value, a1: Value, a2: Value) -> ValueResult;
    fn invoke_func(self: &mut Self, f: Value, args: &Vec<Value>) -> ValueResult;

    fn invoke_eval(self: &mut Self, s: &String) -> ValueResult;

    // Wrapped IO
    fn println0(self: &mut Self);
    fn println(self: &mut Self, str: String);
    fn print(self: &mut Self, str: String);

    // Stack Manipulation
    fn peek(self: &Self, offset: usize) -> &Value;
    fn pop(self: &mut Self) -> Value;
    fn popn(self: &mut Self, n: usize) -> Vec<Value>;
    fn push(self: &mut Self, value: Value);
}




#[derive(Debug)]
pub struct CallFrame {
    /// The return address
    /// When the function returns, execution transfers to this location
    pub return_ip: usize,

    /// A pointer into the current runtime stack, where this function's locals are stored
    /// The local at index 0 will be the function itself, local 1, ...N will be the N parameters, locals after that will be local variables to the function
    frame_pointer: usize,
}


impl<R, W> VirtualMachine<R, W> where
    R: BufRead,
    W: Write {

    pub fn new(result: CompileResult, read: R, write: W) -> VirtualMachine<R, W> {
        VirtualMachine {
            ip: 0,
            code: result.code,
            stack: Vec::with_capacity(256), // Just guesses, not hard limits
            call_stack: vec![CallFrame { return_ip: 0, frame_pointer: 0 }],
            global_count: 0,
            open_upvalues: HashMap::new(),
            strings: result.strings,
            globals: result.globals,
            constants: result.constants,
            functions: result.functions,
            line_numbers: result.line_numbers,
            read,
            write,
        }
    }

    /// Bridge method to `compiler::incremental_compile`
    pub fn incremental_compile(self: &mut Self, source: &String, text: &String, locals: &mut Vec<Locals>) -> IncrementalCompileResult {
        compiler::incremental_compile(source, text, &mut self.code, locals, &mut self.strings, &mut self.constants, &mut self.functions, &mut self.line_numbers, &mut self.globals)
    }

    /// Bridge method to `compiler::eval_compile`
    pub fn eval_compile(self: &mut Self, text: &String) -> AnyResult {
        compiler::eval_compile(text, &mut self.code, &mut self.strings, &mut self.constants, &mut self.functions, &mut self.line_numbers, &mut self.globals)
    }

    pub fn run_until_completion(self: &mut Self) -> ExitType {
        match self.run().map_err(|u| *u) {
            Ok(_) => ExitType::Return,
            Err(RuntimeExit) => ExitType::Exit,
            Err(RuntimeYield) => ExitType::Yield,
            Err(e) => ExitType::Error(error::detail_runtime_error(e, self.ip - 1, &self.call_stack, &self.functions, &self.line_numbers)),
        }
    }

    /// Recovers the VM into an operational state, in case previous instructions terminated in an error or in the middle of a function
    pub fn run_recovery(self: &mut Self, locals: usize) {
        self.call_stack.truncate(1);
        self.stack.truncate(locals);
        self.ip = self.code.len();
    }

    /// Runs until the current call frame is dropped. Used to invoke a user function from native code.
    fn run_frame(self: &mut Self) -> AnyResult {
        let drop_frame: usize = self.call_stack.len() - 1;
        loop {
            let op: Opcode = self.next_op();
            self.run_instruction(op)?;
            if drop_frame == self.call_stack.len() {
                return Ok(())
            }
        }
    }

    fn run(self: &mut Self) -> AnyResult {
        loop {
            let op: Opcode = self.next_op();
            self.run_instruction(op)?;
        }
    }

    /// Executes a single instruction
    fn run_instruction(self: &mut Self, op: Opcode) -> AnyResult {
        macro_rules! operator {
            ($op:expr, $a1:ident, $tr:expr) => {{
                trace::trace_interpreter!("op unary {}", $tr);
                let $a1: Value = self.pop();
                match $op {
                    Ok(v) => self.push(v),
                    Err(e) => return e.err(),
                }
            }};
            ($op:path, $a1:ident, $a2:ident, $tr:expr) => {{
                trace::trace_interpreter!("op binary {}", $tr);
                let $a2: Value = self.pop();
                let $a1: Value = self.pop();
                match $op($a1, $a2) {
                    Ok(v) => self.push(v),
                    Err(e) => return e.err(),
                }
            }};
        }

        macro_rules! operator_unchecked {
            ($op:path, $a1:ident, $tr:expr) => {{
                trace::trace_interpreter!("op unary {}", $tr);
                let $a1: Value = self.pop();
                self.push($op($a1));
            }};
            ($op:path, $a1:ident, $a2:ident, $tr:expr) => {{
                trace::trace_interpreter!("op binary {}", $tr);
                let $a2: Value = self.pop();
                let $a1: Value = self.pop();
                self.push($op($a1, $a2));
            }};
        }

        match op {
            Noop => panic!("Noop should only be emitted as a temporary instruction"),

            // Flow Control
            // All jumps are absolute (because we don't have variable length instructions and it's easy to do so)
            // todo: relative jumps? theoretically allows us more than u32.max instructions ~ 65k
            JumpIfFalse(ip) => {
                trace::trace_interpreter!("jump if false {} -> {}", self.stack.last().unwrap().as_debug_str(), ip);
                let jump: usize = ip as usize;
                let a1: &Value = self.peek(0);
                if !a1.as_bool() {
                    self.ip = jump;
                }
            },
            JumpIfFalsePop(ip) => {
                trace::trace_interpreter!("jump if false pop {} -> {}", self.stack.last().unwrap().as_debug_str(), ip);
                let jump: usize = ip as usize;
                let a1: Value = self.pop();
                if !a1.as_bool() {
                    self.ip = jump;
                }
            },
            JumpIfTrue(ip) => {
                trace::trace_interpreter!("jump if true {} -> {}", self.stack.last().unwrap().as_debug_str(), ip);
                let jump: usize = ip as usize;
                let a1: &Value = self.peek(0);
                if a1.as_bool() {
                    self.ip = jump;
                }
            },
            JumpIfTruePop(ip) => {
                trace::trace_interpreter!("jump if true pop {} -> {}", self.stack.last().unwrap().as_debug_str(), ip);
                let jump: usize = ip as usize;
                let a1: Value = self.pop();
                if a1.as_bool() {
                    self.ip = jump;
                }
            },
            Jump(ip) => {
                trace::trace_interpreter!("jump -> {}", ip);
                self.ip = ip as usize;
            },
            Return => {
                trace::trace_interpreter!("return -> {}", self.return_ip());
                // Functions leave their return value as the top of the stack
                // Below that will be the functions locals, and the function itself
                // The frame pointer points to the first local of the function
                // [prev values ... function, local0, local1, ... localN, ret_val ]
                //                            ^frame pointer
                // So, we pop the return value, splice the difference between the frame pointer and the top, then push the return value
                let ret: Value = self.pop();
                self.stack.truncate(self.frame_pointer() - 1);
                trace::trace_interpreter_stack!("drop frame");
                trace::trace_interpreter_stack!("{}", self.debug_stack());
                self.push(ret);

                self.ip = self.return_ip();
                self.call_stack.pop().unwrap();
                trace::trace_interpreter!("call stack = {}", self.debug_call_stack());
            },

            // Stack Manipulations
            Pop => {
                trace::trace_interpreter!("stack pop {}", self.stack.last().unwrap().as_debug_str());
                self.pop();
            },
            PopN(n) => {
                trace::trace_interpreter!("stack popn {}", n);
                for _ in 0..n {
                    self.pop();
                }
            },
            Dup => {
                trace::trace_interpreter!("dup {}", self.stack.last().unwrap().as_debug_str());
                self.push(self.peek(0).clone());
            },
            Swap => {
                trace::trace_interpreter!("swap {} <-> {}", self.stack.last().unwrap().as_debug_str(), self.stack[self.stack.len() - 2].as_debug_str());
                let a1: Value = self.pop();
                let a2: Value = self.pop();
                self.push(a1);
                self.push(a2);
            },

            PushLocal(local) => {
                let local = self.frame_pointer() + local as usize;
                trace::trace_interpreter!("push local {} : {}", local, self.stack[local].as_debug_str());
                self.push(self.stack[local].clone());
            }
            StoreLocal(local) => {
                let local: usize = self.frame_pointer() + local as usize;
                trace::trace_interpreter!("store local {} : {} -> {}", local, self.stack.last().unwrap().as_debug_str(), self.stack[local].as_debug_str());
                self.stack[local] = self.peek(0).clone();
            },
            PushGlobal(local, is_local) => {
                // Globals are fancy locals that don't use the frame pointer to offset their local variable ID
                // 'global' just means a variable declared outside of an enclosing function
                // As we allow late binding for globals, we need to check that this has been declared first, based on the globals count.
                let local: usize = local as usize;
                trace::trace_interpreter!("push global {} : {}", local, self.stack[local].as_debug_str());
                if local < self.global_count || is_local {
                    self.push(self.stack[local].clone());
                } else {
                    return ValueErrorVariableNotDeclaredYet(self.globals[local].clone()).err()
                }
            },
            StoreGlobal(local, is_local) => {
                let local: usize = local as usize;
                trace::trace_interpreter!("store global {} : {} -> {}", local, self.stack.last().unwrap().as_debug_str(), self.stack[local].as_debug_str());
                if local < self.global_count || is_local {
                    self.stack[local] = self.peek(0).clone();
                } else {
                    return ValueErrorVariableNotDeclaredYet(self.globals[local].clone()).err()
                }
            },
            PushUpValue(index) => {
                let fp = self.frame_pointer() - 1;
                let upvalue: Rc<Cell<UpValue>> = match &mut self.stack[fp] {
                    Value::Closure(c) => c.get(index as usize),
                    _ => panic!("Malformed bytecode"),
                };

                // Reasons why this is convoluted:
                // - We cannot use `.get()` (as it requires `Value` to be `Copy`)
                // - We cannot use `get_mut()` (as even if we have `&mut ClosureImpl`, unboxing the `Rc<>` only gives us `&Cell`)
                let unboxed: UpValue = (*upvalue).replace(UpValue::Open(0));
                (*upvalue).replace(unboxed.clone());

                let value: Value = match unboxed {
                    UpValue::Open(index) => self.stack[index].clone(),
                    UpValue::Closed(value) => value,
                };
                trace::trace_interpreter!("push upvalue {} : {}", index, value.as_debug_str());
                self.push(value);
            },
            StoreUpValue(index) => {
                trace::trace_interpreter!("store upvalue {} : {} -> {}", index, self.stack.last().unwrap().as_debug_str(), self.stack[index as usize].as_debug_str());
                let fp = self.frame_pointer() - 1;
                let value = self.peek(0).clone();
                let upvalue: Rc<Cell<UpValue>> = match &mut self.stack[fp] {
                    Value::Closure(c) => c.get(index as usize),
                    _ => panic!("Malformed bytecode"),
                };

                // Reasons why this is convoluted:
                // - We cannot use `.get()` (as it requires `Value` to be `Copy`)
                // - We cannot use `get_mut()` (as even if we have `&mut ClosureImpl`, unboxing the `Rc<>` only gives us `&Cell`)
                let unboxed: UpValue = (*upvalue).replace(UpValue::Open(0));
                let modified: UpValue = match unboxed {
                    UpValue::Open(stack_index) => {
                        let ret = UpValue::Open(stack_index); // And return the upvalue, unmodified
                        self.stack[stack_index] = value; // Mutate on the stack
                        ret
                    },
                    UpValue::Closed(_) => UpValue::Closed(value), // Mutate on the heap
                };
                (*upvalue).replace(modified);
            },

            StoreArray => {
                trace::trace_interpreter!("store array array = {}, index = {}, value = {}", self.stack[self.stack.len() - 3].as_debug_str(), self.stack[self.stack.len() - 2].as_debug_str(), self.stack.last().unwrap().as_debug_str());
                let a3: Value = self.pop();
                let a2: Value = self.pop();
                let a1: &Value = self.peek(0); // Leave this on the stack when done
                stdlib::set_index(a1, a2, a3)?;
            },

            IncGlobalCount => {
                trace::trace_interpreter!("inc global count -> {}", self.global_count + 1);
                self.global_count += 1;
            },

            Closure => {
                trace::trace_interpreter!("close closure {}", self.stack.last().unwrap().as_debug_str());
                match self.pop() {
                    Value::Function(f) => self.push(Value::closure(f)),
                    _ => panic!("Malformed bytecode"),
                }
            },

            CloseLocal(index) => {
                let local: usize = self.frame_pointer() + index as usize;
                trace::trace_interpreter!("close local {} -> stack[{}] = {} into {}", index, local, self.stack[local].as_debug_str(), self.stack.last().unwrap().as_debug_str());
                let upvalue: Rc<Cell<UpValue>> = self.open_upvalues.entry(local)
                    .or_insert_with(|| Rc::new(Cell::new(UpValue::Open(local))))
                    .clone();
                match self.stack.last_mut().unwrap() {
                    Value::Closure(c) => c.push(upvalue),
                    _ => panic!("Malformed bytecode"),
                }
            },
            CloseUpValue(index) => {
                trace::trace_interpreter!("close upvalue {} into {} from {}", index, self.stack.last().unwrap().as_debug_str(), &self.stack[self.frame_pointer() - 1].as_debug_str());
                let fp = self.frame_pointer() - 1;
                let index: usize = index as usize;
                let upvalue: Rc<Cell<UpValue>> = match &mut self.stack[fp] {
                    Value::Closure(c) => c.get(index),
                    _ => panic!("Malformed bytecode"),
                };

                match self.stack.last_mut().unwrap() {
                    Value::Closure(c) => c.push(upvalue.clone()),
                    _ => panic!("Malformed bytecode"),
                }
            },

            LiftUpValue(index) => {
                trace::trace_interpreter!("lift upvalue {}", self.stack.last().unwrap().as_debug_str());
                let index = self.frame_pointer() + index as usize;
                match self.open_upvalues.remove(&index) {
                    Some(upvalue) => {
                        let value: Value = self.stack[index].clone();
                        let unboxed: UpValue = (*upvalue).replace(UpValue::Open(0));
                        let closed: UpValue = match unboxed {
                            UpValue::Open(_) => UpValue::Closed(value),
                            UpValue::Closed(_) => panic!("Tried to life an already closed upvalue"),
                        };
                        (*upvalue).replace(closed);
                    },
                    None => {}, // This upvalue was never captured - the code path that would've created a closure didn't run, so we don't need to lift it
                }
            },

            InitIterable => {
                trace::trace_interpreter!("init iterable {}", self.stack.last().unwrap().as_debug_str());
                let iter = self.pop().as_iter()?;
                self.push(Value::Iter(Box::new(iter)));
            },
            TestIterable => {
                trace::trace_interpreter!("test iterable {}", self.stack.last().unwrap().as_debug_str());
                let top: usize = self.stack.len() - 1;
                let iter = &mut self.stack[top];
                match iter {
                    Value::Iter(it) => match it.next() {
                        Some(value) => {
                            self.push(value);
                            self.push(Value::Bool(true));
                        },
                        None => {
                            self.push(Value::Bool(false));
                        }
                    },
                    _ => panic!("Malformed bytecode"),
                }
            },

            // Push Operations
            Nil => {
                trace::trace_interpreter!("push nil");
                self.push(Value::Nil);
            },
            True => {
                trace::trace_interpreter!("push true");
                self.push(true.to_value());
            },
            False => {
                trace::trace_interpreter!("push false");
                self.push(false.to_value());
            },
            Int(cid) => {
                let cid: usize = cid as usize;
                let value: i64 = self.constants[cid];
                trace::trace_interpreter!("push constant {} -> {}", cid, value);
                self.push(value.to_value());
            }
            Str(sid) => {
                let sid: usize = sid as usize;
                let str: String = self.strings[sid].clone();
                trace::trace_interpreter!("push {} -> '{}'", sid, str);
                self.push(str.to_value());
            },
            Function(fid) => {
                let fid: usize = fid as usize;
                let func = Value::Function(Rc::clone(&self.functions[fid]));
                trace::trace_interpreter!("push {} -> {}", fid, func.as_debug_str());
                self.push(func);
            },
            NativeFunction(b) => {
                trace::trace_interpreter!("push {}", Value::NativeFunction(b).as_debug_str());
                self.push(Value::NativeFunction(b));
            },
            List(cid) => {
                // List values are present on the stack in-order
                // So we need to splice the last n values of the stack into it's own list
                let length: usize = self.constants[cid as usize] as usize;
                trace::trace_interpreter!("push list n={}", length);
                let start: usize = self.stack.len() - length;
                let end: usize = self.stack.len();
                trace::trace_interpreter_stack!("stack splice {}..{} into list", start, end);
                let list: Value = self.stack.splice(start..end, std::iter::empty()).to_list();
                self.push(list);
            },
            Vector(cid) => {
                // Vector values are present on the stack in-order
                // So we need to splice the last n values of the stack into it's own vector
                let length: usize = self.constants[cid as usize] as usize;
                trace::trace_interpreter!("push vector n={}", length);
                let start: usize = self.stack.len() - length;
                let end: usize = self.stack.len();
                trace::trace_interpreter_stack!("stack splice {}..{} into list", start, end);
                let vector: Value = self.stack.splice(start..end, std::iter::empty()).to_vector();
                self.push(vector);
            },
            Set(cid) => {
                // Set values are present on the stack in-order
                // So we need to splice the last n values of the stack into it's own set
                let length: usize = self.constants[cid as usize] as usize;
                trace::trace_interpreter!("push set n={}", length);
                let start: usize = self.stack.len() - length;
                let end: usize = self.stack.len();
                trace::trace_interpreter_stack!("stack splice {}..{} into set", start, end);
                let set: Value = self.stack.splice(start..end, std::iter::empty()).to_set();
                self.push(set);
            },
            Dict(cid) => {
                // Dict values are present on the stack in-order, in flat key-value order.
                // So we need to splice the last n*2 values of the stack into it's own dict.
                let length: usize = (self.constants[cid as usize] as usize) * 2;
                trace::trace_interpreter!("push set n={}", length);
                let start: usize = self.stack.len() - length;
                let end: usize = self.stack.len();
                trace::trace_interpreter_stack!("stack splice {}..{} into dict", start, end);
                let dict: Value = self.stack.splice(start..end, std::iter::empty()).tuples().to_dict();
                self.push(dict);
            },

            OpFuncEval(nargs) => {
                trace::trace_interpreter!("op function evaluate n = {}", nargs);
                match self.invoke_func_eval(nargs) {
                    Err(e) => return e.err(),
                    _ => {},
                }
            },

            CheckLengthGreaterThan(len) => {
                let len = self.constants[len as usize] as usize;
                trace::trace_interpreter!("check len > {}", len);
                let a1: &Value = self.peek(0);
                let actual = match a1.len() {
                    Ok(len) => len,
                    Err(e) => return Err(e),
                };
                if len > actual {
                    return ValueErrorCannotUnpackLengthMustBeGreaterThan(len, actual, a1.clone()).err()
                }
            },
            CheckLengthEqualTo(len) => {
                let len = self.constants[len as usize] as usize;
                trace::trace_interpreter!("check len == {}", len);
                let a1: &Value = self.peek(0);
                let actual = match a1.len() {
                    Ok(len) => len,
                    Err(e) => return Err(e),
                };
                if len != actual {
                    return ValueErrorCannotUnpackLengthMustBeEqual(len, actual, a1.clone()).err()
                }
            },

            OpIndex => {
                trace::trace_interpreter!("op []");
                let a2: Value = self.pop();
                let a1: Value = self.pop();
                let ret = stdlib::get_index(&a1, &a2)?;
                self.push(ret);
            },
            OpIndexPeek => {
                trace::trace_interpreter!("op [] peek");
                let a2: &Value = self.peek(0);
                let a1: &Value = self.peek(1);
                let ret = stdlib::get_index(a1, a2)?;
                self.push(ret);
            },
            OpSlice => {
                trace::trace_interpreter!("op [:]");
                let a3: Value = self.pop();
                let a2: Value = self.pop();
                let a1: Value = self.pop();
                let ret = stdlib::get_slice(a1, a2, a3, Value::Int(1))?;
                self.push(ret);
            },
            OpSliceWithStep => {
                trace::trace_interpreter!("op [::]");
                let a4: Value = self.pop();
                let a3: Value = self.pop();
                let a2: Value = self.pop();
                let a1: Value = self.pop();
                let ret = stdlib::get_slice(a1, a2, a3, a4)?;
                self.push(ret);
            },

            // Unary Operators
            UnarySub => operator!(operator::unary_sub(a1), a1, "-"),
            UnaryLogicalNot => operator!(operator::unary_logical_not(a1), a1, "!"),
            UnaryBitwiseNot => operator!(operator::unary_bitwise_not(a1), a1, "~"),

            // Binary Operators
            OpMul => operator!(operator::binary_mul, a1, a2, "*"),
            OpDiv => operator!(operator::binary_div, a1, a2, "/"),
            OpMod => operator!(operator::binary_mod, a1, a2, "%"),
            OpPow => operator!(operator::binary_pow, a1, a2, "**"),
            OpIs => operator!(operator::binary_is, a1, a2, "is"),
            OpAdd => operator!(operator::binary_add, a1, a2, "+"),
            OpSub => operator!(operator::binary_sub, a1, a2, "-"),
            OpLeftShift => operator!(operator::binary_left_shift, a1, a2, "<<"),
            OpRightShift => operator!(operator::binary_right_shift, a1, a2, ">>"),
            OpBitwiseAnd => operator!(operator::binary_bitwise_and, a1, a2, "&"),
            OpBitwiseOr => operator!(operator::binary_bitwise_or, a1, a2, "|"),
            OpBitwiseXor => operator!(operator::binary_bitwise_xor, a1, a2, "^"),
            OpIn => operator!(operator::binary_in, a1, a2, "in"),
            OpLessThan => operator_unchecked!(operator::binary_less_than, a1, a2, "<"),
            OpGreaterThan => operator_unchecked!(operator::binary_greater_than, a1, a2, ">"),
            OpLessThanEqual => operator_unchecked!(operator::binary_less_than_or_equal, a1, a2, "<="),
            OpGreaterThanEqual => operator_unchecked!(operator::binary_greater_than_or_equal, a1, a2, ">="),
            OpEqual => operator_unchecked!(operator::binary_equals, a1, a2, "=="),
            OpNotEqual => operator_unchecked!(operator::binary_not_equals, a1, a2, "!="),
            OpMax => operator_unchecked!(std::cmp::max, a1, a2, "max"),
            OpMin => operator_unchecked!(std::cmp::min, a1, a2, "min"),

            Exit => return RuntimeExit.err(),
            Yield => return RuntimeYield.err(),
        }
        Ok(())
    }


    // ===== Basic Ops ===== //

    /// Returns the current `frame_pointer`
    fn frame_pointer(self: &Self) -> usize {
        self.call_stack[self.call_stack.len() - 1].frame_pointer
    }

    /// Returns the current `return_ip`
    fn return_ip(self: &Self) -> usize {
        self.call_stack[self.call_stack.len() - 1].return_ip
    }

    /// Returns the next opcode and increments `ip`
    fn next_op(self: &mut Self) -> Opcode {
        let op: Opcode = self.code[self.ip];
        self.ip += 1;
        op
    }

    fn invoke_func_and_spin(self: &mut Self, nargs: u8) -> ValueResult {
        match self.invoke_func_eval(nargs)? {
            FunctionType::Native => {},
            FunctionType::User => self.run_frame()?
        }
        Ok(self.pop())
    }

    /// Invokes the action of an `OpFuncEval(nargs)` opcode.
    ///
    /// The stack must be setup as `[..., f, arg1, arg2, ... argN ]`, where `f` is the function to be invoked with arguments `arg1, arg2, ... argN`.
    /// The arguments and function will be popped and the return value will be left on the top of the stack.
    ///
    /// Returns a `Result` which may contain an error which occurred during function evaluation.
    fn invoke_func_eval(self: &mut Self, nargs: u8) -> Result<FunctionType, Box<RuntimeError>> {
        let f: &Value = self.peek(nargs as usize);
        match f {
            f @ (Value::Function(_) | Value::Closure(_)) => {
                trace::trace_interpreter!("invoke_func_eval -> {}, nargs = {}", f.as_debug_str(), nargs);

                let func = f.unbox_func();
                if func.nargs == nargs {
                    // Evaluate directly
                    self.call_function(func.head, func.nargs);
                } else if func.nargs > nargs && nargs > 0 {
                    // Evaluate as a partial function
                    let arg: Vec<Value> = self.popn(nargs as usize);
                    let func: Value = self.pop();
                    let partial: Value = Value::partial(func, arg);
                    self.push(partial);
                } else {
                    return IncorrectNumberOfFunctionArguments((**func).clone(), nargs).err();
                }
                Ok(FunctionType::User)
            },
            Value::PartialFunction(_) => {
                trace::trace_interpreter!("invoke_func_eval -> {}, nargs = {}", f.as_debug_str(), nargs);
                // Surgically extract the partial binding from the stack
                let i: usize = self.stack.len() - nargs as usize - 1;
                let mut partial: PartialFunctionImpl = match std::mem::replace(&mut self.stack[i], Value::Nil) {
                    Value::PartialFunction(x) => *x,
                    _ => panic!("Stack corruption")
                };
                let func = partial.func.unbox_func();
                let total_nargs: u8 = partial.args.len() as u8 + nargs;
                if func.nargs > total_nargs {
                    // Not enough arguments, so pop the argument and push a new partial function
                    let top = self.stack.len();
                    for arg in self.stack.splice(top - nargs as usize..top, std::iter::empty()) {
                        partial.args.push(Box::new(arg));
                    }
                    self.pop(); // Should pop the `Nil` we swapped earlier
                    self.push(Value::PartialFunction(Box::new(partial)));
                } else if func.nargs == total_nargs {
                    // Exactly enough arguments to invoke the function
                    // Before we call, we need to pop-push to reorder the arguments and setup partial arguments, so we have the correct calling convention
                    let args: Vec<Value> = self.popn(nargs as usize);
                    let head: usize = func.head;
                    self.pop(); // Should pop the `Nil` we swapped earlier
                    self.push(partial.func);
                    for par in partial.args {
                        self.push(*par);
                    }
                    for arg in args {
                        self.push(arg);
                    }
                    self.call_function(head, total_nargs);

                } else {
                    return IncorrectNumberOfFunctionArguments((**func).clone(), total_nargs).err()
                }
                Ok(FunctionType::User)
            },
            Value::NativeFunction(b) => {
                trace::trace_interpreter!("invoke_func_eval -> {}, nargs = {}", Value::NativeFunction(b.clone()).as_debug_str(), nargs);
                match stdlib::invoke(*b, nargs, self) {
                    Ok(v) => {
                        self.pop();
                        self.push(v)
                    },
                    Err(e) => return Err(e),
                }
                Ok(FunctionType::Native)
            },
            Value::PartialNativeFunction(b, _) => {
                trace::trace_interpreter!("invoke_func_eval -> {}, nargs = {}", Value::NativeFunction(b.clone()).as_debug_str(), nargs);
                // Need to consume the arguments and set up the stack for calling as if all partial arguments were just pushed
                // Surgically extract the binding via std::mem::replace
                let binding: NativeFunction = *b;
                let i: usize = self.stack.len() - 1 - nargs as usize;
                let args: Vec<Value> = match std::mem::replace(&mut self.stack[i], Value::Nil) {
                    Value::PartialNativeFunction(_, x) => *x,
                    _ => panic!("Stack corruption")
                };

                // Splice the args from the binding back into the stack, in the correct order
                // When evaluating a partial function the vm stack will contain [..., argN+1, ... argM]
                // The partial args will contain the vector [argN, argN-1, ... arg1]
                // After this, the vm stack should contain the args [..., arg1, arg2, ... argM]
                let j: usize = self.stack.len() - nargs as usize;
                let partial_args: u8 = args.len() as u8;
                self.stack.splice(j..j, args.into_iter().rev());

                match stdlib::invoke(binding, nargs + partial_args, self) {
                    Ok(v) => {
                        self.pop();
                        self.push(v);
                    },
                    Err(e) => return Err(e),
                }
                Ok(FunctionType::Native)
            },
            _ => return ValueIsNotFunctionEvaluable(f.clone()).err(),
        }
    }

    /// Calls a user function by building a `CallFrame` and jumping to the function's `head` IP
    fn call_function(self: &mut Self, head: usize, nargs: u8) {
        let frame = CallFrame {
            return_ip: self.ip,
            frame_pointer: self.stack.len() - (nargs as usize),
        };
        self.ip = head;
        self.call_stack.push(frame);
        trace::trace_interpreter!("call stack = {}", self.debug_call_stack());
    }


    // ===== Debug Methods ===== //

    pub fn debug_stack(self: &Self) -> String {
        format!(": [{}]", self.stack.iter().rev().map(|t| t.as_debug_str()).collect::<Vec<String>>().join(", "))
    }

    pub fn debug_call_stack(self: &Self) -> String {
        format!(": [{}]", self.call_stack.iter().rev().map(|t| format!("{{fp: {}, ret: {}}}", t.frame_pointer, t.return_ip)).collect::<Vec<String>>().join(", "))
    }
}


impl <R, W> VirtualInterface for VirtualMachine<R, W> where
    R : BufRead,
    W : Write
{
    // ===== Calling Functions External Interface ===== //

    fn invoke_func1(self: &mut Self, f: Value, a1: Value) -> ValueResult {
        self.push(f);
        self.push(a1);
        self.invoke_func_and_spin(1)
    }

    fn invoke_func2(self: &mut Self, f: Value, a1: Value, a2: Value) -> ValueResult {
        self.push(f);
        self.push(a1);
        self.push(a2);
        self.invoke_func_and_spin(2)
    }

    fn invoke_func(self: &mut Self, f: Value, args: &Vec<Value>) -> ValueResult {
        self.push(f);
        for arg in args {
            self.push(arg.clone());
        }
        self.invoke_func_and_spin(args.len() as u8)
    }

    fn invoke_eval(self: &mut Self, text: &String) -> ValueResult {
        let eval_head: usize = self.code.len();

        self.eval_compile(text)?;
        self.call_function(eval_head, 0);
        self.run_frame()?;
        let ret = self.pop();
        self.push(Value::Nil); // `eval` executes as a user function but is called like a native function, this prevents stack fuckery
        Ok(ret)
    }

    // ===== IO Methods ===== //

    fn println0(self: &mut Self) { writeln!(&mut self.write, "").unwrap(); }
    fn println(self: &mut Self, str: String) { writeln!(&mut self.write, "{}", str).unwrap(); }
    fn print(self: &mut Self, str: String) { write!(&mut self.write, "{}", str).unwrap(); }


    // ===== Stack Manipulations ===== //

    /// Peeks at the top element of the stack, or an element `offset` down from the top
    fn peek(self: &Self, offset: usize) -> &Value {
        trace::trace_interpreter_stack!("peek({}) -> {}", offset, self.stack[self.stack.len() - 1 - offset].as_debug_str());
        let ret = self.stack.get(self.stack.len() - 1 - offset).unwrap();
        trace::trace_interpreter_stack!("{}", self.debug_stack());
        ret
    }

    /// Pops the top of the stack
    fn pop(self: &mut Self) -> Value {
        trace::trace_interpreter_stack!("pop() -> {}", self.stack.last().unwrap().as_debug_str());
        let ret = self.stack.pop().unwrap();
        trace::trace_interpreter_stack!("{}", self.debug_stack());
        ret
    }

    /// Pops the top N values off the stack, in order
    fn popn(self: &mut Self, n: usize) -> Vec<Value> {
        trace::trace_interpreter_stack!("popn({}) -> {}, ...", n, self.stack.last().unwrap().as_debug_str());
        let length: usize = self.stack.len();
        let ret = self.stack.splice(length - n..length, std::iter::empty()).collect();
        trace::trace_interpreter_stack!("{}", self.debug_stack());
        ret
    }

    /// Push a value onto the stack
    fn push(self: &mut Self, value: Value) {
        trace::trace_interpreter_stack!("push({})", value.as_debug_str());
        self.stack.push(value);
        trace::trace_interpreter_stack!("{}", self.debug_stack());
    }
}


#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::{compiler, reporting, trace};
    use crate::reporting::ErrorReporter;
    use crate::vm::{ExitType, VirtualMachine};

    #[test] fn test_str_empty() { run_str("", ""); }
    #[test] fn test_str_hello_world() { run_str("print('hello world!')", "hello world!\n"); }
    #[test] fn test_str_empty_print() { run_str("print()", "\n"); }
    #[test] fn test_str_strings() { run_str("print('first', 'second', 'third')", "first second third\n"); }
    #[test] fn test_str_other_args() { run_str("print(nil, -1, 1, true, false, 'test', print)", "nil -1 1 true false test print\n"); }
    #[test] fn test_str_unary_ops() { run_str("print(-1, --1, ---1, ~3, ~~3, !true, !!true)", "-1 1 -1 -4 3 false true\n"); }
    #[test] fn test_str_add_str() { run_str("print(('a' + 'b') + (3 + 4) + (' hello' + 3) + (' and' + true + nil))", "ab7 hello3 andtruenil\n"); }
    #[test] fn test_str_mul_str() { run_str("print('abc' * 3)", "abcabcabc\n"); }
    #[test] fn test_str_add_sub_mul_div_int() { run_str("print(5 - 3, 12 + 5, 3 * 9, 16 / 3)", "2 17 27 5\n"); }
    #[test] fn test_str_div_mod_int() { run_str("print(3 / 2, 3 / 3, -3 / 2, 10 % 3, 11 % 3, 12 % 3)", "1 1 -2 1 2 0\n"); }
    #[test] fn test_str_div_by_zero() { run_str("print(15 / 0)", "TypeError: Cannot divide '15' of type 'int' and '0' of type 'int'\n    at: `print(15 / 0)` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_str_mod_by_zero() { run_str("print(15 % 0)", "TypeError: Cannot modulo '15' of type 'int' and '0' of type 'int'\n    at: `print(15 % 0)` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_str_left_right_shift() { run_str("print(1 << 10, 16 >> 1, 16 << -1, 1 >> -10)", "1024 8 8 1024\n"); }
    #[test] fn test_str_compare_ints_1() { run_str("print(1 < 3, -5 < -10, 6 > 7, 6 > 4)", "true false false true\n"); }
    #[test] fn test_str_compare_ints_2() { run_str("print(1 <= 3, -5 < -10, 3 <= 3, 2 >= 2, 6 >= 7, 6 >= 4, 6 <= 6, 8 >= 8)", "true false true true false true true true\n"); }
    #[test] fn test_str_equal_ints() { run_str("print(1 == 3, -5 == -10, 3 != 3, 2 == 2, 6 != 7)", "false false false true true\n"); }
    #[test] fn test_str_compare_bools_1() { run_str("print(false < false, false < true, true < false, true < true)", "false true false false\n"); }
    #[test] fn test_str_compare_bools_2() { run_str("print(false <= false, false >= true, true >= false, true <= true)", "true false true true\n"); }
    #[test] fn test_str_bitwise_ops() { run_str("print(0b111 & 0b100, 0b1100 | 0b1010, 0b1100 ^ 0b1010)", "4 14 6\n"); }
    #[test] fn test_str_compose() { run_str("print . print", "print\n"); }
    #[test] fn test_str_compose_str() { run_str("'hello world' . print", "hello world\n"); }
    #[test] fn test_str_if_01() { run_str("if 1 < 2 { print('yes') } else { print ('no') }", "yes\n"); }
    #[test] fn test_str_if_02() { run_str("if 1 < -2 { print('yes') } else { print ('no') }", "no\n"); }
    #[test] fn test_str_if_03() { run_str("if true { print('yes') } print('and also')", "yes\nand also\n"); }
    #[test] fn test_str_if_04() { run_str("if 1 < -2 { print('yes') } print('and also')", "and also\n"); }
    #[test] fn test_str_if_05() { run_str("if 0 { print('yes') }", ""); }
    #[test] fn test_str_if_06() { run_str("if 1 { print('yes') }", "yes\n"); }
    #[test] fn test_str_if_07() { run_str("if 'string' { print('yes') }", "yes\n"); }
    #[test] fn test_str_if_08() { run_str("if 1 < 0 { print('yes') } elif 1 { print('hi') } else { print('hehe') }", "hi\n"); }
    #[test] fn test_str_if_09() { run_str("if 1 < 0 { print('yes') } elif 2 < 0 { print('hi') } else { print('hehe') }", "hehe\n"); }
    #[test] fn test_str_if_10() { run_str("if 1 { print('yes') } elif true { print('hi') } else { print('hehe') }", "yes\n"); }
    #[test] fn test_str_short_circuiting_1() { run_str("if true and print('yes') { print('no') }", "yes\n"); }
    #[test] fn test_str_short_circuiting_2() { run_str("if false and print('also no') { print('no') }", ""); }
    #[test] fn test_str_short_circuiting_3() { run_str("if true and (print('yes') or true) { print('also yes') }", "yes\nalso yes\n"); }
    #[test] fn test_str_short_circuiting_4() { run_str("if false or print('yes') { print('no') }", "yes\n"); }
    #[test] fn test_str_short_circuiting_5() { run_str("if true or print('no') { print('yes') }", "yes\n"); }
    #[test] fn test_str_partial_func_1() { run_str("'apples and bananas' . replace ('a', 'o') . print", "opples ond bononos\n"); }
    #[test] fn test_str_partial_func_2() { run_str("'apples and bananas' . replace ('a') ('o') . print", "opples ond bononos\n"); }
    #[test] fn test_str_partial_func_3() { run_str("print('apples and bananas' . replace ('a') ('o'))", "opples ond bononos\n"); }
    #[test] fn test_str_partial_func_4() { run_str("let x = replace ('a', 'o') ; 'apples and bananas' . x . print", "opples ond bononos\n"); }
    #[test] fn test_str_partial_func_5() { run_str("let x = replace ('a', 'o') ; print(x('apples and bananas'))", "opples ond bononos\n"); }
    #[test] fn test_str_partial_func_6() { run_str("('o' . replace('a')) ('apples and bananas') . print", "opples ond bononos\n"); }
    #[test] fn test_str_list_len() { run_str("[1, 2, 3] . len . print", "3\n"); }
    #[test] fn test_str_str_len() { run_str("'12345' . len . print", "5\n"); }
    #[test] fn test_str_list_print() { run_str("[1, 2, '3'] . print", "[1, 2, '3']\n"); }
    #[test] fn test_str_list_repr_print() { run_str("['1', 2, '3'] . repr . print", "['1', 2, '3']\n"); }
    #[test] fn test_str_list_add() { run_str("[1, 2, 3] + [4, 5, 6] . print", "[1, 2, 3, 4, 5, 6]\n"); }
    #[test] fn test_str_empty_list() { run_str("[] . print", "[]\n"); }
    #[test] fn test_str_list_and_index() { run_str("[1, 2, 3] [1] . print", "2\n"); }
    #[test] fn test_str_list_index_out_of_bounds() { run_str("[1, 2, 3] [3] . print", "Index '3' is out of bounds for list of length [0, 3)\n    at: `[1, 2, 3] [3] . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_str_list_index_negative() { run_str("[1, 2, 3] [-1] . print", "3\n"); }
    #[test] fn test_str_list_slice_01() { run_str("[1, 2, 3, 4] [:] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_02() { run_str("[1, 2, 3, 4] [::] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_03() { run_str("[1, 2, 3, 4] [::1] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_04() { run_str("[1, 2, 3, 4] [1:] . print", "[2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_05() { run_str("[1, 2, 3, 4] [:2] . print", "[1, 2]\n"); }
    #[test] fn test_str_list_slice_06() { run_str("[1, 2, 3, 4] [0:] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_07() { run_str("[1, 2, 3, 4] [:4] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_08() { run_str("[1, 2, 3, 4] [1:3] . print", "[2, 3]\n"); }
    #[test] fn test_str_list_slice_09() { run_str("[1, 2, 3, 4] [2:4] . print", "[3, 4]\n"); }
    #[test] fn test_str_list_slice_10() { run_str("[1, 2, 3, 4] [0:2] . print", "[1, 2]\n"); }
    #[test] fn test_str_list_slice_11() { run_str("[1, 2, 3, 4] [:-1] . print", "[1, 2, 3]\n"); }
    #[test] fn test_str_list_slice_12() { run_str("[1, 2, 3, 4] [:-2] . print", "[1, 2]\n"); }
    #[test] fn test_str_list_slice_13() { run_str("[1, 2, 3, 4] [-2:] . print", "[3, 4]\n"); }
    #[test] fn test_str_list_slice_14() { run_str("[1, 2, 3, 4] [-3:] . print", "[2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_15() { run_str("[1, 2, 3, 4] [::2] . print", "[1, 3]\n"); }
    #[test] fn test_str_list_slice_16() { run_str("[1, 2, 3, 4] [::3] . print", "[1, 4]\n"); }
    #[test] fn test_str_list_slice_17() { run_str("[1, 2, 3, 4] [::4] . print", "[1]\n"); }
    #[test] fn test_str_list_slice_18() { run_str("[1, 2, 3, 4] [1::2] . print", "[2, 4]\n"); }
    #[test] fn test_str_list_slice_19() { run_str("[1, 2, 3, 4] [1:3:2] . print", "[2]\n"); }
    #[test] fn test_str_list_slice_20() { run_str("[1, 2, 3, 4] [:-1:2] . print", "[1, 3]\n"); }
    #[test] fn test_str_list_slice_21() { run_str("[1, 2, 3, 4] [1:-1:3] . print", "[2]\n"); }
    #[test] fn test_str_list_slice_22() { run_str("[1, 2, 3, 4] [::-1] . print", "[4, 3, 2, 1]\n"); }
    #[test] fn test_str_list_slice_23() { run_str("[1, 2, 3, 4] [1::-1] . print", "[2, 1]\n"); }
    #[test] fn test_str_list_slice_24() { run_str("[1, 2, 3, 4] [:2:-1] . print", "[4]\n"); }
    #[test] fn test_str_list_slice_25() { run_str("[1, 2, 3, 4] [3:1:-1] . print", "[4, 3]\n"); }
    #[test] fn test_str_list_slice_26() { run_str("[1, 2, 3, 4] [-1:-2:-1] . print", "[4]\n"); }
    #[test] fn test_str_list_slice_27() { run_str("[1, 2, 3, 4] [-2::-1] . print", "[3, 2, 1]\n"); }
    #[test] fn test_str_list_slice_28() { run_str("[1, 2, 3, 4] [:-3:-1] . print", "[4, 3]\n"); }
    #[test] fn test_str_list_slice_29() { run_str("[1, 2, 3, 4] [::-2] . print", "[4, 2]\n"); }
    #[test] fn test_str_list_slice_30() { run_str("[1, 2, 3, 4] [::-3] . print", "[4, 1]\n"); }
    #[test] fn test_str_list_slice_31() { run_str("[1, 2, 3, 4] [::-4] . print", "[4]\n"); }
    #[test] fn test_str_list_slice_32() { run_str("[1, 2, 3, 4] [-2::-2] . print", "[3, 1]\n"); }
    #[test] fn test_str_list_slice_33() { run_str("[1, 2, 3, 4] [-3::-2] . print", "[2]\n"); }
    #[test] fn test_str_list_slice_34() { run_str("[1, 2, 3, 4] [1:1] . print", "[]\n"); }
    #[test] fn test_str_list_slice_35() { run_str("[1, 2, 3, 4] [-1:-1] . print", "[]\n"); }
    #[test] fn test_str_list_slice_36() { run_str("[1, 2, 3, 4] [-1:1:] . print", "[]\n"); }
    #[test] fn test_str_list_slice_37() { run_str("[1, 2, 3, 4] [1:1:-1] . print", "[]\n"); }
    #[test] fn test_str_list_slice_38() { run_str("[1, 2, 3, 4] [-2:2:-3] . print", "[]\n"); }
    #[test] fn test_str_list_slice_39() { run_str("[1, 2, 3, 4] [-1:1:-1] . print", "[4, 3]\n"); }
    #[test] fn test_str_list_slice_40() { run_str("[1, 2, 3, 4] [1:-1:-1] . print", "[]\n"); }
    #[test] fn test_str_list_slice_41() { run_str("[1, 2, 3, 4] [1:10:1] . print", "[2, 3, 4]\n"); }
    #[test] fn test_str_list_slice_42() { run_str("[1, 2, 3, 4] [10:1:-1] . print", "[4, 3]\n"); }
    #[test] fn test_str_list_slice_43() { run_str("[1, 2, 3, 4] [-10:1] . print", "[1]\n"); }
    #[test] fn test_str_list_slice_44() { run_str("[1, 2, 3, 4] [1:-10:-1] . print", "[2, 1]\n"); }
    #[test] fn test_str_list_slice_45() { run_str("[1, 2, 3, 4] [::0]", "ValueError: 'step' argument cannot be zero\n    at: `[1, 2, 3, 4] [::0]` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_str_list_slice_46() { run_str("[1, 2, 3, 4][:-1] . print", "[1, 2, 3]\n"); }
    #[test] fn test_str_list_slice_47() { run_str("[1, 2, 3, 4][:0] . print", "[]\n"); }
    #[test] fn test_str_list_slice_48() { run_str("[1, 2, 3, 4][:1] . print", "[1]\n"); }
    #[test] fn test_str_list_slice_49() { run_str("[1, 2, 3, 4][5:] . print", "[]\n"); }
    #[test] fn test_str_sum_list() { run_str("[1, 2, 3, 4] . sum . print", "10\n"); }
    #[test] fn test_str_sum_values() { run_str("sum(1, 3, 5, 7) . print", "16\n"); }
    #[test] fn test_str_sum_no_arg() { run_str("sum()", "Function 'sum' requires at least 1 parameter but none were present.\n    at: `sum()` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_str_sum_empty_list() { run_str("[] . sum . print", "0\n"); }
    #[test] fn test_local_vars_01() { run_str("let x=0 { x.print }", "0\n"); }
    #[test] fn test_local_vars_02() { run_str("let x=0 { let x=1; x.print }", "1\n"); }
    #[test] fn test_local_vars_03() { run_str("let x=0 { x.print let x=1 }", "0\n"); }
    #[test] fn test_local_vars_04() { run_str("let x=0 { let x=1 } x.print", "0\n"); }
    #[test] fn test_local_vars_05() { run_str("let x=0 { x=1 } x.print", "1\n"); }
    #[test] fn test_local_vars_06() { run_str("let x=0 { x=1 { x=2; x.print } }", "2\n"); }
    #[test] fn test_local_vars_07() { run_str("let x=0 { x=1 { x=2 } x.print }", "2\n"); }
    #[test] fn test_local_vars_08() { run_str("let x=0 { let x=1 { x=2 } x.print }", "2\n"); }
    #[test] fn test_local_vars_09() { run_str("let x=0 { let x=1 { let x=2 } x.print }", "1\n"); }
    #[test] fn test_local_vars_10() { run_str("let x=0 { x=1 { let x=2 } x.print }", "1\n"); }
    #[test] fn test_local_vars_11() { run_str("let x=0 { x=1 { let x=2 } } x.print", "1\n"); }
    #[test] fn test_local_vars_12() { run_str("let x=0 { let x=1 { let x=2 } } x.print", "0\n"); }
    #[test] fn test_local_vars_14() { run_str("let x=3 { let x=x; x.print }", "3\n"); }
    #[test] fn test_functions_01() { run_str("fn foo() { 'hello' . print } ; foo();", "hello\n"); }
    #[test] fn test_functions_02() { run_str("fn foo() { 'hello' . print } ; foo() ; foo()", "hello\nhello\n"); }
    #[test] fn test_functions_03() { run_str("fn foo(a) { 'hello' . print } ; foo(1)", "hello\n"); }
    #[test] fn test_functions_04() { run_str("fn foo(a) { 'hello ' + a . print } ; foo(1)", "hello 1\n"); }
    #[test] fn test_functions_05() { run_str("fn foo(a, b, c) { a + b + c . print } ; foo(1, 2, 3)", "6\n"); }
    #[test] fn test_functions_06() { run_str("fn foo() { 'hello' . print } ; fn bar() { foo() } bar()", "hello\n"); }
    #[test] fn test_functions_07() { run_str("fn foo() { 'hello' . print } ; fn bar(a) { foo() } bar(1)", "hello\n"); }
    #[test] fn test_functions_08() { run_str("fn foo(a) { a . print } ; fn bar(a, b, c) { foo(a + b + c) } bar(1, 2, 3)", "6\n"); }
    #[test] fn test_functions_09() { run_str("fn foo(h, w) { h + ' ' + w . print } ; fn bar(w) { foo('hello', w) } bar('world')", "hello world\n"); }
    #[test] fn test_functions_10() { run_str("let x = 'hello' ; fn foo(x) { x . print } foo(x)", "hello\n"); }
    #[test] fn test_functions_11() { run_str("{ let x = 'hello' ; fn foo(x) { x . print } foo(x) }", "hello\n"); }
    #[test] fn test_functions_12() { run_str("{ let x = 'hello' ; { fn foo(x) { x . print } foo(x) } }", "hello\n"); }
    #[test] fn test_functions_13() { run_str("let x = 'hello' ; { fn foo() { x . print } foo() }", "hello\n"); }
    #[test] fn test_functions_14() { run_str("fn foo(x) { 'hello ' + x . print } 'world' . foo", "hello world\n"); }
    #[test] fn test_function_implicit_return_01() { run_str("fn foo() { } foo() . print", "nil\n"); }
    #[test] fn test_function_implicit_return_02() { run_str("fn foo() { 'hello' } foo() . print", "hello\n"); }
    #[test] fn test_function_implicit_return_03() { run_str("fn foo(x) { if x > 1 {} else {} } foo(2) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_04() { run_str("fn foo(x) { if x > 1 { true } else { false } } foo(2) . print", "true\n"); }
    #[test] fn test_function_implicit_return_05() { run_str("fn foo(x) { if x > 1 { true } else { false } } foo(0) . print", "false\n"); }
    #[test] fn test_function_implicit_return_06() { run_str("fn foo(x) { if x > 1 { } else { false } } foo(2) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_07() { run_str("fn foo(x) { if x > 1 { } else { false } } foo(0) . print", "false\n"); }
    #[test] fn test_function_implicit_return_08() { run_str("fn foo(x) { if x > 1 { true } else { } } foo(2) . print", "true\n"); }
    #[test] fn test_function_implicit_return_09() { run_str("fn foo(x) { if x > 1 { true } else { } } foo(0) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_10() { run_str("fn foo(x) { if x > 1 { 'hello' } } foo(2) . print", "hello\n"); }
    #[test] fn test_function_implicit_return_11() { run_str("fn foo(x) { if x > 1 { 'hello' } } foo(0) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_12() { run_str("fn foo(x) { if x > 1 { if true { 'hello' } } } foo(2) . print", "hello\n"); }
    #[test] fn test_function_implicit_return_13() { run_str("fn foo(x) { if x > 1 { if true { 'hello' } } } foo(0) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_14() { run_str("fn foo(x) { loop { if x > 1 { break } } } foo(2) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_15() { run_str("fn foo(x) { loop { if x > 1 { continue } else { break } } } foo(0) . print", "nil\n"); }
    #[test] fn test_closures_01() { run_str("fn foo() { let x = 'hello' ; fn bar() { x . print } bar() } foo()", "hello\n"); }
    #[test] fn test_closures_02() { run_str("{ fn foo() { 'hello' . print } ; fn bar() { foo() } bar() }", "hello\n"); }
    #[test] fn test_closures_03() { run_str("{ fn foo() { 'hello' . print } ; { fn bar() { foo() } bar() } }", "hello\n"); }
    #[test] fn test_closures_04() { run_str("{ fn foo() { 'hello' . print } ; fn bar(a) { foo() } bar(1) }", "hello\n"); }
    #[test] fn test_closures_05() { run_str("{ fn foo() { 'hello' . print } ; { fn bar(a) { foo() } bar(1) } }", "hello\n"); }
    #[test] fn test_closures_06() { run_str("{ let x = 'hello' ; { fn foo() { x . print } foo() } }", "hello\n"); }
    #[test] fn test_closures_07() { run_str("{ let x = 'hello' ; { { fn foo() { x . print } foo() } } }", "hello\n"); }
    #[test] fn test_closures_08() { run_str("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; x } foo() . print", "hello\n"); }
    #[test] fn test_closures_09() { run_str("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; (fn() -> x = 'goodbye')() ; x } foo() . print", "goodbye\n"); }
    #[test] fn test_closures_10() { run_str("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; let y = (fn() -> x)() ; y } foo() . print", "hello\n"); }
    #[test] fn test_closures_11() { run_str("fn foo() { let x = 'hello' ; (fn() -> x += ' world')() ; x } foo() . print", "hello world\n"); }
    #[test] fn test_closures_12() { run_str("fn foo() { let x = 'hello' ; (fn() -> x += ' world')() ; (fn() -> x)() } foo() . print", "hello world\n"); }
    #[test] fn test_function_return_1() { run_str("fn foo() { return 3 } foo() . print", "3\n"); }
    #[test] fn test_function_return_2() { run_str("fn foo() { let x = 3; return x } foo() . print", "3\n"); }
    #[test] fn test_function_return_3() { run_str("fn foo() { let x = 3; { return x } } foo() . print", "3\n"); }
    #[test] fn test_function_return_4() { run_str("fn foo() { let x = 3; { let x; } return x } foo() . print", "3\n"); }
    #[test] fn test_function_return_5() { run_str("fn foo() { let x; { let x = 3; return x } } foo() . print", "3\n"); }
    #[test] fn test_partial_function_composition_1() { run_str("fn foo(a, b, c) { c . print } (3 . (2 . (1 . foo)))", "3\n"); }
    #[test] fn test_partial_function_composition_2() { run_str("fn foo(a, b, c) { c . print } (2 . (1 . foo)) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_3() { run_str("fn foo(a, b, c) { c . print } (1 . foo) (2) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_4() { run_str("fn foo(a, b, c) { c . print } foo (1) (2) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_5() { run_str("fn foo(a, b, c) { c . print } foo (1, 2) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_6() { run_str("fn foo(a, b, c) { c . print } foo (1) (2, 3)", "3\n"); }
    #[test] fn test_operator_functions_01() { run_str("(+3) . print", "(+)\n"); }
    #[test] fn test_operator_functions_02() { run_str("4 . (+3) . print", "7\n"); }
    #[test] fn test_operator_functions_03() { run_str("4 . (-) . print", "-4\n"); }
    #[test] fn test_operator_functions_04() { run_str("true . (!) . print", "false\n"); }
    #[test] fn test_operator_functions_05() { run_str("let f = (/5) ; 15 . f . print", "3\n"); }
    #[test] fn test_operator_functions_06() { run_str("let f = (*5) ; f(3) . print", "15\n"); }
    #[test] fn test_operator_functions_07() { run_str("let f = (<5) ; 3 . f . print", "true\n"); }
    #[test] fn test_operator_functions_08() { run_str("let f = (>5) ; 3 . f . print", "false\n"); }
    #[test] fn test_operator_functions_09() { run_str("2 . (**5) . print", "32\n"); }
    #[test] fn test_operator_functions_10() { run_str("7 . (%3) . print", "1\n"); }
    #[test] fn test_arrow_functions_01() { run_str("fn foo() -> 3 ; foo() . print", "3\n"); }
    #[test] fn test_arrow_functions_02() { run_str("fn foo() -> 3 ; foo() . print", "3\n"); }
    #[test] fn test_arrow_functions_03() { run_str("fn foo(a) -> 3 * a ; 5 . foo . print", "15\n"); }
    #[test] fn test_arrow_functions_04() { run_str("fn foo(x, y, z) -> x + y + z ; foo(1, 2, 4) . print", "7\n"); }
    #[test] fn test_arrow_functions_05() { run_str("fn foo() -> (fn() -> 123) ; foo() . print", "_\n"); }
    #[test] fn test_arrow_functions_06() { run_str("fn foo() -> (fn() -> 123) ; foo() . repr . print", "fn _()\n"); }
    #[test] fn test_arrow_functions_07() { run_str("fn foo() -> (fn(a, b, c) -> 123) ; foo() . print", "_\n"); }
    #[test] fn test_arrow_functions_08() { run_str("fn foo() -> (fn(a, b, c) -> 123) ; foo() . repr . print", "fn _(a, b, c)\n"); }
    #[test] fn test_arrow_functions_09() { run_str("fn foo() -> (fn() -> 123) ; foo()() . print", "123\n"); }
    #[test] fn test_arrow_functions_10() { run_str("let x = fn() -> 3 ; x() . print", "3\n"); }
    #[test] fn test_arrow_functions_11() { run_str("let x = fn() -> fn() -> 4 ; x() . print", "_\n"); }
    #[test] fn test_arrow_functions_12() { run_str("let x = fn() -> fn() -> 4 ; x() . repr .print", "fn _()\n"); }
    #[test] fn test_arrow_functions_13() { run_str("let x = fn() -> fn() -> 4 ; x()() . print", "4\n"); }
    #[test] fn test_arrow_functions_14() { run_str("fn foo() { if true { return 123 } else { return 321 } } ; foo() . print", "123\n"); }
    #[test] fn test_arrow_functions_15() { run_str("fn foo() { if false { return 123 } else { return 321 } } ; foo() . print", "321\n"); }
    #[test] fn test_arrow_functions_16() { run_str("fn foo() { let x = 1234; x } ; foo() . print", "1234\n"); }
    #[test] fn test_arrow_functions_17() { run_str("fn foo() { let x, y; x + ' and ' + y } ; foo() . print", "nil and nil\n"); }
    #[test] fn test_arrow_functions_18() { run_str("fn foo() { fn bar() -> 3 ; bar } ; foo() . print", "bar\n"); }
    #[test] fn test_arrow_functions_19() { run_str("fn foo() { fn bar() -> 3 ; bar } ; foo() . repr . print", "fn bar()\n"); }
    #[test] fn test_arrow_functions_20() { run_str("fn foo() { fn bar() -> 3 ; bar } ; foo()() . print", "3\n"); }
    #[test] fn test_builtin_map() { run_str("[1, 2, 3] . map(str) . repr . print", "['1', '2', '3']\n") }
    #[test] fn test_builtin_map_lambda() { run_str("[-1, 2, -3] . map(fn(x) -> x . abs) . print", "[1, 2, 3]\n") }
    #[test] fn test_builtin_filter() { run_str("[2, 3, 4, 5, 6] . filter (>3) . print", "[4, 5, 6]\n") }
    #[test] fn test_builtin_filter_lambda() { run_str("[2, 3, 4, 5, 6] . filter (fn(x) -> x % 2 == 0) . print", "[2, 4, 6]\n") }
    #[test] fn test_builtin_reduce() { run_str("[1, 2, 3, 4, 5, 6] . reduce (*) . print", "720\n"); }
    #[test] fn test_builtin_reduce_lambda() { run_str("[1, 2, 3, 4, 5, 6] . reduce (fn(a, b) -> a * b) . print", "720\n"); }
    #[test] fn test_builtin_reduce_with_builtin() { run_str("[1, 2, 3, 4, 5, 6] . reduce (sum) . print", "21\n"); }
    #[test] fn test_builtin_reduce_with_empty() { run_str("[] . reduce(+) . print", "ValueError: Expected value to be a non empty iterable\n    at: `[] . reduce(+) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_builtin_sorted() { run_str("[6, 2, 3, 7, 2, 1] . sort . print", "[1, 2, 2, 3, 6, 7]\n"); }
    #[test] fn test_builtin_reversed() { run_str("[8, 1, 2, 6, 3, 2, 3] . reverse . print", "[3, 2, 3, 6, 2, 1, 8]\n"); }
    #[test] fn test_bare_operator_eval() { run_str("(+)(1, 2) . print", "3\n"); }
    #[test] fn test_bare_operator_partial_eval() { run_str("(+)(1)(2) . print", "3\n"); }
    #[test] fn test_bare_operator_compose_and_eval() { run_str("2 . (+)(1) . print", "3\n"); }
    #[test] fn test_bare_operator_compose() { run_str("1 . (2 . (+)) . print", "3\n"); }
    #[test] fn test_reduce_list_1() { run_str("[1, 2, 3] . reduce (+) . print", "6\n"); }
    #[test] fn test_reduce_list_2() { run_str("[1, 2, 3] . reduce (!) . print", "Function '(!)' requires 2 parameters but 1 were present.\n    at: `[1, 2, 3] . reduce (!) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_str_to_list() { run_str("'funny beans' . list . print", "['f', 'u', 'n', 'n', 'y', ' ', 'b', 'e', 'a', 'n', 's']\n"); }
    #[test] fn test_str_to_set() { run_str("'funny beans' . set . print", "{'f', 'u', 'n', 'y', ' ', 'b', 'e', 'a', 's'}\n"); }
    #[test] fn test_str_to_set_to_sorted() { run_str("'funny' . set . sort . print", "['f', 'n', 'u', 'y']\n"); }
    #[test] fn test_chained_assignments() { run_str("let a, b, c; a = b = c = 3; [a, b, c] . print", "[3, 3, 3]\n"); }
    #[test] fn test_array_assignment_1() { run_str("let a = [1, 2, 3]; a[0] = 3; a . print", "[3, 2, 3]\n"); }
    #[test] fn test_array_assignment_2() { run_str("let a = [1, 2, 3]; a[2] = 1; a . print", "[1, 2, 1]\n"); }
    #[test] fn test_array_assignment_negative_index_1() { run_str("let a = [1, 2, 3]; a[-1] = 6; a . print", "[1, 2, 6]\n"); }
    #[test] fn test_array_assignment_negative_index_2() { run_str("let a = [1, 2, 3]; a[-3] = 6; a . print", "[6, 2, 3]\n"); }
    #[test] fn test_nested_array_assignment_1() { run_str("let a = [[1, 2], [3, 4]]; a[0][1] = 6; a . print", "[[1, 6], [3, 4]]\n"); }
    #[test] fn test_nested_array_assignment_2() { run_str("let a = [[1, 2], [3, 4]]; a[1][0] = 6; a . print", "[[1, 2], [6, 4]]\n"); }
    #[test] fn test_nested_array_assignment_negative_index_1() { run_str("let a = [[1, 2], [3, 4]]; a[0][-1] = 6; a . print", "[[1, 6], [3, 4]]\n"); }
    #[test] fn test_nested_array_assignment_negative_index_2() { run_str("let a = [[1, 2], [3, 4]]; a[-1][-2] = 6; a . print", "[[1, 2], [6, 4]]\n"); }
    #[test] fn test_chained_operator_assignment() { run_str("let a = 1, b; a += b = 4; [a, b] . print", "[5, 4]\n"); }
    #[test] fn test_operator_array_assignment() { run_str("let a = [12]; a[0] += 4; a[0] . print", "16\n"); }
    #[test] fn test_nested_operator_array_assignment() { run_str("let a = [[12]]; a[0][-1] += 4; a . print", "[[16]]\n"); }
    #[test] fn test_weird_assignment() { run_str("let a = [[12]], b = 3; fn f() -> a; f()[0][-1] += b = 5; [f(), b] . print", "[[[17]], 5]\n"); }
    #[test] fn test_mutable_array_in_array_1() { run_str("let a = [0], b = [a]; b[0] = 'hi'; b. print", "['hi']\n"); }
    #[test] fn test_mutable_array_in_array_2() { run_str("let a = [0], b = [a]; b[0][0] = 'hi'; b. print", "[['hi']]\n"); }
    #[test] fn test_list_mul_int() { run_str("[1, 2, 3] * 3 . print", "[1, 2, 3, 1, 2, 3, 1, 2, 3]\n"); }
    #[test] fn test_int_mul_list() { run_str("3 * [1, 2, 3] . print", "[1, 2, 3, 1, 2, 3, 1, 2, 3]\n"); }
    #[test] fn test_mutable_arrays_in_assignments() { run_str("let a = [0], b = [a, a, a]; b[0][0] = 5; b . print", "[[5], [5], [5]]\n"); }
    #[test] fn test_test_nested_array_multiplication() { run_str("let a = [[1]] * 3; a[0][0] = 2; a . print", "[[2], [2], [2]]\n"); }
    #[test] fn test_heap_from_list() { run_str("let h = [1, 7, 3, 2, 7, 6] . heap; h . print", "[1, 2, 3, 7, 7, 6]\n"); }
    #[test] fn test_heap_pop() { run_str("let h = [1, 7, 3, 2, 7, 6] . heap; [h.pop, h.pop, h.pop] . print", "[1, 2, 3]\n"); }
    #[test] fn test_heap_push() { run_str("let h = [1, 7, 3, 2, 7, 6] . heap; h.push(3); h.push(-1); h.push(16); h . print", "[-1, 1, 3, 2, 7, 6, 3, 7, 16]\n"); }
    #[test] fn test_range_1() { run_str("range(3) . list . print", "[0, 1, 2]\n"); }
    #[test] fn test_range_2() { run_str("range(3, 7) . list . print", "[3, 4, 5, 6]\n"); }
    #[test] fn test_range_3() { run_str("range(1, 9, 3) . list . print", "[1, 4, 7]\n"); }
    #[test] fn test_range_4() { run_str("range(6, 3) . list . print", "[]\n"); }
    #[test] fn test_range_5() { run_str("range(10, 4, -2) . list . print", "[10, 8, 6]\n"); }
    #[test] fn test_range_6() { run_str("range(0, 20, -1) . list . print", "[]\n"); }
    #[test] fn test_range_7() { run_str("range(10, 0, 3) . list . print", "[]\n"); }
    #[test] fn test_range_8() { run_str("range(1, 1, 1) . list . print", "[]\n"); }
    #[test] fn test_range_9() { run_str("range(1, 1, 0) . list . print", "ValueError: 'step' argument cannot be zero\n    at: `range(1, 1, 0) . list . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_enumerate_1() { run_str("[] . enumerate . list . print", "[]\n"); }
    #[test] fn test_enumerate_2() { run_str("[1, 2, 3] . enumerate . list . print", "[(0, 1), (1, 2), (2, 3)]\n"); }
    #[test] fn test_enumerate_3() { run_str("'foobar' . enumerate . list . print", "[(0, 'f'), (1, 'o'), (2, 'o'), (3, 'b'), (4, 'a'), (5, 'r')]\n"); }
    #[test] fn test_for_loop_no_intrinsic_with_list() { run_str("for x in ['a', 'b', 'c'] { x . print }", "a\nb\nc\n") }
    #[test] fn test_for_loop_no_intrinsic_with_set() { run_str("for x in 'foobar' . set { x . print }", "f\no\nb\na\nr\n") }
    #[test] fn test_for_loop_no_intrinsic_with_str() { run_str("for x in 'hello' { x . print }", "h\ne\nl\nl\no\n") }
    #[test] fn test_for_loop_range_stop() { run_str("for x in range(5) { x . print }", "0\n1\n2\n3\n4\n"); }
    #[test] fn test_for_loop_range_start_stop() { run_str("for x in range(3, 6) { x . print }", "3\n4\n5\n"); }
    #[test] fn test_for_loop_range_start_stop_step_positive() { run_str("for x in range(1, 10, 3) { x . print }", "1\n4\n7\n"); }
    #[test] fn test_for_loop_range_start_stop_step_negative() { run_str("for x in range(11, 0, -4) { x . print }", "11\n7\n3\n"); }
    #[test] fn test_for_loop_range_start_stop_step_zero() { run_str("for x in range(1, 2, 0) { x . print }", "ValueError: 'step' argument cannot be zero\n    at: `for x in range(1, 2, 0) { x . print }` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_list_literal_empty() { run_str("[] . print", "[]\n"); }
    #[test] fn test_list_literal_len_1() { run_str("['hello'] . print", "['hello']\n"); }
    #[test] fn test_list_literal_len_2() { run_str("['hello', 'world'] . print", "['hello', 'world']\n"); }
    #[test] fn test_sqrt() { run_str("[0, 1, 4, 9, 25, 3, 6, 8, 13] . map(sqrt) . print", "[0, 1, 2, 3, 5, 1, 2, 2, 3]\n"); }
    #[test] fn test_very_large_sqrt() { run_str("[1 << 62, (1 << 62) + 1, (1 << 62) - 1] . map(sqrt) . print", "[2147483648, 2147483648, 2147483647]\n"); }
    #[test] fn test_gcd() { run_str("gcd(12, 8) . print", "4\n"); }
    #[test] fn test_gcd_iter() { run_str("[12, 18, 16] . gcd . print", "2\n"); }
    #[test] fn test_lcm() { run_str("lcm(9, 7) . print", "63\n"); }
    #[test] fn test_lcm_iter() { run_str("[12, 10, 18] . lcm . print", "180\n"); }
    #[test] fn test_if_then_else_1() { run_str("(if true then 'hello' else 'goodbye') . print", "hello\n"); }
    #[test] fn test_if_then_else_2() { run_str("(if false then 'hello' else 'goodbye') . print", "goodbye\n"); }
    #[test] fn test_if_then_else_3() { run_str("(if [] then 'hello' else 'goodbye') . print", "goodbye\n"); }
    #[test] fn test_if_then_else_4() { run_str("(if 3 then 'hello' else 'goodbye') . print", "hello\n"); }
    #[test] fn test_if_then_else_5() { run_str("(if false then (fn() -> 'hello' . print)() else 'nope') . print", "nope\n"); }
    #[test] fn test_pattern_in_let_works() { run_str("let x, y = [1, 2] ; [x, y] . print", "[1, 2]\n"); }
    #[test] fn test_pattern_in_let_too_long() { run_str("let x, y, z = [1, 2] ; [x, y] . print", "ValueError: Cannot unpack '[1, 2]' of type 'list' with length 2, expected exactly 3 elements\n    at: `let x, y, z = [1, 2] ; [x, y] . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_pattern_in_let_too_short() { run_str("let x, y = [1, 2, 3] ; [x, y] . print", "ValueError: Cannot unpack '[1, 2, 3]' of type 'list' with length 3, expected exactly 2 elements\n    at: `let x, y = [1, 2, 3] ; [x, y] . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_pattern_in_let_with_var_at_end() { run_str("let x, *y = [1, 2, 3, 4] ; [x, y] . print", "[1, [2, 3, 4]]\n"); }
    #[test] fn test_pattern_in_let_with_var_at_start() { run_str("let *x, y = [1, 2, 3, 4] ; [x, y] . print", "[[1, 2, 3], 4]\n"); }
    #[test] fn test_pattern_in_let_with_var_at_middle() { run_str("let x, *y, z = [1, 2, 3, 4] ; [x, y, z] . print", "[1, [2, 3], 4]\n"); }
    #[test] fn test_pattern_in_let_with_var_zero_len_at_end() { run_str("let x, *y = [1] ; [x, y] . print", "[1, []]\n"); }
    #[test] fn test_pattern_in_let_with_var_zero_len_at_start() { run_str("let *x, y = [1] ; [x, y] . print", "[[], 1]\n"); }
    #[test] fn test_pattern_in_let_with_var_zero_len_at_middle() { run_str("let x, *y, z = [1, 2] ; [x, y, z] . print", "[1, [], 2]\n"); }
    #[test] fn test_pattern_in_let_with_var_one_len_at_end() { run_str("let x, *y = [1, 2] ; [x, y] . print", "[1, [2]]\n"); }
    #[test] fn test_pattern_in_let_with_var_one_len_at_start() { run_str("let *x, y = [1, 2] ; [x, y] . print", "[[1], 2]\n"); }
    #[test] fn test_pattern_in_let_with_var_one_len_at_middle() { run_str("let x, *y, z = [1, 2, 3] ; [x, y, z] . print", "[1, [2], 3]\n"); }
    #[test] fn test_pattern_in_let_with_only_empty() { run_str("let _ = 'hello' . print", "hello\n"); }
    #[test] fn test_pattern_in_let_empty_x3() { run_str("let _, _, _ = [1, 2, 3]", ""); }
    #[test] fn test_pattern_in_let_empty_x3_too_long() { run_str("let _, _, _ = [1, 2, 3, 4]", "ValueError: Cannot unpack '[1, 2, 3, 4]' of type 'list' with length 4, expected exactly 3 elements\n    at: `let _, _, _ = [1, 2, 3, 4]` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_pattern_in_let_empty_x3_too_short() { run_str("let _, _, _ = [1, 2]", "ValueError: Cannot unpack '[1, 2]' of type 'list' with length 2, expected exactly 3 elements\n    at: `let _, _, _ = [1, 2]` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_pattern_in_let_empty_x2_at_end() { run_str("let _, _, x = [1, 2, 3] ; x . print", "3\n"); }
    #[test] fn test_pattern_in_let_empty_x2_at_middle() { run_str("let _, x, _ = [1, 2, 3] ; x . print", "2\n"); }
    #[test] fn test_pattern_in_let_empty_x2_at_start() { run_str("let x, _, _ = [1, 2, 3] ; x . print", "1\n"); }
    #[test] fn test_pattern_in_let_empty_at_middle() { run_str("let x, _, y = [1, 2, 3] ; [x, y] . print", "[1, 3]\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_at_end() { run_str("let x, *_ = [1, 2, 3, 4] ; x . print", "1\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_at_middle() { run_str("let x, *_, y = [1, 2, 3, 4] ; [x, y] . print", "[1, 4]\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_at_start() { run_str("let *_, x = [1, 2, 3, 4] ; x . print", "4\n"); }
    #[test] fn test_pattern_in_let_with_varargs_var_at_end() { run_str("let _, *x = [1, 2, 3, 4] ; x . print", "[2, 3, 4]\n"); }
    #[test] fn test_pattern_in_let_with_varargs_var_at_middle() { run_str("let _, *x, _ = [1, 2, 3, 4] ; x . print", "[2, 3]\n"); }
    #[test] fn test_pattern_in_let_with_varargs_var_at_start() { run_str("let *x, _ = [1, 2, 3, 4] ; x . print", "[1, 2, 3]\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_to_empty() { run_str("let *_ = []", ""); }
    #[test] fn test_pattern_in_let_with_varargs_empty_to_var_at_end_too_short() { run_str("let *_, x = []", "ValueError: Cannot unpack '[]' of type 'list' with length 0, expected at least 1 elements\n    at: `let *_, x = []` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_to_var_at_start_too_short() { run_str("let x, *_ = []", "ValueError: Cannot unpack '[]' of type 'list' with length 0, expected at least 1 elements\n    at: `let x, *_ = []` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_to_var_at_end() { run_str("let *_, x = [1] ; x . print", "1\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_to_var_at_start() { run_str("let x, *_ = [1] ; x . print", "1\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_to_var_at_middle() { run_str("let x, *_, y = [1, 2] ; [x, y] . print", "[1, 2]\n"); }
    #[test] fn test_pattern_in_let_with_nested_pattern() { run_str("let x, (y, _) = [[1, 2], [3, 4]] ; [x, y] . print", "[[1, 2], 3]\n"); }
    #[test] fn test_pattern_in_let_with_parens_on_one_empty() { run_str("let (_) = [[nil]]", ""); }
    #[test] fn test_pattern_in_let_with_parens_on_one_empty_one_var() { run_str("let (_, x) = [[1, 2]] ; x . print", "2\n"); }
    #[test] fn test_pattern_in_let_with_complex_patterns_1() { run_str("let *_, (_, x, _), _ = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ; x . print", "5\n"); }
    #[test] fn test_pattern_in_let_with_complex_patterns_2() { run_str("let _, (_, (_, (_, (x, *_)))) = [1, [2, [3, [4, [5, [6, [7, [8, [9, nil]]]]]]]]] ; x . print", "5\n"); }
    #[test] fn test_pattern_in_let_with_complex_patterns_3() { run_str("let ((*x, _), (_, (*y, _), _), *_) = [[[1, 2, 3], [[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1], [2], [3]]]] ; [x, y] . print", "[[1, 2], [2, 3]]\n"); }
    #[test] fn test_pattern_in_function() { run_str("fn f((a, b)) -> [b, a] . print ; f([1, 2])", "[2, 1]\n"); }
    #[test] fn test_multiple_patterns_in_function() { run_str("fn f((a, b), (c, d)) -> [a, b, c, d] . print ; f([1, 2], [3, 4])", "[1, 2, 3, 4]\n"); }
    #[test] fn test_pattern_in_function_before_args() { run_str("fn f((a, b, c), d, e) -> [a, b, c, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_in_function_between_args() { run_str("fn f(a, (b, c, d), e) -> [a, b, c, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_in_function_after_args() { run_str("fn f(a, b, (c, d, e)) -> [a, b, c, d, e] . print ; f(1, 2, [3, 4, 5])", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_empty_in_function_before_args() { run_str("fn f((_, b, _), d, e) -> [1, b, 3, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_empty_in_function_between_args() { run_str("fn f(a, (_, _, d), e) -> [a, 2, 3, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_empty_in_function_after_args() { run_str("fn f(a, b, (c, _, _)) -> [a, b, c, 4, 5] . print ; f(1, 2, [3, 4, 5])", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_var_in_function_before_args() { run_str("fn f((a, *_), d, e) -> [a, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 4, 5]\n"); }
    #[test] fn test_pattern_with_var_in_function_between_args() { run_str("fn f(a, (*_, d), e) -> [a, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 4, 5]\n"); }
    #[test] fn test_pattern_with_var_in_function_after_args() { run_str("fn f(a, b, (*c, _, _)) -> [a, b, c] . print ; f(1, 2, [3, 4, 5])", "[1, 2, [3]]\n"); }
    #[test] fn test_index_in_strings() { run_str("'hello'[1] . print", "e\n"); }
    #[test] fn test_slice_in_strings_start() { run_str("'hello'[1:] . print", "ello\n"); }
    #[test] fn test_slice_in_strings_stop() { run_str("'hello'[:3] . print", "hel\n"); }
    #[test] fn test_slice_in_strings_start_stop() { run_str("'hello'[1:3] . print", "el\n"); }
    #[test] fn test_pattern_in_for_with_enumerate() { run_str("for i, x in 'hello' . enumerate { [i, x] . print }", "[0, 'h']\n[1, 'e']\n[2, 'l']\n[3, 'l']\n[4, 'o']\n")}
    #[test] fn test_pattern_in_for_with_empty() { run_str("for _ in range(5) { 'hello' . print }", "hello\nhello\nhello\nhello\nhello\n"); }
    #[test] fn test_pattern_in_for_with_strings() { run_str("for a, *_, b in ['hello', 'world'] { print(a + b) }", "ho\nwd\n")}
    #[test] fn test_construct_vector() { run_str("vector(1, 2, 3) . print", "(1, 2, 3)\n"); }
    #[test] fn test_add_vectors() { run_str("vector(1, 2, 3) + vector(6, 3, 2) . print", "(7, 5, 5)\n"); }
    #[test] fn test_add_vector_and_constant() { run_str("vector(1, 2, 3) + 3 . print", "(4, 5, 6)\n"); }
    #[test] fn test_empty_str() { run_str("str() . print", "\n"); }
    #[test] fn test_empty_list() { run_str("list() . print", "[]\n"); }
    #[test] fn test_empty_set() { run_str("set() . print", "{}\n"); }
    #[test] fn test_empty_dict() { run_str("dict() . print", "{}\n"); }
    #[test] fn test_empty_heap() { run_str("heap() . print", "[]\n"); }
    #[test] fn test_empty_vector() { run_str("vector() . print", "()\n"); }
    #[test] fn test_str_in_str_yes() { run_str("'hello' in 'hey now, hello world' . print", "true\n"); }
    #[test] fn test_str_in_str_no() { run_str("'hello' in 'hey now, \\'ello world' . print", "false\n"); }
    #[test] fn test_int_in_list_yes() { run_str("13 in [10, 11, 12, 13, 14, 15] . print", "true\n"); }
    #[test] fn test_int_in_list_no() { run_str("3 in [10, 11, 12, 13, 14, 15] . print", "false\n"); }
    #[test] fn test_int_in_range_yes() { run_str("13 in range(10, 15) . print", "true\n"); }
    #[test] fn test_int_in_range_no() { run_str("3 in range(10, 15) . print", "false\n"); }
    #[test] fn test_dict_get_and_set() { run_str("let d = dict() ; d['hi'] = 'yes' ; d['hi'] . print", "yes\n"); }
    #[test] fn test_dict_get_when_not_present() { run_str("let d = dict() ; d['hello']", "ValueError: Key 'hello' of type 'str' not found in dictionary\n    at: `let d = dict() ; d['hello']` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_dict_get_when_not_present_with_default() { run_str("let d = dict() . default('haha') ; d['hello'] . print", "haha\n"); }
    #[test] fn test_flat_map_identity() { run_str("['hi', 'bob'] . flat_map(fn(i) -> i) . print", "['h', 'i', 'b', 'o', 'b']\n"); }
    #[test] fn test_flat_map_with_func() { run_str("['hello', 'bob'] . flat_map(fn(i) -> i[2:]) . print", "['l', 'l', 'o', 'b']\n"); }
    #[test] fn test_concat() { run_str("[[], [1], [2, 3], [4, 5, 6], [7, 8, 9, 0]] . concat . print", "[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]\n"); }
    #[test] fn test_zip() { run_str("zip([1, 2, 3, 4, 5], 'hello') . print", "[(1, 'h'), (2, 'e'), (3, 'l'), (4, 'l'), (5, 'o')]\n"); }
    #[test] fn test_zip_with_empty() { run_str("zip('hello', []) . print", "[]\n"); }
    #[test] fn test_zip_with_longer_last() { run_str("zip('hi', 'hello', 'hello the world!') . print", "[('h', 'h', 'h'), ('i', 'e', 'e')]\n"); }
    #[test] fn test_zip_with_longer_first() { run_str("zip('hello the world!', 'hello', 'hi') . print", "[('h', 'h', 'h'), ('e', 'e', 'i')]\n"); }
    #[test] fn test_zip_of_list() { run_str("[[1, 2, 3], [4, 5, 6], [7, 8, 9]] . zip . print", "[(1, 4, 7), (2, 5, 8), (3, 6, 9)]\n"); }
    #[test] fn test_dict_keys() { run_str("[[1, 'a'], [2, 'b'], [3, 'c']] . dict . keys . print", "{1, 2, 3}\n"); }
    #[test] fn test_dict_values() { run_str("[[1, 'a'], [2, 'b'], [3, 'c']] . dict . values . print", "['a', 'b', 'c']\n"); }
    #[test] fn test_empty_literal_is_dict() { run_str("let _ = {} is dict . print", "true\n"); }
    #[test] fn test_dict_literal_singleton() { run_str("let _ = {'hello': 'world'} . print", "{'hello': 'world'}\n"); }
    #[test] fn test_set_literal_singleton() { run_str("let _ = {'hello'} . print", "{'hello'}\n"); }
    #[test] fn test_dict_literal_multiple() { run_str("let _ = {1: 'a', 2: 'b', 3: 'c'} . print", "{1: 'a', 2: 'b', 3: 'c'}\n"); }
    #[test] fn test_set_literal_multiple() { run_str("let _ = {1, 2, 3, 4} . print", "{1, 2, 3, 4}\n"); }
    #[test] fn test_permutations_empty() { run_str("[] . permutations(3) . print", "[]\n"); }
    #[test] fn test_permutations_n_larger_than_size() { run_str("[1, 2, 3] . permutations(5) . print", "[]\n"); }
    #[test] fn test_permutations() { run_str("[1, 2, 3] . permutations(2) . print", "[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]\n"); }
    #[test] fn test_combinations_empty() { run_str("[] . combinations(3) . print", "[]\n"); }
    #[test] fn test_combinations_n_larger_than_size() { run_str("[1, 2, 3] . combinations(5) . print", "[]\n"); }
    #[test] fn test_combinations() { run_str("[1, 2, 3] . combinations(2) . print", "[(1, 2), (1, 3), (2, 3)]\n"); }
    #[test] fn test_find_value_empty() { run_str("[] . find(1) . print", "nil\n"); }
    #[test] fn test_find_func_empty() { run_str("[] . find(==3) . print", "nil\n"); }
    #[test] fn test_find_value_not_found() { run_str("[1, 3, 5, 7] . find(6) . print", "nil\n"); }
    #[test] fn test_find_func_not_found() { run_str("[1, 3, 5, 7] . find(fn(i) -> i % 2 == 0) . print", "nil\n"); }
    #[test] fn test_find_value_found() { run_str("[1, 3, 5, 7] . find(5) . print", "5\n"); }
    #[test] fn test_find_func_found() { run_str("[1, 3, 5, 7] . find(>3) . print", "5\n"); }
    #[test] fn test_find_value_found_multiple() { run_str("[1, 3, 5, 5, 7, 5] . find(5) . print", "5\n"); }
    #[test] fn test_find_func_found_multiple() { run_str("[1, 3, 5, 5, 7, 5] . find(>3) . print", "5\n"); }
    #[test] fn test_rfind_value_empty() { run_str("[] . rfind(1) . print", "nil\n"); }
    #[test] fn test_rfind_func_empty() { run_str("[] . rfind(==3) . print", "nil\n"); }
    #[test] fn test_rfind_value_not_found() { run_str("[1, 3, 5, 7] . rfind(6) . print", "nil\n"); }
    #[test] fn test_rfind_func_not_found() { run_str("[1, 3, 5, 7] . rfind(fn(i) -> i % 2 == 0) . print", "nil\n"); }
    #[test] fn test_rfind_value_found() { run_str("[1, 3, 5, 7] . rfind(5) . print", "5\n"); }
    #[test] fn test_rfind_func_found() { run_str("[1, 3, 5, 7] . rfind(>3) . print", "7\n"); }
    #[test] fn test_rfind_value_found_multiple() { run_str("[1, 3, 5, 5, 7, 5, 3, 1] . rfind(5) . print", "5\n"); }
    #[test] fn test_rfind_func_found_multiple() { run_str("[1, 3, 5, 5, 7, 5, 3, 1] . rfind(>3) . print", "5\n"); }
    #[test] fn test_index_of_value_empty() { run_str("[] . index_of(1) . print", "-1\n"); }
    #[test] fn test_index_of_func_empty() { run_str("[] . index_of(==3) . print", "-1\n"); }
    #[test] fn test_index_of_value_not_found() { run_str("[1, 3, 5, 7] . index_of(6) . print", "-1\n"); }
    #[test] fn test_index_of_func_not_found() { run_str("[1, 3, 5, 7] . index_of(fn(i) -> i % 2 == 0) . print", "-1\n"); }
    #[test] fn test_index_of_value_found() { run_str("[1, 3, 5, 7] . index_of(5) . print", "2\n"); }
    #[test] fn test_index_of_func_found() { run_str("[1, 3, 5, 7] . index_of(>3) . print", "2\n"); }
    #[test] fn test_index_of_value_found_multiple() { run_str("[1, 3, 5, 5, 7, 5] . index_of(5) . print", "2\n"); }
    #[test] fn test_index_of_func_found_multiple() { run_str("[1, 3, 5, 5, 7, 5] . index_of(>3) . print", "2\n"); }
    #[test] fn test_rindex_of_value_empty() { run_str("[] . rindex_of(1) . print", "-1\n"); }
    #[test] fn test_rindex_of_func_empty() { run_str("[] . rindex_of(==3) . print", "-1\n"); }
    #[test] fn test_rindex_of_value_not_found() { run_str("[1, 3, 5, 7] . rindex_of(6) . print", "-1\n"); }
    #[test] fn test_rindex_of_func_not_found() { run_str("[1, 3, 5, 7] . rindex_of(fn(i) -> i % 2 == 0) . print", "-1\n"); }
    #[test] fn test_rindex_of_value_found() { run_str("[1, 3, 5, 7] . rindex_of(5) . print", "2\n"); }
    #[test] fn test_rindex_of_func_found() { run_str("[1, 3, 5, 7] . rindex_of(>3) . print", "3\n"); }
    #[test] fn test_rindex_of_value_found_multiple() { run_str("[1, 3, 5, 5, 7, 5, 3, 1] . rindex_of(5) . print", "5\n"); }
    #[test] fn test_rindex_of_func_found_multiple() { run_str("[1, 3, 5, 5, 7, 5, 3, 1] . rindex_of(>3) . print", "5\n"); }
    #[test] fn test_not_in_yes() { run_str("3 not in [1, 2, 3] . print", "false\n"); }
    #[test] fn test_not_in_no() { run_str("3 not in [1, 5, 8] . print", "true\n"); }
    #[test] fn test_min_by_key() { run_str("[[1, 5], [2, 3], [6, 4]] . min_by(fn(i) -> i[1]) . print", "[2, 3]\n"); }
    #[test] fn test_min_by_cmp() { run_str("[[1, 5], [2, 3], [6, 4]] . min_by(fn(a, b) -> a[1] - b[1]) . print", "[2, 3]\n"); }
    #[test] fn test_min_by_wrong_fn() { run_str("[[1, 5], [2, 3], [6, 4]] . min_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n    at: `[[1, 5], [2, 3], [6, 4]] . min_by(fn() -> 1) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_max_by_key() { run_str("[[1, 5], [2, 3], [6, 4]] . max_by(fn(i) -> i[1]) . print", "[1, 5]\n"); }
    #[test] fn test_max_by_cmp() { run_str("[[1, 5], [2, 3], [6, 4]] . max_by(fn(a, b) -> a[1] - b[1]) . print", "[1, 5]\n"); }
    #[test] fn test_max_by_wrong_fn() { run_str("[[1, 5], [2, 3], [6, 4]] . max_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n    at: `[[1, 5], [2, 3], [6, 4]] . max_by(fn() -> 1) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_sort_by_key() { run_str("[[1, 5], [2, 3], [6, 4]] . sort_by(fn(i) -> i[1]) . print", "[[2, 3], [6, 4], [1, 5]]\n"); }
    #[test] fn test_sort_by_cmp() { run_str("[[1, 5], [2, 3], [6, 4]] . sort_by(fn(a, b) -> a[1] - b[1]) . print", "[[2, 3], [6, 4], [1, 5]]\n"); }
    #[test] fn test_sort_by_wrong_fn() { run_str("[[1, 5], [2, 3], [6, 4]] . sort_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n    at: `[[1, 5], [2, 3], [6, 4]] . sort_by(fn() -> 1) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_ord() { run_str("'a' . ord . print", "97\n"); }
    #[test] fn test_char() { run_str("97 . char . repr . print", "'a'\n"); }
    #[test] fn test_eval_nil() { run_str("'nil' . eval . print", "nil\n"); }
    #[test] fn test_eval_bool() { run_str("'true' . eval . print", "true\n"); }
    #[test] fn test_eval_int_expression() { run_str("'3 + 4' . eval . print", "7\n"); }
    #[test] fn test_eval_zero_equals_zero() { run_str("'0==0' . eval . print", "true\n"); }
    #[test] fn test_operator_in_expr() { run_str("(1 < 2) . print", "true\n"); }
    #[test] fn test_operator_partial_right() { run_str("((<2)(1)) . print", "true\n"); }
    #[test] fn test_operator_partial_left() { run_str("((1<)(2)) . print", "true\n"); }
    #[test] fn test_operator_partial_twice() { run_str("((<)(1)(2)) . print", "true\n"); }
    #[test] fn test_operator_as_prefix() { run_str("((<)(1, 2)) . print", "true\n"); }
    #[test] fn test_operator_partial_right_with_composition() { run_str("(1 . (<2)) . print", "true\n"); }
    #[test] fn test_operator_partial_left_with_composition() { run_str("(2 . (1<)) . print", "true\n"); }
    #[test] fn test_int_to_hex() { run_str("1234 . hex . print", "4d2\n"); }
    #[test] fn test_int_to_bin() { run_str("1234 . bin . print", "10011010010\n"); }
    #[test] fn test_single_element_vector() { run_str("(1,) . print", "(1)\n"); }
    #[test] fn test_multi_element_vector() { run_str("(1,2,3) . print", "(1, 2, 3)\n"); }
    #[test] fn test_multi_element_vector_trailing_comma() { run_str("(1,2,3,) . print", "(1, 2, 3)\n"); }
    #[test] fn test_while_false_if_false() { run_str("while false { if false { } }", ""); }
    #[test] fn test_binary_max_yes() { run_str("let a = 3 ; a max= 6; a . print", "6\n"); }
    #[test] fn test_binary_max_no() { run_str("let a = 3 ; a max= 2; a . print", "3\n"); }
    #[test] fn test_binary_min_yes() { run_str("let a = 3 ; a min= 1; a . print", "1\n"); }
    #[test] fn test_binary_min_no() { run_str("let a = 3 ; a min= 5; a . print", "3\n"); }
    #[test] fn test_all_yes_all() { run_str("[1, 3, 4, 5] . all(>0) . print", "true\n"); }
    #[test] fn test_all_yes_some() { run_str("[1, 3, 4, 5] . all(>3) . print", "false\n"); }
    #[test] fn test_all_yes_none() { run_str("[1, 3, 4, 5] . all(<0) . print", "false\n"); }
    #[test] fn test_any_yes_all() { run_str("[1, 3, 4, 5] . any(>0) . print", "true\n"); }
    #[test] fn test_any_yes_some() { run_str("[1, 3, 4, 5] . any(>3) . print", "true\n"); }
    #[test] fn test_any_yes_none() { run_str("[1, 3, 4, 5] . any(<0) . print", "false\n"); }
    #[test] fn test_format_with_percent_no_args() { run_str("'100 %%' % vector() . print", "100 %\n"); }
    #[test] fn test_format_with_one_int_arg() { run_str("'an int: %d' % (123,) . print", "an int: 123\n"); }
    #[test] fn test_format_with_one_neg_int_arg() { run_str("'an int: %d' % (-123,) . print", "an int: -123\n"); }
    #[test] fn test_format_with_one_zero_pad_int_arg() { run_str("'an int: %05d' % (123,) . print", "an int: 00123\n"); }
    #[test] fn test_format_with_one_zero_pad_neg_int_arg() { run_str("'an int: %05d' % (-123,) . print", "an int: -0123\n"); }
    #[test] fn test_format_with_one_space_pad_int_arg() { run_str("'an int: %5d' % (123,) . print", "an int:   123\n"); }
    #[test] fn test_format_with_one_space_pad_neg_int_arg() { run_str("'an int: %5d' % (-123,) . print", "an int:  -123\n"); }
    #[test] fn test_format_with_one_hex_arg() { run_str("'an int: %x' % (123,) . print", "an int: 7b\n"); }
    #[test] fn test_format_with_one_zero_pad_hex_arg() { run_str("'an int: %04x' % (123,) . print", "an int: 007b\n"); }
    #[test] fn test_format_with_one_space_pad_hex_arg() { run_str("'an int: %4x' % (123,) . print", "an int:   7b\n"); }
    #[test] fn test_format_with_one_bin_arg() { run_str("'an int: %b' % (123,) . print", "an int: 1111011\n"); }
    #[test] fn test_format_with_one_zero_pad_bin_arg() { run_str("'an int: %012b' % (123,) . print", "an int: 000001111011\n"); }
    #[test] fn test_format_with_one_space_pad_bin_arg() { run_str("'an int: %12b' % (123,) . print", "an int:      1111011\n"); }
    #[test] fn test_format_with_many_args() { run_str("'%d %s %x %b ALL THE THINGS %%!' % (10, 'fifteen', 0xff, 0b10101) . print", "10 fifteen ff 10101 ALL THE THINGS %!\n"); }
    #[test] fn test_format_with_solo_arg_nil() { run_str("'hello %s' % nil . print", "hello nil\n"); }
    #[test] fn test_format_with_solo_arg_int() { run_str("'hello %s' % 123 . print", "hello 123\n"); }
    #[test] fn test_format_with_solo_arg_str() { run_str("'hello %s' % 'world' . print", "hello world\n"); }
    #[test] fn test_format_nested_0() { run_str("'%s w%sld %s' % ('hello', 'or', '!') . print", "hello world !\n"); }
    #[test] fn test_format_nested_1() { run_str("'%%%s%%s%s %%s' % ('s w', 'ld') % ('hello', 'or', '!') . print", "hello world !\n"); }
    #[test] fn test_format_nested_2() { run_str("'%ss%%%%s%s%s%ss' % ('%'*3, '%s', ' ', '%'*2) % ('s w', 'ld') % ('hello', 'or', '!') . print", "hello world !\n"); }
    #[test] fn test_format_too_many_args() { run_str("'%d %d %d' % (1, 2)", "ValueError: Not enough arguments for format string\n    at: `'%d %d %d' % (1, 2)` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_format_too_few_args() { run_str("'%d %d %d' % (1, 2, 3, 4)", "ValueError: Not all arguments consumed in format string, next: '4' of type 'int'\n    at: `'%d %d %d' % (1, 2, 3, 4)` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_format_incorrect_character() { run_str("'%g' % (1,)", "ValueError: Invalid format character 'g' in format string\n    at: `'%g' % (1,)` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_format_incorrect_width() { run_str("'%00' % (1,)", "ValueError: Invalid format character '0' in format string\n    at: `'%00' % (1,)` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_list_pop_empty() { run_str("let x = [] , y = x . pop ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n    at: `let x = [] , y = x . pop ; (x, y) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_list_pop() { run_str("let x = [1, 2, 3] , y = x . pop ; (x, y) . print", "([1, 2], 3)\n"); }
    #[test] fn test_set_pop_empty() { run_str("let x = set() , y = x . pop ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n    at: `let x = set() , y = x . pop ; (x, y) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_set_pop() { run_str("let x = {1, 2, 3} , y = x . pop ; (x, y) . print", "({1, 2}, 3)\n"); }
    #[test] fn test_dict_pop_empty() { run_str("let x = dict() , y = x . pop ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n    at: `let x = dict() , y = x . pop ; (x, y) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_dict_pop() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'} , y = x . pop ; (x, y) . print", "({1: 'a', 2: 'b'}, (3, 'c'))\n"); }
    #[test] fn test_list_pop_front_empty() { run_str("let x = [], y = x . pop_front ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n    at: `let x = [], y = x . pop_front ; (x, y) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_list_pop_front() { run_str("let x = [1, 2, 3], y = x . pop_front ; (x, y) . print", "([2, 3], 1)\n"); }
    #[test] fn test_list_push() { run_str("let x = [1, 2, 3] ; x . push(4) ; x . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_set_push() { run_str("let x = {1, 2, 3} ; x . push(4) ; x . print", "{1, 2, 3, 4}\n"); }
    #[test] fn test_list_push_front() { run_str("let x = [1, 2, 3] ; x . push_front(4) ; x . print", "[4, 1, 2, 3]\n"); }
    #[test] fn test_list_insert_front() { run_str("let x = [1, 2, 3] ; x . insert(0, 4) ; x . print", "[4, 1, 2, 3]\n"); }
    #[test] fn test_list_insert_middle() { run_str("let x = [1, 2, 3] ; x . insert(1, 4) ; x . print", "[1, 4, 2, 3]\n"); }
    #[test] fn test_list_insert_end() { run_str("let x = [1, 2, 3] ; x . insert(2, 4) ; x . print", "[1, 2, 4, 3]\n"); }
    #[test] fn test_list_insert_out_of_bounds() { run_str("let x = [1, 2, 3] ; x . insert(4, 4) ; x . print", "Index '4' is out of bounds for list of length [0, 3)\n    at: `let x = [1, 2, 3] ; x . insert(4, 4) ; x . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_dict_insert() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'} ; x . insert(4, 'd') ; x . print", "{1: 'a', 2: 'b', 3: 'c', 4: 'd'}\n"); }
    #[test] fn test_list_remove_front() { run_str("let x = [1, 2, 3] , y = x . remove(0) ; (x, y) . print", "([2, 3], 1)\n"); }
    #[test] fn test_list_remove_middle() { run_str("let x = [1, 2, 3] , y = x . remove(1) ; (x, y) . print", "([1, 3], 2)\n"); }
    #[test] fn test_list_remove_end() { run_str("let x = [1, 2, 3] , y = x . remove(2) ; (x, y) . print", "([1, 2], 3)\n"); }
    #[test] fn test_set_remove_yes() { run_str("let x = {1, 2, 3}, y = x . remove(2) ; (x, y) . print", "({1, 3}, true)\n"); }
    #[test] fn test_set_remove_no() { run_str("let x = {1, 2, 3}, y = x . remove(5) ; (x, y) . print", "({1, 2, 3}, false)\n"); }
    #[test] fn test_dict_remove_yes() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . remove(2) ; (x, y) . print", "({1: 'a', 3: 'c'}, true)\n"); }
    #[test] fn test_dict_remove_no() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . remove(5) ; (x, y) . print", "({1: 'a', 2: 'b', 3: 'c'}, false)\n"); }
    #[test] fn test_list_clear() { run_str("let x = [1, 2, 3] ; x . clear ; x . print", "[]\n"); }
    #[test] fn test_set_clear() { run_str("let x = {1, 2, 3} ; x . clear ; x . print", "{}\n"); }
    #[test] fn test_dict_clear() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'} ; x . clear ; x . print", "{}\n"); }
    #[test] fn test_dict_from_enumerate() { run_str("'hey' . enumerate . dict . print", "{0: 'h', 1: 'e', 2: 'y'}\n"); }
    #[test] fn test_list_peek() { run_str("let x = [1, 2, 3], y = x . peek ; (x, y) . print", "([1, 2, 3], 1)\n"); }
    #[test] fn test_set_peek() { run_str("let x = {1, 2, 3}, y = x . peek ; (x, y) . print", "({1, 2, 3}, 1)\n"); }
    #[test] fn test_dict_peek() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . peek ; (x, y) . print", "({1: 'a', 2: 'b', 3: 'c'}, (1, 'a'))\n"); }
    #[test] fn test_vector_set_index() { run_str("let x = (1, 2, 3) ; x[0] = 3 ; x . print", "(3, 2, 3)\n"); }
    #[test] fn test_annotation_named_func_with_name() { run_str("fn par(f) -> (fn(x) -> f('hello')) ; @par fn do(x) -> print(x) ; do('goodbye')", "hello\n"); }
    #[test] fn test_annotation_expression_func_with_name() { run_str("fn par(f) -> (fn(x) -> f('hello')) ; (@par fn(x) -> print(x))('goodbye')", "hello\n"); }
    #[test] fn test_annotation_named_func_with_expression() { run_str("fn par(a, f) -> (fn(x) -> f(a)) ; @par('hello') fn do(x) -> print(x) ; do('goodbye')", "hello\n"); }
    #[test] fn test_annotation_expression_func_with_expression() { run_str("fn par(a, f) -> (fn(x) -> f(a)) ; (@par('hello') fn(x) -> print(x))('goodbye')", "hello\n"); }
    #[test] fn test_annotation_iife() { run_str("fn iife(f) -> f() ; @iife fn do() -> print('hello')", "hello\n"); }
    #[test] fn test_dot_equals() { run_str("let x = 'hello' ; x .= sort ; x .= reduce(+) ; x . print", "ehllo\n"); }
    #[test] fn test_dot_equals_operator_function() { run_str("let x = 3 ; x .= (+4) ; x . print", "7\n"); }
    #[test] fn test_dot_equals_anonymous_function() { run_str("let x = 'hello' ; x .= fn(x) -> x[0] * len(x) ; x . print", "hhhhh\n"); }
    #[test] fn test_exit_in_expression() { run_str("'this will not print' + exit . print", ""); }
    #[test] fn test_exit_in_ternary() { run_str("print(if 3 > 2 then exit else 'hello')", ""); }
    #[test] fn test_iterables_is_iterable() { run_str("[[], '123', set(), dict()] . all(is iterable) . print", "true\n"); }
    #[test] fn test_non_iterables_is_iterable() { run_str("[true, false, nil, 123, fn() -> {}] . any(is iterable) . print", "false\n"); }
    #[test] fn test_any_is_any() { run_str("[[], '123', set(), dict(), 123, true, false, nil, fn() -> nil] . all(is any) . print", "true\n"); }
    #[test] fn test_function_is_function() { run_str("(fn() -> nil) is function . print", "true\n"); }
    #[test] fn test_non_function_is_function() { run_str("[nil, true, 123, '123', [], set()] . any(is function) . print", "false\n"); }
    #[test] fn test_operator_sub_as_unary() { run_str("(-)(3) . print", "-3\n"); }
    #[test] fn test_operator_sub_as_binary() { run_str("(-)(5, 2) . print", "3\n"); }
    #[test] fn test_operator_sub_as_partial_not_allowed() { run_str("(-3) . print", "-3\n"); }


    #[test] fn test_aoc_2022_01_01() { run("aoc_2022_01_01"); }
    #[test] fn test_append_large_lists() { run("append_large_lists"); }
    #[test] fn test_closure_instead_of_global_variable() { run("closure_instead_of_global_variable"); }
    #[test] fn test_closure_of_loop_variable() { run("closure_of_loop_variable"); }
    #[test] fn test_closure_of_partial_function() { run("closure_of_partial_function"); }
    #[test] fn test_closure_with_non_unique_values() { run("closure_with_non_unique_values"); }
    #[test] fn test_closure_without_stack_semantics() { run("closure_without_stack_semantics"); }
    #[test] fn test_closures_are_poor_mans_classes() { run("closures_are_poor_mans_classes"); }
    #[test] fn test_closures_inner_multiple_functions_read_stack() { run("closures_inner_multiple_functions_read_stack"); }
    #[test] fn test_closures_inner_multiple_functions_read_stack_and_heap() { run("closures_inner_multiple_functions_read_stack_and_heap"); }
    #[test] fn test_closures_inner_multiple_variables_read_heap() { run("closures_inner_multiple_variables_read_heap"); }
    #[test] fn test_closures_inner_multiple_variables_read_stack() { run("closures_inner_multiple_variables_read_stack"); }
    #[test] fn test_closures_inner_write_heap_read_heap() { run("closures_inner_write_heap_read_heap"); }
    #[test] fn test_closures_inner_write_stack_read_heap() { run("closures_inner_write_stack_read_heap"); }
    #[test] fn test_closures_inner_write_stack_read_return() { run("closures_inner_write_stack_read_return"); }
    #[test] fn test_closures_inner_write_stack_read_stack() { run("closures_inner_write_stack_read_stack"); }
    #[test] fn test_closures_inner_read_heap() { run("closures_inner_read_heap"); }
    #[test] fn test_closures_inner_read_stack() { run("closures_inner_read_stack"); }
    #[test] fn test_closures_nested_inner_read_heap() { run("closures_nested_inner_read_heap"); }
    #[test] fn test_closures_nested_inner_read_heap_x2() { run("closures_nested_inner_read_heap_x2"); }
    #[test] fn test_closures_nested_inner_read_stack() { run("closures_nested_inner_read_stack"); }
    #[test] fn test_fibonacci() { run("fibonacci"); }
    #[test] fn test_for_loop_modify_loop_variable() { run("for_loop_modify_loop_variable"); }
    #[test] fn test_for_loop_range_map() { run("for_loop_range_map"); }
    #[test] fn test_for_loop_with_multiple_references() { run("for_loop_with_multiple_references"); }
    #[test] fn test_function_capture_from_inner_scope() { run("function_capture_from_inner_scope"); }
    #[test] fn test_function_capture_from_outer_scope() { run("function_capture_from_outer_scope"); }
    #[test] fn test_late_bound_global() { run("late_bound_global"); }
    #[test] fn test_late_bound_global_invalid() { run("late_bound_global_invalid"); }
    #[test] fn test_map_loop_with_multiple_references() { run("map_loop_with_multiple_references"); }
    #[test] fn test_memoize() { run("memoize"); }
    #[test] fn test_memoize_recursive() { run("memoize_recursive"); }
    #[test] fn test_memoize_recursive_as_annotation() { run("memoize_recursive_as_annotation"); }
    #[test] fn test_range_used_twice() { run("range_used_twice"); }
    #[test] fn test_runtime_error_with_trace() { run("runtime_error_with_trace"); }
    #[test] fn test_upvalue_never_captured() { run("upvalue_never_captured"); }


    fn run_str(code: &'static str, expected: &'static str) {
        let text: &String = &String::from(code);
        let source: &String = &String::from("<test>");
        let compile = compiler::compile(source, text);

        if compile.is_err() {
            assert_eq!(format!("Compile Error:\n\n{}", compile.err().unwrap().join("\n")).as_str(), expected);
            return
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut vm = VirtualMachine::new(compile.ok().unwrap(), &b""[..], &mut buf);

        let result: ExitType = vm.run_until_completion();
        assert!(vm.stack.is_empty() || result.is_early_exit());

        let mut output: String = String::from_utf8(buf).unwrap();

        if let ExitType::Error(err) = result {
            output.push_str(ErrorReporter::new(text, source).format_runtime_error(err).as_str());
        }

        assert_eq!(output.as_str(), expected);
    }

    fn run(path: &'static str) {
        let root: &PathBuf = &trace::test::get_test_resource_path("compiler", path);
        let text: &String = &trace::test::get_test_resource_src(&root);
        let source = &String::from(path);
        let compile= compiler::compile(source, text);

        if compile.is_err() {
            assert_eq!(format!("Compile Error:\n\n{}", compile.err().unwrap().join("\n")).as_str(), "Compiled");
            return
        }

        let compile = compile.unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let mut vm = VirtualMachine::new(compile, &b""[..], &mut buf);

        let result: ExitType = vm.run_until_completion();
        assert!(vm.stack.is_empty() || result.is_early_exit());

        let mut output: String = String::from_utf8(buf).unwrap();

        if let ExitType::Error(e) = result {
            output.push_str(reporting::format_runtime_error(&text.split("\n").collect(), source, e).as_str());
        }

        trace::test::compare_test_resource_content(&root, output.split("\n").map(|s| String::from(s)).collect::<Vec<String>>());
    }
}
