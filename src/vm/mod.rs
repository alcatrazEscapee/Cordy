use std::io::{BufRead, Write};
use std::rc::Rc;
use error::{DetailRuntimeError, RuntimeError};

use crate::{stdlib, trace};
use crate::compiler::CompileResult;
use crate::stdlib::StdBinding;
use crate::vm::opcode::Opcode;
use crate::vm::value::Value;
use crate::vm::value::{FunctionImpl, PartialFunctionImpl};

use Opcode::{*};
use RuntimeError::{*};

type AnyResult = Result<(), Box<RuntimeError>>;

pub mod value;
pub mod opcode;
pub mod error;
pub mod operator;


pub struct VirtualMachine<R, W> {
    ip: usize,
    code: Vec<Opcode>,
    stack: Vec<Value>,
    call_stack: Vec<CallFrame>,
    global_count: usize,

    strings: Vec<String>,
    globals: Vec<String>,
    constants: Vec<i64>,
    functions: Vec<Rc<FunctionImpl>>,
    line_numbers: Vec<u16>,

    read: R,
    write: W,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum FunctionType {
    Native, User
}


pub trait VirtualInterface {
    // Invoking Functions

    /// Invokes the action of an `OpFuncEval(nargs)` opcode.
    ///
    /// The stack must be setup as `[..., f, arg1, arg2, ... argN ]`, where `f` is the function to be invoked with arguments `arg1, arg2, ... argN`.
    /// The arguments and function will be popped and the return value will be left on the top of the stack.
    ///
    /// Returns a `Result` which may contain an error which occurred during function evaluation.
    fn invoke_func_eval(self: &mut Self, nargs: u8) -> Result<FunctionType, Box<RuntimeError>>;

    /// Breaks back into the VM main loop and starts executing until the current call frame is dropped.
    /// Used after native code invokes either of the above two functions, to start a runtime loop for user functions.
    fn run_after_invoke(self: &mut Self, function_type: FunctionType) -> AnyResult;

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
            call_stack: Vec::with_capacity(32),
            global_count: 0,
            strings: result.strings,
            globals: result.globals,
            constants: result.constants,
            functions: result.functions.into_iter().map(|f| Rc::new(f)).collect(),
            line_numbers: result.line_numbers,
            read,
            write,
        }
    }

    pub fn run_until_completion(self: &mut Self) -> Result<(), DetailRuntimeError> {
        if let Err(e) = self.run() {
            match *e {
                RuntimeExit => {},
                _ => {
                    return Err(error::detail_runtime_error(*e, self.ip - 1, &self.call_stack, &self.functions, &self.line_numbers))
                }
            }
        }
        trace::trace_interpreter_stack!("final {}", self.debug_stack());
        Ok(())
    }

    fn run_until_frame(self: &mut Self) -> AnyResult {
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
        self.call_stack.push(CallFrame {
            return_ip: 0,
            frame_pointer: 0
        });

        loop {
            let op = self.next_op();
            self.run_instruction(op)?;
        }
    }

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
            // todo: relative jumps? theoretically allows us more than u16.max instructions ~ 65k
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

            StoreArray => {
                trace::trace_interpreter!("store array array = {}, index = {}, value = {}", self.stack[self.stack.len() - 3].as_debug_str(), self.stack[self.stack.len() - 2].as_debug_str(), self.stack.last().unwrap().as_debug_str());
                let a3: Value = self.pop();
                let a2: Value = self.pop();
                let a1: Value = self.peek(0).clone(); // Leave this on the stack when done
                match (a1, a2) {
                    (Value::List(l), Value::Int(r)) => match stdlib::list_set_index(l, r, a3) {
                        Ok(_) => {}, // No push, as we left the previous value on top of the stack
                        Err(e) => return Err(e),
                    },
                    (l, r) => return TypeErrorBinaryOp(OpIndex, l, r).err()
                }
            },

            PushUpValue(index, _) => {
                let value: Value = match &self.stack[self.frame_pointer() - 1] {
                    Value::Closure(c) => c.environment[index as usize].clone(),
                    _ => panic!("Malformed bytecode"),
                };
                self.push(value);
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
                trace::trace_interpreter!("close local {} into {}", index, self.stack.last().unwrap().as_debug_str());
                let local: usize = self.frame_pointer() + index as usize;
                let value: Value = self.stack[local].clone();
                match self.stack.last_mut().unwrap() {
                    Value::Closure(c) => c.environment.push(value),
                    _ => panic!("Malformed bytecode"),
                }
            },
            CloseUpValue(index) => {
                trace::trace_interpreter!("close upvalue {} into {} from {}", index, self.stack.last().unwrap().as_debug_str(), &self.stack[self.frame_pointer() - 1].as_debug_str());
                let index: usize = index as usize;
                let value: Value = match &self.stack[self.frame_pointer() - 1] {
                    Value::Closure(c) => c.environment[index].clone(),
                    _ => panic!("Malformed bytecode"),
                };

                match self.stack.last_mut().unwrap() {
                    Value::Closure(c) => c.environment.push(value),
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
                self.push(Value::Bool(true));
            },
            False => {
                trace::trace_interpreter!("push false");
                self.push(Value::Bool(false));
            },
            Int(cid) => {
                let cid: usize = cid as usize;
                let value: i64 = self.constants[cid];
                trace::trace_interpreter!("push constant {} -> {}", cid, value);
                self.push(Value::Int(value));
            }
            Str(sid) => {
                let sid: usize = sid as usize;
                let str: String = self.strings[sid].clone();
                trace::trace_interpreter!("push {} -> '{}'", sid, str);
                self.push(Value::Str(Box::new(str)));
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
                trace::trace_interpreter!("push [n={}]", cid);
                // List values are present on the stack in-order
                // So we need to splice the last n values of the stack into it's own list
                let cid: usize = cid as usize;
                let length: usize = self.constants[cid] as usize;
                let start: usize = self.stack.len() - length;
                let end: usize = self.stack.len();
                trace::trace_interpreter_stack!("stack splice {}..{} into [...]", start, end);
                let list: Value = Value::iter_list(self.stack.splice(start..end, std::iter::empty()));
                self.push(list);
            },

            OpFuncEval(nargs) => {
                trace::trace_interpreter!("op function evaluate n = {}", nargs);
                match self.invoke_func_eval(nargs) {
                    Err(e) => return e.err(),
                    _ => {},
                }
            }

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
            OpLessThan => operator_unchecked!(operator::binary_less_than, a1, a2, "<"),
            OpGreaterThan => operator_unchecked!(operator::binary_greater_than, a1, a2, ">"),
            OpLessThanEqual => operator_unchecked!(operator::binary_less_than_or_equal, a1, a2, "<="),
            OpGreaterThanEqual => operator_unchecked!(operator::binary_greater_than_or_equal, a1, a2, ">="),
            OpEqual => operator_unchecked!(operator::binary_equals, a1, a2, "=="),
            OpNotEqual => operator_unchecked!(operator::binary_not_equals, a1, a2, "!="),
            OpBitwiseAnd => operator!(operator::binary_bitwise_and, a1, a2, "&"),
            OpBitwiseOr => operator!(operator::binary_bitwise_or, a1, a2, "|"),
            OpBitwiseXor => operator!(operator::binary_bitwise_xor, a1, a2, "^"),

            OpIndex => {
                trace::trace_interpreter!("op []");
                let a2: Value = self.pop();
                let a1: Value = self.pop();
                match (a1, a2) {
                    (Value::List(l), Value::Int(r)) => match stdlib::list_get_index(l, r) {
                        Ok(v) => self.push(v),
                        Err(e) => return Err(e),
                    },
                    (l, r) => return TypeErrorBinaryOp(OpIndex, l, r).err()
                }
            },
            OpIndexPeek => {
                trace::trace_interpreter!("op [] peek");
                let a2: Value = self.peek(0).clone();
                let a1: Value = self.peek(1).clone();
                match (a1, a2) {
                    (Value::List(l), Value::Int(r)) => match stdlib::list_get_index(l, r) {
                        Ok(v) => self.push(v),
                        Err(e) => return Err(e),
                    },
                    (l, r) => return TypeErrorBinaryOp(OpIndex, l, r).err()
                }
            },
            OpSlice => {
                trace::trace_interpreter!("op [:]");
                let a3: Value = self.pop();
                let a2: Value = self.pop();
                let a1: Value = self.pop();
                match stdlib::list_slice(a1, a2, a3, Value::Int(1)) {
                    Ok(v) => self.push(v),
                    Err(e) => return Err(e),
                }
            },
            OpSliceWithStep => {
                trace::trace_interpreter!("op [::]");
                let a4: Value = self.pop();
                let a3: Value = self.pop();
                let a2: Value = self.pop();
                let a1: Value = self.pop();
                match stdlib::list_slice(a1, a2, a3, a4) {
                    Ok(v) => self.push(v),
                    Err(e) => return Err(e),
                }
            },

            Exit => return RuntimeExit.err(),
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

    #[cfg(trace_interpreter_stack = "on")]
    fn debug_stack(self: &Self) -> String {
        format!(": [{}]", self.stack.iter().rev().map(|t| t.as_debug_str()).collect::<Vec<String>>().join(", "))
    }

    #[cfg(trace_interpreter = "on")]
    fn debug_call_stack(self: &Self) -> String {
        format!(": [{}]", self.call_stack.iter().rev().map(|t| format!("{{fp: {}, ret: {}}}", t.frame_pointer, t.return_ip)).collect::<Vec<String>>().join(", "))
    }
}


impl <R, W> VirtualInterface for VirtualMachine<R, W> where
    R : BufRead,
    W : Write
{
    // ===== Calling Functions External Interface ===== //

    fn invoke_func_eval(self: &mut Self, nargs: u8) -> Result<FunctionType, Box<RuntimeError>> {
        let f: &Value = self.peek(nargs as usize);
        match f {
            f @ (Value::Function(_) | Value::Closure(_)) => {
                trace::trace_interpreter!("invoke_func_eval -> {}, nargs = {}", f.as_debug_str(), nargs);

                let func = f.as_function();
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
                let func = partial.func.as_function();
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
                let binding: StdBinding = *b;
                let i: usize = self.stack.len() - 1 - nargs as usize;
                let args: Vec<Box<Value>> = match std::mem::replace(&mut self.stack[i], Value::Nil) {
                    Value::PartialNativeFunction(_, x) => *x,
                    _ => panic!("Stack corruption")
                };

                // Splice the args from the binding back into the stack, in the correct order
                // When evaluating a partial function the vm stack will contain [..., argN+1, ... argM]
                // The partial args will contain the vector [argN, argN-1, ... arg1]
                // After this, the vm stack should contain the args [..., arg1, arg2, ... argM]
                let j: usize = self.stack.len() - nargs as usize;
                let partial_args: u8 = args.len() as u8;
                self.stack.splice(j..j, args.into_iter().map(|t| *t).rev());

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

    fn run_after_invoke(self: &mut Self, function_type: FunctionType) -> AnyResult {
        match function_type {
            FunctionType::Native => Ok(()),
            FunctionType::User => self.run_until_frame()
        }
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
    use crate::{compiler, ErrorReporter, reporting, trace, VirtualMachine};
    use crate::vm::error::DetailRuntimeError;

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
    #[test] fn test_str_if_08() { run_str("if 1 < 0 { print('yes') } elif 1 { print('hi') } else { print('hehe')", "hi\n"); }
    #[test] fn test_str_if_09() { run_str("if 1 < 0 { print('yes') } elif 2 < 0 { print('hi') } else { print('hehe')", "hehe\n"); }
    #[test] fn test_str_if_10() { run_str("if 1 { print('yes') } elif true { print('hi') } else { print('hehe')", "yes\n"); }
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
    #[test] fn test_functions_13() { run_str("let x = 'hello' ; { fn foo() { x . print } foo()", "hello\n"); }
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
    #[test] fn test_closures_06() { run_str("{ let x = 'hello' ; { fn foo() { x . print } foo() }", "hello\n"); }
    #[test] fn test_closures_07() { run_str("{ let x = 'hello' ; { { fn foo() { x . print } foo() } }", "hello\n"); }
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
    #[test] fn test_builtin_sorted() { run_str("[6, 2, 3, 7, 2, 1] . sorted . print", "[1, 2, 2, 3, 6, 7]\n"); }
    #[test] fn test_builtin_reversed() { run_str("[8, 1, 2, 6, 3, 2, 3] . reversed . print", "[3, 2, 3, 6, 2, 1, 8]\n"); }
    #[test] fn test_bare_operator_eval() { run_str("(+)(1, 2) . print", "3\n"); }
    #[test] fn test_bare_operator_partial_eval() { run_str("(+)(1)(2) . print", "3\n"); }
    #[test] fn test_bare_operator_compose_and_eval() { run_str("2 . (+)(1) . print", "3\n"); }
    #[test] fn test_bare_operator_compose() { run_str("1 . (2 . (+)) . print", "3\n"); }
    #[test] fn test_reduce_list_1() { run_str("[1, 2, 3] . reduce (+) . print", "6\n"); }
    #[test] fn test_reduce_list_2() { run_str("[1, 2, 3] . reduce (-) . print", "Function '(-)' requires 2 parameters but 1 were present.\n    at: `[1, 2, 3] . reduce (-) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_str_to_list() { run_str("'funny beans' . list . print", "['f', 'u', 'n', 'n', 'y', ' ', 'b', 'e', 'a', 'n', 's']\n"); }
    #[test] fn test_str_to_set() { run_str("'funny beans' . set . print", "{'f', 'u', 'y', ' ', 'b', 'e', 'a', 'n', 's'}\n"); }
    #[test] fn test_str_to_set_to_sorted() { run_str("'funny' . set . sorted . print", "['f', 'n', 'u', 'y']\n"); }
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
    #[test] fn test_range_1() { run_str("range(3) . print", "[0, 1, 2]\n"); }
    #[test] fn test_range_2() { run_str("range(3, 7) . print", "[3, 4, 5, 6]\n"); }
    #[test] fn test_range_3() { run_str("range(1, 9, 3) . print", "[1, 4, 7]\n"); }
    #[test] fn test_range_4() { run_str("range(6, 3) . print", "[]\n"); }
    #[test] fn test_range_5() { run_str("range(10, 4, -2) . print", "[10, 8, 6]\n"); }
    #[test] fn test_range_6() { run_str("range(0, 20, -1) . print", "[]\n"); }
    #[test] fn test_range_7() { run_str("range(10, 0, 3) . print", "[]\n"); }
    #[test] fn test_range_8() { run_str("range(1, 1, 1) . print", "[]\n"); }
    #[test] fn test_range_9() { run_str("range(1, 1, 0) . print", "ValueError: 'step' argument cannot be zero\n    at: `range(1, 1, 0) . print` (line 1)\n    at: execution of script '<test>'\n"); }
    #[test] fn test_enumerate_1() { run_str("[] . enumerate . print", "[]\n"); }
    #[test] fn test_enumerate_2() { run_str("[1, 2, 3] . enumerate . print", "[[0, 1], [1, 2], [2, 3]]\n"); }
    #[test] fn test_enumerate_3() { run_str("'foobar' . enumerate . print", "[[0, 'f'], [1, 'o'], [2, 'o'], [3, 'b'], [4, 'a'], [5, 'r']]\n"); }
    #[test] fn test_for_loop_no_intrinsic_with_list() { run_str("for x in ['a', 'b', 'c'] { x . print }", "a\nb\nc\n") }
    #[test] fn test_for_loop_no_intrinsic_with_set() { run_str("for x in 'foobar' . set { x . print }", "f\no\nb\na\nr\n") }
    #[test] fn test_for_loop_no_intrinsic_with_str() { run_str("for x in 'hello' { x . print }", "h\ne\nl\nl\no\n") }
    #[test] fn test_for_loop_intrinsic_range_stop() { run_str("for x in range(5) { x . print }", "0\n1\n2\n3\n4\n"); }
    #[test] fn test_for_loop_intrinsic_range_start_stop() { run_str("for x in range(3, 6) { x . print }", "3\n4\n5\n"); }
    #[test] fn test_list_literal_empty() { run_str("[] . print", "[]\n"); }
    #[test] fn test_list_literal_len_1() { run_str("['hello'] . print", "['hello']\n"); }
    #[test] fn test_list_literal_len_2() { run_str("['hello', 'world'] . print", "['hello', 'world']\n"); }
    #[test] fn test_sqrt() { run_str("[0, 1, 4, 9, 25, 3, 6, 8, 13] . map(sqrt) . print", "[0, 1, 2, 3, 5, 1, 2, 2, 3]\n"); }
    #[test] fn test_very_large_sqrt() { run_str("[1 << 62, (1 << 62) + 1, (1 << 62) - 1] . map(sqrt) . print", "[2147483648, 2147483648, 2147483647]\n"); }
    #[test] fn test_gcd() { run_str("gcd(12, 8) . print", "4\n"); }
    #[test] fn test_gcd_iter() { run_str("[12, 18, 16] . gcd . print", "2\n"); }
    #[test] fn test_lcm() { run_str("lcm(9, 7) . print", "63\n"); }
    #[test] fn test_lcm_iter() { run_str("[12, 10, 18] . lcm . print", "180\n"); }


    #[test] fn test_aoc_2022_01_01() { run("aoc_2022_01_01"); }
    #[test] fn test_append_large_lists() { run("append_large_lists"); }
    #[test] fn test_closure_instead_of_global_variable() { run("closure_instead_of_global_variable"); }
    #[test] fn test_closure_of_partial_function() { run("closure_of_partial_function"); }
    #[test] fn test_closure_with_non_unique_values() { run("closure_with_non_unique_values"); }
    #[test] fn test_closure_without_stack_semantics() { run("closure_without_stack_semantics"); }
    #[test] fn test_fibonacci() { run("fibonacci"); }
    #[test] fn test_function_capture_from_inner_scope() { run("function_capture_from_inner_scope"); }
    #[test] fn test_function_capture_from_outer_scope() { run("function_capture_from_outer_scope"); }
    #[test] fn test_late_bound_global() { run("late_bound_global"); }
    #[test] fn test_late_bound_global_invalid() { run("late_bound_global_invalid"); }
    #[test] fn test_runtime_error_with_trace() { run("runtime_error_with_trace"); }


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

        let result: Result<(), DetailRuntimeError> = vm.run_until_completion();
        assert!(vm.stack.is_empty() || result.is_err());

        let mut output: String = String::from_utf8(buf).unwrap();

        match result {
            Ok(_) => {},
            Err(err) => output.push_str(ErrorReporter::new(text, source).format_runtime_error(err).as_str()),
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

        let result: Result<(), DetailRuntimeError> = vm.run_until_completion();
        assert!(vm.stack.is_empty() || result.is_err());

        let mut output: String = String::from_utf8(buf).unwrap();

        if let Err(e) = result {
            output.push_str(reporting::format_runtime_error(&text.split("\n").collect(), source, e).as_str());
        }

        trace::test::compare_test_resource_content(&root, output.split("\n").map(|s| String::from(s)).collect::<Vec<String>>());
    }
}
