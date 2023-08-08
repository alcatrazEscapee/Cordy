use std::cell::Cell;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::rc::Rc;
use fxhash::FxBuildHasher;

use crate::{compiler, core, trace, util};
use crate::compiler::{CompileParameters, CompileResult, Fields, IncrementalCompileResult, Locals};
use crate::reporting::{Location, SourceView};
use crate::util::OffsetAdd;
use crate::vm::value::{Field, Literal, UpValue, ValueStructType};
use crate::core::Pattern;

pub use crate::vm::error::{DetailRuntimeError, RuntimeError};
pub use crate::vm::opcode::{Opcode, StoreOp};
pub use crate::vm::value::{C64, FunctionImpl, guard_recursive_hash, IntoDictValue, IntoIterableValue, IntoValue, Iterable, LiteralType, MAX_INT, MIN_INT, StructTypeImpl, Type, ValueOption, ValuePtr, ValueResult, ErrorResult, AnyResult, Prefix};

use Opcode::{*};
use RuntimeError::{*};

pub mod operator;

mod value;
mod opcode;
mod error;

/// Per-test, how many instructions should be allowed to execute.
/// This primarily prevents infinite-loop tests from causing tests to hang, allowing easier debugging.
#[cfg(test)]
const TEST_EXECUTION_LIMIT: usize = 1000;


pub struct VirtualMachine<R, W> {
    ip: usize,
    code: Vec<Opcode>,
    stack: Vec<ValuePtr>,
    call_stack: Vec<CallFrame>,
    literal_stack: Vec<Literal>,
    global_count: usize,
    open_upvalues: HashMap<usize, Rc<Cell<UpValue>>, FxBuildHasher>,
    unroll_stack: Vec<i32>,

    constants: Vec<ValuePtr>,
    patterns: Vec<Rc<Pattern>>,
    globals: Vec<String>,
    locations: Vec<Location>,
    fields: Fields,

    view: SourceView,
    read: R,
    write: W,
    args: ValuePtr,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum FunctionType {
    Native, User
}

#[derive(Debug)]
pub enum ExitType {
    Exit, Return, Yield, Error(DetailRuntimeError)
}

impl ExitType {
    pub fn is_early_exit(&self) -> bool {
        matches!(self, ExitType::Exit | ExitType::Error(_))
    }

    fn of<R: BufRead, W: Write>(vm: &VirtualMachine<R, W>, result: AnyResult) -> ExitType {
        match result.map_err(|e| e.value) {
            Ok(_) => ExitType::Return,
            Err(RuntimeExit) => ExitType::Exit,
            Err(RuntimeYield) => ExitType::Yield,
            Err(error) => ExitType::Error(error.with_stacktrace(vm.ip - 1, &vm.call_stack, &vm.constants, &vm.locations)),
        }
    }
}


pub trait VirtualInterface {
    // Invoking Functions

    fn invoke_func0(&mut self, f: ValuePtr) -> ValueResult;
    fn invoke_func1(&mut self, f: ValuePtr, a1: ValuePtr) -> ValueResult;
    fn invoke_func2(&mut self, f: ValuePtr, a1: ValuePtr, a2: ValuePtr) -> ValueResult;
    fn invoke_func(&mut self, f: ValuePtr, args: &Vec<ValuePtr>) -> ValueResult;

    fn invoke_eval(&mut self, s: &String) -> ValueResult;

    /// Executes a `StoreOp`, storing the value `value`
    fn store(&mut self, op: StoreOp, value: ValuePtr) -> AnyResult;

    // Wrapped IO
    fn println0(&mut self);
    fn println(&mut self, str: String);
    fn print(&mut self, str: String);

    fn read_line(&mut self) -> String;
    fn read(&mut self) -> String;

    fn get_envs(&self) -> ValuePtr;
    fn get_env(&self, name: &String) -> ValuePtr;
    fn get_args(&self) -> ValuePtr;

    // Stack Manipulation
    fn peek(&self, offset: usize) -> &ValuePtr;
    fn pop(&mut self) -> ValuePtr;
    fn popn(&mut self, n: u32) -> Vec<ValuePtr>;
    fn push(&mut self, value: ValuePtr);
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

    pub fn new(result: CompileResult, view: SourceView, read: R, write: W, args: Vec<String>) -> VirtualMachine<R, W> {
        VirtualMachine {
            ip: 0,
            code: result.code,
            stack: Vec::with_capacity(256), // Just guesses, not hard limits
            call_stack: vec![CallFrame { return_ip: 0, frame_pointer: 0 }],
            literal_stack: Vec::with_capacity(16),
            global_count: 0,
            open_upvalues: HashMap::with_hasher(FxBuildHasher::default()),
            unroll_stack: Vec::new(),

            constants: result.constants,
            patterns: result.patterns,
            globals: result.globals,
            locations: result.locations,
            fields: result.fields,

            view,
            read,
            write,
            args: args.into_iter().map(|u| u.to_value()).to_list(),
        }
    }

    pub fn view(&self) -> &SourceView {
        &self.view
    }

    pub fn view_mut(&mut self) -> &mut SourceView {
        &mut self.view
    }

    /// Bridge method to `compiler::incremental_compile`
    pub fn incremental_compile(&mut self, locals: &mut Vec<Locals>) -> IncrementalCompileResult {
        compiler::incremental_compile(self.as_compile_parameters(false, locals))
    }

    /// Bridge method to `compiler::eval_compile`
    pub fn eval_compile(&mut self, text: &String) -> AnyResult {
        let mut locals = Locals::empty();
        compiler::eval_compile(text, self.as_compile_parameters(false, &mut locals))
    }

    fn as_compile_parameters<'a, 'b: 'a, 'c: 'a>(&'b mut self, enable_optimization: bool, locals: &'c mut Vec<Locals>) -> CompileParameters<'a> {
        CompileParameters::new(enable_optimization, &mut self.code, &mut self.constants, &mut self.patterns, &mut self.globals, &mut self.locations, &mut self.fields, locals, &mut self.view)
    }

    pub fn run_until_completion(&mut self) -> ExitType {
        let result = self.run();
        ExitType::of(self, result)
    }

    /// Recovers the VM into an operational state, in case previous instructions terminated in an error or in the middle of a function
    pub fn run_recovery(&mut self, locals: usize) {
        self.call_stack.truncate(1);
        self.stack.truncate(locals);
        self.literal_stack.clear();
        self.ip = self.code.len();
    }

    fn run(&mut self) -> AnyResult {
        #[cfg(test)]
        let mut limit = 0;
        let drop_frame: usize = self.call_stack.len() - 1;
        loop {
            #[cfg(test)]
            {
                limit += 1;
                if limit == TEST_EXECUTION_LIMIT {
                    panic!("Execution limit reached");
                }
            }
            let op: Opcode = self.next_op();
            self.run_instruction(op)?;
            if drop_frame == self.call_stack.len() {
                return Ok(())
            }
        }
    }

    /// Executes a single instruction
    #[inline(always)]
    fn run_instruction(&mut self, op: Opcode) -> AnyResult {
        trace::trace_interpreter!("vm::run op={:?}", op);
        match op {
            Noop => panic!("Noop should only be emitted as a temporary instruction"),

            // Flow Control
            JumpIfFalse(ip) => {
                let jump: usize = self.ip.add_offset(ip);
                let a1: &ValuePtr = self.peek(0);
                if !a1.to_bool() {
                    self.ip = jump;
                }
            },
            JumpIfFalsePop(ip) => {
                let jump: usize = self.ip.add_offset(ip);
                let a1: ValuePtr = self.pop();
                if !a1.to_bool() {
                    self.ip = jump;
                }
            },
            JumpIfTrue(ip) => {
                let jump: usize = self.ip.add_offset(ip);
                let a1: &ValuePtr = self.peek(0);
                if a1.to_bool() {
                    self.ip = jump;
                }
            },
            JumpIfTruePop(ip) => {
                let jump: usize = self.ip.add_offset(ip);
                let a1: ValuePtr = self.pop();
                if a1.to_bool() {
                    self.ip = jump;
                }
            },
            Jump(ip) => {
                let jump: usize = self.ip.add_offset(ip);
                self.ip = jump;
            },
            Return => {
                // Functions leave their return value as the top of the stack
                // Below that will be the functions locals, and the function itself
                // The frame pointer points to the first local of the function
                // [prev values ... function, local0, local1, ... localN, ret_val ]
                //                            ^frame pointer
                // So, we pop the return value, truncate the difference between the frame pointer and the top, then push the return value
                trace::trace_interpreter_stack!("drop frame {}", self.debug_stack());

                let frame: CallFrame = self.call_stack.pop().unwrap(); // Pop the call frame

                self.stack.swap_remove(frame.frame_pointer - 1); // This removes the function, and drops it, and it gets automatically replaced with the return value
                self.stack.truncate(frame.frame_pointer); // Drop all values above the frame pointer
                self.ip = frame.return_ip; // And jump to the return address
            },

            // Stack Manipulations
            Pop => {
                self.pop();
            },
            PopN(n) => {
                let len: usize = self.stack.len();
                self.stack.truncate(len - n as usize);
                trace::trace_interpreter_stack!("PopN {}", self.debug_stack());
            },
            Swap => {
                let len: usize = self.stack.len();
                self.stack.swap(len - 1, len - 2);
                trace::trace_interpreter_stack!("Swap {}", self.debug_stack());
            },

            PushLocal(local) => {
                // Locals are offset by the frame pointer, and don't need to check existence, as we don't allow late binding.
                let local = self.frame_pointer() + local as usize;
                trace::trace_interpreter!("vm::run PushLocal index={}, local={}", local, self.stack[local].as_debug_str());
                self.push(self.stack[local].clone());
            }
            StoreLocal(local, pop) => {
                trace::trace_interpreter!("vm::run StoreLocal index={}, value={}, prev={}", local, self.stack.last().unwrap().as_debug_str(), self.stack[local as usize].as_debug_str());
                let value = if pop { self.pop() } else { self.peek(0).clone() };
                self.store_local(local, value);
            },
            PushGlobal(local) => {
                // Globals are absolute offsets, and allow late binding, which means we have to check the global count before referencing.
                let local: usize = local as usize;
                trace::trace_interpreter!("vm::run PushGlobal index={}, value={}", local, self.stack[local].as_debug_str());
                if local < self.global_count {
                    self.push(self.stack[local].clone());
                } else {
                    return ValueErrorVariableNotDeclaredYet(self.globals[local].clone()).err()
                }
            },
            StoreGlobal(local, pop) => {
                trace::trace_interpreter!("vm::run StoreGlobal index={}, value={}, prev={}", local, self.stack.last().unwrap().as_debug_str(), self.stack[local as usize].as_debug_str());
                let value = if pop { self.pop() } else { self.peek(0).clone() };
                self.store_global(local, value)?;
            },
            PushUpValue(index) => {
                let fp = self.frame_pointer() - 1;
                let upvalue: Rc<Cell<UpValue>> = self.stack[fp].as_closure().borrow().get(index as usize);

                let interior = (*upvalue).take();
                upvalue.set(interior.clone()); // Replace back the original value

                let value: ValuePtr = match interior {
                    UpValue::Open(index) => self.stack[index].clone(),
                    UpValue::Closed(value) => value,
                };
                trace::trace_interpreter!("vm::run PushUpValue index={}, value={}", index, value.as_debug_str());
                self.push(value);
            },
            StoreUpValue(index) => {
                trace::trace_interpreter!("vm::run StoreUpValue index={}, value={}, prev={}", index, self.stack.last().unwrap().as_debug_str(), self.stack[index as usize].as_debug_str());
                let value = self.peek(0).clone();
                self.store_upvalue(index, value);
            },

            StoreArray => {
                trace::trace_interpreter!("vm::run StoreArray array={}, index={}, value={}", self.stack[self.stack.len() - 3].as_debug_str(), self.stack[self.stack.len() - 2].as_debug_str(), self.stack.last().unwrap().as_debug_str());
                let a3: ValuePtr = self.pop();
                let a2: ValuePtr = self.pop();
                let a1: &ValuePtr = self.peek(0); // Leave this on the stack when done
                core::set_index(a1, a2, a3)?;
            },

            InitGlobal => {
                self.global_count += 1;
            }

            Closure => {
                let f = self.pop();
                self.push(ValuePtr::closure(f));
            },

            CloseLocal(index) => {
                let local: usize = self.frame_pointer() + index as usize;
                trace::trace_interpreter!("vm::run CloseLocal index={}, local={}, value={}, closure={}", index, local, self.stack[local].as_debug_str(), self.stack.last().unwrap().as_debug_str());
                let upvalue: Rc<Cell<UpValue>> = self.open_upvalues.entry(local)
                    .or_insert_with(|| Rc::new(Cell::new(UpValue::Open(local))))
                    .clone();
                self.stack.last()
                    .unwrap()
                    .as_closure()
                    .borrow_mut()
                    .push(upvalue);
            },
            CloseUpValue(index) => {
                trace::trace_interpreter!("vm::run CloseUpValue index={}, value={}, closure={}", index, self.stack.last().unwrap().as_debug_str(), &self.stack[self.frame_pointer() - 1].as_debug_str());
                let fp = self.frame_pointer() - 1;
                let index: usize = index as usize;
                let upvalue: Rc<Cell<UpValue>> = self.stack[fp].as_closure().borrow().get(index);

                self.stack.last()
                    .unwrap()
                    .as_closure()
                    .borrow_mut()
                    .push(upvalue.clone());
            },

            LiftUpValue(index) => {
                let index = self.frame_pointer() + index as usize;
                if let Some(upvalue) = self.open_upvalues.remove(&index) {
                    let value: ValuePtr = self.stack[index].clone();
                    let unboxed: UpValue = (*upvalue).replace(UpValue::Open(0));
                    let closed: UpValue = match unboxed {
                        UpValue::Open(_) => UpValue::Closed(value),
                        UpValue::Closed(_) => panic!("Tried to lift an already closed upvalue"),
                    };
                    (*upvalue).replace(closed);
                }
            },

            InitIterable => {
                let iter = self.pop().to_iter()?;
                self.push(iter.to_value());
            },
            TestIterable(ip) => {
                let top: usize = self.stack.len() - 1;
                let iter = self.stack[top].as_iterable_mut();
                match iter.next() {
                    Some(value) => self.push(value),
                    None => self.ip = self.ip.add_offset(ip),
                }
            },

            ExecPattern(index) => {
                // I would do away with the `.clone()`s here, and the `Rc<>` on patterns, but we need no borrows to call `apply(&mut VM)`
                let top = self.peek(0).clone();
                let pattern = self.patterns[index as usize].clone();
                pattern.apply(self, &top)?;
            },

            // Push Operations
            Nil => self.push(ValuePtr::nil()),
            True => self.push(true.to_value()),
            False => self.push(false.to_value()),
            NativeFunction(native) => self.push(native.to_value()),
            Constant(id) => {
                self.push(self.constants[id as usize].clone());
            },

            LiteralBegin(op, length) => {
                self.literal_stack.push(Literal::new(op, length));
            },
            LiteralAcc(length) => {
                let top = self.literal_stack.last_mut().unwrap();
                top.accumulate(splice(&mut self.stack, length));
            },
            LiteralUnroll => {
                let arg = self.pop();
                let top = self.literal_stack.last_mut().unwrap();
                top.unroll(arg.to_iter()?)?;
            },
            LiteralEnd => {
                let top = self.literal_stack.pop().unwrap();
                self.push(top.to_value());
            },

            Unroll(first) => {
                if first {
                    self.unroll_stack.push(0);
                }
                let arg: ValuePtr = self.pop();
                let mut len: i32 = -1; // An empty unrolled argument contributes an offset of -1 + <number of elements unrolled>
                for e in arg.to_iter()? {
                    self.push(e);
                    len += 1;
                }
                *self.unroll_stack.last_mut().unwrap() += len;
            }

            Call(mut nargs, any_unroll) => {
                if any_unroll {
                    let unrolled_nargs: i32 = self.unroll_stack.pop().unwrap();
                    nargs = nargs.add_offset(unrolled_nargs);
                }
                self.invoke(nargs)?;
            },

            OpIndex => {
                let a2: ValuePtr = self.pop();
                let a1: ValuePtr = self.pop();
                let ret = core::get_index(self, &a1, a2)?;
                self.push(ret);
            },
            OpIndexPeek => {
                let a2: ValuePtr = self.peek(0).clone();
                let a1: ValuePtr = self.peek(1).clone();
                let ret = core::get_index(self, &a1, a2)?;
                self.push(ret);
            },
            OpSlice => {
                let a3: ValuePtr = self.pop();
                let a2: ValuePtr = self.pop();
                let a1: ValuePtr = self.pop();
                let ret = core::get_slice(&a1, a2, a3, 1i64.to_value())?;
                self.push(ret);
            },
            OpSliceWithStep => {
                let a4: ValuePtr = self.pop();
                let a3: ValuePtr = self.pop();
                let a2: ValuePtr = self.pop();
                let a1: ValuePtr = self.pop();
                let ret = core::get_slice(&a1, a2, a3, a4)?;
                self.push(ret);
            },

            GetField(field_index) => {
                let a1: ValuePtr = self.pop();
                let ret: ValuePtr = a1.get_field(&self.fields, field_index)?;
                self.push(ret);
            },
            GetFieldPeek(field_index) => {
                let a1: ValuePtr = self.peek(0).clone();
                let ret: ValuePtr = a1.get_field(&self.fields, field_index)?;
                self.push(ret);
            },
            GetFieldFunction(field_index) => {
                self.push(ValuePtr::from(Field(field_index)));
            },
            SetField(field_index) => {
                let a2: ValuePtr = self.pop();
                let a1: ValuePtr = self.pop();
                let ret: ValuePtr = a1.set_field(&self.fields, field_index, a2)?;
                self.push(ret);
            },

            Unary(op) => {
                let a1: ValuePtr = self.pop();
                let ret: ValuePtr = op.apply(a1)?;
                self.push(ret);
            },
            Binary(op) => {
                let a2: ValuePtr = self.pop();
                let a1: ValuePtr = self.pop();
                let ret: ValuePtr = op.apply(a1, a2)?;
                self.push(ret);
            },

            Slice => {
                let arg2: ValuePtr = self.pop();
                let arg1: ValuePtr = self.pop();
                self.push(ValuePtr::slice(arg1, arg2, ValuePtr::nil())?);
            },
            SliceWithStep => {
                let arg3: ValuePtr = self.pop();
                let arg2: ValuePtr = self.pop();
                let arg1: ValuePtr = self.pop();
                self.push(ValuePtr::slice(arg1, arg2, arg3)?);
            }

            Exit => return RuntimeExit.err(),
            Yield => {
                // First, jump to the end of current code, so when we startup again, we are in the right location
                self.ip = self.code.len();
                return RuntimeYield.err()
            },
            AssertFailed => {
                let ret: ValuePtr = self.pop();
                return RuntimeAssertFailed(ret.to_str()).err()
            },
        }
        Ok(())
    }

    // ===== Store Implementations ===== //

    fn store_local(&mut self, index: u32, value: ValuePtr) {
        let local: usize = self.frame_pointer() + index as usize;
        self.stack[local] = value;
    }

    fn store_global(&mut self, index: u32, value: ValuePtr) -> AnyResult {
        let local: usize = index as usize;
        if local < self.global_count {
            self.stack[local] = value;
            Ok(())
        } else {
            ValueErrorVariableNotDeclaredYet(self.globals[local].clone()).err()
        }
    }

    fn store_upvalue(&mut self, index: u32, value: ValuePtr) {
        let fp = self.frame_pointer() - 1;
        let upvalue: Rc<Cell<UpValue>> = self.stack[fp].as_closure().borrow().get(index as usize);

        // Reasons why this is convoluted:
        // - We cannot use `.get()` (as it requires `ValuePtr` to be `Copy`)
        // - We cannot use `get_mut()` (as even if we have `&mut ClosureImpl`, unboxing the `Rc<>` only gives us `&Cell`)
        let unboxed: UpValue = (*upvalue).take();
        let modified: UpValue = match unboxed {
            UpValue::Open(stack_index) => {
                let ret = UpValue::Open(stack_index); // And return the upvalue, unmodified
                self.stack[stack_index] = value; // Mutate on the stack
                ret
            },
            UpValue::Closed(_) => UpValue::Closed(value), // Mutate on the heap
        };
        (*upvalue).set(modified);
    }


    // ===== Basic Ops ===== //

    /// Returns the current `frame_pointer`
    fn frame_pointer(&self) -> usize {
        self.call_stack[self.call_stack.len() - 1].frame_pointer
    }

    /// Returns the next opcode and increments `ip`
    fn next_op(&mut self) -> Opcode {
        let op: Opcode = self.code[self.ip];
        self.ip += 1;
        op
    }

    fn invoke_and_spin(&mut self, nargs: u32) -> ValueResult {
        match self.invoke(nargs)? {
            FunctionType::Native => {},
            FunctionType::User => self.run()?
        }
        self.pop().ok()
    }

    /// Invokes the action of an `OpFuncEval(nargs)` opcode.
    ///
    /// The stack must be setup as `[..., f, arg1, arg2, ... argN ]`, where `f` is the function to be invoked with arguments `arg1, arg2, ... argN`.
    /// The arguments and function will be popped and the return value will be left on the top of the stack.
    ///
    /// Returns a `Result` which may contain an error which occurred during function evaluation.
    fn invoke(&mut self, nargs: u32) -> ErrorResult<FunctionType> {
        let f: &ValuePtr = self.peek(nargs as usize);
        trace::trace_interpreter!("vm::invoke func={:?}, nargs={}", f, nargs);
        match f.ty() {
            Type::Function | Type::Closure => {
                let func = f.get_function();
                if func.in_range(nargs) {
                    // Evaluate directly
                    self.call_function(func.jump_offset(nargs), nargs, func.num_var_args(nargs));
                    Ok(FunctionType::User)
                } else if func.min_args() > nargs {
                    // Evaluate as a partial function
                    // Special case if nargs == 0, we can avoid creating a partial wrapper and doing any stack manipulations
                    if nargs > 0 {
                        let arg: Vec<ValuePtr> = self.popn(nargs);
                        let func: ValuePtr = self.pop();
                        let partial: ValuePtr = ValuePtr::partial(func, arg);
                        self.push(partial);
                    }
                    // Partial functions are already evaluated, so we return native, since we don't need to spin
                    Ok(FunctionType::Native)
                } else {
                    IncorrectArgumentsUserFunction(func.clone(), nargs).err()
                }
            },
            Type::PartialFunction => {
                // Surgically extract the partial binding from the stack
                let i: usize = self.stack.len() - nargs as usize - 1;
                let mut partial = std::mem::replace(&mut self.stack[i], ValuePtr::nil()).as_partial_function().value;
                let func = partial.func.get();
                let total_nargs: u32 = partial.args.len() as u32 + nargs;
                if func.min_args() > total_nargs {
                    // Not enough arguments, so pop the argument and push a new partial function
                    for arg in splice(&mut self.stack, nargs) {
                        partial.args.push(arg);
                    }
                    self.pop(); // Should pop the `Nil` we swapped earlier
                    self.push(partial.to_value());
                    // Partial functions are already evaluated, so we return native, since we don't need to spin
                    Ok(FunctionType::Native)
                } else if func.in_range(total_nargs) {
                    // Exactly enough arguments to invoke the function
                    // Before we call, we need to pop-push to reorder the arguments and setup partial arguments, so we have the correct calling convention
                    let head: usize = func.jump_offset(total_nargs);
                    let num_var_args: Option<u32> = func.num_var_args(nargs);
                    self.stack[i] = partial.func.inner(); // Replace the `Nil` from earlier
                    insert(&mut self.stack, partial.args.into_iter(), nargs);
                    self.call_function(head, total_nargs, num_var_args);
                    Ok(FunctionType::User)
                } else {
                    IncorrectArgumentsUserFunction(func.clone(), total_nargs).err()
                }
            },
            Type::NativeFunction => {
                let ret = core::invoke_stack(f.as_native(), nargs, self)?;

                self.pop();
                self.push(ret);

                Ok(FunctionType::Native)
            },
            Type::PartialNativeFunction => {
                // Need to consume the arguments and set up the stack for calling as if all partial arguments were just pushed
                // Surgically extract the binding via std::mem::replace
                let i: usize = self.stack.len() - 1 - nargs as usize;
                let partial = std::mem::replace(&mut self.stack[i], ValuePtr::nil()).as_partial_native().value;

                let ret = core::invoke_partial(partial.func, partial.partial, nargs, self)?;

                self.pop();
                self.push(ret);

                Ok(FunctionType::Native)
            }
            Type::List => {
                // This is somewhat horrifying, but it could be optimized in constant cases later, in all cases where this syntax is actually used
                // As a result this code should almost never enter as it should be optimized away.
                if nargs != 1 {
                    return ValueIsNotFunctionEvaluable(f.clone()).err();
                }
                let arg = self.pop();
                let func = self.pop();
                let list = func.as_list().borrow();
                if list.list.len() != 1 {
                    return ValueErrorEvalListMustHaveUnitLength(list.list.len()).err()
                }
                let index = list.list[0].clone();
                let result = core::get_index(self, &arg, index)?;
                self.push(result);
                Ok(FunctionType::Native)
            },
            Type::Slice => {
                if nargs != 1 {
                    return ValueIsNotFunctionEvaluable(f.clone()).err();
                }
                let arg = self.pop();
                let slice = self.pop().as_slice().value;
                self.push(slice.apply(&arg)?);
                Ok(FunctionType::Native)
            }
            Type::StructType => {
                let type_impl = f.as_struct_type().borrow_const();
                let expected_args = type_impl.field_names.len() as u32;
                if nargs != expected_args {
                    return IncorrectArgumentsStruct(type_impl.clone(), nargs).err()
                }

                let args: Vec<ValuePtr> = self.popn(nargs);
                let struct_type = self.pop();
                let instance: ValuePtr = ValuePtr::instance(ValueStructType::new(struct_type), args);

                self.push(instance);

                Ok(FunctionType::Native)
            },
            Type::GetField => {
                let field_index = f.as_field();
                if nargs != 1 {
                    return IncorrectArgumentsGetField(self.fields.get_field_name(field_index), nargs).err()
                }

                let arg: ValuePtr = self.pop();
                let ret: ValuePtr = arg.get_field(&self.fields, field_index)?;

                self.pop(); // The get field
                self.push(ret);

                Ok(FunctionType::Native)
            },
            Type::Memoized => {
                // Bounce directly to `core::invoke_memoized`
                let ret: ValuePtr = core::invoke_memoized(self, nargs)?;
                self.push(ret);
                Ok(FunctionType::Native)
            },
            _ => ValueIsNotFunctionEvaluable(f.clone()).err(),
        }
    }

    /// Calls a user function by building a `CallFrame` and jumping to the function's `head` IP
    fn call_function(&mut self, head: usize, nargs: u32, num_var_args: Option<u32>) {
        let frame = CallFrame {
            return_ip: self.ip,
            frame_pointer: self.stack.len() - (nargs as usize),
        };
        self.ip = head;
        self.call_stack.push(frame);

        if let Some(num_var_args) = num_var_args {
            let args = splice(&mut self.stack, num_var_args).to_vector();
            self.push(args);
        }
    }


    // ===== Debug Methods ===== //

    pub fn debug_stack(&self) -> String {
        format!(": [{}]", self.stack.iter().rev().map(|t| t.as_debug_str()).collect::<Vec<String>>().join(", "))
    }

    pub fn debug_call_stack(&self) -> String {
        format!(": [{}]", self.call_stack.iter().rev().map(|t| format!("{{fp: {}, ret: {}}}", t.frame_pointer, t.return_ip)).collect::<Vec<String>>().join(", "))
    }
}


impl <R, W> VirtualInterface for VirtualMachine<R, W> where
    R : BufRead,
    W : Write
{
    // ===== Calling Functions External Interface ===== //

    fn invoke_func0(&mut self, f: ValuePtr) -> ValueResult {
        self.push(f);
        self.invoke_and_spin(0)
    }

    fn invoke_func1(&mut self, f: ValuePtr, a1: ValuePtr) -> ValueResult {
        self.push(f);
        self.push(a1);
        self.invoke_and_spin(1)
    }

    fn invoke_func2(&mut self, f: ValuePtr, a1: ValuePtr, a2: ValuePtr) -> ValueResult {
        self.push(f);
        self.push(a1);
        self.push(a2);
        self.invoke_and_spin(2)
    }

    fn invoke_func(&mut self, f: ValuePtr, args: &Vec<ValuePtr>) -> ValueResult {
        self.push(f);
        for arg in args {
            self.push(arg.clone());
        }
        self.invoke_and_spin(args.len() as u32)
    }

    fn invoke_eval(&mut self, text: &String) -> ValueResult {
        let eval_head: usize = self.code.len();

        self.eval_compile(text)?;
        self.call_function(eval_head, 0, None);
        self.run()?;
        let ret = self.pop();
        self.push(ValuePtr::nil()); // `eval` executes as a user function but is called like a native function, this prevents stack fuckery
        ret.ok()
    }

    fn store(&mut self, op: StoreOp, value: ValuePtr) -> AnyResult {
        match op {
            StoreOp::Local(index) => self.store_local(index, value),
            StoreOp::Global(index) => self.store_global(index, value)?,
            StoreOp::UpValue(index) => self.store_upvalue(index, value),
        }
        Ok(())
    }

    // ===== IO Methods ===== //

    fn println0(&mut self) { writeln!(&mut self.write).unwrap(); }
    fn println(&mut self, str: String) { writeln!(&mut self.write, "{}", str).unwrap(); }
    fn print(&mut self, str: String) { write!(&mut self.write, "{}", str).unwrap(); }

    fn read_line(&mut self) -> String {
        let mut buf = String::new();
        self.read.read_line(&mut buf).unwrap();
        util::strip_line_ending(&mut buf);
        buf
    }

    fn read(&mut self) -> String {
        let mut buf = String::new();
        self.read.read_to_string(&mut buf).unwrap();
        buf
    }

    fn get_envs(&self) -> ValuePtr {
        std::env::vars().map(|(k, v)| (k.to_value(), v.to_value())).to_dict()
    }

    fn get_env(&self, name: &String) -> ValuePtr {
        std::env::var(name).map_or(ValuePtr::nil(), |u| u.to_value())
    }

    fn get_args(&self) -> ValuePtr {
        self.args.clone()
    }


    // ===== Stack Manipulations ===== //

    /// Peeks at the top element of the stack, or an element `offset` down from the top
    fn peek(&self, offset: usize) -> &ValuePtr {
        trace::trace_interpreter_stack!("peek({}) -> {}", offset, self.stack[self.stack.len() - 1 - offset].as_debug_str());
        let ret = self.stack.get(self.stack.len() - 1 - offset).unwrap();
        trace::trace_interpreter_stack!("{}", self.debug_stack());
        ret
    }

    /// Pops the top of the stack
    fn pop(&mut self) -> ValuePtr {
        trace::trace_interpreter_stack!("pop() -> {}", self.stack.last().unwrap().as_debug_str());
        let ret = self.stack.pop().unwrap();
        trace::trace_interpreter_stack!("{}", self.debug_stack());
        ret
    }

    /// Pops the top N values off the stack, in order
    fn popn(&mut self, n: u32) -> Vec<ValuePtr> {
        let ret = splice(&mut self.stack, n).collect();
        trace::trace_interpreter_stack!("{}", self.debug_stack());
        ret
    }

    /// Push a value onto the stack
    fn push(&mut self, value: ValuePtr) {
        trace::trace_interpreter_stack!("push({})", value.as_debug_str());
        self.stack.push(value);
        trace::trace_interpreter_stack!("{}", self.debug_stack());
    }
}

/// Iterates the top `n` values of the provided stack, in-order, removing them from the stack.
///
/// **N.B.** This is not implemented as a method on `VirtualMachine` as we want to take a partial borrow only of
/// `&mut self.stack` when called, and which means we can interact with other methods on the VM (e.g. the literal stack).
#[inline]
fn splice(stack: &mut Vec<ValuePtr>, n: u32) -> impl Iterator<Item=ValuePtr> + '_ {
    let start: usize = stack.len() - n as usize;
    let end: usize = stack.len();
    stack.splice(start..end, std::iter::empty())
}

/// Inserts the contents of `args`, at the top of the stack minus `n`.
/// Uses `Vec::splice` to be optimal.
///
/// Where : `n = N`, `args = [b0, b1, ... bM]`
/// <br>Before : `stack = [..., a0, a1, ... aN]`
/// <br>After : `stack = [..., b0, b1, ... bM, a0, a1, ... aN]`
#[inline]
fn insert<I : Iterator<Item=ValuePtr>>(stack: &mut Vec<ValuePtr>, args: I, n: u32) {
    let at: usize = stack.len() - n as usize;
    stack.splice(at..at, args);
}



#[cfg(test)]
mod tests {
    use crate::{compiler, test_util};
    use crate::reporting::SourceView;
    use crate::vm::{ExitType, VirtualMachine};

    #[test] fn test_empty() { run_str("", ""); }
    #[test] fn test_compose_1() { run_str("print . print", "print\n"); }
    #[test] fn test_compose_2() { run_str("'hello world' . print", "hello world\n"); }
    #[test] fn test_if_01() { run_str("if 1 < 2 { print('yes') } else { print ('no') }", "yes\n"); }
    #[test] fn test_if_02() { run_str("if 1 < -2 { print('yes') } else { print ('no') }", "no\n"); }
    #[test] fn test_if_03() { run_str("if true { print('yes') } print('and also')", "yes\nand also\n"); }
    #[test] fn test_if_04() { run_str("if 1 < -2 { print('yes') } print('and also')", "and also\n"); }
    #[test] fn test_if_05() { run_str("if 0 { print('yes') }", ""); }
    #[test] fn test_if_06() { run_str("if 1 { print('yes') }", "yes\n"); }
    #[test] fn test_if_07() { run_str("if 'string' { print('yes') }", "yes\n"); }
    #[test] fn test_if_08() { run_str("if 1 < 0 { print('yes') } elif 1 { print('hi') } else { print('hehe') }", "hi\n"); }
    #[test] fn test_if_09() { run_str("if 1 < 0 { print('yes') } elif 2 < 0 { print('hi') } else { print('hehe') }", "hehe\n"); }
    #[test] fn test_if_10() { run_str("if 1 { print('yes') } elif true { print('hi') } else { print('hehe') }", "yes\n"); }
    #[test] fn test_if_short_circuiting_1() { run_str("if true and print('yes') { print('no') }", "yes\n"); }
    #[test] fn test_if_short_circuiting_2() { run_str("if false and print('also no') { print('no') }", ""); }
    #[test] fn test_if_short_circuiting_3() { run_str("if true and (print('yes') or true) { print('also yes') }", "yes\nalso yes\n"); }
    #[test] fn test_if_short_circuiting_4() { run_str("if false or print('yes') { print('no') }", "yes\n"); }
    #[test] fn test_if_short_circuiting_5() { run_str("if true or print('no') { print('yes') }", "yes\n"); }
    #[test] fn test_if_then_else_1() { run_str("(if true then 'hello' else 'goodbye') . print", "hello\n"); }
    #[test] fn test_if_then_else_2() { run_str("(if false then 'hello' else 'goodbye') . print", "goodbye\n"); }
    #[test] fn test_if_then_else_3() { run_str("(if [] then 'hello' else 'goodbye') . print", "goodbye\n"); }
    #[test] fn test_if_then_else_4() { run_str("(if 3 then 'hello' else 'goodbye') . print", "hello\n"); }
    #[test] fn test_if_then_else_5() { run_str("(if false then (fn() -> 'hello' . print)() else 'nope') . print", "nope\n"); }
    #[test] fn test_if_then_else_top_level() { run_str("if true then print('hello') else print('goodbye')", "hello\n"); }
    #[test] fn test_if_then_else_top_level_in_loop() { run_str("for x in range(2) { if x then x else x }", ""); }
    #[test] fn test_while_false_if_false() { run_str("while false { if false { } }", ""); }
    #[test] fn test_while_else_no_loop() { run_str("while false { break } else { print('hello') }", "hello\n"); }
    #[test] fn test_while_else_break() { run_str("while true { break } else { print('hello') } print('world')", "world\n"); }
    #[test] fn test_while_else_no_break() { run_str("let x = true ; while x { x = false } else { print('hello') }", "hello\n"); }
    #[test] fn test_do_while_1() { run_str("do { 'test' . print } while false", "test\n"); }
    #[test] fn test_do_while_2() { run_str("let i = 0 ; do { i . print ; i += 1 } while i < 3", "0\n1\n2\n"); }
    #[test] fn test_do_while_3() { run_str("let i = 0 ; do { i += 1 ; i . print } while i < 3", "1\n2\n3\n"); }
    #[test] fn test_do_while_4() { run_str("let i = 5 ; do { i . print } while i < 3", "5\n"); }
    #[test] fn test_do_without_while() { run_str("do { 'test' . print }", "test\n"); }
    #[test] fn test_do_while_else_1() { run_str("do { 'loop' . print } while false else { 'else' . print }", "loop\nelse\n"); }
    #[test] fn test_do_while_else_2() { run_str("do { 'loop' . print ; break } while false else { 'else' . print }", "loop\n"); }
    #[test] fn test_do_while_else_3() { run_str("let i = 0 ; do { i . print ; i += 1 ; if i > 2 { break } } while 1 else { 'end' . print }", "0\n1\n2\n"); }
    #[test] fn test_do_while_else_4() { run_str("let i = 0 ; do { i . print ; i += 1 ; if i > 2 { break } } while i < 2 else { 'end' . print }", "0\n1\nend\n"); }
    #[test] fn test_for_loop_no_intrinsic_with_list() { run_str("for x in ['a', 'b', 'c'] { x . print }", "a\nb\nc\n") }
    #[test] fn test_for_loop_no_intrinsic_with_set() { run_str("for x in 'foobar' . set { x . print }", "f\no\nb\na\nr\n") }
    #[test] fn test_for_loop_no_intrinsic_with_str() { run_str("for x in 'hello' { x . print }", "h\ne\nl\nl\no\n") }
    #[test] fn test_for_loop_range_stop() { run_str("for x in range(5) { x . print }", "0\n1\n2\n3\n4\n"); }
    #[test] fn test_for_loop_range_start_stop() { run_str("for x in range(3, 6) { x . print }", "3\n4\n5\n"); }
    #[test] fn test_for_loop_range_start_stop_step_positive() { run_str("for x in range(1, 10, 3) { x . print }", "1\n4\n7\n"); }
    #[test] fn test_for_loop_range_start_stop_step_negative() { run_str("for x in range(11, 0, -4) { x . print }", "11\n7\n3\n"); }
    #[test] fn test_for_loop_range_start_stop_step_zero() { run_str("for x in range(1, 2, 0) { x . print }", "ValueError: 'step' argument cannot be zero\n  at: line 1 (<test>)\n\n1 | for x in range(1, 2, 0) { x . print }\n2 |               ^^^^^^^^^\n"); }
    #[test] fn test_for_else_no_loop() { run_str("for _ in [] { print('hello') ; break } else { print('world') }", "world\n"); }
    #[test] fn test_for_else_break() { run_str("for c in 'abcd' { if c == 'b' { break } } else { print('hello') } print('world')", "world\n"); }
    #[test] fn test_for_else_no_break() { run_str("for c in 'abcd' { if c == 'B' { break } } else { print('hello') }", "hello\n"); }
    #[test] fn test_struct_str_of_struct_instance() { run_str("struct Foo(a, b) Foo(1, 2) . print", "Foo(a=1, b=2)\n"); }
    #[test] fn test_struct_str_of_struct_constructor() { run_str("struct Foo(a, b) Foo . print", "struct Foo(a, b)\n"); }
    #[test] fn test_struct_get_field_of_struct() { run_str("struct Foo(a, b) Foo(1, 2) -> a . print", "1\n"); }
    #[test] fn test_struct_get_field_of_struct_wrong_name() { run_str("struct Foo(a, b) struct Bar(c, d) Foo(1, 2) -> c . print", "TypeError: Cannot get field 'c' on struct Foo(a, b)\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) struct Bar(c, d) Foo(1, 2) -> c . print\n2 |                                             ^^^^\n"); }
    #[test] fn test_struct_get_field_of_not_struct() { run_str("struct Foo(a, b) (1, 2) -> a . print", "TypeError: Cannot get field 'a' on '(1, 2)' of type 'vector'\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) (1, 2) -> a . print\n2 |                         ^^^^\n"); }
    #[test] fn test_struct_get_field_with_overlapping_offsets() { run_str("struct Foo(a, b) struct Bar(b, a) Foo(1, 2) -> b . print", "2\n"); }
    #[test] fn test_struct_set_field_of_struct() { run_str("struct Foo(a, b) let x = Foo(1, 2) ; x->a = 3 ; x->a . print", "3\n"); }
    #[test] fn test_struct_set_field_of_struct_wrong_name() { run_str("struct Foo(a, b) struct Bar(c, d) let x = Foo(1, 2) ; x->c = 3", "TypeError: Cannot get field 'c' on struct Foo(a, b)\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) struct Bar(c, d) let x = Foo(1, 2) ; x->c = 3\n2 |                                                            ^\n"); }
    #[test] fn test_struct_set_field_of_not_struct() { run_str("struct Foo(a, b) (1, 2)->a = 3", "TypeError: Cannot get field 'a' on '(1, 2)' of type 'vector'\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b) (1, 2)->a = 3\n2 |                            ^\n"); }
    #[test] fn test_struct_op_set_field_of_struct() { run_str("struct Foo(a, b) let x = Foo(1, 2) ; x->a += 3 ; x->a . print", "4\n"); }
    #[test] fn test_struct_partial_get_field_in_bare_method() { run_str("struct Foo(a, b) let x = Foo(2, 3), f = (->b) ; x . f . print", "3\n"); }
    #[test] fn test_struct_partial_get_field_in_function_eval() { run_str("struct Foo(a, b) [Foo(1, 2), Foo(2, 3)] . map(->b) . print", "[2, 3]\n"); }
    #[test] fn test_struct_more_partial_get_field() { run_str("struct Foo(foo) ; let x = Foo('hello') ; print([x, Foo('')] . filter(->foo) . len)", "1\n"); }
    #[test] fn test_struct_recursive_repr() { run_str("struct S(x) ; let x = S(nil) ; x->x = x ; x.print", "S(x=S(...))\n"); }
    #[test] fn test_struct_operator_is() { run_str("struct A() ; struct B() let a = A(), b = B() ; [a is A, A is function, a is B, A is A, a is function] . print", "[true, true, false, false, false]\n"); }
    #[test] fn test_struct_construct_not_enough_arguments() { run_str("struct Foo(a, b, c) ; Foo(1)(2) . print ; ", "Incorrect number of arguments for struct Foo(a, b, c), got 1\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b, c) ; Foo(1)(2) . print ; \n2 |                          ^^^\n"); }
    #[test] fn test_struct_construct_too_many_arguments() { run_str("struct Foo(a, b, c) ; Foo(1, 2, 3, 4) . print", "Incorrect number of arguments for struct Foo(a, b, c), got 4\n  at: line 1 (<test>)\n\n1 | struct Foo(a, b, c) ; Foo(1, 2, 3, 4) . print\n2 |                          ^^^^^^^^^^^^\n"); }
    #[test] fn test_local_vars_01() { run_str("let x=0 do { x.print }", "0\n"); }
    #[test] fn test_local_vars_02() { run_str("let x=0 do { let x=1; x.print }", "1\n"); }
    #[test] fn test_local_vars_03() { run_str("let x=0 do { x.print let x=1 }", "0\n"); }
    #[test] fn test_local_vars_04() { run_str("let x=0 do { let x=1 } x.print", "0\n"); }
    #[test] fn test_local_vars_05() { run_str("let x=0 do { x=1 } x.print", "1\n"); }
    #[test] fn test_local_vars_06() { run_str("let x=0 do { x=1 do { x=2; x.print } }", "2\n"); }
    #[test] fn test_local_vars_07() { run_str("let x=0 do { x=1 do { x=2 } x.print }", "2\n"); }
    #[test] fn test_local_vars_08() { run_str("let x=0 do { let x=1 do { x=2 } x.print }", "2\n"); }
    #[test] fn test_local_vars_09() { run_str("let x=0 do { let x=1 do { let x=2 } x.print }", "1\n"); }
    #[test] fn test_local_vars_10() { run_str("let x=0 do { x=1 do { let x=2 } x.print }", "1\n"); }
    #[test] fn test_local_vars_11() { run_str("let x=0 do { x=1 do { let x=2 } } x.print", "1\n"); }
    #[test] fn test_local_vars_12() { run_str("let x=0 do { let x=1 do { let x=2 } } x.print", "0\n"); }
    #[test] fn test_local_vars_14() { run_str("let x=3 do { let x=x; x.print }", "3\n"); }
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
    #[test] fn test_mutable_arrays_in_assignments() { run_str("let a = [0], b = [a, a, a]; b[0][0] = 5; b . print", "[[5], [5], [5]]\n"); }
    #[test] fn test_pattern_in_let_works() { run_str("let x, y = [1, 2] ; [x, y] . print", "[1, 2]\n"); }
    #[test] fn test_pattern_in_let_too_long() { run_str("let x, y, z = [1, 2] ; [x, y] . print", "ValueError: Cannot unpack '[1, 2]' of type 'list' with length 2, expected exactly 3 elements\n  at: line 1 (<test>)\n\n1 | let x, y, z = [1, 2] ; [x, y] . print\n2 |                    ^\n"); }
    #[test] fn test_pattern_in_let_too_short() { run_str("let x, y = [1, 2, 3] ; [x, y] . print", "ValueError: Cannot unpack '[1, 2, 3]' of type 'list' with length 3, expected exactly 2 elements\n  at: line 1 (<test>)\n\n1 | let x, y = [1, 2, 3] ; [x, y] . print\n2 |                    ^\n"); }
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
    #[test] fn test_pattern_in_let_empty_x3_too_long() { run_str("let _, _, _ = [1, 2, 3, 4]", "ValueError: Cannot unpack '[1, 2, 3, 4]' of type 'list' with length 4, expected exactly 3 elements\n  at: line 1 (<test>)\n\n1 | let _, _, _ = [1, 2, 3, 4]\n2 |                          ^\n"); }
    #[test] fn test_pattern_in_let_empty_x3_too_short() { run_str("let _, _, _ = [1, 2]", "ValueError: Cannot unpack '[1, 2]' of type 'list' with length 2, expected exactly 3 elements\n  at: line 1 (<test>)\n\n1 | let _, _, _ = [1, 2]\n2 |                    ^\n"); }
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
    #[test] fn test_pattern_in_let_with_varargs_empty_to_var_at_end_too_short() { run_str("let *_, x = []", "ValueError: Cannot unpack '[]' of type 'list' with length 0, expected at least 1 elements\n  at: line 1 (<test>)\n\n1 | let *_, x = []\n2 |              ^\n"); }
    #[test] fn test_pattern_in_let_with_varargs_empty_to_var_at_start_too_short() { run_str("let x, *_ = []", "ValueError: Cannot unpack '[]' of type 'list' with length 0, expected at least 1 elements\n  at: line 1 (<test>)\n\n1 | let x, *_ = []\n2 |              ^\n"); }
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
    #[test] fn test_pattern_in_function_multiple() { run_str("fn f((a, b), (c, d)) -> [a, b, c, d] . print ; f([1, 2], [3, 4])", "[1, 2, 3, 4]\n"); }
    #[test] fn test_pattern_in_function_before_args() { run_str("fn f((a, b, c), d, e) -> [a, b, c, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_in_function_between_args() { run_str("fn f(a, (b, c, d), e) -> [a, b, c, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_in_function_after_args() { run_str("fn f(a, b, (c, d, e)) -> [a, b, c, d, e] . print ; f(1, 2, [3, 4, 5])", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_empty_in_function_before_args() { run_str("fn f((_, b, _), d, e) -> [1, b, 3, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_empty_in_function_between_args() { run_str("fn f(a, (_, _, d), e) -> [a, 2, 3, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_empty_in_function_after_args() { run_str("fn f(a, b, (c, _, _)) -> [a, b, c, 4, 5] . print ; f(1, 2, [3, 4, 5])", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_pattern_with_var_in_function_before_args() { run_str("fn f((a, *_), d, e) -> [a, d, e] . print ; f([1, 2, 3], 4, 5)", "[1, 4, 5]\n"); }
    #[test] fn test_pattern_with_var_in_function_between_args() { run_str("fn f(a, (*_, d), e) -> [a, d, e] . print ; f(1, [2, 3, 4], 5)", "[1, 4, 5]\n"); }
    #[test] fn test_pattern_with_var_in_function_after_args() { run_str("fn f(a, b, (*c, _, _)) -> [a, b, c] . print ; f(1, 2, [3, 4, 5])", "[1, 2, [3]]\n"); }
    #[test] fn test_pattern_in_for_with_enumerate() { run_str("for i, x in 'hello' . enumerate { [i, x] . print }", "[0, 'h']\n[1, 'e']\n[2, 'l']\n[3, 'l']\n[4, 'o']\n")}
    #[test] fn test_pattern_in_for_with_empty() { run_str("for _ in range(5) { 'hello' . print }", "hello\nhello\nhello\nhello\nhello\n"); }
    #[test] fn test_pattern_in_for_with_strings() { run_str("for a, *_, b in ['hello', 'world'] { print(a + b) }", "ho\nwd\n"); }
    #[test] fn test_pattern_in_expression() { run_str("let x, y, z ; x, y, z = 'abc' ; print(x, y, z)", "a b c\n"); }
    #[test] fn test_pattern_in_expression_nested() { run_str("let x, y, z ; z = x, y = (1, 2) ; print(x, y, z)", "1 2 (1, 2)\n"); }
    #[test] fn test_pattern_in_expression_locals() { run_str("do { let x, y, z ; z = x, y = (1, 2) ; print(x, y, z) }", "1 2 (1, 2)\n"); }
    #[test] fn test_pattern_in_expression_return_value() { run_str("let x, y, z ; print(x, y, z = 'abc') ; print(x, y, z)", "abc\na b c\n"); }
    #[test] fn test_pattern_in_expression_with_variadic() { run_str("let x, y ; *x, y = 'hello' ; print(x, y)", "hell o\n"); }
    #[test] fn test_pattern_in_expression_with_nested_and_empty() { run_str("let x, y ; (x, *_), (*_, y) = ('hello', 'world') ; print(x, y)", "h d\n"); }
    #[test] fn test_pattern_in_expression_empty() { run_str("_ = nil ; _, _, _ = (1, 2, 3)", ""); }
    #[test] fn test_pattern_in_expression_empty_variadic() { run_str("*_ = 'hello world'", ""); }
    #[test] fn test_function_repr() { run_str("(fn((_, *_), x) -> nil) . repr . print", "fn _((_, *_), x)\n"); }
    #[test] fn test_function_repr_partial() { run_str("(fn((_, *_), x) -> nil)(1) . repr . print", "fn _((_, *_), x)\n"); }
    #[test] fn test_function_closure_repr() { run_str("fn box(x) -> fn((_, *_), y) -> x ; box(nil) . repr . print", "fn _((_, *_), y)\n"); }
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
    #[test] fn test_functions_11() { run_str("do { let x = 'hello' ; fn foo(x) { x . print } foo(x) }", "hello\n"); }
    #[test] fn test_functions_12() { run_str("do { let x = 'hello' ; do { fn foo(x) { x . print } foo(x) } }", "hello\n"); }
    #[test] fn test_functions_13() { run_str("let x = 'hello' ; do { fn foo() { x . print } foo() }", "hello\n"); }
    #[test] fn test_functions_14() { run_str("fn foo(x) { 'hello ' + x . print } 'world' . foo", "hello world\n"); }
    #[test] fn test_function_implicit_return_01() { run_str("fn foo() { } foo() . print", "nil\n"); }
    #[test] fn test_function_implicit_return_02() { run_str("fn foo() { 'hello' } foo() . print", "hello\n"); }
    #[test] fn test_function_implicit_return_03() { run_str("fn foo(x) { if x > 1 then nil else 'nope' } foo(2) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_04() { run_str("fn foo(x) { if x > 1 then true else false } foo(2) . print", "true\n"); }
    #[test] fn test_function_implicit_return_05() { run_str("fn foo(x) { if x > 1 then true else false } foo(0) . print", "false\n"); }
    #[test] fn test_function_implicit_return_06() { run_str("fn foo(x) { if x > 1 then nil else false } foo(2) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_07() { run_str("fn foo(x) { if x > 1 then nil else false } foo(0) . print", "false\n"); }
    #[test] fn test_function_implicit_return_08() { run_str("fn foo(x) { if x > 1 then true else nil } foo(2) . print", "true\n"); }
    #[test] fn test_function_implicit_return_09() { run_str("fn foo(x) { if x > 1 then true else nil } foo(0) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_10() { run_str("fn foo(x) { if x > 1 then 'hello' else nil } foo(2) . print", "hello\n"); }
    #[test] fn test_function_implicit_return_11() { run_str("fn foo(x) { if x > 1 then 'hello' else nil } foo(0) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_12() { run_str("fn foo(x) { if x > 1 then if true then 'hello' else nil else nil } foo(2) . print", "hello\n"); }
    #[test] fn test_function_implicit_return_13() { run_str("fn foo(x) { if x > 1 then if true then 'hello' else nil else nil } foo(0) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_14() { run_str("fn foo(x) { loop { if x > 1 { break } } } foo(2) . print", "nil\n"); }
    #[test] fn test_function_implicit_return_15() { run_str("fn foo(x) { loop { if x > 1 { continue } else { break } } } foo(0) . print", "nil\n"); }
    #[test] fn test_closures_01() { run_str("fn foo() { let x = 'hello' ; fn bar() { x . print } bar() } foo()", "hello\n"); }
    #[test] fn test_closures_02() { run_str("do { fn foo() { 'hello' . print } ; fn bar() { foo() } bar() }", "hello\n"); }
    #[test] fn test_closures_03() { run_str("do { fn foo() { 'hello' . print } ; do { fn bar() { foo() } bar() } }", "hello\n"); }
    #[test] fn test_closures_04() { run_str("do { fn foo() { 'hello' . print } ; fn bar(a) { foo() } bar(1) }", "hello\n"); }
    #[test] fn test_closures_05() { run_str("do { fn foo() { 'hello' . print } ; do { fn bar(a) { foo() } bar(1) } }", "hello\n"); }
    #[test] fn test_closures_06() { run_str("do { let x = 'hello' ; do { fn foo() { x . print } foo() } }", "hello\n"); }
    #[test] fn test_closures_07() { run_str("do { let x = 'hello' ; do { do { fn foo() { x . print } foo() } } }", "hello\n"); }
    #[test] fn test_closures_08() { run_str("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; x } foo() . print", "hello\n"); }
    #[test] fn test_closures_09() { run_str("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; (fn() -> x = 'goodbye')() ; x } foo() . print", "goodbye\n"); }
    #[test] fn test_closures_10() { run_str("fn foo() { let x = 'before' ; (fn() -> x = 'hello')() ; let y = (fn() -> x)() ; y } foo() . print", "hello\n"); }
    #[test] fn test_closures_11() { run_str("fn foo() { let x = 'hello' ; (fn() -> x += ' world')() ; x } foo() . print", "hello world\n"); }
    #[test] fn test_closures_12() { run_str("fn foo() { let x = 'hello' ; (fn() -> x += ' world')() ; (fn() -> x)() } foo() . print", "hello world\n"); }
    #[test] fn test_function_return_1() { run_str("fn foo() { return 3 } foo() . print", "3\n"); }
    #[test] fn test_function_return_2() { run_str("fn foo() { let x = 3; return x } foo() . print", "3\n"); }
    #[test] fn test_function_return_3() { run_str("fn foo() { let x = 3; do { return x } } foo() . print", "3\n"); }
    #[test] fn test_function_return_4() { run_str("fn foo() { let x = 3; do { let x; } return x } foo() . print", "3\n"); }
    #[test] fn test_function_return_5() { run_str("fn foo() { let x; do { let x = 3; return x } } foo() . print", "3\n"); }
    #[test] fn test_function_return_no_value() { run_str("fn foo() { print('hello') ; return ; print('world') } foo() . print", "hello\nnil\n"); }
    #[test] fn test_partial_func_1() { run_str("'apples and bananas' . replace ('a', 'o') . print", "opples ond bononos\n"); }
    #[test] fn test_partial_func_2() { run_str("'apples and bananas' . replace ('a') ('o') . print", "opples ond bononos\n"); }
    #[test] fn test_partial_func_3() { run_str("print('apples and bananas' . replace ('a') ('o'))", "opples ond bononos\n"); }
    #[test] fn test_partial_func_4() { run_str("let x = replace ('a', 'o') ; 'apples and bananas' . x . print", "opples ond bononos\n"); }
    #[test] fn test_partial_func_5() { run_str("let x = replace ('a', 'o') ; print(x('apples and bananas'))", "opples ond bononos\n"); }
    #[test] fn test_partial_func_6() { run_str("('o' . replace('a')) ('apples and bananas') . print", "opples ond bononos\n"); }
    #[test] fn test_partial_function_composition_1() { run_str("fn foo(a, b, c) { c . print } (3 . (2 . (1 . foo)))", "3\n"); }
    #[test] fn test_partial_function_composition_2() { run_str("fn foo(a, b, c) { c . print } (2 . (1 . foo)) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_3() { run_str("fn foo(a, b, c) { c . print } (1 . foo) (2) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_4() { run_str("fn foo(a, b, c) { c . print } foo (1) (2) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_5() { run_str("fn foo(a, b, c) { c . print } foo (1, 2) (3)", "3\n"); }
    #[test] fn test_partial_function_composition_6() { run_str("fn foo(a, b, c) { c . print } foo (1) (2, 3)", "3\n"); }
    #[test] fn test_partial_function_zero_arg_user_function() { run_str("fn foo(a, b) {} ; foo() . repr . print", "fn foo(a, b)\n"); }
    #[test] fn test_partial_function_zero_arg_native_function() { run_str("len() . repr . print", "fn len(x)\n"); }
    #[test] fn test_partial_function_zero_arg_operator_function() { run_str("(+)() . repr . print", "fn (+)(lhs, rhs)\n"); }
    #[test] fn test_partial_function_zero_arg_partial_user_function() { run_str("fn foo(a, b) {} ; foo(1)() . repr . print", "fn foo(a, b)\n"); }
    #[test] fn test_partial_function_zero_arg_partial_native_function() { run_str("push(1)() . repr . print", "fn push(value, collection)\n"); }
    #[test] fn test_partial_function_zero_arg_partial_operator_function() { run_str("(+)(1)() . repr . print", "fn (+)(lhs, rhs)\n"); }
    #[test] fn test_partial_function_zero_arg_user_not_optimized() { run_str("fn f(x) -> x() ; f(f(f(f))) . repr . print", "fn f(x)\n"); }
    #[test] fn test_partial_function_zero_arg_native_not_optimized() { run_str("fn f(x) -> x() ; f(f(f(len))) . repr . print", "fn len(x)\n"); }
    #[test] fn test_partial_function_zero_arg_operator_not_optimized() { run_str("fn f(x) -> x() ; f(f(f(+))) . repr . print", "fn (+)(lhs, rhs)\n"); }
    #[test] fn test_partial_user_functions_1() { run_str("fn foo(x) -> print(x) ; foo()('hi')", "hi\n"); }
    #[test] fn test_partial_user_functions_2() { run_str("fn foo(x, y) -> print(x, y) ; foo()('hi', 'there')", "hi there\n"); }
    #[test] fn test_partial_user_functions_3() { run_str("fn foo(x, y) -> print(x, y) ; foo('hi')('there')", "hi there\n"); }
    #[test] fn test_partial_user_functions_4() { run_str("fn foo(x, y) -> print(x, y) ; foo('hi')()('there')", "hi there\n"); }
    #[test] fn test_partial_user_functions_5() { run_str("fn foo(x, y) -> print(x, y) ; [1, 2] . map(foo('hello'))", "hello 1\nhello 2\n"); }
    #[test] fn test_partial_user_functions_6() { run_str("fn add(x, y) -> x + y ; [1, 2, 3] . map(add(3)) . print", "[4, 5, 6]\n"); }
    #[test] fn test_partial_user_functions_7() { run_str("fn add(x, y, z) -> x + y ; [1, 2, 3] . map(add(3)) . print", "[fn add(x, y, z), fn add(x, y, z), fn add(x, y, z)]\n"); }
    #[test] fn test_partial_user_functions_8() { run_str("fn add(x, y) -> x + y ; add(1)(2) . print", "3\n"); }
    #[test] fn test_function_with_one_default_arg() { run_str("fn foo(a, b?) { print(a, b) } ; foo('test') ; foo('test', 'bar')", "test nil\ntest bar\n"); }
    #[test] fn test_function_with_one_default_arg_not_enough() { run_str("fn foo(a, b?) { print(a, b) } ; foo()", ""); }
    #[test] fn test_function_with_one_default_arg_too_many() { run_str("fn foo(a, b?) { print(a, b) } ; foo(1, 2, 3)", "Incorrect number of arguments for fn foo(a, b), got 3\n  at: line 1 (<test>)\n\n1 | fn foo(a, b?) { print(a, b) } ; foo(1, 2, 3)\n2 |                                    ^^^^^^^^^\n"); }
    #[test] fn test_function_many_default_args() { run_str("fn foo(a, b = 1, c = 1 + 1, d = 1 * 3) { print(a, b, c, d) } foo('test') ; foo('and', 11) ; foo('other', 11, 22) ; foo('things', 11, 22, 33)", "test 1 2 3\nand 11 2 3\nother 11 22 3\nthings 11 22 33\n"); }
    #[test] fn test_function_unroll_1() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(...['hello', 'the', 'world'])", "hello the world\n"); }
    #[test] fn test_function_unroll_2() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, 3, ...[])", "1 2 3\n"); }
    #[test] fn test_function_unroll_3() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(1, ...[], 2, ...[], 3)", "1 2 3\n"); }
    #[test] fn test_function_unroll_4() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(...'ab', 'c')", "a b c\n"); }
    #[test] fn test_function_unroll_5() { run_str("fn foo(a, b, c, d) -> print(a, b, c, d) ; foo(...'ab', ...'cd')", "a b c d\n"); }
    #[test] fn test_function_unroll_6() { run_str("fn foo(a, b, c, d) -> print(a, b, c, d) ; foo(...'a', ...'bc', ...'d')", "a b c d\n"); }
    #[test] fn test_function_unroll_7() { run_str("fn foo(a, b, c, d) -> print(a, b, c, d) ; foo('a', ...'bc', 'd')", "a b c d\n"); }
    #[test] fn test_function_unroll_8() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(1, ...'ab')", "1 a b\n"); }
    #[test] fn test_function_unroll_9() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(...'ab', 3)", "a b 3\n"); }
    #[test] fn test_function_unroll_10() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, ...[3, 4])", "Incorrect number of arguments for fn foo(a, b, c), got 4\n  at: line 1 (<test>)\n\n1 | fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, ...[3, 4])\n2 |                                        ^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_function_unroll_11() { run_str("fn foo(a, b, c) -> print(a, b, c) ; foo(1, 2, ...[]) is function . print", "true\n"); }
    #[test] fn test_function_unroll_12() { run_str("sum([1, 2, 3, 4, 5]) . print", "15\n"); }
    #[test] fn test_function_unroll_13() { run_str("sum(...[1, 2, 3, 4, 5]) . print", "15\n"); }
    #[test] fn test_function_unroll_14() { run_str("print(...[1, 2, 3])", "1 2 3\n"); }
    #[test] fn test_function_unroll_15() { run_str("print(...[print(...[1, 2, 3])])", "1 2 3\nnil\n"); }
    #[test] fn test_function_unroll_16() { run_str("print(...[], ...[print(...[], 'second', ...[], ...[print('first', ...[])])], ...[], ...[print('third')])", "first\nsecond nil\nthird\nnil nil\n"); }
    #[test] fn test_function_unroll_17() { run_str("print(1, ...[2, print('a', ...[1, 2, 3], 'e'), -2], 3)", "a 1 2 3 e\n1 2 nil -2 3\n"); }
    #[test] fn test_function_var_args_1() { run_str("fn foo(*a) -> print(a) ; foo()", "()\n"); }
    #[test] fn test_function_var_args_2() { run_str("fn foo(*a) -> print(a) ; foo(1)", "(1)\n"); }
    #[test] fn test_function_var_args_3() { run_str("fn foo(*a) -> print(a) ; foo(1, 2)", "(1, 2)\n"); }
    #[test] fn test_function_var_args_4() { run_str("fn foo(*a) -> print(a) ; foo(1, 2, 3)", "(1, 2, 3)\n"); }
    #[test] fn test_function_var_args_5() { run_str("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1)", "1 nil ()\n"); }
    #[test] fn test_function_var_args_6() { run_str("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2)", "1 2 ()\n"); }
    #[test] fn test_function_var_args_7() { run_str("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2, 3)", "1 2 (3)\n"); }
    #[test] fn test_function_var_args_8() { run_str("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2, 3, 4)", "1 2 (3, 4)\n"); }
    #[test] fn test_function_var_args_9() { run_str("fn foo(a, b?, *c) -> print(a, b, c) ; foo(1, 2, 3, 4, 5)", "1 2 (3, 4, 5)\n"); }
    #[test] fn test_function_call_with_over_u8_arguments() { run_str("sum(...range(1 + 1000)) . print", "500500\n"); }
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
    #[test] fn test_operator_functions_eval() { run_str("(+)(1, 2) . print", "3\n"); }
    #[test] fn test_operator_functions_partial_eval() { run_str("(+)(1)(2) . print", "3\n"); }
    #[test] fn test_operator_functions_compose_and_eval() { run_str("2 . (+)(1) . print", "3\n"); }
    #[test] fn test_operator_functions_compose() { run_str("1 . (2 . (+)) . print", "3\n"); }
    #[test] fn test_operator_in_expr() { run_str("(1 < 2) . print", "true\n"); }
    #[test] fn test_operator_partial_right() { run_str("((<2)(1)) . print", "true\n"); }
    #[test] fn test_operator_partial_left() { run_str("((1<)(2)) . print", "true\n"); }
    #[test] fn test_operator_partial_twice() { run_str("((<)(1)(2)) . print", "true\n"); }
    #[test] fn test_operator_as_prefix() { run_str("((<)(1, 2)) . print", "true\n"); }
    #[test] fn test_operator_partial_right_with_composition() { run_str("(1 . (<2)) . print", "true\n"); }
    #[test] fn test_operator_partial_left_with_composition() { run_str("(2 . (1<)) . print", "true\n"); }
    #[test] fn test_operator_binary_max_yes() { run_str("let a = 3 ; a max= 6; a . print", "6\n"); }
    #[test] fn test_operator_binary_max_no() { run_str("let a = 3 ; a max= 2; a . print", "3\n"); }
    #[test] fn test_operator_binary_min_yes() { run_str("let a = 3 ; a min= 1; a . print", "1\n"); }
    #[test] fn test_operator_binary_min_no() { run_str("let a = 3 ; a min= 5; a . print", "3\n"); }
    #[test] fn test_operator_dot_equals() { run_str("let x = 'hello' ; x .= sort ; x .= reduce(+) ; x . print", "ehllo\n"); }
    #[test] fn test_operator_dot_equals_operator_function() { run_str("let x = 3 ; x .= (+4) ; x . print", "7\n"); }
    #[test] fn test_operator_dot_equals_anonymous_function() { run_str("let x = 'hello' ; x .= fn(x) -> x[0] * len(x) ; x . print", "hhhhh\n"); }
    #[test] fn test_operator_in() { run_str("let f = (in) ; f(1, [1]) . print", "true\n"); }
    #[test] fn test_operator_in_partial_left() { run_str("let f = (1 in) ; f([1]) . print", "true\n"); }
    #[test] fn test_operator_in_partial_right() { run_str("let f = (in [1]) ; f(1) . print", "true\n"); }
    #[test] fn test_operator_not_in() { run_str("let f = (not in) ; f(1, []) . print", "true\n"); }
    #[test] fn test_operator_not_in_partial_left() { run_str("let f = (1 not in) ; f([]) . print", "true\n"); }
    #[test] fn test_operator_not_in_partial_right() { run_str("let f = (not in []) ; f(1) . print", "true\n"); }
    #[test] fn test_operator_is() { run_str("let f = (is) ; f(1, int) . print", "true\n"); }
    #[test] fn test_operator_is_iterable_yes() { run_str("[[], '123', set(), dict()] . all(is iterable) . print", "true\n"); }
    #[test] fn test_operator_is_iterable_no() { run_str("[true, false, nil, 123, fn() -> {}] . any(is iterable) . print", "false\n"); }
    #[test] fn test_operator_is_any_yes() { run_str("[[], '123', set(), dict(), 123, true, false, nil, fn() -> nil] . all(is any) . print", "true\n"); }
    #[test] fn test_operator_is_function_yes() { run_str("(fn() -> nil) is function . print", "true\n"); }
    #[test] fn test_operator_is_function_no() { run_str("[nil, true, 123, '123', [], set()] . any(is function) . print", "false\n"); }
    #[test] fn test_operator_is_partial_left() { run_str("let f = (1 is) ; f(int) . print", "true\n"); }
    #[test] fn test_operator_is_partial_right() { run_str("let f = (is int) ; f(1) . print", "true\n"); }
    #[test] fn test_operator_not_is() { run_str("let f = (is not) ; f(1, str) . print", "true\n"); }
    #[test] fn test_operator_not_is_partial_left() { run_str("let f = (1 is not) ; f(str) . print", "true\n"); }
    #[test] fn test_operator_not_is_partial_right() { run_str("let f = (is not str) ; f(1) . print", "true\n"); }
    #[test] fn test_operator_sub_as_unary() { run_str("(-)(3) . print", "-3\n"); }
    #[test] fn test_operator_sub_as_binary() { run_str("(-)(5, 2) . print", "3\n"); }
    #[test] fn test_operator_sub_as_partial_not_allowed() { run_str("(-3) . print", "-3\n"); }
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
    #[test] fn test_annotation_named_func_with_name() { run_str("fn par(f) -> (fn(x) -> f('hello')) ; @par fn foo(x) -> print(x) ; foo('goodbye')", "hello\n"); }
    #[test] fn test_annotation_named_func_with_expression() { run_str("fn par(a, f) -> (fn(x) -> f(a)) ; @par('hello') fn foo(x) -> print(x) ; foo('goodbye')", "hello\n"); }
    #[test] fn test_annotation_expression_func_with_name() { run_str("fn par(f) -> (fn(x) -> f('hello')) ; par(fn(x) -> print(x))('goodbye')", "hello\n"); }
    #[test] fn test_annotation_expression_func_with_expression() { run_str("fn par(a, f) -> (fn(x) -> f(a)) ; par('hello', fn(x) -> print(x))('goodbye')", "hello\n"); }
    #[test] fn test_annotation_iife() { run_str("fn iife(f) -> f() ; @iife fn foo() -> print('hello')", "hello\n"); }
    #[test] fn test_function_call_on_list() { run_str("'hello' . [0] . print", "h\n"); }
    #[test] fn test_function_compose_on_list() { run_str("[-1]('hello') . print", "o\n"); }
    #[test] fn test_slice_literal_2_no_nil() { run_str("let x = [1:2] ; x . print", "[1:2]\n"); }
    #[test] fn test_slice_literal_2_all_nil() { run_str("let x = [:] ; x . print", "[:]\n"); }
    #[test] fn test_slice_literal_3_no_nil() { run_str("let x = [1:2:3] ; x . print", "[1:2:3]\n"); }
    #[test] fn test_slice_literal_3_all_nil() { run_str("let x = [::] ; x . print", "[:]\n"); }
    #[test] fn test_slice_literal_3_last_not_nil() { run_str("let x = [::-1] ; x . print", "[::-1]\n"); }
    #[test] fn test_slice_literal_not_int() { run_str("let x = ['hello':'world'] ; x . print", "TypeError: Expected 'hello' of type 'str' to be a int\n  at: line 1 (<test>)\n\n1 | let x = ['hello':'world'] ; x . print\n2 |         ^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_slice_in_expr_1() { run_str("'1234' . [::-1] . print", "4321\n"); }
    #[test] fn test_slice_in_expr_2() { run_str("let x = [::-1] ; '1234' . x . print", "4321\n"); }
    #[test] fn test_slice_in_expr_3() { run_str("'hello the world!' . split(' ') . map([2:]) . print", "['llo', 'e', 'rld!']\n"); }
    #[test] fn test_int_operators() { run_str("print(5 - 3, 12 + 5, 3 * 9, 16 / 3)", "2 17 27 5\n"); }
    #[test] fn test_int_div_mod() { run_str("print(3 / 2, 3 / 3, -3 / 2, 10 % 3, 11 % 3, 12 % 3)", "1 1 -2 1 2 0\n"); }
    #[test] fn test_int_div_by_zero() { run_str("print(15 / 0)", "Compile Error:\n\nValueError: Expected value to be non-zero\n  at: line 1 (<test>)\n\n1 | print(15 / 0)\n2 |          ^\n"); }
    #[test] fn test_int_left_right_shift() { run_str("print(1 << 10, 16 >> 1, 16 << -1, 1 >> -10)", "1024 8 8 1024\n"); }
    #[test] fn test_int_comparisons_1() { run_str("print(1 < 3, -5 < -10, 6 > 7, 6 > 4)", "true false false true\n"); }
    #[test] fn test_int_comparisons_2() { run_str("print(1 <= 3, -5 < -10, 3 <= 3, 2 >= 2, 6 >= 7, 6 >= 4, 6 <= 6, 8 >= 8)", "true false true true false true true true\n"); }
    #[test] fn test_int_equality() { run_str("print(1 == 3, -5 == -10, 3 != 3, 2 == 2, 6 != 7)", "false false false true true\n"); }
    #[test] fn test_int_bitwise_operators() { run_str("print(0b111 & 0b100, 0b1100 | 0b1010, 0b1100 ^ 0b1010)", "4 14 6\n"); }
    #[test] fn test_int_to_hex() { run_str("1234 . hex . print", "4d2\n"); }
    #[test] fn test_int_to_bin() { run_str("1234 . bin . print", "10011010010\n"); }
    #[test] fn test_int_default_value_yes() { run_str("int('123', 567) . print", "123\n"); }
    #[test] fn test_int_default_value_no() { run_str("int('yes', 567) . print", "567\n"); }
    #[test] fn test_int_min_and_max() { run_str("[int.min, max(int)] . print", "[-4611686018427387904, 4611686018427387903]\n") }
    #[test] fn test_complex_add() { run_str("(1 + 2i) + (3 + 4j) . print", "4 + 6i\n"); }
    #[test] fn test_complex_mul() { run_str("(1 + 2i) * (3 + 4j) . print", "-5 + 10i\n"); }
    #[test] fn test_complex_str() { run_str("1 + 1i . print", "1 + 1i\n"); }
    #[test] fn test_complex_str_no_real_part() { run_str("123i . print", "123i\n"); }
    #[test] fn test_complex_typeof() { run_str("123i . typeof . print", "complex\n"); }
    #[test] fn test_complex_no_real_part_is_int() { run_str("1i * 1i . typeof . print", "int\n"); }
    #[test] fn test_complex_to_vector() { run_str("1 + 3i . vector . print", "(1, 3)\n"); }
    #[test] fn test_bool_comparisons_1() { run_str("print(false < false, false < true, true < false, true < true)", "false true false false\n"); }
    #[test] fn test_bool_comparisons_2() { run_str("print(false <= false, false >= true, true >= false, true <= true)", "true false true true\n"); }
    #[test] fn test_bool_operator_add() { run_str("true + true + false + false . print", "2\n"); }
    #[test] fn test_bool_sum() { run_str("range(10) . map(>3) . sum . print", "6\n"); }
    #[test] fn test_bool_reduce_add() { run_str("range(10) . map(>3) . reduce(+) . print", "6\n"); }
    #[test] fn test_str_empty() { run_str("'' . print", "\n"); }
    #[test] fn test_str_add() { run_str("print(('a' + 'b') + (3 + 4) + (' hello' + 3) + (' and' + true + nil))", "ab7 hello3 andtruenil\n"); }
    #[test] fn test_str_partial_left_add() { run_str("'world ' . (+'hello') . print", "world hello\n"); }
    #[test] fn test_str_partial_right_add() { run_str("' world' . ('hello'+) . print", "hello world\n"); }
    #[test] fn test_str_mul() { run_str("print('abc' * 3)", "abcabcabc\n"); }
    #[test] fn test_str_index() { run_str("'hello'[1] . print", "e\n"); }
    #[test] fn test_str_slice_start() { run_str("'hello'[1:] . print", "ello\n"); }
    #[test] fn test_str_slice_stop() { run_str("'hello'[:3] . print", "hel\n"); }
    #[test] fn test_str_slice_start_stop() { run_str("'hello'[1:3] . print", "el\n"); }
    #[test] fn test_str_operator_in_yes() { run_str("'hello' in 'hey now, hello world' . print", "true\n"); }
    #[test] fn test_str_operator_in_no() { run_str("'hello' in 'hey now, \\'ello world' . print", "false\n"); }
    #[test] fn test_str_format_with_percent_no_args() { run_str("'100 %%' % vector() . print", "100 %\n"); }
    #[test] fn test_str_format_with_one_int_arg() { run_str("'an int: %d' % (123,) . print", "an int: 123\n"); }
    #[test] fn test_str_format_with_one_neg_int_arg() { run_str("'an int: %d' % (-123,) . print", "an int: -123\n"); }
    #[test] fn test_str_format_with_one_zero_pad_int_arg() { run_str("'an int: %05d' % (123,) . print", "an int: 00123\n"); }
    #[test] fn test_str_format_with_one_zero_pad_neg_int_arg() { run_str("'an int: %05d' % (-123,) . print", "an int: -0123\n"); }
    #[test] fn test_str_format_with_one_space_pad_int_arg() { run_str("'an int: %5d' % (123,) . print", "an int:   123\n"); }
    #[test] fn test_str_format_with_one_space_pad_neg_int_arg() { run_str("'an int: %5d' % (-123,) . print", "an int:  -123\n"); }
    #[test] fn test_str_format_with_one_hex_arg() { run_str("'an int: %x' % (123,) . print", "an int: 7b\n"); }
    #[test] fn test_str_format_with_one_zero_pad_hex_arg() { run_str("'an int: %04x' % (123,) . print", "an int: 007b\n"); }
    #[test] fn test_str_format_with_one_space_pad_hex_arg() { run_str("'an int: %4x' % (123,) . print", "an int:   7b\n"); }
    #[test] fn test_str_format_with_one_bin_arg() { run_str("'an int: %b' % (123,) . print", "an int: 1111011\n"); }
    #[test] fn test_str_format_with_one_zero_pad_bin_arg() { run_str("'an int: %012b' % (123,) . print", "an int: 000001111011\n"); }
    #[test] fn test_str_format_with_one_space_pad_bin_arg() { run_str("'an int: %12b' % (123,) . print", "an int:      1111011\n"); }
    #[test] fn test_str_format_with_many_args() { run_str("'%d %s %x %b ALL THE THINGS %%!' % (10, 'fifteen', 0xff, 0b10101) . print", "10 fifteen ff 10101 ALL THE THINGS %!\n"); }
    #[test] fn test_str_format_with_solo_arg_nil() { run_str("'hello %s' % nil . print", "hello nil\n"); }
    #[test] fn test_str_format_with_solo_arg_int() { run_str("'hello %s' % 123 . print", "hello 123\n"); }
    #[test] fn test_str_format_with_solo_arg_str() { run_str("'hello %s' % 'world' . print", "hello world\n"); }
    #[test] fn test_str_format_nested_0() { run_str("'%s w%sld %s' % ('hello', 'or', '!') . print", "hello world !\n"); }
    #[test] fn test_str_format_nested_1() { run_str("'%%%s%%s%s %%s' % ('s w', 'ld') % ('hello', 'or', '!') . print", "hello world !\n"); }
    #[test] fn test_str_format_nested_2() { run_str("'%ss%%%%s%s%s%ss' % ('%'*3, '%s', ' ', '%'*2) % ('s w', 'ld') % ('hello', 'or', '!') . print", "hello world !\n"); }
    #[test] fn test_str_format_too_many_args() { run_str("'%d %d %d' % (1, 2)", "ValueError: Not enough arguments for format string\n  at: line 1 (<test>)\n\n1 | '%d %d %d' % (1, 2)\n2 |            ^\n"); }
    #[test] fn test_str_format_too_few_args() { run_str("'%d %d %d' % (1, 2, 3, 4)", "ValueError: Not all arguments consumed in format string, next: '4' of type 'int'\n  at: line 1 (<test>)\n\n1 | '%d %d %d' % (1, 2, 3, 4)\n2 |            ^\n"); }
    #[test] fn test_str_format_incorrect_character() { run_str("'%g' % (1,)", "ValueError: Invalid format character 'g' in format string\n  at: line 1 (<test>)\n\n1 | '%g' % (1,)\n2 |      ^\n"); }
    #[test] fn test_str_format_incorrect_width() { run_str("'%00' % (1,)", "ValueError: Invalid format character '0' in format string\n  at: line 1 (<test>)\n\n1 | '%00' % (1,)\n2 |       ^\n"); }
    #[test] fn test_list_empty_constructor() { run_str("list() . print", "[]\n"); }
    #[test] fn test_list_literal_empty() { run_str("[] . print", "[]\n"); }
    #[test] fn test_list_literal_len_1() { run_str("['hello'] . print", "['hello']\n"); }
    #[test] fn test_list_literal_len_2() { run_str("['hello', 'world'] . print", "['hello', 'world']\n"); }
    #[test] fn test_list_literal_unroll_at_start() { run_str("[...[1, 2, 3], 4, 5] . print", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_list_literal_unroll_at_end() { run_str("[0, ...[1, 2, 3]] . print", "[0, 1, 2, 3]\n"); }
    #[test] fn test_list_literal_unroll_once() { run_str("[...[1, 2, 3]] . print", "[1, 2, 3]\n"); }
    #[test] fn test_list_literal_unroll_multiple() { run_str("[...[1, 2, 3], ...[4, 5]] . print", "[1, 2, 3, 4, 5]\n"); }
    #[test] fn test_list_literal_unroll_multiple_and_empty() { run_str("[...[], 0, ...[1, 2, 3], ...[4, 5], ...[], 6] . print", "[0, 1, 2, 3, 4, 5, 6]\n"); }
    #[test] fn test_list_from_str() { run_str("'funny beans' . list . print", "['f', 'u', 'n', 'n', 'y', ' ', 'b', 'e', 'a', 'n', 's']\n"); }
    #[test] fn test_list_add() { run_str("[1, 2, 3] + [4, 5, 6] . print", "[1, 2, 3, 4, 5, 6]\n"); }
    #[test] fn test_list_multiply_left() { run_str("[1, 2, 3] * 3 . print", "[1, 2, 3, 1, 2, 3, 1, 2, 3]\n"); }
    #[test] fn test_list_multiply_right() { run_str("3 * [1, 2, 3] . print", "[1, 2, 3, 1, 2, 3, 1, 2, 3]\n"); }
    #[test] fn test_list_multiply_nested() { run_str("let a = [[1]] * 3; a[0][0] = 2; a . print", "[[2], [2], [2]]\n"); }
    #[test] fn test_list_operator_in_yes() { run_str("13 in [10, 11, 12, 13, 14, 15] . print", "true\n"); }
    #[test] fn test_list_operator_in_no() { run_str("3 in [10, 11, 12, 13, 14, 15] . print", "false\n"); }
    #[test] fn test_list_operator_not_in_yes() { run_str("3 not in [1, 2, 3] . print", "false\n"); }
    #[test] fn test_list_operator_not_in_no() { run_str("3 not in [1, 5, 8] . print", "true\n"); }
    #[test] fn test_list_index() { run_str("[1, 2, 3] [1] . print", "2\n"); }
    #[test] fn test_list_index_out_of_bounds() { run_str("[1, 2, 3] [3] . print", "Index '3' is out of bounds for list of length [0, 3)\n  at: line 1 (<test>)\n\n1 | [1, 2, 3] [3] . print\n2 |           ^^^\n"); }
    #[test] fn test_list_index_negative() { run_str("[1, 2, 3] [-1] . print", "3\n"); }
    #[test] fn test_list_slice_01() { run_str("[1, 2, 3, 4] [:] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_list_slice_02() { run_str("[1, 2, 3, 4] [::] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_list_slice_03() { run_str("[1, 2, 3, 4] [::1] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_list_slice_04() { run_str("[1, 2, 3, 4] [1:] . print", "[2, 3, 4]\n"); }
    #[test] fn test_list_slice_05() { run_str("[1, 2, 3, 4] [:2] . print", "[1, 2]\n"); }
    #[test] fn test_list_slice_06() { run_str("[1, 2, 3, 4] [0:] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_list_slice_07() { run_str("[1, 2, 3, 4] [:4] . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_list_slice_08() { run_str("[1, 2, 3, 4] [1:3] . print", "[2, 3]\n"); }
    #[test] fn test_list_slice_09() { run_str("[1, 2, 3, 4] [2:4] . print", "[3, 4]\n"); }
    #[test] fn test_list_slice_10() { run_str("[1, 2, 3, 4] [0:2] . print", "[1, 2]\n"); }
    #[test] fn test_list_slice_11() { run_str("[1, 2, 3, 4] [:-1] . print", "[1, 2, 3]\n"); }
    #[test] fn test_list_slice_12() { run_str("[1, 2, 3, 4] [:-2] . print", "[1, 2]\n"); }
    #[test] fn test_list_slice_13() { run_str("[1, 2, 3, 4] [-2:] . print", "[3, 4]\n"); }
    #[test] fn test_list_slice_14() { run_str("[1, 2, 3, 4] [-3:] . print", "[2, 3, 4]\n"); }
    #[test] fn test_list_slice_15() { run_str("[1, 2, 3, 4] [::2] . print", "[1, 3]\n"); }
    #[test] fn test_list_slice_16() { run_str("[1, 2, 3, 4] [::3] . print", "[1, 4]\n"); }
    #[test] fn test_list_slice_17() { run_str("[1, 2, 3, 4] [::4] . print", "[1]\n"); }
    #[test] fn test_list_slice_18() { run_str("[1, 2, 3, 4] [1::2] . print", "[2, 4]\n"); }
    #[test] fn test_list_slice_19() { run_str("[1, 2, 3, 4] [1:3:2] . print", "[2]\n"); }
    #[test] fn test_list_slice_20() { run_str("[1, 2, 3, 4] [:-1:2] . print", "[1, 3]\n"); }
    #[test] fn test_list_slice_21() { run_str("[1, 2, 3, 4] [1:-1:3] . print", "[2]\n"); }
    #[test] fn test_list_slice_22() { run_str("[1, 2, 3, 4] [::-1] . print", "[4, 3, 2, 1]\n"); }
    #[test] fn test_list_slice_23() { run_str("[1, 2, 3, 4] [1::-1] . print", "[2, 1]\n"); }
    #[test] fn test_list_slice_24() { run_str("[1, 2, 3, 4] [:2:-1] . print", "[4]\n"); }
    #[test] fn test_list_slice_25() { run_str("[1, 2, 3, 4] [3:1:-1] . print", "[4, 3]\n"); }
    #[test] fn test_list_slice_26() { run_str("[1, 2, 3, 4] [-1:-2:-1] . print", "[4]\n"); }
    #[test] fn test_list_slice_27() { run_str("[1, 2, 3, 4] [-2::-1] . print", "[3, 2, 1]\n"); }
    #[test] fn test_list_slice_28() { run_str("[1, 2, 3, 4] [:-3:-1] . print", "[4, 3]\n"); }
    #[test] fn test_list_slice_29() { run_str("[1, 2, 3, 4] [::-2] . print", "[4, 2]\n"); }
    #[test] fn test_list_slice_30() { run_str("[1, 2, 3, 4] [::-3] . print", "[4, 1]\n"); }
    #[test] fn test_list_slice_31() { run_str("[1, 2, 3, 4] [::-4] . print", "[4]\n"); }
    #[test] fn test_list_slice_32() { run_str("[1, 2, 3, 4] [-2::-2] . print", "[3, 1]\n"); }
    #[test] fn test_list_slice_33() { run_str("[1, 2, 3, 4] [-3::-2] . print", "[2]\n"); }
    #[test] fn test_list_slice_34() { run_str("[1, 2, 3, 4] [1:1] . print", "[]\n"); }
    #[test] fn test_list_slice_35() { run_str("[1, 2, 3, 4] [-1:-1] . print", "[]\n"); }
    #[test] fn test_list_slice_36() { run_str("[1, 2, 3, 4] [-1:1:] . print", "[]\n"); }
    #[test] fn test_list_slice_37() { run_str("[1, 2, 3, 4] [1:1:-1] . print", "[]\n"); }
    #[test] fn test_list_slice_38() { run_str("[1, 2, 3, 4] [-2:2:-3] . print", "[]\n"); }
    #[test] fn test_list_slice_39() { run_str("[1, 2, 3, 4] [-1:1:-1] . print", "[4, 3]\n"); }
    #[test] fn test_list_slice_40() { run_str("[1, 2, 3, 4] [1:-1:-1] . print", "[]\n"); }
    #[test] fn test_list_slice_41() { run_str("[1, 2, 3, 4] [1:10:1] . print", "[2, 3, 4]\n"); }
    #[test] fn test_list_slice_42() { run_str("[1, 2, 3, 4] [10:1:-1] . print", "[4, 3]\n"); }
    #[test] fn test_list_slice_43() { run_str("[1, 2, 3, 4] [-10:1] . print", "[1]\n"); }
    #[test] fn test_list_slice_44() { run_str("[1, 2, 3, 4] [1:-10:-1] . print", "[2, 1]\n"); }
    #[test] fn test_list_slice_45() { run_str("[1, 2, 3, 4] [::0]", "ValueError: 'step' argument cannot be zero\n  at: line 1 (<test>)\n\n1 | [1, 2, 3, 4] [::0]\n2 |              ^^^^^\n"); }
    #[test] fn test_list_slice_46() { run_str("[1, 2, 3, 4][:-1] . print", "[1, 2, 3]\n"); }
    #[test] fn test_list_slice_47() { run_str("[1, 2, 3, 4][:0] . print", "[]\n"); }
    #[test] fn test_list_slice_48() { run_str("[1, 2, 3, 4][:1] . print", "[1]\n"); }
    #[test] fn test_list_slice_49() { run_str("[1, 2, 3, 4][5:] . print", "[]\n"); }
    #[test] fn test_list_pop_empty() { run_str("let x = [] , y = x . pop ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n  at: line 1 (<test>)\n\n1 | let x = [] , y = x . pop ; (x, y) . print\n2 |                    ^^^^^\n"); }
    #[test] fn test_list_pop() { run_str("let x = [1, 2, 3] , y = x . pop ; (x, y) . print", "([1, 2], 3)\n"); }
    #[test] fn test_list_pop_front_empty() { run_str("let x = [], y = x . pop_front ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n  at: line 1 (<test>)\n\n1 | let x = [], y = x . pop_front ; (x, y) . print\n2 |                   ^^^^^^^^^^^\n"); }
    #[test] fn test_list_pop_front() { run_str("let x = [1, 2, 3], y = x . pop_front ; (x, y) . print", "([2, 3], 1)\n"); }
    #[test] fn test_list_push() { run_str("let x = [1, 2, 3] ; x . push(4) ; x . print", "[1, 2, 3, 4]\n"); }
    #[test] fn test_list_push_front() { run_str("let x = [1, 2, 3] ; x . push_front(4) ; x . print", "[4, 1, 2, 3]\n"); }
    #[test] fn test_list_insert_front() { run_str("let x = [1, 2, 3] ; x . insert(0, 4) ; x . print", "[4, 1, 2, 3]\n"); }
    #[test] fn test_list_insert_middle() { run_str("let x = [1, 2, 3] ; x . insert(1, 4) ; x . print", "[1, 4, 2, 3]\n"); }
    #[test] fn test_list_insert_end() { run_str("let x = [1, 2, 3] ; x . insert(2, 4) ; x . print", "[1, 2, 4, 3]\n"); }
    #[test] fn test_list_insert_out_of_bounds() { run_str("let x = [1, 2, 3] ; x . insert(4, 4) ; x . print", "Index '4' is out of bounds for list of length [0, 3)\n  at: line 1 (<test>)\n\n1 | let x = [1, 2, 3] ; x . insert(4, 4) ; x . print\n2 |                       ^^^^^^^^^^^^^^\n"); }
    #[test] fn test_list_remove_front() { run_str("let x = [1, 2, 3] , y = x . remove(0) ; (x, y) . print", "([2, 3], 1)\n"); }
    #[test] fn test_list_remove_middle() { run_str("let x = [1, 2, 3] , y = x . remove(1) ; (x, y) . print", "([1, 3], 2)\n"); }
    #[test] fn test_list_remove_end() { run_str("let x = [1, 2, 3] , y = x . remove(2) ; (x, y) . print", "([1, 2], 3)\n"); }
    #[test] fn test_list_clear() { run_str("let x = [1, 2, 3] ; x . clear ; x . print", "[]\n"); }
    #[test] fn test_list_peek() { run_str("let x = [1, 2, 3], y = x . peek ; (x, y) . print", "([1, 2, 3], 1)\n"); }
    #[test] fn test_list_str() { run_str("[1, 2, '3'] . print", "[1, 2, '3']\n"); }
    #[test] fn test_list_repr() { run_str("['1', 2, '3'] . repr . print", "['1', 2, '3']\n"); }
    #[test] fn test_list_recursive_repr() { run_str("let x = [] ; x.push(x) ; x.print", "[[...]]\n"); }
    #[test] fn test_list_recursive_knot_repr() { run_str("let x = [] ; let y = [x] ; x.push(y) ; x.print", "[[[...]]]\n"); }
    #[test] fn test_list_recursive_complex_repr() { run_str("struct S(x) ; let x = [S(nil)] ; x[0]->x = [S(x)] ; x.print", "[S(x=[S(x=[...])])]\n"); }
    #[test] fn test_vector_empty_constructor() { run_str("vector() . print", "()\n"); }
    #[test] fn test_vector_empty_iterable_constructor() { run_str("vector([]) . print", "()\n"); }
    #[test] fn test_vector_iterable_constructor() { run_str("vector([1, 2, 3]) . print", "(1, 2, 3)\n"); }
    #[test] fn test_vector_multiple_constructor() { run_str("vector(1, 2, 3) . print", "(1, 2, 3)\n"); }
    #[test] fn test_vector_literal_single() { run_str("(1,) . print", "(1)\n"); }
    #[test] fn test_vector_literal_multiple() { run_str("(1,2,3) . print", "(1, 2, 3)\n"); }
    #[test] fn test_vector_literal_multiple_trailing_comma() { run_str("(1,2,3,) . print", "(1, 2, 3)\n"); }
    #[test] fn test_vector_literal_unroll_at_start() { run_str("(...(1, 2, 3), 4, 5) . print", "(1, 2, 3, 4, 5)\n"); }
    #[test] fn test_vector_literal_unroll_at_end() { run_str("(0, ...(1, 2, 3)) . print", "(0, 1, 2, 3)\n"); }
    #[test] fn test_vector_literal_unroll_once() { run_str("(...(1, 2, 3)) . print", "(1, 2, 3)\n"); }
    #[test] fn test_vector_literal_unroll_multiple() { run_str("(...(1, 2, 3), ...(4, 5)) . print", "(1, 2, 3, 4, 5)\n"); }
    #[test] fn test_vector_literal_unroll_multiple_and_empty() { run_str("(...vector(), 0, ...(1, 2, 3), ...(4, 5), ...vector(), 6) . print", "(0, 1, 2, 3, 4, 5, 6)\n"); }
    #[test] fn test_vector() { run_str("vector(1, 2, 3) . print", "(1, 2, 3)\n"); }
    #[test] fn test_vector_add() { run_str("vector(1, 2, 3) + vector(6, 3, 2) . print", "(7, 5, 5)\n"); }
    #[test] fn test_vector_add_constant() { run_str("vector(1, 2, 3) + 3 . print", "(4, 5, 6)\n"); }
    #[test] fn test_set_empty_constructor() { run_str("set() . print", "{}\n"); }
    #[test] fn test_vector_array_assign() { run_str("let x = (1, 2, 3) ; x[0] = 3 ; x . print", "(3, 2, 3)\n"); }
    #[test] fn test_vector_recursive_repr() { run_str("let x = (nil,) ; x[0] = x ; x.print", "((...))\n"); }
    #[test] fn test_set_literal_empty() { run_str("{} is set . print ; {} . print", "true\n{}\n"); }
    #[test] fn test_set_literal_single() { run_str("{'hello'} . print", "{'hello'}\n"); }
    #[test] fn test_set_literal_multiple() { run_str("{1, 2, 3, 4} . print", "{1, 2, 3, 4}\n"); }
    #[test] fn test_set_literal_unroll_at_start() { run_str("{...{1, 2, 3}, 4, 5} . print", "{1, 2, 3, 4, 5}\n"); }
    #[test] fn test_set_literal_unroll_at_end() { run_str("{0, ...{1, 2, 3}} . print", "{0, 1, 2, 3}\n"); }
    #[test] fn test_set_literal_unroll_once() { run_str("{...{1, 2, 3}} . print", "{1, 2, 3}\n"); }
    #[test] fn test_set_literal_unroll_multiple() { run_str("{...{1, 2, 3}, ...{4, 5}} . print", "{1, 2, 3, 4, 5}\n"); }
    #[test] fn test_set_literal_unroll_multiple_and_empty() { run_str("{...{}, 0, ...{1, 2, 3}, ...{4, 5}, ...set(), 6} . print", "{0, 1, 2, 3, 4, 5, 6}\n"); }
    #[test] fn test_set_literal_unroll_from_dict_implicit() { run_str("{...{(1, 1), (2, 2)}} . print", "{(1, 1), (2, 2)}\n"); }
    #[test] fn test_set_literal_unroll_from_dict_explicit() { run_str("{...{(1, 1), (2, 2)}, 3} . print", "{(1, 1), (2, 2), 3}\n"); }
    #[test] fn test_set_from_str() { run_str("'funny beans' . set . print", "{'f', 'u', 'n', 'y', ' ', 'b', 'e', 'a', 's'}\n"); }
    #[test] fn test_set_pop_empty() { run_str("let x = set() , y = x . pop ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n  at: line 1 (<test>)\n\n1 | let x = set() , y = x . pop ; (x, y) . print\n2 |                       ^^^^^\n"); }
    #[test] fn test_set_pop() { run_str("let x = {1, 2, 3} , y = x . pop ; (x, y) . print", "({1, 2}, 3)\n"); }
    #[test] fn test_set_push() { run_str("let x = {1, 2, 3} ; x . push(4) ; x . print", "{1, 2, 3, 4}\n"); }
    #[test] fn test_set_remove_yes() { run_str("let x = {1, 2, 3}, y = x . remove(2) ; (x, y) . print", "({1, 3}, true)\n"); }
    #[test] fn test_set_remove_no() { run_str("let x = {1, 2, 3}, y = x . remove(5) ; (x, y) . print", "({1, 2, 3}, false)\n"); }
    #[test] fn test_set_clear() { run_str("let x = {1, 2, 3} ; x . clear ; x . print", "{}\n"); }
    #[test] fn test_set_peek() { run_str("let x = {1, 2, 3}, y = x . peek ; (x, y) . print", "({1, 2, 3}, 1)\n"); }
    #[test] fn test_set_insert_self() { run_str("let x = set() ; x.push(x)", "ValueError: Cannot create recursive hash based collection from '{{...}}' of type 'set'\n  at: line 1 (<test>)\n\n1 | let x = set() ; x.push(x)\n2 |                  ^^^^^^^^\n"); }
    #[test] fn test_set_indirect_insert_self() { run_str("let x = set() ; x.push([x])", "ValueError: Cannot create recursive hash based collection from '{[{...}]}' of type 'set'\n  at: line 1 (<test>)\n\n1 | let x = set() ; x.push([x])\n2 |                  ^^^^^^^^^^\n"); }
    #[test] fn test_set_recursive_repr() { run_str("let x = set() ; x.push(x) ; x.print", "ValueError: Cannot create recursive hash based collection from '{{...}}' of type 'set'\n  at: line 1 (<test>)\n\n1 | let x = set() ; x.push(x) ; x.print\n2 |                  ^^^^^^^^\n"); }
    #[test] fn test_dict_empty_constructor() { run_str("dict() . print", "{}\n"); }
    #[test] fn test_dict_literal_single() { run_str("{'hello': 'world'} . print", "{'hello': 'world'}\n"); }
    #[test] fn test_dict_literal_multiple() { run_str("{1: 'a', 2: 'b', 3: 'c'} . print", "{1: 'a', 2: 'b', 3: 'c'}\n"); }
    #[test] fn test_dict_literal_unroll_at_start() { run_str("{...{1: 1, 2: 2}, 3: 3} . print", "{1: 1, 2: 2, 3: 3}\n"); }
    #[test] fn test_dict_literal_unroll_at_end() { run_str("{0: 0, ...{1: 1, 2: 2}} . print", "{0: 0, 1: 1, 2: 2}\n"); }
    #[test] fn test_dict_literal_unroll_multiple() { run_str("{...{1: 1, 2: 2}, 3: 3, ...{4: 4}} . print", "{1: 1, 2: 2, 3: 3, 4: 4}\n"); }
    #[test] fn test_dict_literal_unroll_multiple_and_empty() { run_str("{...{}, 0: 0, ...{1: 1, 2: 2, 3: 3}, ...{4: 4, 5: 5}, ...set(), ...dict(), 6: 6} . print", "{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}\n"); }
    #[test] fn test_dict_literal_unroll_from_set() { run_str("{...{(1, 1), (2, 2)}, 3: 3} . print", "{1: 1, 2: 2, 3: 3}\n"); }
    #[test] fn test_dict_literal_unroll_from_not_pair() { run_str("{...{1, 2, 3}, 4: 4}", "ValueError: Cannot collect key-value pair '1' of type 'int' into a dict\n  at: line 1 (<test>)\n\n1 | {...{1, 2, 3}, 4: 4}\n2 |  ^^^\n"); }
    #[test] fn test_dict_get_and_set() { run_str("let d = dict() ; d['hi'] = 'yes' ; d['hi'] . print", "yes\n"); }
    #[test] fn test_dict_get_when_not_present() { run_str("let d = dict() ; d['hello']", "ValueError: Key 'hello' of type 'str' not found in dictionary\n  at: line 1 (<test>)\n\n1 | let d = dict() ; d['hello']\n2 |                   ^^^^^^^^^\n"); }
    #[test] fn test_dict_get_when_not_present_with_default() { run_str("let d = dict() . default('haha') ; d['hello'] . print", "haha\n"); }
    #[test] fn test_dict_keys() { run_str("[[1, 'a'], [2, 'b'], [3, 'c']] . dict . keys . print", "{1, 2, 3}\n"); }
    #[test] fn test_dict_values() { run_str("[[1, 'a'], [2, 'b'], [3, 'c']] . dict . values . print", "['a', 'b', 'c']\n"); }
    #[test] fn test_dict_pop_empty() { run_str("let x = dict() , y = x . pop ; (x, y) . print", "ValueError: Expected value to be a non empty iterable\n  at: line 1 (<test>)\n\n1 | let x = dict() , y = x . pop ; (x, y) . print\n2 |                        ^^^^^\n"); }
    #[test] fn test_dict_pop() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'} , y = x . pop ; (x, y) . print", "({1: 'a', 2: 'b'}, (3, 'c'))\n"); }
    #[test] fn test_dict_insert() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'} ; x . insert(4, 'd') ; x . print", "{1: 'a', 2: 'b', 3: 'c', 4: 'd'}\n"); }
    #[test] fn test_dict_remove_yes() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . remove(2) ; (x, y) . print", "({1: 'a', 3: 'c'}, true)\n"); }
    #[test] fn test_dict_remove_no() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . remove(5) ; (x, y) . print", "({1: 'a', 2: 'b', 3: 'c'}, false)\n"); }
    #[test] fn test_dict_clear() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'} ; x . clear ; x . print", "{}\n"); }
    #[test] fn test_dict_from_enumerate() { run_str("'hey' . enumerate . dict . print", "{0: 'h', 1: 'e', 2: 'y'}\n"); }
    #[test] fn test_dict_peek() { run_str("let x = {1: 'a', 2: 'b', 3: 'c'}, y = x . peek ; (x, y) . print", "({1: 'a', 2: 'b', 3: 'c'}, (1, 'a'))\n"); }
    #[test] fn test_dict_default_with_query() { run_str("let d = dict() . default(3) ; d[0] ; d.print", "{0: 3}\n"); }
    #[test] fn test_dict_default_with_function() { run_str("let d = dict() . default(list) ; d[0].push(2) ; d[1].push(3) ; d.print", "{0: [2], 1: [3]}\n"); }
    #[test] fn test_dict_default_with_mutable_default() { run_str("let d = dict() . default([]) ; d[0].push(2) ; d[1].push(3) ; d.print", "{0: [2, 3], 1: [2, 3]}\n"); }
    #[test] fn test_dict_default_with_self_entry() { run_str("let d ; d = dict() . default(fn() { d['count'] += 1 ; d['hello'] = 'special' ; 'otherwise' }) ; d['count'] = 0 ; d['hello'] ; d['world'] ; d.print", "{'count': 2, 'hello': 'special', 'world': 'otherwise'}\n"); }
    #[test] fn test_dict_increment() { run_str("let d = dict() . default(fn() -> 3) ; d[0] . print ; d[0] += 1 ; d . print ; d[0] += 1 ; d . print", "3\n{0: 4}\n{0: 5}\n"); }
    #[test] fn test_dict_insert_self_as_key() { run_str("let x = dict() ; x[x] = 'yes'", "ValueError: Cannot create recursive hash based collection from '{{...}: 'yes'}' of type 'dict'\n  at: line 1 (<test>)\n\n1 | let x = dict() ; x[x] = 'yes'\n2 |                       ^\n"); }
    #[test] fn test_dict_insert_self_as_value() { run_str("let x = dict() ; x['yes'] = x", ""); }
    #[test] fn test_dict_recursive_key_index() { run_str("let x = dict() ; x[x] = 'yes' ; x.print", "ValueError: Cannot create recursive hash based collection from '{{...}: 'yes'}' of type 'dict'\n  at: line 1 (<test>)\n\n1 | let x = dict() ; x[x] = 'yes' ; x.print\n2 |                       ^\n"); }
    #[test] fn test_dict_recursive_key_insert() { run_str("let x = dict() ; x.insert(x, 'yes') ; x.print", "ValueError: Cannot create recursive hash based collection from '{{...}: 'yes'}' of type 'dict'\n  at: line 1 (<test>)\n\n1 | let x = dict() ; x.insert(x, 'yes') ; x.print\n2 |                   ^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_dict_recursive_value_repr() { run_str("let x = dict() ; x['yes'] = x ; x.print", "{'yes': {...}}\n"); }
    #[test] fn test_heap_empty_constructor() { run_str("heap() . print", "[]\n"); }
    #[test] fn test_heap_from_list() { run_str("let h = [1, 7, 3, 2, 7, 6] . heap; h . print", "[1, 2, 3, 7, 7, 6]\n"); }
    #[test] fn test_heap_pop() { run_str("let h = [1, 7, 3, 2, 7, 6] . heap; [h.pop, h.pop, h.pop] . print", "[1, 2, 3]\n"); }
    #[test] fn test_heap_push() { run_str("let h = [1, 7, 3, 2, 7, 6] . heap; h.push(3); h.push(-1); h.push(16); h . print", "[-1, 1, 3, 2, 7, 6, 3, 7, 16]\n"); }
    #[test] fn test_heap_recursive_repr() { run_str("let x = heap() ; x.push(x) ; x.print", "[[...]]\n"); }
    #[test] fn test_print_hello_world() { run_str("print('hello world!')", "hello world!\n"); }
    #[test] fn test_print_empty() { run_str("print()", "\n"); }
    #[test] fn test_print_strings() { run_str("print('first', 'second', 'third')", "first second third\n"); }
    #[test] fn test_print_other_things() { run_str("print(nil, -1, 1, true, false, 'test', print)", "nil -1 1 true false test print\n"); }
    #[test] fn test_print_unary_operators() { run_str("print(-1, --1, ---1, !3, !!3, !true, !!true)", "-1 1 -1 -4 3 false true\n"); }
    #[test] fn test_exit_in_expression() { run_str("'this will not print' + exit . print", ""); }
    #[test] fn test_exit_in_ternary() { run_str("print(if 3 > 2 then exit else 'hello')", ""); }
    #[test] fn test_assert_pass() { run_str("assert [1, 2] . len . (==2) ; print('yes!')", "yes!\n")}
    #[test] fn test_assert_pass_with_no_message() { run_str("assert [1, 2] .len . (==2) : print('should not show') ; print('should show')", "should show\n"); }
    #[test] fn test_assert_fail() { run_str("assert 1 + 2 != 3", "Assertion Failed: nil\n  at: line 1 (<test>)\n\n1 | assert 1 + 2 != 3\n2 |        ^^^^^^^^^^\n"); }
    #[test] fn test_assert_fail_with_message() { run_str("assert 'here' in 'the goose is gone' : 'goose issues are afoot'", "Assertion Failed: goose issues are afoot\n  at: line 1 (<test>)\n\n1 | assert 'here' in 'the goose is gone' : 'goose issues are afoot'\n2 |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_assert_messages_are_lazy() { run_str("assert true : exit ; print('should reach here')", "should reach here\n"); }
    #[test] fn test_len_list() { run_str("[1, 2, 3] . len . print", "3\n"); }
    #[test] fn test_len_str() { run_str("'12345' . len . print", "5\n"); }
    #[test] fn test_sum_list() { run_str("[1, 2, 3, 4] . sum . print", "10\n"); }
    #[test] fn test_sum_values() { run_str("sum(1, 3, 5, 7) . print", "16\n"); }
    #[test] fn test_sum_no_arg() { run_str("sum()", "Incorrect number of arguments for fn sum(...), got 0\n  at: line 1 (<test>)\n\n1 | sum()\n2 |    ^^\n"); }
    #[test] fn test_sum_empty_list() { run_str("[] . sum . print", "0\n"); }
    #[test] fn test_map() { run_str("[1, 2, 3] . map(str) . repr . print", "['1', '2', '3']\n") }
    #[test] fn test_map_lambda() { run_str("[-1, 2, -3] . map(fn(x) -> x . abs) . print", "[1, 2, 3]\n") }
    #[test] fn test_filter() { run_str("[2, 3, 4, 5, 6] . filter (>3) . print", "[4, 5, 6]\n") }
    #[test] fn test_filter_lambda() { run_str("[2, 3, 4, 5, 6] . filter (fn(x) -> x % 2 == 0) . print", "[2, 4, 6]\n") }
    #[test] fn test_reduce_with_operator() { run_str("[1, 2, 3, 4, 5, 6] . reduce (*) . print", "720\n"); }
    #[test] fn test_reduce_with_function() { run_str("[1, 2, 3, 4, 5, 6] . reduce (fn(a, b) -> a * b) . print", "720\n"); }
    #[test] fn test_reduce_with_unary_operator() { run_str("[1, 2, 3] . reduce (!) . print", "Incorrect number of arguments for fn (!)(x), got 2\n  at: line 1 (<test>)\n\n1 | [1, 2, 3] . reduce (!) . print\n2 |           ^^^^^^^^^^^^\n"); }
    #[test] fn test_reduce_with_sum() { run_str("[1, 2, 3, 4, 5, 6] . reduce (sum) . print", "21\n"); }
    #[test] fn test_reduce_with_empty() { run_str("[] . reduce(+) . print", "ValueError: Expected value to be a non empty iterable\n  at: line 1 (<test>)\n\n1 | [] . reduce(+) . print\n2 |    ^^^^^^^^^^^\n"); }
    #[test] fn test_sorted() { run_str("[6, 2, 3, 7, 2, 1] . sort . print", "[1, 2, 2, 3, 6, 7]\n"); }
    #[test] fn test_sorted_with_set_of_str() { run_str("'funny' . set . sort . print", "['f', 'n', 'u', 'y']\n"); }
    #[test] fn test_group_by_int_negative() { run_str("group_by(-1, [1, 2, 3, 4]) . print", "ValueError: Expected value '-1: int' to be positive\n  at: line 1 (<test>)\n\n1 | group_by(-1, [1, 2, 3, 4]) . print\n2 |         ^^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_group_by_int_zero() { run_str("group_by(0, [1, 2, 3, 4]) . print", "ValueError: Expected value '0: int' to be positive\n  at: line 1 (<test>)\n\n1 | group_by(0, [1, 2, 3, 4]) . print\n2 |         ^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_group_by_int_by_one() { run_str("group_by(1, [1, 2, 3, 4]) . print", "[(1), (2), (3), (4)]\n"); }
    #[test] fn test_group_by_int_by_three() { run_str("[1, 2, 3, 4, 5, 6] . group_by(3) . print", "[(1, 2, 3), (4, 5, 6)]\n"); }
    #[test] fn test_group_by_int_by_one_empty_iterable() { run_str("[] . group_by(1) . print", "[]\n"); }
    #[test] fn test_group_by_int_by_three_empty_iterable() { run_str("[] . group_by(3) . print", "[]\n"); }
    #[test] fn test_group_by_int_by_three_with_remainder() { run_str("[1, 2, 3, 4] . group_by(3) . print", "[(1, 2, 3), (4)]\n"); }
    #[test] fn test_group_by_int_by_three_not_enough() { run_str("[1, 2] . group_by(3) . print", "[(1, 2)]\n"); }
    #[test] fn test_group_by_function_empty_iterable() { run_str("[] . group_by(fn(x) -> nil) . print", "{}\n"); }
    #[test] fn test_group_by_function_all_same_keys() { run_str("[1, 2, 3, 4] . group_by(fn(x) -> nil) . print", "{nil: (1, 2, 3, 4)}\n"); }
    #[test] fn test_group_by_function_all_different_keys() { run_str("[1, 2, 3, 4] . group_by(fn(x) -> x) . print", "{1: (1), 2: (2), 3: (3), 4: (4)}\n"); }
    #[test] fn test_group_by_function_remainder_by_three() { run_str("[1, 2, 3, 4, 5] . group_by(%3) . print", "{1: (1, 4), 2: (2, 5), 0: (3)}\n"); }
    #[test] fn test_reverse() { run_str("[8, 1, 2, 6, 3, 2, 3] . reverse . print", "[3, 2, 3, 6, 2, 1, 8]\n"); }
    #[test] fn test_range_1() { run_str("range(3) . list . print", "[0, 1, 2]\n"); }
    #[test] fn test_range_2() { run_str("range(3, 7) . list . print", "[3, 4, 5, 6]\n"); }
    #[test] fn test_range_3() { run_str("range(1, 9, 3) . list . print", "[1, 4, 7]\n"); }
    #[test] fn test_range_4() { run_str("range(6, 3) . list . print", "[]\n"); }
    #[test] fn test_range_5() { run_str("range(10, 4, -2) . list . print", "[10, 8, 6]\n"); }
    #[test] fn test_range_6() { run_str("range(0, 20, -1) . list . print", "[]\n"); }
    #[test] fn test_range_7() { run_str("range(10, 0, 3) . list . print", "[]\n"); }
    #[test] fn test_range_8() { run_str("range(1, 1, 1) . list . print", "[]\n"); }
    #[test] fn test_range_9() { run_str("range(1, 1, 0) . list . print", "ValueError: 'step' argument cannot be zero\n  at: line 1 (<test>)\n\n1 | range(1, 1, 0) . list . print\n2 |      ^^^^^^^^^\n"); }
    #[test] fn test_range_operator_in_yes() { run_str("13 in range(10, 15) . print", "true\n"); }
    #[test] fn test_range_operator_in_no() { run_str("3 in range(10, 15) . print", "false\n"); }
    #[test] fn test_enumerate_1() { run_str("[] . enumerate . list . print", "[]\n"); }
    #[test] fn test_enumerate_2() { run_str("[1, 2, 3] . enumerate . list . print", "[(0, 1), (1, 2), (2, 3)]\n"); }
    #[test] fn test_enumerate_3() { run_str("'foobar' . enumerate . list . print", "[(0, 'f'), (1, 'o'), (2, 'o'), (3, 'b'), (4, 'a'), (5, 'r')]\n"); }
    #[test] fn test_sqrt() { run_str("[0, 1, 4, 9, 25, 3, 6, 8, 13] . map(sqrt) . print", "[0, 1, 2, 3, 5, 1, 2, 2, 3]\n"); }
    #[test] fn test_sqrt_very_large() { run_str("[1 << 61, (1 << 61) + 1, (1 << 61) - 1] . map(sqrt) . print", "[1518500249, 1518500249, 1518500249]\n"); }
    #[test] fn test_gcd() { run_str("gcd(12, 8) . print", "4\n"); }
    #[test] fn test_gcd_iter() { run_str("[12, 18, 16] . gcd . print", "2\n"); }
    #[test] fn test_lcm() { run_str("lcm(9, 7) . print", "63\n"); }
    #[test] fn test_lcm_iter() { run_str("[12, 10, 18] . lcm . print", "180\n"); }
    #[test] fn test_flat_map_identity() { run_str("['hi', 'bob'] . flat_map(fn(i) -> i) . print", "['h', 'i', 'b', 'o', 'b']\n"); }
    #[test] fn test_flat_map_with_func() { run_str("['hello', 'bob'] . flat_map(fn(i) -> i[2:]) . print", "['l', 'l', 'o', 'b']\n"); }
    #[test] fn test_concat() { run_str("[[], [1], [2, 3], [4, 5, 6], [7, 8, 9, 0]] . concat . print", "[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]\n"); }
    #[test] fn test_zip() { run_str("zip([1, 2, 3, 4, 5], 'hello') . print", "[(1, 'h'), (2, 'e'), (3, 'l'), (4, 'l'), (5, 'o')]\n"); }
    #[test] fn test_zip_with_empty() { run_str("zip('hello', []) . print", "[]\n"); }
    #[test] fn test_zip_with_longer_last() { run_str("zip('hi', 'hello', 'hello the world!') . print", "[('h', 'h', 'h'), ('i', 'e', 'e')]\n"); }
    #[test] fn test_zip_with_longer_first() { run_str("zip('hello the world!', 'hello', 'hi') . print", "[('h', 'h', 'h'), ('e', 'e', 'i')]\n"); }
    #[test] fn test_zip_of_list() { run_str("[[1, 2, 3], [4, 5, 6], [7, 8, 9]] . zip . print", "[(1, 4, 7), (2, 5, 8), (3, 6, 9)]\n"); }
    #[test] fn test_permutations_empty() { run_str("[] . permutations(3) . print", "[]\n"); }
    #[test] fn test_permutations_n_larger_than_size() { run_str("[1, 2, 3] . permutations(5) . print", "[]\n"); }
    #[test] fn test_permutations() { run_str("[1, 2, 3] . permutations(2) . print", "[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]\n"); }
    #[test] fn test_combinations_empty() { run_str("[] . combinations(3) . print", "[]\n"); }
    #[test] fn test_combinations_n_larger_than_size() { run_str("[1, 2, 3] . combinations(5) . print", "[]\n"); }
    #[test] fn test_combinations() { run_str("[1, 2, 3] . combinations(2) . print", "[(1, 2), (1, 3), (2, 3)]\n"); }
    #[test] fn test_replace_regex_1() { run_str("'apples and bananas' . replace('[abe]+', 'o') . print", "opplos ond ononos\n"); }
    #[test] fn test_replace_regex_2() { run_str("'[a] [b] [c] [d]' . replace('[ac]', '$0$0') . print", "[aa] [b] [cc] [d]\n"); }
    #[test] fn test_replace_regex_with_function() { run_str("'apples and bananas' . replace('apples', fn((c, *_)) -> c . to_upper) . print", "APPLES and bananas\n"); }
    #[test] fn test_replace_regex_with_wrong_function() { run_str("'apples and bananas' . replace('apples', argv) . print", "Incorrect number of arguments for fn argv(), got 1\n  at: line 1 (<test>)\n\n1 | 'apples and bananas' . replace('apples', argv) . print\n2 |                      ^^^^^^^^^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_replace_regex_with_capture_group() { run_str("'apples and bananas' . replace('([a-z])([a-z]+)', 'yes') . print", "yes yes yes\n"); }
    #[test] fn test_replace_regex_with_capture_group_function() { run_str("'apples and bananas' . replace('([a-z])([a-z]+)', fn((_, a, b)) -> to_upper(a) + b) . print", "Apples And Bananas\n"); }
    #[test] fn test_replace_regex_implicit_newline() { run_str("'first\nsecond\nthird\nfourth' . replace('\\n', ', ') . print", "first, second, third, fourth\n"); }
    #[test] fn test_replace_regex_explicit_newline() { run_str("'first\nsecond\nthird\nfourth' . replace('\n', ', ') . print", "first, second, third, fourth\n"); }
    #[test] fn test_search_regex_match_all_yes() { run_str("'test' . search('test') . print", "[('test')]\n"); }
    #[test] fn test_search_regex_match_all_no() { run_str("'test' . search('nope') . print", "[]\n"); }
    #[test] fn test_search_regex_match_partial_yes() { run_str("'any and nope and nothing' . search('nope') . print", "[('nope')]\n"); }
    #[test] fn test_search_regex_match_partial_no() { run_str("'any and nope and nothing' . search('some') . print", "[]\n"); }
    #[test] fn test_search_regex_match_partial_no_start() { run_str("'any and nope and nothing' . search('^some') . print", "[]\n"); }
    #[test] fn test_search_regex_match_partial_no_end() { run_str("'any and nope and nothing' . search('some$') . print", "[]\n"); }
    #[test] fn test_search_regex_match_partial_no_start_and_end() { run_str("'any and nope and nothing' . search('^some$') . print", "[]\n"); }
    #[test] fn test_search_regex_capture_group_match_none() { run_str("'some WORDS with CAPITAL letters' . search('[A-Z]([a-z]+)') . print", "[]\n"); }
    #[test] fn test_search_regex_capture_group_match_one() { run_str("'some WORDS with Capital letters' . search('[A-Z]([a-z]+)') . print", "[('Capital', 'apital')]\n"); }
    #[test] fn test_search_regex_capture_group_match_some() { run_str("'some Words With Capital letters' . search('[A-Z]([a-z]+)') . print", "[('Words', 'ords'), ('With', 'ith'), ('Capital', 'apital')]\n"); }
    #[test] fn test_search_regex_many_capture_groups_match_none() { run_str("'some WORDS with CAPITAL letters' . search('([A-Z])([a-z]+)') . print", "[]\n"); }
    #[test] fn test_search_regex_many_capture_groups_match_one() { run_str("'some WORDS with Capital letters' . search('([A-Z])[a-z]([a-z]+)') . print", "[('Capital', 'C', 'pital')]\n"); }
    #[test] fn test_search_regex_many_capture_groups_match_some() { run_str("'some Words With Capital letters' . search('([A-Z])[a-z]([a-z]+)') . print", "[('Words', 'W', 'rds'), ('With', 'W', 'th'), ('Capital', 'C', 'pital')]\n"); }
    #[test] fn test_search_regex_cannot_compile() { run_str("'test' . search('missing close bracket lol ( this one') . print", "ValueError: Cannot compile regex 'missing close bracket lol ( this one'\n            Parsing error at position 36: Opening parenthesis without closing parenthesis\n  at: line 1 (<test>)\n\n1 | 'test' . search('missing close bracket lol ( this one') . print\n2 |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_split_regex_empty_str() { run_str("'abc' . split('') . print", "['a', 'b', 'c']\n"); }
    #[test] fn test_split_regex_space() { run_str("'a b c' . split(' ') . print", "['a', 'b', 'c']\n"); }
    #[test] fn test_split_regex_space_duplicates() { run_str("' a  b   c' . split(' ') . print", "['', 'a', '', 'b', '', '', 'c']\n"); }
    #[test] fn test_split_regex_space_any_whitespace() { run_str("' a  b   c' . split(' +') . print", "['', 'a', 'b', 'c']\n"); }
    #[test] fn test_split_regex_space_any_with_trim() { run_str("' \nabc  \rabc \\r\\n  abc \\t  \t  \t' . trim . split('\\s+') . print", "['abc', 'abc', 'abc']\n"); }
    #[test] fn test_split_regex_on_substring() { run_str("'the horse escaped the barn' . split('the') . print", "['', ' horse escaped ', ' barn']\n"); }
    #[test] fn test_split_regex_on_substring_with_or() { run_str("'the horse escaped the barn' . split('(the| )') . print", "['', '', 'horse', 'escaped', '', '', 'barn']\n"); }
    #[test] fn test_split_regex_on_substring_with_wildcard() { run_str("'the horse escaped the barn' . split(' *e *') . print", "['th', 'hors', '', 'scap', 'd th', 'barn']\n"); }
    #[test] fn test_join_empty() { run_str("[] . join('test') . print", "\n"); }
    #[test] fn test_join_single() { run_str("['apples'] . join('test') . print", "apples\n"); }
    #[test] fn test_join_strings() { run_str("'test' . join(' ') . print", "t e s t\n"); }
    #[test] fn test_join_ints() { run_str("[1, 3, 5, 7, 9] . join('') . print", "13579\n"); }
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
    #[test] fn test_min_by_key() { run_str("[[1, 5], [2, 3], [6, 4]] . min_by(fn(i) -> i[1]) . print", "[2, 3]\n"); }
    #[test] fn test_min_by_cmp() { run_str("[[1, 5], [2, 3], [6, 4]] . min_by(fn(a, b) -> a[1] - b[1]) . print", "[2, 3]\n"); }
    #[test] fn test_min_by_wrong_fn() { run_str("[[1, 5], [2, 3], [6, 4]] . min_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n  at: line 1 (<test>)\n\n1 | [[1, 5], [2, 3], [6, 4]] . min_by(fn() -> 1) . print\n2 |                          ^^^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_max_by_key() { run_str("[[1, 5], [2, 3], [6, 4]] . max_by(fn(i) -> i[1]) . print", "[1, 5]\n"); }
    #[test] fn test_max_by_cmp() { run_str("[[1, 5], [2, 3], [6, 4]] . max_by(fn(a, b) -> a[1] - b[1]) . print", "[1, 5]\n"); }
    #[test] fn test_max_by_wrong_fn() { run_str("[[1, 5], [2, 3], [6, 4]] . max_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n  at: line 1 (<test>)\n\n1 | [[1, 5], [2, 3], [6, 4]] . max_by(fn() -> 1) . print\n2 |                          ^^^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_sort_by_key() { run_str("[[1, 5], [2, 3], [6, 4]] . sort_by(fn(i) -> i[1]) . print", "[[2, 3], [6, 4], [1, 5]]\n"); }
    #[test] fn test_sort_by_cmp() { run_str("[[1, 5], [2, 3], [6, 4]] . sort_by(fn(a, b) -> a[1] - b[1]) . print", "[[2, 3], [6, 4], [1, 5]]\n"); }
    #[test] fn test_sort_by_wrong_fn() { run_str("[[1, 5], [2, 3], [6, 4]] . sort_by(fn() -> 1) . print", "TypeError: Expected '_' of type 'function' to be a '<A, B> fn key(A) -> B' or '<A> cmp(A, A) -> int' function\n  at: line 1 (<test>)\n\n1 | [[1, 5], [2, 3], [6, 4]] . sort_by(fn() -> 1) . print\n2 |                          ^^^^^^^^^^^^^^^^^^^^\n"); }
    #[test] fn test_ord() { run_str("'a' . ord . print", "97\n"); }
    #[test] fn test_char() { run_str("97 . char . repr . print", "'a'\n"); }
    #[test] fn test_eval_nil() { run_str("'nil' . eval . print", "nil\n"); }
    #[test] fn test_eval_bool() { run_str("'true' . eval . print", "true\n"); }
    #[test] fn test_eval_int_expression() { run_str("'3 + 4' . eval . print", "7\n"); }
    #[test] fn test_eval_zero_equals_zero() { run_str("'0==0' . eval . print", "true\n"); }
    #[test] fn test_eval_create_new_function() { run_str("eval('fn() { print . print }')()", "print\n"); }
    #[test] fn test_eval_overwrite_function() { run_str("fn foo() {} ; foo = eval('fn() { print . print }') ; foo()", "print\n"); }
    #[test] fn test_eval_with_runtime_error_in_different_source() { run_str("eval('%sprint + 1' % (' ' * 100))", "TypeError: Cannot add 'print' of type 'native function' and '1' of type 'int'\n  at: line 1 (<eval>)\n  at: `<script>` (line 1)\n\n1 |                                                                                                     print + 1\n2 |                                                                                                           ^\n"); }
    #[test] fn test_eval_function_with_runtime_error_in_different_source() { run_str("eval('%sfn() -> print + 1' % (' ' * 100))()", "TypeError: Cannot add 'print' of type 'native function' and '1' of type 'int'\n  at: line 1 (<eval>)\n  at: `fn _()` (line 1)\n\n1 |                                                                                                     fn() -> print + 1\n2 |                                                                                                                   ^\n"); }
    #[test] fn test_all_yes_all() { run_str("[1, 3, 4, 5] . all(>0) . print", "true\n"); }
    #[test] fn test_all_yes_some() { run_str("[1, 3, 4, 5] . all(>3) . print", "false\n"); }
    #[test] fn test_all_yes_none() { run_str("[1, 3, 4, 5] . all(<0) . print", "false\n"); }
    #[test] fn test_any_yes_all() { run_str("[1, 3, 4, 5] . any(>0) . print", "true\n"); }
    #[test] fn test_any_yes_some() { run_str("[1, 3, 4, 5] . any(>3) . print", "true\n"); }
    #[test] fn test_any_yes_none() { run_str("[1, 3, 4, 5] . any(<0) . print", "false\n"); }
    #[test] fn test_typeof_of_basic_types() { run_str("[nil, 0, false, 'test', [], {1}, {1: 2}, heap(), (1, 2), range(30), enumerate([])] . map(typeof) . map(print)", "nil\nint\nbool\nstr\nlist\nset\ndict\nheap\nvector\nrange\nenumerate\n"); }
    #[test] fn test_typeof_functions() { run_str("[range, fn() -> nil, push(3), ((fn(a, b) -> nil)(1))] . map(typeof) . all(==function) . print", "true\n"); }
    #[test] fn test_typeof_struct_constructor() { run_str("struct Foo(a, b) Foo . typeof . print", "function\n"); }
    #[test] fn test_typeof_struct_instance() { run_str("struct Foo(a, b) Foo(1, 2) . typeof . print", "struct Foo(a, b)\n"); }
    #[test] fn test_typeof_slice() { run_str("[:] . typeof . print", "function\n"); }
    #[test] fn test_count_ones() { run_str("0b11011011 . count_ones . print", "6\n"); }
    #[test] fn test_count_zeros() { run_str("0 . count_zeros . print", "64\n"); }
    #[test] fn test_env_exists() { run_str("env . repr . print", "fn env(...)\n"); }
    #[test] fn test_argv_exists() { run_str("argv . repr . print", "fn argv()\n"); }
    #[test] fn test_argv_is_empty() { run_str("argv() . repr . print", "[]\n"); }


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
    #[test] fn test_late_bound_global_assignment() { run("late_bound_global_assignment"); }
    #[test] fn test_late_bound_global_invalid() { run("late_bound_global_invalid"); }
    #[test] fn test_map_loop_with_multiple_references() { run("map_loop_with_multiple_references"); }
    #[test] fn test_memoize() { run("memoize"); }
    #[test] fn test_memoize_recursive() { run("memoize_recursive"); }
    #[test] fn test_memoize_recursive_as_annotation() { run("memoize_recursive_as_annotation"); }
    #[test] fn test_quine() { run("quine"); }
    #[test] fn test_range_used_twice() { run("range_used_twice"); }
    #[test] fn test_runtime_error_with_trace() { run("runtime_error_with_trace"); }
    #[test] fn test_upvalue_never_captured() { run("upvalue_never_captured"); }


    fn run_str(text: &'static str, expected: &'static str) {
        let view: SourceView = SourceView::new(String::from("<test>"), String::from(text));
        let compile = compiler::compile(true, &view);

        if compile.is_err() {
            assert_eq!(format!("Compile Error:\n\n{}", compile.err().unwrap().join("\n")).as_str(), expected);
            return
        }

        let compile = compile.unwrap();
        println!("[-d] === Compiled ===");
        for line in compile.disassemble(&view, true) {
            println!("[-d] {}", line);
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut vm = VirtualMachine::new(compile, view, &b""[..], &mut buf, vec![]);

        let result: ExitType = vm.run_until_completion();
        assert!(vm.stack.is_empty() || result.is_early_exit());

        let view: SourceView = vm.view;
        let mut output: String = String::from_utf8(buf).unwrap();

        if let ExitType::Error(error) = result {
            output.push_str(view.format(&error).as_str());
        }

        assert_eq!(output.as_str(), expected);
    }

    fn run(path: &'static str) {
        let resource = test_util::get_resource("compiler", path);
        let view: SourceView = resource.view();
        let compile= compiler::compile(true, &view);

        if compile.is_err() {
            assert_eq!(format!("Compile Error:\n\n{}", compile.err().unwrap().join("\n")).as_str(), "Compiled");
            return
        }

        let compile = compile.unwrap();
        println!("[-d] === Compiled ===");
        for line in compile.disassemble(&view, true) {
            println!("[-d] {}", line);
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut vm = VirtualMachine::new(compile, view, &b""[..], &mut buf, vec![]);

        let result: ExitType = vm.run_until_completion();
        assert!(vm.stack.is_empty() || result.is_early_exit());

        let view: SourceView = vm.view;
        let mut output: String = String::from_utf8(buf).unwrap();

        if let ExitType::Error(error) = result {
            output.push_str(view.format(&error).as_str());
        }

        resource.compare(output.split("\n").map(String::from).collect::<Vec<String>>());
    }
}
