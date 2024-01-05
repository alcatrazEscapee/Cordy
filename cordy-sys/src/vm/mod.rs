use std::cell::Cell;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::rc::Rc;
use fxhash::FxBuildHasher;

use crate::{AsError, compiler, core, trace, util};
use crate::compiler::{CompileParameters, CompileResult, Fields, FunctionLibrary, IncrementalCompileResult, Locals};
use crate::reporting::{Location, SourceView};
use crate::util::{Noop, OffsetAdd};
use crate::vm::value::{Literal, UpValue};
use crate::core::Pattern;

pub use crate::vm::error::{DetailRuntimeError, RuntimeError};
pub use crate::vm::opcode::{Opcode, StoreOp};
pub use crate::vm::value::{AnyResult, ErrorPtr, ErrorResult, Function, IntoDictValue, IntoIterableValue, IntoValue, Iterable, LiteralType, MAX_INT, MIN_INT, StructTypeImpl, Type, ValuePtr, ValueResult, Method, ComplexType, PartialNativeFunction, RationalType};

use Opcode::{*};
use RuntimeError::{*};
use crate::vm::operator::BinaryOp;


pub mod operator;

mod opcode;
mod value;
mod error;

#[cfg(test)]
mod tests;

/// Per-test, how many instructions should be allowed to execute.
/// This primarily prevents infinite-loop tests from causing tests to hang, allowing easier debugging.
#[cfg(test)]
const TEST_EXECUTION_LIMIT: usize = 1000;


pub struct VirtualMachine<R, W, F> {
    ip: usize,
    code: Vec<Opcode>,
    stack: Vec<ValuePtr>,
    call_stack: Vec<CallFrame>,
    literal_stack: Vec<Literal>,
    global_count: usize,
    open_upvalues: HashMap<usize, Rc<Cell<UpValue>>, FxBuildHasher>,
    unroll_stack: Vec<i32>,

    constants: Vec<ValuePtr>,

    /// N.B. This field must not be modified by any mutable reference to the VM
    /// Non-mutable references (via `unsafe`) are handed out to `self.patterns` while a `&mut` reference is kept of the VM.
    patterns: Vec<Pattern<StoreOp>>,
    globals: Vec<String>,
    locations: Vec<Location>,
    fields: Fields,
    functions: FunctionLibrary,

    view: SourceView,
    read: R,
    write: W,
    ffi: F,
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

    fn of<R: BufRead, W: Write, F : FunctionInterface>(vm: &VirtualMachine<R, W, F>, result: AnyResult) -> ExitType {
        match result {
            Ok(_) => ExitType::Return,
            Err(error) => match error.as_err() {
                RuntimeExit => ExitType::Exit,
                RuntimeYield => ExitType::Yield,
                _ => ExitType::Error(RuntimeError::with_stacktrace(error, vm.ip - 1, &vm.call_stack, &vm.constants, &vm.locations))
            },
        }
    }
}


pub trait FunctionInterface {
    /// Handles a foreign function call from the given ID.
    fn handle(&mut self, functions: &FunctionLibrary, handle_id: u32, args: Vec<ValuePtr>) -> ValueResult;
}

impl FunctionInterface for Noop {
    fn handle(&mut self, _ : &FunctionLibrary, _: u32, _: Vec<ValuePtr>) -> ValueResult {
        panic!("Natives are not supported for Noop");
    }
}


pub trait VirtualInterface {
    // Invoking Functions

    fn invoke_func0(&mut self, f: ValuePtr) -> ValueResult;
    fn invoke_func1(&mut self, f: ValuePtr, a1: ValuePtr) -> ValueResult;
    fn invoke_func2(&mut self, f: ValuePtr, a1: ValuePtr, a2: ValuePtr) -> ValueResult;
    fn invoke_func(&mut self, f: ValuePtr, args: &[ValuePtr]) -> ValueResult;

    fn invoke_eval(&mut self, s: String) -> ValueResult;
    fn invoke_monitor(&mut self, cmd: &str) -> ValueResult;

    /// Executes a `StoreOp`, storing the value `value`
    fn store(&mut self, op: StoreOp, value: ValuePtr) -> AnyResult;

    // Wrapped IO
    fn println0(&mut self);
    fn println(&mut self, str: &str);
    fn print(&mut self, str: &str);

    fn read_line(&mut self) -> String;
    fn read(&mut self) -> String;

    fn get_envs(&self) -> ValuePtr;
    fn get_env(&self, name: &str) -> ValuePtr;
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


impl VirtualMachine<Noop, Vec<u8>, Noop> {
    /// Creates a new `VirtualMachine` with empty external read/write capabilities.
    pub fn default(result: CompileResult, view: SourceView) -> VirtualMachine<Noop, Vec<u8>, Noop> {
        VirtualMachine::new(result, view, Noop, vec![], Noop)
    }
}

impl<W : Write> VirtualMachine<Noop, W, Noop> {
    /// Creates a new `VirtualMachine` with the given write capability but an empty read capability.
    pub fn with(result: CompileResult, view: SourceView, write: W) -> VirtualMachine<Noop, W, Noop> {
        VirtualMachine::new(result, view, Noop, write, Noop)
    }
}


impl<R : BufRead, W : Write, F : FunctionInterface> VirtualMachine<R, W, F> {
    /// Creates a new `VirtualMachine` with the given read and write capabilities.
    pub fn new(result: CompileResult, view: SourceView, read: R, write: W, ffi: F) -> VirtualMachine<R, W, F> {
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
            functions: result.functions,

            view,
            read,
            write,
            ffi,
            args: ValuePtr::nil(),
        }
    }

    pub fn with_args(&mut self, args: Vec<String>) {
        self.args = args.into_iter().map(|u| u.to_value()).to_list()
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
    pub fn eval_compile(&mut self, text: String) -> AnyResult {
        let mut locals = Locals::empty();
        compiler::eval_compile(text, self.as_compile_parameters(false, &mut locals))
    }

    fn as_compile_parameters<'a, 'b: 'a, 'c: 'a>(&'b mut self, enable_optimization: bool, locals: &'c mut Vec<Locals>) -> CompileParameters<'a> {
        CompileParameters::new(enable_optimization, &mut self.code, &mut self.constants, &mut self.patterns, &mut self.globals, &mut self.locations, &mut self.fields, &mut self.functions, locals, &mut self.view)
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

    /// Core Interpreter loop. This runs until the current frame is dropped via a `Return` opcode.
    ///
    /// **Implementation Note:** This function is absolutely, entirely essential performance wise.
    fn run(&mut self) -> AnyResult {
        #[cfg(test)]
        let mut limit: usize = 0;
        let drop_frame: usize = self.call_stack.len() - 1;
        loop {
            // Anti-infinite-loop protection for tests, which are all short-running programs
            #[cfg(test)]
            {
                limit += 1;
                if limit == TEST_EXECUTION_LIMIT {
                    panic!("Execution limit reached");
                }
            }

            // Fetch the current opcode, and always increment the IP immediately after
            let op: Opcode = unsafe {
                // Skipping the bounds check gave a ~3% performance improvement on a spin-loop benchmark, so this is worth doing.
                debug_assert!(self.ip < self.code.len());
                *self.code.get_unchecked(self.ip)
            };
            self.ip += 1;

            // Trace current opcode before we begin executing it
            trace::trace_interpreter!("vm::run op={:?}", op);

            // Execute the current instruction
            match op {
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

                    if drop_frame == self.call_stack.len() { // Once we dropped the frame, we may exit to the calling function
                        return Ok(())
                    }
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
                    self.push(f.to_closure());
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
                    let mut iter = self.stack[top].as_synthetic_iterable().borrow_mut();
                    match iter.next() {
                        Some(value) => {
                            drop(iter);
                            self.push(value)
                        },
                        None => self.ip = self.ip.add_offset(ip),
                    }
                },

                ExecPattern(index) => {
                    let top = self.peek(0).clone();

                    // Here, we would like to split the VM struct into a `& self.patterns` and a `&mut self.<everything else>`
                    // But, there's no mechanism for the compiler to understand that this is legal
                    // So we invoke some unsafe code here to split the mutable and immutable reference
                    // This just copies the reference, which lets us have a separate `&VM` that we treat as only pointing to `& VM.patterns`
                    let pattern = unsafe {
                        std::mem::transmute_copy::<&Pattern<StoreOp>, &Pattern<StoreOp>>(&&self.patterns[index as usize])
                    };
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

                CallNative(handle_id) => {
                    let nargs = self.functions.lookup(handle_id).nargs;
                    let args = self.popn(nargs as u32);
                    let ret = self.ffi.handle(&self.functions, handle_id, args)?;
                    self.push(ret);
                }

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
                    let ret: ValuePtr = a1.get_field(&self.fields, &self.constants, field_index)?;
                    self.push(ret);
                },
                GetFieldPeek(field_index) => {
                    let a1: &ValuePtr = self.peek(0);
                    let ret: ValuePtr = a1.get_field(&self.fields, &self.constants, field_index)?;
                    self.push(ret);
                },
                GetFieldFunction(field_index) => {
                    self.push(ValuePtr::field(field_index));
                },
                SetField(field_index) => {
                    let a2: ValuePtr = self.pop();
                    let a1: ValuePtr = self.pop();
                    let ret: ValuePtr = a1.set_field(&self.fields, field_index, a2)?;
                    self.push(ret);
                },

                GetMethod(index) => {
                    let a1: ValuePtr = self.pop();
                    let ret: ValuePtr = self.constants[index as usize].clone().to_partial(vec![a1]);
                    self.push(ret);
                }

                Unary(op) => {
                    let arg: ValuePtr = self.pop();
                    let ret: ValuePtr = op.apply(arg)?;
                    self.push(ret);
                },
                Binary(op) => {
                    let rhs: ValuePtr = self.pop();
                    let lhs: ValuePtr = self.pop();
                    let ret: ValuePtr = op.apply(lhs, rhs)?;
                    self.push(ret);
                },
                Compare(op, offset) => {
                    let rhs: ValuePtr = self.pop();
                    let lhs: ValuePtr = self.pop();
                    if op.apply(&lhs, &rhs) {
                        // If true, then push back `rhs`, and continue
                        self.push(rhs);
                    } else {
                        // If false, then push `false` and jump to end of chain
                        let jump: usize = self.ip.add_offset(offset);
                        self.push(false.to_value());
                        self.ip = jump;
                    }
                }

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

                AssertTest(ip) => {
                    let ret = self.pop();
                    if ret.to_bool() {
                        let jump: usize = self.ip.add_offset(ip);
                        self.ip = jump;
                    } else {
                        self.push(ValuePtr::nil()); // No additional message
                    }
                },
                AssertCompare(op, ip) => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    if op.apply(&lhs, &rhs) {
                        let jump: usize = self.ip.add_offset(ip);
                        self.ip = jump;
                    } else {
                        self.push(format!("Expected {} {} {}", lhs.to_repr_str(), BinaryOp::from(op).as_error(), rhs.to_repr_str()).to_value());
                    }
                },
                AssertFailed => {
                    let ret: ValuePtr = self.pop();
                    let message: ValuePtr = self.pop();
                    return RuntimeAssertFailed(ret, message).err()
                },
            }
        }
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
                let func = f.as_function_or_closure();
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
                        let partial: ValuePtr = func.to_partial(arg);
                        self.push(partial);
                    }
                    // Partial functions are already evaluated, so we return native, since we don't need to spin
                    Ok(FunctionType::Native)
                } else {
                    IncorrectArgumentsUserFunction(func.clone(), nargs).err()
                }
            },
            Type::PartialFunction => {
                // Extract the partial function from the stack, and then check if we can evaluate it.
                // - If we have enough arguments, push all arguments onto the stack and evaluate,
                // - If we don't, then create a new partial function with the combined arguments
                let i: usize = self.stack.len() - nargs as usize - 1;
                let ptr = std::mem::take(&mut self.stack[i]);
                let partial = ptr.as_partial_function();
                let func = partial.as_function();
                let total_nargs: u32 = partial.nargs() + nargs;
                if func.min_args() > total_nargs {
                    // Not enough arguments, so pop the argument and push a new partial function
                    let partial = partial.with(splice(&mut self.stack, nargs));

                    self.pop(); // Should pop the `Nil` we swapped earlier
                    self.push(partial.to_value());

                    // Partial functions are already evaluated, so we return native, since we don't need to spin
                    Ok(FunctionType::Native)
                } else if func.in_range(total_nargs) {
                    // Exactly enough arguments to invoke the function
                    // We need to set up the stack correctly to call the function
                    let head: usize = func.jump_offset(total_nargs);
                    let num_var_args: Option<u32> = func.num_var_args(nargs);
                    let (ptr, args) = partial.consume();
                    self.stack[i] = ptr; // Replace the `Nil` from earlier
                    insert(&mut self.stack, args, nargs);
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
                let ptr = std::mem::take(&mut self.stack[i]);
                let ret = core::invoke_partial(ptr.as_partial_native(), nargs, self)?;

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
                if list.len() != 1 {
                    return ValueErrorEvalListMustHaveUnitLength(list.len()).err()
                }
                let index = list[0].clone();
                let result = core::get_index(self, &arg, index)?;
                self.push(result);
                Ok(FunctionType::Native)
            },
            Type::Slice => {
                if nargs != 1 {
                    return ValueIsNotFunctionEvaluable(f.clone()).err();
                }
                let arg = self.pop();
                let slice = self.pop();
                self.push(slice.as_slice().apply(&arg)?);
                Ok(FunctionType::Native)
            }
            Type::StructType => {
                let owner = f.as_struct_type();
                if owner.is_module() {
                    return ValueIsNotFunctionEvaluable(f.clone()).err()
                }
                if nargs != owner.num_fields() {
                    return IncorrectArgumentsStruct(owner.as_str(), nargs).err()
                }

                let args: Vec<ValuePtr> = self.popn(nargs);
                let owner = self.pop();
                let instance: ValuePtr = ValuePtr::instance(owner, args);

                self.push(instance);

                Ok(FunctionType::Native)
            }
            Type::GetField => {
                let field_index = f.as_field();
                if nargs != 1 {
                    return IncorrectArgumentsGetField(self.fields.get_name(field_index), nargs).err()
                }

                let arg: ValuePtr = self.pop();
                let ret: ValuePtr = arg.get_field(&self.fields, &self.constants, field_index)?;

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


impl <R : BufRead, W : Write, F : FunctionInterface> VirtualInterface for VirtualMachine<R, W, F>
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

    fn invoke_func(&mut self, f: ValuePtr, args: &[ValuePtr]) -> ValueResult {
        self.push(f);
        for arg in args {
            self.push(arg.clone());
        }
        self.invoke_and_spin(args.len() as u32)
    }

    fn invoke_eval(&mut self, text: String) -> ValueResult {
        let eval_head: usize = self.code.len();

        self.eval_compile(text)?;
        self.call_function(eval_head, 0, None);
        self.run()?;
        let ret = self.pop();
        self.push(ValuePtr::nil()); // `eval` executes as a user function but is called like a native function, this prevents stack corruption
        ret.ok()
    }

    fn invoke_monitor(&mut self, cmd: &str) -> ValueResult {
        match cmd {
            "stack" => self.stack.iter().cloned().to_list().ok(),
            "call-stack" => self.call_stack.iter()
                .map(|frame| vec![frame.frame_pointer.to_value(), frame.return_ip.to_value()].to_value())
                .to_list()
                .ok(),
            "code" => self.code.iter()
                .enumerate()
                .map(|(ip, op)| op.disassembly(ip, &mut std::iter::empty(), &self.fields, &self.constants).to_value())
                .to_list()
                .ok(),
            _ => MonitorError(String::from(cmd)).err(),
        }
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
    fn println(&mut self, str: &str) { writeln!(&mut self.write, "{}", str).unwrap(); }
    fn print(&mut self, str: &str) { write!(&mut self.write, "{}", str).unwrap(); }

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

    fn get_env(&self, name: &str) -> ValuePtr {
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
