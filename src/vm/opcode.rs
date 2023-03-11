use crate::stdlib::NativeFunction;
use crate::vm::operator::{BinaryOp, UnaryOp};
use crate::misc::OffsetAdd;

use Opcode::{*};

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum Opcode {

    Noop,

    /// The parameter is an offset value, based on the IP *after* the instruction is executed. So an instruction:
    ///
    /// `005: Jump(-2)`
    ///
    /// will jump to 005 + 1 (since we finish executing this instruction) - 2 = 004
    ///
    /// `Jump(-1)` is a no-op.
    JumpIfFalse(i32),
    JumpIfFalsePop(i32),
    JumpIfTrue(i32),
    JumpIfTruePop(i32),
    Jump(i32),

    Return,

    // Stack Operations
    Pop,
    PopN(u32),
    Dup,
    Swap,

    // Note: Local + Global does not, for the VM, mean in terms of block scoped
    // Rather, it means if this variable is accessed in the stack relative to the stack frame, or relative to the stack bottom
    // In this regard it refers to variables outside of functions (top level, even within block scopes), and variables within functions
    PushLocal(u32),
    StoreLocal(u32),
    PushGlobal(u32),
    StoreGlobal(u32),

    PushUpValue(u32), // index
    StoreUpValue(u32), // index

    StoreArray,

    // Increments the count of currently declared global variables. This is checked on every `CheckGlobalCount` to verify that no global is referenced before it is initialized
    // Due to late binding allowed in the parser, we cannot ensure this does not happen at runtime, so it needs to be checked.
    IncGlobalCount,

    // Create a closure from a function (with no upvalues yet)
    Closure,

    // Take a closure, and add the respective local or upvalue to the closure
    CloseLocal(u32),
    CloseUpValue(u32),

    /// Lifts an UpValue from a stack slot (offset by the frame pointer) to the heap
    /// It does so by boxing it into a `Rc<Cell<Value>>`, stored on the closure's `environment` array. Each closure references the same `UpValue`, and hence will see all mutations.
    /// Takes a local index of an upvalue to lift.
    LiftUpValue(u32),

    /// Converts the top of the stack to an `Value::Iter()`.
    InitIterable,

    /// Expects the top of the stack to contain a `Value::Iter()`. Tests if this has reached the end of the iterable.
    /// If yes, it will push `false`, and do nothing else.
    /// If no, it will push the next value in the iterable, followed by `true`.
    /// This is intended to be followed by a `JumpIfFalsePop(end of loop)`
    TestIterable,

    // Push
    Nil,
    True,
    False,
    Int(u32),
    Str(u32),
    Function(u32),
    NativeFunction(NativeFunction),
    // Note that `List`, `Vector`, `Set`, `Dict`, are different from invoking native functions
    // 1. They don't require a function evaluation and resolution (efficient)
    // 2. They allow zero and one element cases to be handled exactly as usual (i.e. `list('no')` is `['no'], not ['n', 'o'])
    List(u32),
    Vector(u32),
    Set(u32),
    Dict(u32),
    /// The parameter corresponds to the `type index` of the struct.
    Struct(u32),

    // Runtime specific type checks

    /// Takes an `Int` constant, and checks that the top of the stack is an iterable with length > the provided constant
    CheckLengthGreaterThan(u32),
    /// Takes an `Int` constant, and checks that the top of the stack is an iterable with length = the provided constant
    CheckLengthEqualTo(u32),

    /// Opcode for function evaluation (either with `()` or with `.`). The `u8` parameter is the number of arguments to the function.
    ///
    /// - The stack must be setup with the function to be called (which must be a callable type, i.e. return `true` to `value.is_function()`, followed by `n` arguments.
    /// - A call frame will be pushed, and the `frame_pointer()` of the VM will point to the first local of the new frame. The function itself can be accessed via `stack[frame_pointer() - 1]`.
    /// - Upon reaching a `Return` opcode, everything above and including `frame_pointer() - 1` will be popped off the stack, and the return value will be pushed onto the stack (replacing the spot where the function was).
    ///
    /// Implementation Note: In order to implement function composition (the `.` operator), this is preceded with a `Swap` opcode to reverse the order of the argument and function to be called.
    OpFuncEval(u8),

    /// Takes a stack of `[index, list, ...]`, pops the top two elements, and pushes `list[index]`
    OpIndex,
    /// Takes a stack of `[index, list, ...]`, and pushes `list[index]` (does not pop any values)
    OpIndexPeek,

    OpSlice,
    OpSliceWithStep,

    GetField(u32),
    GetFieldPeek(u32),
    GetFieldFunction(u32),
    SetField(u32),

    Unary(UnaryOp),
    Binary(BinaryOp),

    // Special
    Exit,
    Yield,
    AssertFailed,
}

impl Opcode {
    /// Replaces the parameter of a relative jump, with an absolute offset. Only used for debugging, i.e. disassembly purposes.
    pub fn to_absolute_jump(self: Self, ip: usize) -> Opcode {
        match self {
            JumpIfFalse(offset) => JumpIfFalse(ip.add_offset(offset) as i32 + 1),
            JumpIfFalsePop(offset) => JumpIfFalsePop(ip.add_offset(offset) as i32 + 1),
            JumpIfTrue(offset) => JumpIfTrue(ip.add_offset(offset) as i32 + 1),
            JumpIfTruePop(offset) => JumpIfTruePop(ip.add_offset(offset) as i32 + 1),
            Jump(offset) => Jump(ip.add_offset(offset) as i32 + 1),
            _ => self,
        }
    }
}


#[cfg(test)]
mod test {
    use crate::vm::opcode::Opcode;

    #[test] fn test_layout() { assert_eq!(8, std::mem::size_of::<Opcode>()); }
}