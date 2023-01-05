use crate::stdlib::StdBinding;

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum Opcode {

    Noop,

    // Flow Control
    // These only peek() the stack
    JumpIfFalse(u16),
    JumpIfFalsePop(u16),
    JumpIfTrue(u16),
    JumpIfTruePop(u16),
    Jump(u16),
    Return,

    // Stack Operations
    Pop,
    PopN(u16),
    Dup,
    Swap,

    // Note: Local + Global does not, for the VM, mean in terms of block scoped
    // Rather, it means if this variable is accessed in the stack relative to the stack frame, or relative to the stack bottom
    // In this regard it refers to variables outside of functions (top level, even within block scopes), and variables within functions
    PushLocal(u16),
    StoreLocal(u16),
    PushGlobal(u16, bool), // bool = is_local = !is_true_global
    StoreGlobal(u16, bool),

    PushUpValue(u16, bool), // index, is_local

    StoreArray,

    // Increments the count of currently declared global variables. This is checked on every `CheckGlobalCount` to verify that no global is referenced before it is initialized
    // Due to late binding allowed in the parser, we cannot ensure this does not happen at runtime, so it needs to be checked.
    IncGlobalCount,

    // Create a closure from a function (with no upvalues yet)
    Closure,

    // Take a closure, and add the respective local or upvalue to the closure
    CloseLocal(u16),
    CloseUpValue(u16),

    // Push
    Nil,
    True,
    False,
    Int(u16),
    Str(u16),
    Function(u16),
    NativeFunction(StdBinding),
    List(u16),

    // Runtime specific type checks
    // Both of these opcodes pop the top (a length), and peek the next, and validate the length of the peek'd variable is greater than or equal to the provided value.
    CheckLengthGreaterThan,
    CheckLengthEqualTo,

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

    // Unary Operators
    UnarySub,
    UnaryLogicalNot,
    UnaryBitwiseNot,

    // Binary Operators
    // Ordered by precedence, highest to lowest

    OpMul,
    OpDiv,
    OpMod,
    OpPow,
    OpIs,

    OpAdd,
    OpSub,

    OpLeftShift,
    OpRightShift,

    OpBitwiseAnd,
    OpBitwiseOr,
    OpBitwiseXor,

    OpLessThan,
    OpGreaterThan,
    OpLessThanEqual,
    OpGreaterThanEqual,
    OpEqual,
    OpNotEqual,

    // Special
    Exit,
}


#[cfg(test)]
mod test {
    use crate::vm::opcode::Opcode;

    #[test] fn test_layout() { assert_eq!(4, std::mem::size_of::<Opcode>()); }
}