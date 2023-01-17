use crate::stdlib::NativeFunction;

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

    PushUpValue(u16), // index
    StoreUpValue(u16), // index

    StoreArray,

    // Increments the count of currently declared global variables. This is checked on every `CheckGlobalCount` to verify that no global is referenced before it is initialized
    // Due to late binding allowed in the parser, we cannot ensure this does not happen at runtime, so it needs to be checked.
    IncGlobalCount,

    // Create a closure from a function (with no upvalues yet)
    Closure,

    // Take a closure, and add the respective local or upvalue to the closure
    CloseLocal(u16),
    CloseUpValue(u16),

    /// Lifts an UpValue from a stack slot (offset by the frame pointer) to the heap
    /// It does so by boxing it into a `Rc<Cell<Value>>`, stored on the closure's `environment` array. Each closure references the same `UpValue`, and hence will see all mutations.
    /// Takes a local index of an upvalue to lift.
    LiftUpValue(u16),

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
    Int(u16),
    Str(u16),
    Function(u16),
    NativeFunction(NativeFunction),
    // Note that `List`, `Vector`, `Set`, `Dict`, are different from invoking native functions
    // 1. They don't require a function evaluation and resolution (efficient)
    // 2. They allow zero and one element cases to be handled exactly as usual (i.e. `list('no')` is `['no'], not ['n', 'o'])
    List(u16),
    Vector(u16),
    Set(u16),
    Dict(u16),

    // Runtime specific type checks

    /// Takes an `Int` constant, and checks that the top of the stack is an iterable with length > the provided constant
    CheckLengthGreaterThan(u16),
    /// Takes an `Int` constant, and checks that the top of the stack is an iterable with length = the provided constant
    CheckLengthEqualTo(u16),

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

    OpIn,
    OpLessThan,
    OpGreaterThan,
    OpLessThanEqual,
    OpGreaterThanEqual,
    OpEqual,
    OpNotEqual,

    OpMax,
    OpMin,

    // Special
    Exit,
    Yield,
}


#[cfg(test)]
mod test {
    use crate::vm::opcode::Opcode;

    #[test] fn test_layout() { assert_eq!(4, std::mem::size_of::<Opcode>()); }
}