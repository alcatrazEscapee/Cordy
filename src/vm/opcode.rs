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
    PushGlobal(u16),
    StoreGlobal(u16),

    StoreArray,

    // Push
    Nil,
    True,
    False,
    Int(u16),
    Str(u16),
    Function(u16),
    Bound(StdBinding),
    List(u16),

    // Unary Operators
    UnarySub,
    UnaryLogicalNot,
    UnaryBitwiseNot,

    // Binary Operators
    // Ordered by precedence, highest to lowest
    OpFuncEval(u8),
    OpIndex,
    OpIndexPeek,
    OpSlice,
    OpSliceWithStep,

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

    OpFuncCompose,

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