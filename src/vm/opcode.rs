use crate::stdlib::StdBinding;

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Opcode {

    Noop,

    // Flow Control
    // These only peek() the stack
    JumpIfFalse(u16),
    JumpIfFalsePop(u16),
    JumpIfTrue(u16),
    Jump(u16),

    // Stack Operations
    Dupe, // ... x, y, z] -> ... x, y, z, z]
    Pop,
    PopN(u16),

    PushLocal(u16),
    StoreLocal(u16),

    // Push
    Nil,
    True,
    False,
    Int(u16),
    Str(u16),
    Bound(StdBinding),
    List(u16),

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
    OpIndex,
    OpSlice,
    OpSliceWithStep,

    // Special
    LineNumber(u16),
    Exit,
}


#[cfg(test)]
mod test {
    use crate::vm::opcode::Opcode;

    #[test] fn test_layout() { assert_eq!(4, std::mem::size_of::<Opcode>()); }
}