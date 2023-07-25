use crate::compiler::parser::semantic::{LValue, LValueReference};
use crate::reporting::Location;
use crate::core::NativeFunction;
use crate::vm::{C64, LiteralType, Opcode, RuntimeError, Value, ValueResult};
use crate::vm::operator::{BinaryOp, UnaryOp};

#[derive(Debug, Clone)]
pub struct Expr(pub Location, pub ExprType);

type Arg = Box<Expr>;

#[derive(Debug, Clone)]
pub enum ExprType {
    // Terminals
    Nil,
    Exit,
    Bool(bool),
    Int(i64),
    Complex(C64),
    Str(String),
    LValue(LValueReference),
    NativeFunction(NativeFunction),
    Function(u32, Vec<Opcode>),
    SliceLiteral(Arg, Arg, Box<Option<Expr>>),

    // Operators + Functions
    Unary(UnaryOp, Arg),

    /// Arguments are `op, lhs, rhs, swap`
    /// If `swap` is `true`, then **both** of the following effects will apply:
    ///
    /// 1. The arguments `lhs` and `rhs` will be evaluated in the opposite semantic order
    /// 2. A `Swap` opcode will be emitted.
    ///
    /// This means that a `binary_with(op, lhs, rhs, swap)` is equivalent to saying that "rhs needs to be evaluated before lhs in the operation op(lhs, rhs)"
    ///
    /// **The side (right vs left) of each argument is still correct!!!**
    Binary(BinaryOp, Arg, Arg, bool),
    Literal(LiteralType, Vec<Expr>),
    Unroll(Arg, bool), // first: bool
    Eval(Arg, Vec<Expr>, bool), // any_unroll: bool
    Compose(Arg, Arg),
    LogicalAnd(Arg, Arg),
    LogicalOr(Arg, Arg),
    Index(Arg, Arg),
    Slice(Arg, Arg, Arg),
    SliceWithStep(Arg, Arg, Arg, Arg),
    IfThenElse(Arg, Arg, Arg),
    GetField(Arg, u32),
    SetField(Arg, u32, Arg),
    SwapField(Arg, u32, Arg, BinaryOp),
    GetFieldFunction(u32),

    // Assignments
    Assignment(LValueReference, Arg),
    ArrayAssignment(Arg, Arg, Arg),

    /// Note that `BinaryOp::NotEqual` is used to indicate this is a `Compose()` operation under the hood
    ArrayOpAssignment(Arg, Arg, BinaryOp, Arg),
    PatternAssignment(LValue, Arg),

    // Error
    RuntimeError(Box<RuntimeError>),
}


impl Expr {

    // N.B.
    // Some of these constructor functions require a `Location` parameter, and others don't.
    // This is based on the fallibility of the operation AT RUNTIME.
    // Location tracking is only necessary if the opcodes corresponding to this specific term may raise a runtime error
    // For instance, `+` (OpAdd) may raise an error, but `1` (Int) does not.

    pub fn nil() -> Expr { Expr(Location::empty(), ExprType::Nil) }
    pub fn exit() -> Expr { Expr(Location::empty(), ExprType::Exit) }
    pub fn bool(it: bool) -> Expr { Expr(Location::empty(), ExprType::Bool(it)) }
    pub fn int(it: i64) -> Expr { Expr(Location::empty(), ExprType::Int(it)) }
    pub fn complex(it: i64) -> Expr { Expr::c64(C64::new(0, it)) }
    pub fn c64(it: C64) -> Expr { Expr(Location::empty(), ExprType::Complex(it)) }
    pub fn str(it: String) -> Expr { Expr(Location::empty(), ExprType::Str(it)) }
    pub fn lvalue(loc: Location, lvalue: LValueReference) -> Expr {
        match lvalue {
            LValueReference::NativeFunction(native) => Expr::native(loc, native),
            lvalue => Expr(loc, ExprType::LValue(lvalue))
        }
    }
    pub fn native(loc: Location, native: NativeFunction) -> Expr { Expr(loc, ExprType::NativeFunction(native)) }
    pub fn function(function: u32, closed_locals: Vec<Opcode>) -> Expr { Expr(Location::empty(), ExprType::Function(function, closed_locals)) }

    pub fn assign_lvalue(loc: Location, lvalue: LValueReference, expr: Expr) -> Expr { Expr(loc, ExprType::Assignment(lvalue, Box::new(expr))) }
    pub fn assign_pattern(loc: Location, lvalue: LValue, expr: Expr) -> Expr { Expr(loc, ExprType::PatternAssignment(lvalue, Box::new(expr))) }
    pub fn assign_array(loc: Location, array: Expr, index: Expr, rhs: Expr) -> Expr { Expr(loc, ExprType::ArrayAssignment(Box::new(array), Box::new(index), Box::new(rhs))) }
    pub fn assign_op_array(loc: Location, array: Expr, index: Expr, op: BinaryOp, rhs: Expr) -> Expr { Expr(loc, ExprType::ArrayOpAssignment(Box::new(array), Box::new(index), op, Box::new(rhs))) }

    pub fn list(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Literal(LiteralType::List, args)) }
    pub fn vector(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Literal(LiteralType::Vector, args)) }
    pub fn set(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Literal(LiteralType::Set, args)) }
    pub fn dict(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Literal(LiteralType::Dict, args)) }

    pub fn raw_slice(loc: Location, arg1: Expr, arg2: Expr, arg3: Option<Expr>) -> Expr { Expr(loc, ExprType::SliceLiteral(Box::new(arg1), Box::new(arg2), Box::new(arg3))) }

    pub fn unary(self: Self, loc: Location, op: UnaryOp) -> Expr { Expr(loc, ExprType::Unary(op, Box::new(self))) }
    pub fn binary(self: Self, loc: Location, op: BinaryOp, rhs: Expr, swap: bool) -> Expr { Expr(loc, ExprType::Binary(op, Box::new(self), Box::new(rhs), swap)) }
    pub fn unroll(self: Self, loc: Location, first: bool) -> Expr { Expr(loc, ExprType::Unroll(Box::new(self), first)) }
    pub fn eval(self: Self, loc: Location, args: Vec<Expr>, any_unroll: bool) -> Expr { Expr(loc, ExprType::Eval(Box::new(self), args, any_unroll)) }
    pub fn compose(self: Self, loc: Location, f: Expr) -> Expr { Expr(loc, ExprType::Compose(Box::new(self), Box::new(f))) }
    pub fn index(self: Self, loc: Location, index: Expr) -> Expr { Expr(loc, ExprType::Index(Box::new(self), Box::new(index))) }
    pub fn slice(self: Self, loc: Location, arg1: Expr, arg2: Expr) -> Expr { Expr(loc, ExprType::Slice(Box::new(self), Box::new(arg1), Box::new(arg2))) }
    pub fn slice_step(self: Self, loc: Location, arg1: Expr, arg2: Expr, arg3: Expr) -> Expr { Expr(loc, ExprType::SliceWithStep(Box::new(self), Box::new(arg1), Box::new(arg2), Box::new(arg3))) }
    pub fn if_then_else(self: Self, loc: Location, if_true: Expr, if_false: Expr) -> Expr { Expr(loc, ExprType::IfThenElse(Box::new(self), Box::new(if_true), Box::new(if_false))) }
    pub fn get_field(self: Self, loc: Location, field_index: u32) -> Expr { Expr(loc, ExprType::GetField(Box::new(self), field_index)) }
    pub fn set_field(self: Self, loc: Location, field_index: u32, rhs: Expr) -> Expr { Expr(loc, ExprType::SetField(Box::new(self), field_index, Box::new(rhs))) }
    pub fn swap_field(self: Self, loc: Location, field_index: u32, rhs: Expr, op: BinaryOp) -> Expr { Expr(loc, ExprType::SwapField(Box::new(self), field_index, Box::new(rhs), op)) }
    pub fn get_field_function(loc: Location, field_index: u32) -> Expr { Expr(loc, ExprType::GetFieldFunction(field_index)) }

    pub fn logical(self: Self, loc: Location, op: BinaryOp, rhs: Expr) -> Expr {
        match op {
            BinaryOp::And => Expr(loc, ExprType::LogicalAnd(Box::new(self), Box::new(rhs))),
            BinaryOp::Or => Expr(loc, ExprType::LogicalOr(Box::new(self), Box::new(rhs))),
            _ => panic!("No logical operator for binary operator {:?}", op),
        }
    }

    pub fn value(value: Value) -> Expr {
        match value {
            Value::Nil => Expr::nil(),
            Value::Bool(it) => Expr::bool(it),
            Value::Int(it) => Expr::int(it),
            Value::Complex(it) => Expr::c64(*it),
            Value::Str(it) => Expr::str((*it).clone()),
            _ => panic!("Not a constant value type"),
        }
    }

    pub fn error(loc: Location, error: Box<RuntimeError>) -> Expr {
        Expr(loc, ExprType::RuntimeError(error))
    }

    pub fn value_result(loc: Location, value: ValueResult) -> Expr {
        match value {
            Ok(value) => Expr::value(value),
            Err(e) => Expr::error(loc, e),
        }
    }

    pub fn is_unroll(self: &Self) -> bool { match self { Expr(_, ExprType::Unroll(_, _)) => true, _ => false } }
}