use crate::compiler::parser::semantic::{LValue, LValueReference};
use crate::reporting::Location;
use crate::stdlib::NativeFunction;
use crate::vm::Opcode;
use crate::vm::operator::{BinaryOp, UnaryOp};

#[derive(Debug, Clone)]
pub struct Expr(pub Location, pub ExprType);

type Arg = Box<Expr>;

#[derive(Debug, Clone)]
pub enum ExprType {
    Nil,
    Exit,
    Bool(bool),
    Int(i64),
    Str(String),
    LValue(LValueReference),
    NativeFunction(NativeFunction),
    Function(u32, Vec<Opcode>),
    Unary(UnaryOp, Arg),
    Binary(BinaryOp, Arg, Arg),
    Sequence(SequenceOp, Vec<Expr>),
    Eval(Arg, Vec<Expr>),
    Compose(Arg, Arg),
    LogicalAnd(Arg, Arg),
    LogicalOr(Arg, Arg),
    Index(Arg, Arg),
    Slice(Arg, Arg, Arg),
    SliceWithStep(Arg, Arg, Arg, Arg),
    IfThenElse(Arg, Arg, Arg),
    Assignment(LValueReference, Arg),
    ArrayAssignment(Arg, Arg, Arg),

    /// Note that `BinaryOp::NotEqual` is used to indicate this is a `Compose()` operation under the hood
    ArrayOpAssignment(Arg, Arg, BinaryOp, Arg),
    PatternAssignment(LValue, Arg),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SequenceOp {
    List, Set, Dict, Vector
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

    pub fn list(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Sequence(SequenceOp::List, args)) }
    pub fn set(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Sequence(SequenceOp::Set, args)) }
    pub fn dict(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Sequence(SequenceOp::Dict, args)) }
    pub fn vector(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Sequence(SequenceOp::Vector, args)) }

    pub fn not(self: Self, loc: Location) -> Expr { self.unary(loc, UnaryOp::Not) }
    pub fn unary(self: Self, loc: Location, op: UnaryOp) -> Expr { Expr(loc, ExprType::Unary(op, Box::new(self))) }
    pub fn binary(self: Self, loc: Location, op: BinaryOp, rhs: Expr) -> Expr { Expr(loc, ExprType::Binary(op, Box::new(self), Box::new(rhs))) }
    pub fn eval(self: Self, loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, ExprType::Eval(Box::new(self), args)) }
    pub fn compose(self: Self, loc: Location, f: Expr) -> Expr { Expr(loc, ExprType::Compose(Box::new(self), Box::new(f))) }
    pub fn index(self: Self, loc: Location, index: Expr) -> Expr { Expr(loc, ExprType::Index(Box::new(self), Box::new(index))) }
    pub fn slice(self: Self, loc: Location, arg1: Expr, arg2: Expr) -> Expr { Expr(loc, ExprType::Slice(Box::new(self), Box::new(arg1), Box::new(arg2))) }
    pub fn slice_step(self: Self, loc: Location, arg1: Expr, arg2: Expr, arg3: Expr) -> Expr { Expr(loc, ExprType::SliceWithStep(Box::new(self), Box::new(arg1), Box::new(arg2), Box::new(arg3))) }
    pub fn if_then_else(self: Self, if_true: Expr, if_false: Expr) -> Expr { Expr(Location::empty(), ExprType::IfThenElse(Box::new(self), Box::new(if_true), Box::new(if_false))) }

    pub fn logical(self: Self, loc: Location, op: BinaryOp, rhs: Expr) -> Expr {
        match op {
            BinaryOp::And => Expr(loc, ExprType::LogicalAnd(Box::new(self), Box::new(rhs))),
            BinaryOp::Or => Expr(loc, ExprType::LogicalOr(Box::new(self), Box::new(rhs))),
            _ => panic!("No logical operator for binary operator {:?}", op),
        }
    }
}