use crate::compiler::parser::semantic::{LValue, LValueReference};
use crate::core::NativeFunction;
use crate::reporting::Location;
use crate::vm::{LiteralType, Opcode, RuntimeError, Type, ValuePtr, ValueResult};
use crate::vm::operator::{BinaryOp, CompareOp, UnaryOp};


#[derive(Debug, Clone)]
pub struct Expr(pub Location, pub ExprType);

type ExprPtr = Box<Expr>;

#[derive(Debug, Clone)]
pub enum ExprType {
    // Terminals
    Nil,
    Exit,
    Bool(bool),
    Int(i64),
    Complex(num_complex::Complex<i64>),
    Str(String),
    LValue(LValueReference),
    NativeFunction(NativeFunction),
    Function(u32, Vec<Opcode>),
    SliceLiteral(ExprPtr, ExprPtr, Box<Option<Expr>>),

    // Operators + Functions
    Unary(UnaryOp, ExprPtr),

    /// Arguments are `op, lhs, rhs, swap`
    /// If `swap` is `true`, then **both** of the following effects will apply:
    ///
    /// 1. The arguments `lhs` and `rhs` will be evaluated in the opposite semantic order
    /// 2. A `Swap` opcode will be emitted.
    ///
    /// This means that a `binary_with(op, lhs, rhs, swap)` is equivalent to saying that "rhs needs to be evaluated before lhs in the operation op(lhs, rhs)"
    ///
    /// **The side (right vs left) of each argument is still correct!!!**
    Binary(BinaryOp, ExprPtr, ExprPtr, bool),
    /// Expression for chained comparison operators
    Compare(ExprPtr, Vec<(Location, CompareOp, Expr)>),
    Literal(LiteralType, Vec<Expr>),
    Unroll(ExprPtr, bool), // first: bool
    Eval(ExprPtr, Vec<Expr>, bool), // any_unroll: bool
    Compose(ExprPtr, ExprPtr),
    LogicalAnd(ExprPtr, ExprPtr),
    LogicalOr(ExprPtr, ExprPtr),
    Index(ExprPtr, ExprPtr),
    Slice(ExprPtr, ExprPtr, ExprPtr),
    SliceWithStep(ExprPtr, ExprPtr, ExprPtr, ExprPtr),
    IfThenElse(ExprPtr, ExprPtr, ExprPtr),
    GetField(ExprPtr, u32),
    SetField(ExprPtr, u32, ExprPtr),
    SwapField(ExprPtr, u32, ExprPtr, BinaryOp),
    GetFieldFunction(u32),

    // Assignments
    Assignment(LValueReference, ExprPtr),
    ArrayAssignment(ExprPtr, ExprPtr, ExprPtr),

    /// Note that `BinaryOp::NotEqual` is used to indicate this is a `Compose()` operation under the hood
    ArrayOpAssignment(ExprPtr, ExprPtr, BinaryOp, ExprPtr),
    PatternAssignment(LValue, ExprPtr),

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
    pub fn complex(it: i64) -> Expr { Expr::c64(num_complex::Complex::new(0, it)) }
    pub fn c64(it: num_complex::Complex<i64>) -> Expr { Expr(Location::empty(), ExprType::Complex(it)) }
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

    pub fn unary(self, loc: Location, op: UnaryOp) -> Expr { Expr(loc, ExprType::Unary(op, Box::new(self))) }
    pub fn binary(self, loc: Location, op: BinaryOp, rhs: Expr, swap: bool) -> Expr { Expr(loc, ExprType::Binary(op, Box::new(self), Box::new(rhs), swap)) }
    pub fn compare(self, mut ops: Vec<(Location, CompareOp, Expr)>) -> Expr {
        match ops.len() {
            0 => self,
            1 => {
                let (loc, op, rhs) = ops.pop().unwrap();
                self.binary(loc, op.to_binary(), rhs, false)
            }
            _ => Expr(Location::empty(), ExprType::Compare(Box::new(self), ops))
        }
    }
    pub fn unroll(self, loc: Location, first: bool) -> Expr { Expr(loc, ExprType::Unroll(Box::new(self), first)) }
    pub fn eval(self, loc: Location, args: Vec<Expr>, any_unroll: bool) -> Expr { Expr(loc, ExprType::Eval(Box::new(self), args, any_unroll)) }
    pub fn compose(self, loc: Location, f: Expr) -> Expr { Expr(loc, ExprType::Compose(Box::new(self), Box::new(f))) }
    pub fn index(self, loc: Location, index: Expr) -> Expr { Expr(loc, ExprType::Index(Box::new(self), Box::new(index))) }
    pub fn slice(self, loc: Location, arg1: Expr, arg2: Expr) -> Expr { Expr(loc, ExprType::Slice(Box::new(self), Box::new(arg1), Box::new(arg2))) }
    pub fn slice_step(self, loc: Location, arg1: Expr, arg2: Expr, arg3: Expr) -> Expr { Expr(loc, ExprType::SliceWithStep(Box::new(self), Box::new(arg1), Box::new(arg2), Box::new(arg3))) }
    pub fn if_then_else(self, loc: Location, if_true: Expr, if_false: Expr) -> Expr { Expr(loc, ExprType::IfThenElse(Box::new(self), Box::new(if_true), Box::new(if_false))) }
    pub fn get_field(self, loc: Location, field_index: u32) -> Expr { Expr(loc, ExprType::GetField(Box::new(self), field_index)) }
    pub fn set_field(self, loc: Location, field_index: u32, rhs: Expr) -> Expr { Expr(loc, ExprType::SetField(Box::new(self), field_index, Box::new(rhs))) }
    pub fn swap_field(self, loc: Location, field_index: u32, rhs: Expr, op: BinaryOp) -> Expr { Expr(loc, ExprType::SwapField(Box::new(self), field_index, Box::new(rhs), op)) }
    pub fn get_field_function(loc: Location, field_index: u32) -> Expr { Expr(loc, ExprType::GetFieldFunction(field_index)) }

    pub fn logical(self, loc: Location, op: BinaryOp, rhs: Expr) -> Expr {
        match op {
            BinaryOp::And => Expr(loc, ExprType::LogicalAnd(Box::new(self), Box::new(rhs))),
            BinaryOp::Or => Expr(loc, ExprType::LogicalOr(Box::new(self), Box::new(rhs))),
            _ => panic!("No logical operator for binary operator {:?}", op),
        }
    }

    pub fn value(value: ValuePtr) -> Expr {
        match value.ty() {
            Type::Nil => Expr::nil(),
            Type::Bool => Expr::bool(value.as_bool()),
            Type::Int => Expr::int(value.as_int()),
            Type::Complex => Expr::c64(value.as_precise_complex().value.inner),
            Type::ShortStr | Type::LongStr => Expr::str(value.as_str_owned()),
            _ => panic!("Not a constant value type"),
        }
    }

    pub fn error(loc: Location, error: Box<RuntimeError>) -> Expr {
        Expr(loc, ExprType::RuntimeError(error))
    }

    pub fn value_result(loc: Location, value: ValueResult) -> Expr {
        match value.as_result() {
            Ok(value) => Expr::value(value),
            Err(e) => Expr::error(loc, Box::new(e.value)),
        }
    }

    pub fn is_unroll(&self) -> bool { matches!(self, Expr(_, ExprType::Unroll(_, _))) }
}