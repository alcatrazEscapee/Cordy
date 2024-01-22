use crate::compiler::parser::semantic::{LValue, LValueReference};
use crate::core::NativeFunction;
use crate::reporting::Location;
use crate::vm::{ComplexType, ErrorPtr, LiteralType, Opcode, Type, ValuePtr, ValueResult};
use crate::vm::operator::{BinaryOp, CompareOp, UnaryOp};

use ExprType::{*};

#[derive(Debug, Clone)]
pub struct Expr(pub Location, pub ExprType);

type ExprPtr = Box<Expr>;

#[derive(Debug, Clone)]
pub enum ExprType {
    /// The constant `nil`
    Nil,
    /// The keyword `exit`, which invokes the `Exit` opcode, but is treated in expressions as a terminal (like `nil`)
    Exit,
    /// A boolean, either `true` or `false`
    Bool(bool),
    /// An integer literal
    Int(i64),
    /// A complex literal
    Complex(ComplexType),
    /// A string literal
    Str(String),
    /// A native function.
    NativeFunction(NativeFunction),
    /// A reference to a field name directly, via a `(->name)` literal.
    Field(u32),

    /// A `_` literal, which may be present in pattern assignment expressions, but is invalid as an expression itself
    Empty,
    /// A `*_` literal, which may be present in pattern assignment expressions, but is invalid as an expression itself
    VarEmpty,

    /// An error that occurred during constant evaluation. This can be triggered from something such as `1 + print`
    /// This does not emit any bytecode, and will rather raise the corresponding error in the parser when emitted
    Error(ErrorPtr),
    /// A user function - directly referenced as part of an expression by virtue of being included as a unnamed function like `(fn() {})`
    Function {
        function_id: u32,
        closed_locals: Vec<Opcode>
    },

    LValue(LValueReference),
    SliceLiteral(ExprPtr, ExprPtr, Box<Option<Expr>>),

    // Operators + Functions
    Unary(UnaryOp, ExprPtr),

    /// Arguments are `op, lhs, rhs, swap`
    /// If `swap` is `true`, then **both** of the following effects will apply:
    ///
    /// 1. The arguments `lhs` and `rhs` will be evaluated in the opposite semantic order
    /// 2. A `Swap` opcode will be emitted.
    ///
    /// This means that a `binary_with(op, lhs, rhs, true)` is equivalent to saying that "rhs needs to be evaluated before lhs in the operation op(lhs, rhs)"
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

    // Assignments
    Assignment(LValueReference, ExprPtr),
    ArrayAssignment(ExprPtr, ExprPtr, ExprPtr),

    /// Note that `BinaryOp::NotEqual` is used to indicate this is a `Compose()` operation under the hood
    ArrayOpAssignment(ExprPtr, ExprPtr, BinaryOp, ExprPtr),
    PatternAssignment(LValue, ExprPtr),
}

impl ExprType {
    fn at(self, loc: Location) -> Expr {
        Expr(loc, self)
    }
}


impl Expr {

    pub fn nil() -> Expr { Expr(Location::empty(), Nil) }
    pub fn exit(loc: Location) -> Expr { Exit.at(loc) }

    pub fn bool(loc: Location, value: bool) -> Expr { Bool(value).at(loc) }
    pub fn int(loc: Location, value: i64) -> Expr { Int(value).at(loc) }
    pub fn complex(loc: Location, value: ComplexType) -> Expr { Complex(value).at(loc) }
    pub fn str(loc: Location, value: String) -> Expr { Str(value).at(loc) }
    pub fn native(loc: Location, value: NativeFunction) -> Expr { NativeFunction(value).at(loc) }
    pub fn field(loc: Location, field_index: u32) -> Expr { Field(field_index).at(loc) }

    pub fn empty(loc: Location) -> Expr { Empty.at(loc) }
    pub fn var_empty(loc: Location) -> Expr { VarEmpty.at(loc) }

    pub fn error(loc: Location, error: ErrorPtr) -> Expr { Error(error).at(loc) }
    pub fn function(function_id: u32, closed_locals: Vec<Opcode>) -> Expr { Expr(Location::empty(), Function { function_id, closed_locals }) }

    pub fn lvalue(loc: Location, lvalue: LValueReference) -> Expr {
        match lvalue {
            LValueReference::NativeFunction(native) => Expr::native(loc, native),
            lvalue => Expr(loc, ExprType::LValue(lvalue))
        }
    }

    pub fn assign_lvalue(loc: Location, lvalue: LValueReference, expr: Expr) -> Expr { Expr(loc, Assignment(lvalue, Box::new(expr))) }
    pub fn assign_pattern(loc: Location, lvalue: LValue, expr: Expr) -> Expr { Expr(loc, PatternAssignment(lvalue, Box::new(expr))) }
    pub fn assign_array(loc: Location, array: Expr, index: Expr, rhs: Expr) -> Expr { Expr(loc, ArrayAssignment(Box::new(array), Box::new(index), Box::new(rhs))) }
    pub fn assign_op_array(loc: Location, array: Expr, index: Expr, op: BinaryOp, rhs: Expr) -> Expr { Expr(loc, ArrayOpAssignment(Box::new(array), Box::new(index), op, Box::new(rhs))) }

    pub fn list(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, Literal(LiteralType::List, args)) }
    pub fn vector(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, Literal(LiteralType::Vector, args)) }
    pub fn set(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, Literal(LiteralType::Set, args)) }
    pub fn dict(loc: Location, args: Vec<Expr>) -> Expr { Expr(loc, Literal(LiteralType::Dict, args)) }

    pub fn raw_slice(loc: Location, arg1: Expr, arg2: Expr, arg3: Option<Expr>) -> Expr { Expr(loc, SliceLiteral(Box::new(arg1), Box::new(arg2), Box::new(arg3))) }

    pub fn unary(self, loc: Location, op: UnaryOp) -> Expr { Expr(loc, Unary(op, Box::new(self))) }
    pub fn binary(self, loc: Location, op: BinaryOp, rhs: Expr, swap: bool) -> Expr { Expr(loc, Binary(op, Box::new(self), Box::new(rhs), swap)) }
    pub fn compare(self, mut ops: Vec<(Location, CompareOp, Expr)>) -> Expr {
        match ops.len() {
            0 => self,
            1 => {
                let (loc, op, rhs) = ops.pop().unwrap();
                self.binary(loc, BinaryOp::from(op), rhs, false)
            }
            _ => Expr(Location::empty(), Compare(Box::new(self), ops))
        }
    }
    pub fn unroll(self, loc: Location, first: bool) -> Expr { Expr(loc, Unroll(Box::new(self), first)) }
    pub fn eval(self, loc: Location, args: Vec<Expr>, any_unroll: bool) -> Expr { Expr(loc, Eval(Box::new(self), args, any_unroll)) }
    pub fn compose(self, loc: Location, f: Expr) -> Expr { Expr(loc, Compose(Box::new(self), Box::new(f))) }
    pub fn index(self, loc: Location, index: Expr) -> Expr { Expr(loc, Index(Box::new(self), Box::new(index))) }
    pub fn slice(self, loc: Location, arg1: Expr, arg2: Expr) -> Expr { Expr(loc, Slice(Box::new(self), Box::new(arg1), Box::new(arg2))) }
    pub fn slice_step(self, loc: Location, arg1: Expr, arg2: Expr, arg3: Expr) -> Expr { Expr(loc, SliceWithStep(Box::new(self), Box::new(arg1), Box::new(arg2), Box::new(arg3))) }
    pub fn if_then_else(self, loc: Location, if_true: Expr, if_false: Expr) -> Expr { Expr(loc, IfThenElse(Box::new(self), Box::new(if_true), Box::new(if_false))) }
    pub fn get_field(self, loc: Location, field_index: u32) -> Expr { Expr(loc, GetField(Box::new(self), field_index)) }
    pub fn set_field(self, loc: Location, field_index: u32, rhs: Expr) -> Expr { Expr(loc, SetField(Box::new(self), field_index, Box::new(rhs))) }
    pub fn swap_field(self, loc: Location, field_index: u32, rhs: Expr, op: BinaryOp) -> Expr { Expr(loc, SwapField(Box::new(self), field_index, Box::new(rhs), op)) }

    pub fn logical(self, loc: Location, op: BinaryOp, rhs: Expr) -> Expr {
        match op {
            BinaryOp::And => Expr(loc, LogicalAnd(Box::new(self), Box::new(rhs))),
            BinaryOp::Or => Expr(loc, LogicalOr(Box::new(self), Box::new(rhs))),
            _ => panic!("No logical operator for binary operator {:?}", op),
        }
    }

    pub fn value(loc: Location, value: ValuePtr) -> Expr {
        match value.ty() {
            Type::Nil => Expr::nil(),
            Type::Bool => Expr::bool(loc, value.as_bool()),
            Type::Int => Expr::int(loc, value.as_int()),
            Type::Complex => Expr::complex(loc, value.to_complex()),
            Type::ShortStr | Type::LongStr => Expr::str(loc, value.as_str_owned()),
            _ => panic!("Not a constant value type"),
        }
    }

    pub fn value_result(loc: Location, value: ValueResult) -> Expr {
        match value.as_result() {
            Ok(value) => Expr::value(loc, value),
            Err(e) => Expr::error(loc, e),
        }
    }

    pub fn is_unroll(&self) -> bool { matches!(self, Expr(_, ExprType::Unroll(_, _))) }
}