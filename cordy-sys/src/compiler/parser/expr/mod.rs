use crate::compiler::parser::semantic::LValueReference;
use crate::core::NativeFunction;
use crate::reporting::Location;
use crate::vm::{ComplexType, ErrorPtr, LiteralType, Opcode, Type, ValuePtr, ValueResult};
use crate::vm::operator::{BinaryOp, CompareOp, UnaryOp};
use crate::compiler::ParserErrorType;

use ExprType::{*};
use ParserErrorType::{*};

mod optimizer;

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

    /// A comma expression represents a series of comma seperated expressions within parenthesis.
    /// It may form one of multiple different expressions depending on context:
    ///
    /// - Resolving precedence
    /// - Vector literals
    /// - Pattern assignment statements
    ///
    /// ### Comma Expression Types
    ///
    /// There are multiple types of comma expressions:
    ///
    /// - `x,`: An _explicit_, _bare_ comma expression, which is only parsed from top-level expressions, and is usable as a vector literal or pattern assignment.
    /// - `(x)`: An _implicit_, comma expression, which may resolve precedence, or used as a pattern assignment.
    /// - `(x,)` An _explicit_, comma expression, which may be as a vector literal, or used as a pattern assignment.
    ///
    /// **N.B.** A _implicit_, _bare_ comma expression would just consist of `x`, so it will never get parsed as such.
    ///
    /// **Note** This is a **transient expression** and should be fully removed by the time `process_expr()` is called, prior to optimization.
    Comma {
        args: Vec<Expr>,
        explicit: bool,
        bare: bool,
    },

    /// A `Call()` represents a function call with the function `ExprPtr` and arguments given in `args`.
    Call {
        f: ExprPtr,
        args: Vec<Expr>,
        unroll: bool
    },

    /// A `Compose()` represents a binary `.` operation, with arguments `lhs . rhs` or `arg . f`
    Compose(ExprPtr, ExprPtr),

    /// Unary `...` applied to an expression as a prefix. It is parsed only from expressions that produce `Comma()` expressions.
    Unroll(ExprPtr),

    /// Unary operator `<op> <expr>`
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

    LogicalAnd(ExprPtr, ExprPtr),
    LogicalOr(ExprPtr, ExprPtr),
    Index(ExprPtr, ExprPtr),
    Slice(ExprPtr, ExprPtr, ExprPtr),
    SliceWithStep(ExprPtr, ExprPtr, ExprPtr, ExprPtr),
    IfThenElse(ExprPtr, ExprPtr, ExprPtr),

    GetField(ExprPtr, u32),
    SetField(ExprPtr, u32, ExprPtr),
    SwapField(ExprPtr, u32, ExprPtr, BinaryOp),

    Literal(LiteralType, Vec<Expr>),
    SliceLiteral(ExprPtr, ExprPtr, Box<Option<Expr>>),

    LValue(LValueReference),

    Assignment(LValueReference, ExprPtr),
    ArrayAssignment(ExprPtr, ExprPtr, ExprPtr),

    /// Note that `BinaryOp::NotEqual` is used to indicate this is a `Compose()` operation under the hood
    ArrayOpAssignment(ExprPtr, ExprPtr, BinaryOp, ExprPtr),
}

/// # Expression Visitor
///
/// This trait can be implemented by a type to provide a visitor, which receives every `Expr` in a nested expression
/// in a post-order traversal.
///
/// Implementing the visitor is done simply, where the visitor may or may not have persistent state to the visitor:
/// ```rust,ignore
/// use crate::compiler::parser::{Visitor, Visitable};
///
/// struct MyVisitor;
///
/// impl Visitor for MyVisitor {
///     fn visit(&mut self, expr: Expr) -> Expr {
///         expr
///     }
/// }
/// ```
///
/// In order to use the visitor and visit all elements, the `Visitable` trait must be used,
/// and the visitor must be passed to a call of `Expr::visit()`
///
/// ```rust,ignore
/// expr.visit(&mut MyVisitor)
/// ```
///
/// **Note:** Calling `Visitor::visit(Expr)` will **not** traverse the entire expression tree, use `Expr::visit(&mut Visitor)` instead.
pub trait Visitor {
    fn visit(&mut self, expr: Expr) -> Expr;
}

/// # Expression Visitable
///
/// This is the trait used to implement the post-order traversal using a `Visitor`. `Expr::visit(&mut Visitor)` is the recursive call, which then recursively traverses through the entire
/// expression tree, calling the same `visit()` method on sub-expressions. After each expression is constructed, each expression is passed through `Visitor::visit(Expr)`, which is the
/// user-visible API which interacts with each expression.
///
/// - `type Output` is intended for types like `ExprPtr`, which is convenient to provide a `Visitable` implementation for, however the output is easier to use as an `Expr`
pub trait Visitable<V : Visitor> {
    type Output;

    fn visit(self, visitor: &mut V) -> Self::Output;
}

impl<V : Visitor> Visitable<V> for Expr {
    type Output = Self;

    fn visit(self, v: &mut V) -> Self {
        let expr = match self {
            Expr(loc, Comma { args, explicit, bare }) => Expr::comma(loc, args.visit(v), explicit, bare),
            Expr(loc, Call { f, args, .. }) => f.visit(v).call(loc, args.visit(v)),
            Expr(loc, Compose(arg, f)) => arg.visit(v).compose(loc, f.visit(v)),
            Expr(loc, Unroll(arg)) => arg.visit(v).unroll(loc),

            Expr(loc, Unary(op, arg)) => arg.visit(v).unary(loc, op),
            Expr(loc, Binary(op, lhs, rhs, swap)) => lhs.visit(v).binary(loc, op, rhs.visit(v), swap),
            Expr(_, Compare(arg, ops)) => arg.visit(v).compare(ops.visit(v)),
            Expr(loc, LogicalAnd(lhs, rhs)) => lhs.visit(v).logical(loc, BinaryOp::And, rhs.visit(v)),
            Expr(loc, LogicalOr(lhs, rhs)) => lhs.visit(v).logical(loc, BinaryOp::Or, rhs.visit(v)),
            Expr(loc, Index(array, index)) => array.visit(v).index(loc, index.visit(v)),
            Expr(loc, Slice(array, arg1, arg2)) => array.visit(v).slice(loc, arg1.visit(v), arg2.visit(v)),
            Expr(loc, SliceWithStep(array, arg1, arg2, arg3)) => array.visit(v).slice_step(loc, arg1.visit(v), arg2.visit(v), arg3.visit(v)),
            Expr(loc, IfThenElse(condition, if_true, if_false)) => condition.visit(v).if_then_else(loc, if_true.visit(v), if_false.visit(v)),

            Expr(loc, GetField(arg, field_index)) => arg.visit(v).get_field(loc, field_index),
            Expr(loc, SetField(arg, field_index, value)) => arg.visit(v).set_field(loc, field_index, value.visit(v)),
            Expr(loc, SwapField(arg, field_index, value, op)) => arg.visit(v).swap_field(loc, field_index, value.visit(v), op),

            Expr(loc, Literal(ty, args)) => Expr::literal(loc, ty, args.visit(v)),
            Expr(loc, SliceLiteral(arg1, arg2, arg3)) => Expr::slice_literal(loc, arg1.visit(v), arg2.visit(v), arg3.visit(v)),

            Expr(loc, Assignment(lvalue, arg)) => Assignment(lvalue, Box::new(arg.visit(v))).at(loc),
            Expr(loc, ArrayAssignment(array, index, value)) => ArrayAssignment(Box::new(array.visit(v)), Box::new(index.visit(v)), Box::new(value.visit(v))).at(loc),
            Expr(loc, ArrayOpAssignment(array, index, op, value)) => ArrayOpAssignment(Box::new(array.visit(v)), Box::new(index.visit(v)), op, Box::new(value.visit(v))).at(loc),

            _ => self,
        };
        v.visit(expr)
    }
}

impl<V : Visitor> Visitable<V> for ExprPtr {
    type Output = Expr;
    fn visit(self, visitor: &mut V) -> Expr {
        (*self).visit(visitor)
    }
}

impl<V : Visitor> Visitable<V> for Box<Option<Expr>> {
    type Output = Option<Expr>;
    fn visit(self, visitor: &mut V) -> Self::Output {
        self.map(|expr| expr.visit(visitor))
    }
}

impl<V : Visitor> Visitable<V> for Vec<Expr> {
    type Output = Self;
    fn visit(self, visitor: &mut V) -> Self {
        self.into_iter().map(|expr| expr.visit(visitor)).collect()
    }
}

impl<T1, T2, V : Visitor> Visitable<V> for Vec<(T1, T2, Expr)> {
    type Output = Self;
    fn visit(self, visitor: &mut V) -> Self::Output {
        self.into_iter().map(|(t1, t2, expr)| (t1, t2, expr.visit(visitor))).collect()
    }
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

    pub fn comma(loc: Location, args: Vec<Expr>, explicit: bool, bare: bool) -> Expr { Comma { args, explicit, bare }.at(loc) }

    /// Given `comma` is a `Comma()` expression, this produces the expression of calling `self` with arguments given by `comma`
    pub fn call_with(self, comma: Expr) -> Expr {
        match comma {
            Expr(loc, Comma { args, .. }) => self.call(loc, args),
            _ => panic!("call() argument must be a Comma() expression")
        }
    }

    pub fn call(self, loc: Location, args: Vec<Expr>) -> Expr {
        let unroll = args.iter().any(|u| u.is_unroll());
        Call { f: Box::new(self), args, unroll }.at(loc)
    }

    pub fn compose(self, loc: Location, f: Expr) -> Expr { Compose(Box::new(self), Box::new(f)).at(loc) }
    pub fn unroll(self, loc: Location) -> Expr { Unroll(Box::new(self)).at(loc) }

    pub fn assign(self, loc: Location, rhs: Expr) -> Result<Expr, ParserErrorType> {
        match self {
            Expr(_, LValue(lvalue @ (
                LValueReference::Local(_) |
                LValueReference::UpValue(_) |
                LValueReference::Global(_) |
                LValueReference::LateBinding(_) |
                LValueReference::ThisField { .. }
            ))) => Ok(Assignment(lvalue, Box::new(rhs)).at(loc)),
            Expr(_, Index(array, index)) => Ok(ArrayAssignment(array, index, Box::new(rhs)).at(loc)),
            Expr(_, GetField(lhs, field_index)) => Ok(SetField(lhs, field_index, Box::new(rhs)).at(loc)),
            Expr(_, Empty | VarEmpty) => Err(AssignmentTargetTrivialEmptyLValue),
            // todo: try and convert to an lvalue pattern expression
            _ => Err(AssignmentTargetInvalid),
        }
    }

    pub fn op_assign(self, loc: Location, op: BinaryOp, rhs: Expr) -> Result<Expr, ParserErrorType> {
        match self {
            lhs @ Expr(_, LValue(
                LValueReference::Local(_) |
                LValueReference::UpValue(_) |
                LValueReference::Global(_) |
                LValueReference::LateBinding(_) |
                LValueReference::ThisField { .. }
            )) => {
                let lvalue = match &lhs { Expr(_, LValue(lv)) => lv.clone(), _ => unreachable!() };
                let assign = match op {
                    BinaryOp::NotEqual => lhs.compose(loc, rhs),
                    op => lhs.binary(loc, op, rhs, false),
                };
                Ok(Assignment(lvalue, Box::new(assign)).at(loc))
            },
            Expr(_, Index(array, index)) => Ok(ArrayOpAssignment(array, index, op, Box::new(rhs)).at(loc)),
            Expr(_, GetField(lhs, field_index)) => Ok(SwapField(lhs, field_index, Box::new(rhs), op).at(loc)),
            Expr(_, Empty | VarEmpty) => Err(AssignmentTargetTrivialEmptyLValue),
            _ => Err(AssignmentTargetInvalid)
        }
    }


    pub fn lvalue(loc: Location, lvalue: LValueReference) -> Expr {
        match lvalue {
            LValueReference::NativeFunction(native) => Expr::native(loc, native),
            lvalue => Expr(loc, LValue(lvalue))
        }
    }

    pub fn list(loc: Location, args: Vec<Expr>) -> Expr { Expr::literal(loc, LiteralType::List, args) }
    pub fn vector(loc: Location, args: Vec<Expr>) -> Expr { Expr::literal(loc, LiteralType::Vector, args) }
    pub fn set(loc: Location, args: Vec<Expr>) -> Expr { Expr::literal(loc, LiteralType::Set, args) }
    pub fn dict(loc: Location, args: Vec<Expr>) -> Expr { Expr::literal(loc, LiteralType::Dict, args) }

    pub fn literal(loc: Location, ty: LiteralType, args: Vec<Expr>) -> Expr { Literal(ty, args).at(loc) }
    pub fn slice_literal(loc: Location, arg1: Expr, arg2: Expr, arg3: Option<Expr>) -> Expr { SliceLiteral(Box::new(arg1), Box::new(arg2), Box::new(arg3)).at(loc) }

    pub fn unary(self, loc: Location, op: UnaryOp) -> Expr { Unary(op, Box::new(self)).at(loc) }
    pub fn binary(self, loc: Location, op: BinaryOp, rhs: Expr, swap: bool) -> Expr { Binary(op, Box::new(self), Box::new(rhs), swap).at(loc) }
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
    pub fn index(self, loc: Location, index: Expr) -> Expr { Index(Box::new(self), Box::new(index)).at(loc) }
    pub fn slice(self, loc: Location, arg1: Expr, arg2: Expr) -> Expr { Slice(Box::new(self), Box::new(arg1), Box::new(arg2)).at(loc) }
    pub fn slice_step(self, loc: Location, arg1: Expr, arg2: Expr, arg3: Expr) -> Expr { SliceWithStep(Box::new(self), Box::new(arg1), Box::new(arg2), Box::new(arg3)).at(loc) }
    pub fn if_then_else(self, loc: Location, if_true: Expr, if_false: Expr) -> Expr { IfThenElse(Box::new(self), Box::new(if_true), Box::new(if_false)).at(loc) }

    pub fn get_field(self, loc: Location, field_index: u32) -> Expr { GetField(Box::new(self), field_index).at(loc) }
    pub fn set_field(self, loc: Location, field_index: u32, rhs: Expr) -> Expr { SetField(Box::new(self), field_index, Box::new(rhs)).at(loc) }
    pub fn swap_field(self, loc: Location, field_index: u32, rhs: Expr, op: BinaryOp) -> Expr { SwapField(Box::new(self), field_index, Box::new(rhs), op).at(loc) }

    pub fn logical(self, loc: Location, op: BinaryOp, rhs: Expr) -> Expr {
        match op {
            BinaryOp::And => LogicalAnd(Box::new(self), Box::new(rhs)).at(loc),
            BinaryOp::Or => LogicalOr(Box::new(self), Box::new(rhs)).at(loc),
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

    pub fn result(loc: Location, value: ValueResult) -> Expr {
        match value.as_result() {
            Ok(value) => Expr::value(loc, value),
            Err(e) => Expr::error(loc, e),
        }
    }

    pub fn is_unroll(&self) -> bool { matches!(self, Expr(_, ExprType::Unroll(_))) }
}