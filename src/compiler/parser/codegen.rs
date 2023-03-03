use crate::compiler::parser::expr::{Expr, ExprType, SequenceOp};
use crate::compiler::parser::Parser;
use crate::vm::Opcode;
use crate::vm::operator::BinaryOp;

use Opcode::{*};

impl<'a> Parser<'a> {

    pub fn emit_expr(self: &mut Self, expr: Expr) {
        match expr {
            Expr(_, ExprType::Nil) => self.push(Nil),
            Expr(_, ExprType::Exit) => self.push(Exit),
            Expr(_, ExprType::Bool(true)) => self.push(True),
            Expr(_, ExprType::Bool(false)) => self.push(False),
            Expr(_, ExprType::Int(it)) => {
                let id = self.declare_constant(it);
                self.push(Int(id));
            },
            Expr(_, ExprType::Str(it)) => {
                let id = self.declare_string(it);
                self.push(Str(id));
            },
            Expr(loc, ExprType::LValue(lvalue)) => self.push_load_lvalue(loc, lvalue),
            Expr(_, ExprType::NativeFunction(native)) => self.push(NativeFunction(native)),
            Expr(_, ExprType::Function(id, closed_locals)) => {
                self.push(Function(id));
                self.emit_closure_and_closed_locals(closed_locals)
            },
            Expr(loc, ExprType::Unary(op, arg)) => {
                self.emit_expr(*arg);
                self.push_with(Unary(op), loc);
            },
            Expr(loc, ExprType::Binary(op, lhs, rhs)) => {
                self.emit_expr(*lhs);
                self.emit_expr(*rhs);
                self.push_with(Binary(op), loc);
            },
            Expr(loc, ExprType::Sequence(op, args)) => {
                let nargs: u32 = args.len() as u32;
                for arg in args {
                    self.emit_expr(arg);
                }
                self.push_with(match op {
                    SequenceOp::List => List(nargs),
                    SequenceOp::Set => Set(nargs),
                    SequenceOp::Dict => Dict(nargs),
                    SequenceOp::Vector => Vector(nargs),
                }, loc)
            },
            Expr(loc, ExprType::Eval(f, args)) => {
                let nargs: u8 = args.len() as u8;
                self.emit_expr(*f);
                for arg in args {
                    self.emit_expr(arg);
                }
                self.push_with(OpFuncEval(nargs), loc);
            },
            Expr(loc, ExprType::Compose(arg, f)) => {
                self.emit_expr(*arg);
                self.emit_expr(*f);
                self.push(Swap);
                self.push_with(OpFuncEval(1), loc);
            },
            Expr(_, ExprType::LogicalAnd(lhs, rhs)) => {
                self.emit_expr(*lhs);
                let jump_if_false = self.reserve();
                self.push(Pop);
                self.emit_expr(*rhs);
                self.fix_jump(jump_if_false, JumpIfFalse)
            },
            Expr(_, ExprType::LogicalOr(lhs, rhs)) => {
                self.emit_expr(*lhs);
                let jump_if_true = self.reserve();
                self.push(Pop);
                self.emit_expr(*rhs);
                self.fix_jump(jump_if_true, JumpIfTrue);
            },
            Expr(loc, ExprType::Index(array, index)) => {
                self.emit_expr(*array);
                self.emit_expr(*index);
                self.push_with(OpIndex, loc);
            },
            Expr(loc, ExprType::Slice(array, arg1, arg2)) => {
                self.emit_expr(*array);
                self.emit_expr(*arg1);
                self.emit_expr(*arg2);
                self.push_with(OpSlice, loc);
            },
            Expr(loc, ExprType::SliceWithStep(array, arg1, arg2, arg3)) => {
                self.emit_expr(*array);
                self.emit_expr(*arg1);
                self.emit_expr(*arg2);
                self.emit_expr(*arg3);
                self.push_with(OpSliceWithStep, loc);
            },
            Expr(_, ExprType::IfThenElse(condition, if_true, if_false)) => {
                self.emit_expr(*condition);
                let jump_if_false_pop = self.reserve();
                self.emit_expr(*if_true);
                let jump = self.reserve();
                self.fix_jump(jump_if_false_pop, JumpIfFalsePop);
                self.emit_expr(*if_false);
                self.fix_jump(jump, Jump);
            },
            Expr(_, ExprType::Assignment(lvalue, rhs)) => {
                self.emit_expr(*rhs);
                self.push_store_lvalue(lvalue);
            },
            Expr(loc, ExprType::ArrayAssignment(array, index, rhs)) => {
                self.emit_expr(*array);
                self.emit_expr(*index);
                self.emit_expr(*rhs);
                self.push_with(StoreArray, loc);
            },
            Expr(loc, ExprType::ArrayOpAssignment(array, index, op, rhs)) => {
                self.emit_expr(*array);
                self.emit_expr(*index);
                self.push_with(OpIndexPeek, loc);
                self.emit_expr(*rhs);
                match op {
                    BinaryOp::NotEqual => { // Marker to indicate this is a `array[index] .= rhs`
                        self.push(Swap);
                        self.push_with(OpFuncEval(1), loc);
                    },
                    op => self.push_with(Binary(op), loc),
                }
                self.push_with(StoreArray, loc);
            },
            Expr(_, ExprType::PatternAssignment(lvalue, rhs)) => {
                self.emit_expr(*rhs);
                lvalue.emit_destructuring(self, false, true);
            },
            Expr(loc, ExprType::RuntimeError(e)) => {
                self.runtime_error(loc, e);
            }
        }
    }

    pub fn emit_closure_and_closed_locals(self: &mut Self, closed_locals: Vec<Opcode>) {
        if !closed_locals.is_empty() {
            self.push(Closure);
            for op in closed_locals {
                self.push(op);
            }
        }
    }
}