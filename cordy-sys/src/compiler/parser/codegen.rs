use crate::compiler::parser::expr::{Expr, ExprType};
use crate::compiler::parser::optimizer::Optimize;
use crate::compiler::parser::Parser;
use crate::reporting::Location;
use crate::vm::Opcode;
use crate::vm::operator::BinaryOp;

use Opcode::{*};


impl<'a> Parser<'a> {

    /// Emits an expression, with optimizations (if enabled)
    pub fn emit_optimized_expr(&mut self, mut expr: Expr) {
        if self.enable_optimization {
            expr = expr.optimize();
        }
        self.emit_expr(expr);
    }

    /// Recursive version of the above.
    /// Does not call optimizations as the expression is already assumed to be optimized.
    fn emit_expr(&mut self, expr: Expr) {
        match expr {
            Expr(_, ExprType::Nil) => self.push(Nil),
            Expr(_, ExprType::Exit) => self.push(Exit),
            Expr(_, ExprType::Bool(true)) => self.push(True),
            Expr(_, ExprType::Bool(false)) => self.push(False),
            Expr(_, ExprType::Int(it)) => {
                let id = self.declare_const(it);
                self.push(Constant(id));
            },
            Expr(_, ExprType::Complex(it)) => {
                let id = self.declare_const(it);
                self.push(Constant(id))
            }
            Expr(_, ExprType::Str(it)) => {
                let id = self.declare_const(it);
                self.push(Constant(id));
            },
            Expr(loc, ExprType::LValue(lvalue)) => self.push_load_lvalue(loc, lvalue),
            Expr(_, ExprType::NativeFunction(native)) => self.push(NativeFunction(native)),
            Expr(_, ExprType::Function(id, closed_locals)) => {
                self.push(Constant(id));
                self.emit_closure_and_closed_locals(closed_locals)
            },
            Expr(loc, ExprType::SliceLiteral(arg1, arg2, arg3)) => {
                self.emit_expr(*arg1);
                self.emit_expr(*arg2);
                if let Some(arg3) = *arg3 {
                    self.emit_expr(arg3);
                    self.push_with(SliceWithStep, loc);
                } else {
                    self.push_with(Slice, loc);
                }
            },
            Expr(loc, ExprType::Unary(op, arg)) => {
                self.emit_expr(*arg);
                self.push_with(Unary(op), loc);
            },
            Expr(loc, ExprType::Binary(op, lhs, rhs, swap)) => {
                if swap {
                    self.emit_expr(*rhs);
                    self.emit_expr(*lhs);
                    self.push(Swap);
                } else {
                    self.emit_expr(*lhs);
                    self.emit_expr(*rhs);
                }
                self.push_with(Binary(op), loc);
            },
            Expr(loc, ExprType::Literal(op, args)) => {
                self.push(LiteralBegin(op, args.len() as u32));

                let mut acc_args: u32 = 0;
                for arg in args {
                    match arg {
                        Expr(arg_loc, ExprType::Unroll(unroll_arg, _)) => {
                            if acc_args > 0 {
                                self.push(LiteralAcc(acc_args));
                                acc_args = 0;
                            }
                            self.emit_expr(*unroll_arg);
                            self.push_with(LiteralUnroll, arg_loc);
                        },
                        _ => {
                            self.emit_expr(arg);
                            acc_args += 1
                        },
                    }
                }

                if acc_args > 0 {
                    self.push(LiteralAcc(acc_args));
                }

                self.push_with(LiteralEnd, loc);
            },
            Expr(loc, ExprType::Unroll(arg, first)) => {
                self.emit_expr(*arg);
                self.push_with(Unroll(first), loc);
            },
            Expr(loc, ExprType::Eval(f, args, any_unroll)) => {
                let nargs: u32 = args.len() as u32;
                self.emit_expr(*f);
                for arg in args {
                    self.emit_expr(arg);
                }
                self.push_with(Call(nargs, any_unroll), loc);
            },
            Expr(loc, ExprType::Compose(arg, f)) => {
                self.emit_expr(*arg);
                self.emit_expr(*f);
                self.push(Swap);
                self.push_with(Call(1, false), loc);
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
                let jump_if_false_pop = self.reserve_with(Location::empty());
                self.emit_expr(*if_true);
                let jump = self.reserve_with(Location::empty());
                self.fix_jump(jump_if_false_pop, JumpIfFalsePop);
                self.emit_expr(*if_false);
                self.fix_jump(jump, Jump);
            },
            Expr(loc, ExprType::GetField(lhs, field_index)) => {
                self.emit_expr(*lhs);
                self.push_with(GetField(field_index), loc);
            },
            Expr(loc, ExprType::SetField(lhs, field_index, rhs)) => {
                self.emit_expr(*lhs);
                self.emit_expr(*rhs);
                self.push_with(SetField(field_index), loc)
            },
            Expr(loc, ExprType::SwapField(lhs, field_index, rhs, op)) => {
                self.emit_expr(*lhs);
                self.push_with(GetFieldPeek(field_index), loc);
                self.emit_expr(*rhs);
                self.push_with(Binary(op), loc);
                self.push_with(SetField(field_index), loc);
            },
            Expr(loc, ExprType::GetFieldFunction(field_index)) => {
                self.push_with(GetFieldFunction(field_index), loc);
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
                        self.push_with(Call(1, false), loc);
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

    pub fn emit_closure_and_closed_locals(&mut self, closed_locals: Vec<Opcode>) {
        if !closed_locals.is_empty() {
            self.push(Closure);
            for op in closed_locals {
                self.push(op);
            }
        }
    }
}