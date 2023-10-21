use crate::compiler::parser::expr::{Expr, ExprType};
use crate::compiler::parser::Parser;
use crate::vm::Opcode;
use crate::vm::operator::{BinaryOp, CompareOp};
use crate::compiler::ParserErrorType;
use crate::compiler::parser::core::{BranchType, ForwardBlockId};
use crate::compiler::optimizer::Optimize;

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
                    self.push_at(SliceWithStep, loc);
                } else {
                    self.push_at(Slice, loc);
                }
            },
            Expr(loc, ExprType::Unary(op, arg)) => {
                self.emit_expr(*arg);
                self.push_at(Unary(op), loc);
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
                self.push_at(Binary(op), loc);
            },
            Expr(_, ExprType::Compare(lhs, mut ops)) => {
                assert!(ops.len() > 1); // Any other situations should be expressed differently

                // Multiple comparison ops need to chain a series of `and` operations, and the last one will be a normal binary op
                // <lhs>
                // <rhs>
                // Compare -> pop both, compare, if true, then push rhs again
                //         -> if false, push false, then jump to end of chain
                // We reserve the compare now, and fix it later, at the top-level compare op
                self.emit_expr(*lhs);

                let mut compare_ops: Vec<(CompareOp, ForwardBlockId)> = Vec::new();
                for (_, op, rhs) in ops.drain(..ops.len() - 1) {
                    self.emit_expr(rhs);
                    compare_ops.push((op, self.branch_forward()));
                }

                // Last op
                // <lhs>
                // <rhs>
                // Binary(op)
                // <- fix all jumps to jump to here
                let (loc, op, rhs) = ops.pop().unwrap();
                self.emit_expr(rhs);
                self.push_at(Binary(op.to_binary()), loc);

                for (op, cmp) in compare_ops {
                    self.join_forward(cmp, BranchType::Compare(op));
                }
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
                            self.push_at(LiteralUnroll, arg_loc);
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

                self.push_at(LiteralEnd, loc);
            },
            Expr(loc, ExprType::Unroll(arg, first)) => {
                self.emit_expr(*arg);
                self.push_at(Unroll(first), loc);
            },
            Expr(loc, ExprType::Eval(f, args, any_unroll)) => {
                let nargs: u32 = args.len() as u32;
                self.emit_expr(*f);
                for arg in args {
                    self.emit_expr(arg);
                }
                self.push_at(Call(nargs, any_unroll), loc);
            },
            Expr(loc, ExprType::Compose(arg, f)) => {
                self.emit_expr(*arg);
                self.emit_expr(*f);
                self.push(Swap);
                self.push_at(Call(1, false), loc);
            },
            Expr(_, ExprType::LogicalAnd(lhs, rhs)) => {
                self.emit_expr(*lhs);
                let branch = self.branch_forward();
                self.push(Pop);
                self.emit_expr(*rhs);
                self.join_forward(branch, BranchType::JumpIfFalse);
            },
            Expr(_, ExprType::LogicalOr(lhs, rhs)) => {
                self.emit_expr(*lhs);
                let branch = self.branch_forward();
                self.push(Pop);
                self.emit_expr(*rhs);
                self.join_forward(branch, BranchType::JumpIfTrue);
            },
            Expr(loc, ExprType::Index(array, index)) => {
                self.emit_expr(*array);
                self.emit_expr(*index);
                self.push_at(OpIndex, loc);
            },
            Expr(loc, ExprType::Slice(array, arg1, arg2)) => {
                self.emit_expr(*array);
                self.emit_expr(*arg1);
                self.emit_expr(*arg2);
                self.push_at(OpSlice, loc);
            },
            Expr(loc, ExprType::SliceWithStep(array, arg1, arg2, arg3)) => {
                self.emit_expr(*array);
                self.emit_expr(*arg1);
                self.emit_expr(*arg2);
                self.emit_expr(*arg3);
                self.push_at(OpSliceWithStep, loc);
            },
            Expr(_, ExprType::IfThenElse(condition, if_true, if_false)) => {
                self.emit_expr(*condition);
                let branch1 = self.branch_forward();
                self.emit_expr(*if_true);
                let branch2 = self.branch_forward();
                self.join_forward(branch1, BranchType::JumpIfFalsePop);
                self.emit_expr(*if_false);
                self.join_forward(branch2, BranchType::Jump);
            },
            Expr(loc, ExprType::GetField(lhs, field_index)) => {
                self.emit_expr(*lhs);
                self.push_at(GetField(field_index), loc);
            },
            Expr(loc, ExprType::SetField(lhs, field_index, rhs)) => {
                self.emit_expr(*lhs);
                self.emit_expr(*rhs);
                self.push_at(SetField(field_index), loc)
            },
            Expr(loc, ExprType::SwapField(lhs, field_index, rhs, op)) => {
                self.emit_expr(*lhs);
                self.push_at(GetFieldPeek(field_index), loc);
                self.emit_expr(*rhs);
                self.push_at(Binary(op), loc);
                self.push_at(SetField(field_index), loc);
            },
            Expr(loc, ExprType::GetFieldFunction(field_index)) => {
                self.push_at(GetFieldFunction(field_index), loc);
            },
            Expr(loc, ExprType::Assignment(lvalue, rhs)) => {
                self.push_store_lvalue_prefix(&lvalue, loc);
                self.emit_expr(*rhs);
                self.push_store_lvalue(lvalue, loc, true);
            },
            Expr(loc, ExprType::ArrayAssignment(array, index, rhs)) => {
                self.emit_expr(*array);
                self.emit_expr(*index);
                self.emit_expr(*rhs);
                self.push_at(StoreArray, loc);
            },
            Expr(loc, ExprType::ArrayOpAssignment(array, index, op, rhs)) => {
                self.emit_expr(*array);
                self.emit_expr(*index);
                self.push_at(OpIndexPeek, loc);
                self.emit_expr(*rhs);
                match op {
                    BinaryOp::NotEqual => { // Marker to indicate this is a `array[index] .= rhs`
                        self.push(Swap);
                        self.push_at(Call(1, false), loc);
                    },
                    op => self.push_at(Binary(op), loc),
                }
                self.push_at(StoreArray, loc);
            },
            Expr(_, ExprType::PatternAssignment(lvalue, rhs)) => {
                self.emit_expr(*rhs);
                lvalue.emit_destructuring(self, false, true);
            },
            Expr(loc, ExprType::RuntimeError(e)) => {
                self.semantic_error_at(loc, ParserErrorType::Runtime(e));
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