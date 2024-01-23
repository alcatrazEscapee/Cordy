use crate::compiler::parser::expr::{Expr, ExprType, Visitable, Visitor};
use crate::compiler::parser::Parser;
use crate::vm::{LiteralType, Opcode};
use crate::vm::operator::{BinaryOp, CompareOp};
use crate::compiler::ParserErrorType;
use crate::compiler::parser::core::{BranchType, ForwardBlockId};


use Opcode::{*};


struct Process<'a, 'b : 'a>(&'a mut Parser<'b>);

impl<'a, 'b> Visitor for Process<'a, 'b> {
    fn visit(&mut self, expr: Expr) -> Expr {
        dbg!(&expr);
        match expr {
            // Both `_` and `*_` are illegal transient fragments and emit errors directly
            Expr(loc, ExprType::Empty) => {
                self.0.semantic_error_at(loc, ParserErrorType::LValueEmptyUsedOutsideAssignment);
                Expr::nil()
            },
            Expr(loc, ExprType::VarEmpty) => {
                self.0.semantic_error_at(loc, ParserErrorType::LValueVarEmptyUsedOutsideAssignment);
                Expr::nil()
            },

            // Comma expressions are either resolving precedence, or are converted to a vector literal
            Expr(_, ExprType::Comma { mut args, explicit: false })
                if args.len() == 1 && !matches!(args[0].1, ExprType::Unroll(_))
                => args.pop().unwrap(),
            Expr(loc, ExprType::Comma { args, ..}) => Expr(loc, ExprType::Literal(LiteralType::Vector, args)),

            _ => expr,
        }
    }
}


impl<'a> Parser<'a> {

    /// Emits an expression, with optimizations (if enabled)
    pub fn emit_optimized_expr(&mut self, mut expr: Expr) {
        expr = self.process_expr(expr);
        if self.enable_optimization {
            expr = self.optimize_expr(expr)
        }
        self.emit_expr(expr);
    }

    /// Processes and transforms expressions _prior_ to optimizing and emitting them. This is mainly to allow for
    /// transient expressions, like `Comma()`, to be transformed and validated into easier to work with expressions,
    /// from the view of the optimizer.
    ///
    /// This will, for many expression fragments, emit semantic errors and return `Expr::nil()`, as the fragments do not
    /// need to be visible to the optimizer in the event an error was raised.
    fn process_expr(&mut self, expr: Expr) -> Expr {
        expr.visit(&mut Process(self))
    }

    /// Recursive version of the above.
    /// Does not call optimizations as the expression is already assumed to be optimized.
    fn emit_expr(&mut self, expr: Expr) {
        match expr {
            Expr(_, ExprType::Nil) => self.push(Nil),
            Expr(loc, ExprType::Exit) => self.push_at(Exit, loc),
            Expr(loc, ExprType::Bool(true)) => self.push_at(True, loc),
            Expr(loc, ExprType::Bool(false)) => self.push_at(False, loc),
            Expr(loc, ExprType::Int(it)) => {
                let id = self.declare_const(it);
                self.push_at(Constant(id), loc);
            }
            Expr(loc, ExprType::Complex(it)) => {
                let id = self.declare_const(it);
                self.push_at(Constant(id), loc)
            }
            Expr(loc, ExprType::Str(it)) => {
                let id = self.declare_const(it);
                self.push_at(Constant(id), loc);
            }
            Expr(loc, ExprType::NativeFunction(native)) => self.push_at(NativeFunction(native), loc),
            Expr(loc, ExprType::Field(field_index)) => self.push_at(GetFieldFunction(field_index), loc),

            Expr(loc, ExprType::Error(e)) => self.semantic_error_at(loc, ParserErrorType::Runtime(e)),
            Expr(loc, ExprType::Function { function_id, closed_locals }) => {
                self.push_at(Constant(function_id), loc);
                self.emit_closure_and_closed_locals(closed_locals)
            }

            Expr(loc, ExprType::Call { f, args, .. }) => {
                let nargs: u32 = args.len() as u32;
                let mut unroll: bool = false;

                self.emit_expr(*f);
                for arg in args {
                    match arg {
                        Expr(loc, ExprType::Unroll(arg)) => {
                            self.emit_expr(*arg);
                            self.push_at(Unroll(!unroll), loc);
                            unroll = true;
                        }
                        _ => self.emit_expr(arg)
                    }
                }
                self.push_at(Call(nargs, unroll), loc);
            }
            Expr(loc, ExprType::Compose(arg, f)) => {
                self.emit_expr(*arg);
                self.emit_expr(*f);
                self.push(Swap);
                self.push_at(Call(1, false), loc);
            }

            Expr(loc, ExprType::LValue(lvalue)) => self.push_load_lvalue(loc, lvalue),



            Expr(loc, ExprType::Unary(op, arg)) => {
                self.emit_expr(*arg);
                self.push_at(Unary(op), loc);
            },
            Expr(loc, ExprType::Binary(op, lhs, rhs, swap)) => {
                self.emit_binary_op_args(*lhs, *rhs, swap);
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
                self.push_at(Binary(BinaryOp::from(op)), loc);

                for (op, cmp) in compare_ops {
                    self.join_forward(cmp, BranchType::Compare(op));
                }
            },
            Expr(loc, ExprType::Literal(ty, args)) => {
                self.push_at(LiteralBegin(ty, args.len() as u32), loc);

                let mut acc: u32 = 0;
                for arg in args {
                    match arg {
                        Expr(loc, ExprType::Unroll(arg)) => {
                            if acc > 0 {
                                self.push_at(LiteralAcc(acc), loc);
                                acc = 0;
                            }
                            self.emit_expr(*arg);
                            self.push_at(LiteralUnroll, loc);
                        }
                        _ => {
                            self.emit_expr(arg);
                            acc += 1
                        },
                    }
                }

                if acc > 0 {
                    self.push(LiteralAcc(acc));
                }

                self.push_at(LiteralEnd, loc);
            }
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

            _ => panic!("emit_expr() not implemented for {:?}", expr)
        }
    }

    pub fn emit_binary_op_args(&mut self, lhs: Expr, rhs: Expr, swap: bool) {
        if swap {
            self.emit_expr(rhs);
            self.emit_expr(lhs);
            self.push(Swap);
        } else {
            self.emit_expr(lhs);
            self.emit_expr(rhs);
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