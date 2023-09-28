use crate::core;
use crate::vm::{AnyResult, IntoValue, StoreOp, ValuePtr, VirtualInterface};
use crate::vm::RuntimeError;

use RuntimeError::{*};


#[derive(Debug, Clone)]
pub struct Pattern<OpType> {
    len: usize,
    variadic: bool,
    terms: Vec<Term<OpType>>
}

#[derive(Debug, Clone)]
enum Term<OpType> {
    Index(i64, OpType),
    Slice(i64, i64, OpType),
    Pattern(i64, Pattern<OpType>),
}


impl<OpType> Pattern<OpType> {
    pub fn new(len: usize, variadic: bool) -> Pattern<OpType> {
        Pattern { len, variadic, terms: Vec::new() }
    }

    pub fn push_index(&mut self, index: i64, op: OpType) {
        self.terms.push(Term::Index(index, op));
    }

    pub fn push_slice(&mut self, low: i64, high: i64, op: OpType) {
        self.terms.push(Term::Slice(low, high, op));
    }

    pub fn push_pattern(&mut self, index: i64, pattern: Pattern<OpType>) {
        self.terms.push(Term::Pattern(index, pattern))
    }

    pub fn visit<Visitor : FnMut(&mut OpType) -> ()>(&mut self, visitor: &mut Visitor) {
        for term in &mut self.terms {
            match term {
                Term::Index(_, op) => visitor(op),
                Term::Slice(_, _, op) => visitor(op),
                Term::Pattern(_, pattern) => pattern.visit(visitor),
            }
        }
    }

    pub fn map<NewType, Visitor : FnMut(OpType) -> NewType>(self, visitor: &mut Visitor) -> Pattern<NewType> {
        Pattern {
            len: self.len,
            variadic: self.variadic,
            terms: self.terms.into_iter().map(|term| {
                match term {
                    Term::Index(index, op) => Term::Index(index, visitor(op)),
                    Term::Slice(lo, hi, op) => Term::Slice(lo, hi, visitor(op)),
                    Term::Pattern(index, pattern) => Term::Pattern(index, pattern.map(visitor)),
                }
            }).collect()
        }
    }
}


/// Implementation for the runtime, which uses `StoreOp`
impl Pattern<StoreOp> {

    /// Executes the pattern against the given value `ptr`
    ///
    /// N.B. This method is considered to take a _partial borrow_ of `vm`, in that it is forbidden from mutating the `patterns` field on the VM
    /// This is of course fine, as the patterns are only modified during compilation, which cannot be invoked as a result of a pattern expression.
    pub fn apply<VM : VirtualInterface>(&self, vm: &mut VM, ptr: &ValuePtr) -> AnyResult {
        self.check_length(ptr)?;

        for term in &self.terms {
            match term {
                Term::Index(index, op) => {
                    let ret = core::get_index(vm, ptr, index.to_value())?;
                    vm.store(*op, ret)?;
                },
                Term::Slice(low, high, op) => {
                    let high = if high == &0 { ValuePtr::nil() } else { high.to_value() };
                    let ret = core::get_slice(ptr, low.to_value(), high, 1i64.to_value())?;
                    vm.store(*op, ret)?;
                },
                Term::Pattern(index, next) => {
                    let ret = core::get_index(vm, ptr, index.to_value())?;
                    next.apply(vm, &ret)?;
                }
            }
        }
        Ok(())
    }

    fn check_length(&self, ptr: &ValuePtr) -> AnyResult {
        let len = ptr.len()?;
        match self.variadic {
            true if self.len > len => ValueErrorCannotUnpackLengthMustBeGreaterThan(self.len as u32, len, ptr.clone()).err(),
            false if self.len != len => ValueErrorCannotUnpackLengthMustBeEqual(self.len as u32, len, ptr.clone()).err(),
            _ => Ok(()),
        }
    }
}