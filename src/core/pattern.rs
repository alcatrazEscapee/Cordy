use crate::core;
use crate::vm::{IntoValue, StoreOp, ValuePtr, VirtualInterface};
use crate::vm::RuntimeError;

use RuntimeError::{*};


#[derive(Debug, Clone)]
pub struct Pattern {
    len: usize,
    variadic: bool,
    terms: Vec<Term>
}

#[derive(Debug, Clone)]
enum Term {
    Index(i64, StoreOp),
    Slice(i64, i64, StoreOp),
    Pattern(i64, Pattern),
}


impl Pattern {

    pub fn new(len: usize, variadic: bool) -> Pattern {
        Pattern { len, variadic, terms: Vec::new() }
    }

    pub fn push_index(&mut self, index: i64, op: StoreOp) {
        self.terms.push(Term::Index(index, op));
    }

    pub fn push_slice(&mut self, low: i64, high: i64, op: StoreOp) {
        self.terms.push(Term::Slice(low, high, op));
    }

    pub fn push_pattern(&mut self, index: i64, pattern: Pattern) {
        self.terms.push(Term::Pattern(index, pattern))
    }

    pub fn apply<VM : VirtualInterface>(&self, vm: &mut VM, ptr: &ValuePtr) -> Result<(), Box<RuntimeError>> {
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

    fn check_length(&self, ptr: &ValuePtr) -> Result<(), Box<RuntimeError>> {
        let len = ptr.len()?;
        match self.variadic {
            true if self.len > len => ValueErrorCannotUnpackLengthMustBeGreaterThan(self.len as u32, len, ptr.clone()).err(),
            false if self.len != len => ValueErrorCannotUnpackLengthMustBeEqual(self.len as u32, len, ptr.clone()).err(),
            _ => Ok(()),
        }
    }
}