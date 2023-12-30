use std::borrow::Cow;
use std::collections::VecDeque;

use crate::core;
use crate::vm::{ErrorResult, IntoValue, Type, ValuePtr, ValueResult};
use crate::vm::RuntimeError::{TypeErrorArgMustBeInt, TypeErrorArgMustBeSliceable};
use crate::vm::value::{ListImpl, VectorImpl};
use crate::vm::value::ptr::Ref;


/// # Slice
///
/// A `slice` is an instance produced by `[::]` literals. When initially evaluated, all arguments are checked for either integer-like types,
/// or `nil` (which will be treated as `None` / not-present).
///
/// A `Slice` is considered immutable, and shared.
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Clone)]
pub struct Slice {
    start: ValuePtr,
    stop: ValuePtr,
    step: ValuePtr,
}

impl ValuePtr {
    /// Creates a new `Slice` from the given values. Raises an error if any value is not `nil` or int-like.
    pub fn slice(start: ValuePtr, stop: ValuePtr, step: ValuePtr) -> ValueResult {
        #[inline]
        fn check(ptr: ValuePtr) -> ValueResult {
            match ptr.is_int() || ptr.is_nil() {
                true => ptr.ok(),
                false => TypeErrorArgMustBeInt(ptr).err()
            }
        }

        Slice::new(check(start)?, check(stop)?, check(step)?).to_value().ok()
    }
}

impl Slice {
    fn new(start: ValuePtr, stop: ValuePtr, step: ValuePtr) -> Slice {
        Slice { start, stop, step }
    }

    pub fn apply(&self, arg: &ValuePtr) -> ValueResult {
        core::get_slice(arg, self.start.clone(), self.stop.clone(), self.step.clone())
    }

    pub(super) fn to_repr_str(&self) -> Cow<str> {
        #[inline]
        fn to_str(i: &ValuePtr) -> String {
            if i.is_nil() {
                String::new()
            } else {
                i.as_int().to_string()
            }
        }

        Cow::from(match self.step.is_nil() {
            false => format!("[{}:{}:{}]", to_str(&self.start), to_str(&self.stop), self.step.as_int()),
            true => format!("[{}:{}]", to_str(&self.start), to_str(&self.stop)),
        })
    }
}


/// # Sliceable
///
/// A `Sliceable` is a builder for taking slices of an object, and accumulating them in another similar argument.
///
/// Lists, vectors, and strings all are capable of being sliced, and they produce slices of identical values (slices of strings are strings).
/// Thus, each variant of `Sliceable` contains a reference to a value, capable of being sliced, plus a builder for new values.
pub enum Sliceable<'a> {
    Str(&'a str, String),
    List(Ref<'a, ListImpl>, VecDeque<ValuePtr>),
    Vector(Ref<'a, VectorImpl>, Vec<ValuePtr>),
}

impl ValuePtr {
    pub fn to_slice(&self) -> ErrorResult<Sliceable> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Ok(Sliceable::Str(self.as_str_slice(), String::new())),
            Type::List => Ok(Sliceable::List(self.as_list().borrow(), VecDeque::new())),
            Type::Vector => Ok(Sliceable::Vector(self.as_vector().borrow(), Vec::new())),
            _ => TypeErrorArgMustBeSliceable(self.clone()).err()
        }
    }
}

impl IntoValue for Sliceable<'_> {
    fn to_value(self) -> ValuePtr {
        match self {
            Sliceable::Str(_, it) => it.to_value(),
            Sliceable::List(_, it) => it.to_value(),
            Sliceable::Vector(_, it) => it.to_value(),
        }
    }
}

impl Sliceable<'_> {

    /// Returns the length of the original value being sliced.
    pub fn len(&self) -> usize {
        match self {
            Sliceable::Str(it, _) => it.len(),
            Sliceable::List(it, _) => it.list.len(),
            Sliceable::Vector(it, _) => it.vector.len(),
        }
    }

    /// Accepts a new value from the slice into the builder, pushing to the back of the new slice.
    pub fn accept(&mut self, index: i64) {
        if index >= 0 && index < self.len() as i64 {
            let index = index as usize;
            match self {
                Sliceable::Str(src, dest) => dest.push(src.chars().nth(index).unwrap()),
                Sliceable::List(src, dest) => dest.push_back(src.list[index].clone()),
                Sliceable::Vector(src, dest) => dest.push(src.vector[index].clone()),
            }
        }
    }
}