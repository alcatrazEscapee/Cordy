use std::convert::Infallible;
use std::ops::{ControlFlow, FromResidual, Try};

use crate::vm::{RuntimeError, ValuePtr};


/// # ErrorPtr
///
/// This is a `ValuePtr` which is guaranteed to represent an error. It is used in two situations:
///
/// - As the error type of `ErrorResult<T>`, which is a type used to convert `ValuePtr` -> `Result<T, ErrorPtr>`
/// - As the error type of `ValueResult`, which is a zero-cost abstraction for `Result<ValuePtr, ErrorPtr>`
///
/// It can be converted to a `&RuntimeError` via the borrow on `as_err()`, although the `ErrorPtr` manages the lifetime of the error.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorPtr {
    ptr: ValuePtr
}

/// A type for representing an error that may occur in converting a `ValuePtr` into a non-Cordy type.
/// i.e. `as_int_checked()` returns an `ErrorResult<i64>`
///
/// As this type represents a non-zero abstraction over `Result<T, ErrorPtr>`, it should only be used and immediately unboxed.
/// Extended return types all should be used as `ValueResult` when returning Cordy-type values.
pub type ErrorResult<T> = Result<T, ErrorPtr>;

/// A type for representing any kind of optional error result, which may be `Ok` or produce an error.
/// i.e. as the return type of the VM's top-level `run()` method.
pub type AnyResult = ErrorResult<()>;


impl ErrorPtr {
    pub fn new(ptr: ValuePtr) -> ErrorPtr {
        debug_assert!(ptr.is_err());
        ErrorPtr { ptr }
    }

    pub fn as_err(&self) -> &RuntimeError {
        self.ptr.as_err()
    }
}

impl From<ErrorPtr> for ValueResult {
    fn from(value: ErrorPtr) -> Self {
        ValueResult::err(value.ptr)
    }
}


/// # ValueResult
///
/// This is an abstraction for `Result<ValuePtr, ErrorPtr>` where:
///
/// - The memory layout is identical to a `ValuePtr`, incurring no overhead from additional size of `Result<T, E>` due to enum discriminants
/// - This indicates that, instead of returning a `ValuePtr` which is not an error, that we need to always check this for a possible error
/// - The `ok()` field of this `ValueResult` is ensured to never be an error
#[derive(Debug, Eq, PartialEq)]
pub struct ValueResult {
    ptr: ValuePtr
}

impl ValueResult {
    pub fn ok(ptr: ValuePtr) -> ValueResult {
        debug_assert!(!ptr.is_err()); // Should never have an error-type in `ok()`
        ValueResult { ptr }
    }

    #[cold]
    pub fn err(ptr: ValuePtr) -> ValueResult {
        debug_assert!(ptr.is_err());
        ValueResult { ptr }
    }

    /// Boxes this `ValueResult` into a traditional Rust `Result<T, E>`. The left will be guaranteed to contain a non-error type, and the right will be guaranteed to contain an error.
    #[inline(always)]
    pub fn as_result(self) -> ErrorResult<ValuePtr> {
        match self.ptr.is_err() {
            true => Err(ErrorPtr::new(self.ptr)),
            false => Ok(self.ptr)
        }
    }

    #[inline]
    pub fn is_ok(&self) -> bool {
        !self.ptr.is_err()
    }

    #[inline]
    pub fn is_err(&self) -> bool {
        self.ptr.is_err()
    }
}


/// Converts a `ErrorResult<T>` returned from a `?` expression into a `ValueResult`
impl FromResidual<ErrorResult<Infallible>> for ValueResult {
    fn from_residual(residual: ErrorResult<Infallible>) -> Self {
        match residual {
            Err(err) => ValueResult::err(err.ptr),
            _ => unreachable!(),
        }
    }
}

/// Converts a `ErrorPtr` returned as the residual from a `ValueResult?` into a `ErrorResult<T>`
impl<T> FromResidual<ErrorPtr> for ErrorResult<T> {
    fn from_residual(residual: ErrorPtr) -> Self {
        Err(residual)
    }
}

/// Allows us to use `?` operator on `check_<T>()?` expressions, which assert that the value is of a given type, and early return if not.
/// This does, unfortunately, require nightly unstable rust, but the code clarity is worth having (and it allows us to seamlessly use `ValueResult`
/// as a zero-cost (memory layout) wise abstraction for `Result<ValuePtr, ErrorPtr>` (which would otherwise be twice the stack size.
impl Try for ValueResult {
    type Output = ValuePtr;
    type Residual = ErrorPtr;

    fn from_output(ptr: ValuePtr) -> ValueResult {
        ValueResult { ptr }
    }

    fn branch(self) -> ControlFlow<ErrorPtr, ValuePtr> {
        match self.ptr.is_err() {
            true => ControlFlow::Break(ErrorPtr::new(self.ptr)),
            false => ControlFlow::Continue(self.ptr),
        }
    }
}

/// Associated type for `Try`
impl FromResidual for ValueResult {
    fn from_residual(residual: ErrorPtr) -> ValueResult {
        ValueResult { ptr: residual.ptr }
    }
}

/// Used by iterators to produce a `Vec<ValuePtr>` from a operation that may return an error
impl FromIterator<ValueResult> for ErrorResult<Vec<ValuePtr>> {
    fn from_iter<T: IntoIterator<Item=ValueResult>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut vec: Vec<ValuePtr> = Vec::with_capacity(iter.size_hint().0);
        for ptr in iter {
            match ptr.as_result() {
                Ok(ptr) => vec.push(ptr),
                Err(err) => return Err(err),
            }
        }
        Ok(vec)
    }
}