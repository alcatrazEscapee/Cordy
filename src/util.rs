use std::ops::{ControlFlow, Try};

use crate::vm::RuntimeError;

pub fn strip_line_ending(buffer: &mut String) {
    if buffer.ends_with('\n') {
        buffer.pop();
        if buffer.ends_with('\r') {
            buffer.pop();
        }
    }
}

/// In combination with `join_result`, used to escape a pattern where Rust requires a closure that returns a `T`, but we might error due to having to invoke non-native function code.
/// This function effectively wraps any function that may return `Result<T, E>`, and passes through a silent `default` value in the case of errors
/// The actual error is accumulated in `err`, which is expected to be initialized to `None`
///
/// **Usage**
///
/// ```rs
/// let mut err = None;
/// let ret = some_builtin_inside_a_closure(util::yield_result(&mut err, || {
///     vm.invoke_func1(f, a1)
/// }))
/// let result = misc::join_result(ret, err);
/// ```
#[inline(always)]
pub fn yield_result<T, E>(err: &mut Option<E>, f: impl FnOnce() -> Result<T, E>, default: T) -> T {
    match f() {
        Ok(e) => e,
        Err(e) => {
            *err = Some(e);
            default
        }
    }
}

/// Used to wrap a function that might normally error, with one that passes the error up (i.e. outside of a closure), and returns a default to the inner function.
///
/// N.B. This did not work without specifying the types `Box<RuntimeError>` exactly. I do not know why.
#[inline(always)]
pub fn catch<T>(err: &mut Option<Box<RuntimeError>>, f: impl FnOnce() -> Result<T, Box<RuntimeError>>, default: T) -> T {
    match f().branch() {
        ControlFlow::Continue(e) => e,
        ControlFlow::Break(Err(e)) => {
            *err = Some(e);
            default
        },
        _ => unreachable!(),
    }
}

/// Joins an `Option<Box<RuntimeError>>` error with a result, using fully generic `Try` trait implementation.
#[inline(always)]
pub fn join<T, E, R : Try<Output=T, Residual=E>>(result: T, err: Option<E>) -> R {
    match err {
        Some(e) => R::from_residual(e),
        None => R::from_output(result)
    }
}


/// Used with `yield_result` to escape a convoluted Rust pattern, and use the stdlib more thoroughly.
///
/// See `yield_result()`
#[inline(always)]
pub fn join_result<T, E>(result: T, err: Option<E>) -> Result<T, E> {
    match err {
        Some(e) => Err(e),
        None => Ok(result)
    }
}


pub trait OffsetAdd<F> {
    fn add_offset(self, offset: F) -> Self;
}

impl OffsetAdd<i32> for u32 { fn add_offset(self, offset: i32) -> Self { (self as i32 + offset) as u32 } }
impl OffsetAdd<i32> for usize { fn add_offset(self, offset: i32) -> Self { (self as isize + offset as isize) as usize } }


macro_rules! impl_partial_ord {
    ($ty:ty) => {
        impl PartialOrd for $ty {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(&other))
            }
        }
    };
}

pub(crate) use impl_partial_ord;