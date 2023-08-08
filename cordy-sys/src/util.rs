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