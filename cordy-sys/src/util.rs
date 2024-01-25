use std::io;
use std::io::{BufRead, Read};
use std::ops::{ControlFlow, Try};

#[cfg(test)] use pretty_assertions::{assert_eq};

use crate::vm::{ErrorPtr, ErrorResult};


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
pub fn catch<T>(err: &mut Option<ErrorPtr>, f: impl FnOnce() -> ErrorResult<T>, default: T) -> T {
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

impl OffsetAdd<i32> for u32 {
    fn add_offset(self, offset: i32) -> Self {
        (self as i32 + offset) as u32
    }
}

impl OffsetAdd<i32> for usize {
    fn add_offset(self, offset: i32) -> Self {
        (self as isize + offset as isize) as usize
    }
}


macro_rules! impl_partial_ord {
    ($ty:ty) => {
        impl PartialOrd for $ty {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(&other))
            }
        }
    };
}

macro_rules! impl_deref {
    ($ty:ty, $deref:ty, $f:ident) => {
        impl Deref for $ty {
            type Target = $deref;

            fn deref(&self) -> &Self::Target {
                &self.$f
            }
        }

        impl DerefMut for $ty {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.$f
            }
        }
    };
}

macro_rules! if_cfg {
    ($feature:literal, $if_true:expr, $if_false:expr) => {
        {
            #[cfg(feature = $feature)]
            {
                $if_true
            }
            #[cfg(not(feature = $feature))]
            {
                $if_false
            }
        }
    };
}

pub(crate) use {impl_partial_ord, impl_deref, if_cfg};


pub struct Noop;

impl Read for Noop {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { Ok(0) }
}

impl BufRead for Noop {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { Ok(&[]) }
    fn consume(&mut self, _: usize) {}
}

/// Version of `assert_eq` with explicit actual and expected parameters, that prints the entire thing including newlines.
pub fn assert_eq(actual: String, expected: String) {
    assert_eq!(actual, expected, "\n=== (Left) Actual ===\n\n{}\n\n=== (Right) Expected ===\n\n{}\n\nActual: {:?}", actual, expected, actual);
}