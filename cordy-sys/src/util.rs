use std::io;
use std::io::{BufRead, Read};
use std::ops::{ControlFlow, Try};

use crate::vm::{ErrorResult, Prefix, RuntimeError};

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
pub fn catch<T>(err: &mut Option<Box<Prefix<RuntimeError>>>, f: impl FnOnce() -> ErrorResult<T>, default: T) -> T {
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


pub struct EmptyRead;

impl Read for EmptyRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { Ok(0) }
}

impl BufRead for EmptyRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { Ok(&[]) }
    fn consume(&mut self, _: usize) {}
}


#[cfg(test)] use std::{env, fs};
#[cfg(test)] use std::path::PathBuf;
#[cfg(test)] use crate::SourceView;

/// Version of `assert_eq` with explicit actual and expected parameters, that prints the entire thing including newlines.
#[cfg(test)]
pub fn assert_eq(actual: String, expected: String) {
    assert_eq!(actual, expected, "\n=== Expected ===\n{}\n=== Actual ===\n{}\n", expected, actual);
}


#[cfg(test)]
pub struct Resource {
    root: PathBuf,
}

#[cfg(test)]
impl Resource {

    pub fn new(resource_type: &'static str, path: &'static str) -> (Resource, SourceView) {
        let root = [env::var("CARGO_MANIFEST_DIR").unwrap().as_str(), "test", resource_type, format!("{}.cor", path).as_str()].iter().collect::<PathBuf>();
        let view = SourceView::new(format!("{}.cor", path), fs::read_to_string(&root).expect(format!("Reading: {:?}", root).as_str()));
        (Resource { root }, view)
    }

    /// Takes `actual`, writes it to `.cor.out`, and compares it against the `.cor.trace` file
    pub fn assert_eq(self: &Self, actual: Vec<String>) {
        let actual: String = actual.join("\n");
        let expected: String = fs::read_to_string(self.root.with_extension("cor.trace"))
            .expect(format!("Reading: {:?}", self.root).as_str());

        fs::write(self.root.with_extension("cor.out"), &actual).unwrap();

        assert_eq(actual, expected.replace("\r", ""));
    }
}
