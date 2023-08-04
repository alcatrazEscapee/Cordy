use crate::vm::Value;

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