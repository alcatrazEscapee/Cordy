use std::cell::Cell;
use std::iter::FusedIterator;
use std::pin::Pin;
use std::str::Chars;

use crate::vm::{Type, ValuePtr};
use crate::vm::value::{ConstValue, SharedValue};
use crate::vm::value::ptr::SharedPrefix;


impl SharedValue for String {}
impl ConstValue for String {}


/// A specialization of either a `String`, or a `&str`
/// This is used in methods like `ValuePtr.to_str()` which in some cases, can return a unmodified view of a `&str`, but in other cases will always need to construct a new `String`
pub enum RefStr<'a> {
    Owned(String),
    Borrow(&'a str),
}


impl<'a> RefStr<'a> {

    pub fn as_slice(&self) -> &str {
        match self {
            RefStr::Owned(value) => value.as_str(),
            RefStr::Borrow(value) => value,
        }
    }

    pub fn as_owned(self) -> String {
        match self {
            RefStr::Owned(value) => value,
            RefStr::Borrow(value) => String::from(value),
        }
    }
}


pub trait IntoRefStr<'a> {
    fn to_ref_str(self) -> RefStr<'a>;
}

impl<'a> IntoRefStr<'a> for &'a str {
    fn to_ref_str(self) -> RefStr<'a> {
        RefStr::Borrow(self)
    }
}

impl<'a> IntoRefStr<'a> for String {
    fn to_ref_str(self) -> RefStr<'a> {
        RefStr::Owned(self)
    }
}


/// An owned iterator over a string obtained from a `ValuePtr`. This reduces the need for a copy, and escapes lifetime constraints.
#[derive(Debug, Clone)]
pub struct IterStr {
    ptr: IterStrType,
    iter: Chars<'static>,
    /// A cached value of `count()`
    /// When initialized this will be `usize::MAX` which is obviously wrong. We specialize `count()` on `IterStr` to query, possibly cache, and return this value
    count: Cell<usize>,
}

#[derive(Debug, Clone)]
pub enum IterStrType {
    Shared(ValuePtr), // Already behind a `Rc<>`
    Inline(Pin<Box<ValuePtr>>), // Need to pin this somewhere so we can prevent it from being moved
}

impl IterStrType {
    fn as_str_slice(&self) -> &str {
        match self {
            IterStrType::Shared(ptr) => ptr.as_long_str().borrow_const().as_str(),
            IterStrType::Inline(pin) => pin.as_short_str()
        }
    }
}


const INVALID: usize = usize::MAX;

impl IterStr {

    /// This is separate from `iterator.count()` because we can take `&self` instead of `self`
    pub fn count(&self) -> usize {
        let mut count = self.count.get();
        if count == INVALID {
            // Need to update the cached count, in the case we call this `count()` again, and the string is lon
            // We can't consume the iterator, so we go into the original `ptr` again and this time iterate over as a string slice
            count = self.ptr.as_str_slice()
                .chars()
                .count();

            // Update the cached count value
            self.count.set(count);
        }
        count
    }
}

/// Implement all iterator types that `Chars` does that just bounce to the held `iter`
impl Iterator for IterStr {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl DoubleEndedIterator for IterStr {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl FusedIterator for IterStr {}


/// Implementation details of string types
///
/// We implement a 'small string' optimization in Cordy, by having two separate string representations:
/// 1. A 'small string', which is an inline 7-byte buffer, stored in a `ValuePtr` inline.
/// 2. A 'large string', which is a shared (so reference counted) heap allocated buffer.
///
/// Both small strings and large strings expose themselves as a `&str` as much as possible, because it simplifies the API
impl ValuePtr {

    pub fn is_str(&self) -> bool {
        // Faster check (short string) is first
        self.is_short_str() || self.is_long_str()
    }

    /// Returns the `ValuePtr` as a string slice.
    ///
    /// The value must either be a `Type::ShortStr` or `Type::LongStr`
    pub fn as_str_slice(&self) -> &str {
        debug_assert!(self.is_str());
        if self.is_short_str() { // Faster path (doesn't have to check `.ty()`)
            self.as_short_str()
        } else {
            self.as_long_str().borrow_const().as_str()
        }
    }

    /// Returns the `ValuePtr` as a heap-allocated, owned `String`.
    ///
    /// **Note**: This will always copy the underlying `String`, even when it is a `Type::LongStr`. Do not use unless an owned `String` is absolutely required.
    pub fn as_str_owned(&self) -> String {
        String::from(self.as_str_slice())
    }

    /// Returns the `ValuePtr` as a owned, character iterator wrapper
    ///
    /// The value must either be a `Type::ShortStr` or `Type::LongStr`
    pub fn as_str_iter(self) -> IterStr {
        debug_assert!(self.is_str());
        let (ptr, iter) = if self.is_short_str() { // Faster path (doesn't have to check `.ty()`)
            let pin = Box::pin(self);
            let iter: Chars<'static> = unsafe {
                std::mem::transmute(pin.as_short_str().chars())
            };
            (IterStrType::Inline(pin), iter)
        } else {
            let iter: Chars<'static> = unsafe {
                std::mem::transmute(self.as_long_str().borrow_const().chars())
            };
            (IterStrType::Shared(self), iter)
        };
        IterStr { ptr, iter, count: Cell::new(INVALID) }
    }

    /// Don't expose this (or `as_short_str`), only interact through `&str`
    fn as_long_str(&self) -> &SharedPrefix<String> {
        debug_assert!(self.ty() == Type::LongStr);
        self.as_shared_ref()
    }

    fn is_long_str(&self) -> bool {
        self.ty() == Type::LongStr
    }
}

#[cfg(test)]
mod tests {
    use crate::vm::{IntoValue, ValuePtr};

    #[test]
    fn test_long_str() {
        let long_str = "this is a long string".to_value();

        assert_eq!(long_str.as_str_slice(), "this is a long string");
        assert!(long_str.is_str());
        assert!(long_str.is_long_str());
    }

    #[test]
    fn test_short_str() {
        let short_str = "abc".to_value();

        assert_eq!(short_str.as_str_slice(), "abc");
        assert!(short_str.is_str());
        assert!(short_str.is_short_str());
    }

    #[test]
    fn test_empty_str() {
        let empty_str = "".to_value();

        assert_eq!(empty_str.as_str_slice(), "");
        assert!(empty_str.is_str());
        assert!(empty_str.is_short_str());
    }

    #[test]
    fn test_raw_empty_str() {
        let empty_str = ValuePtr::empty_str();

        assert_eq!(empty_str.as_str_slice(), "");
        assert!(empty_str.is_str());
        assert!(empty_str.is_short_str());
    }

    #[test]
    fn test_iter_through_short_str() {
        let mut short_str = "abc".to_value().as_str_iter();

        assert_eq!(short_str.next(), Some('a'));
        assert_eq!(short_str.next(), Some('b'));
        assert_eq!(short_str.next(), Some('c'));
        assert_eq!(short_str.next(), None);
    }
}