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
        debug_assert!(matches!(self.ty(), Type::ShortStr | Type::LongStr));
        if self.is_short_str() { // Faster path (doesn't have to check `.ty()`) check
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
}