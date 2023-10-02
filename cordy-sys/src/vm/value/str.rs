use crate::vm::ValuePtr;


/// Implementation details of string types
///
/// We implement a 'small string' optimization in Cordy, by having two separate string representations:
/// 1. A 'small string', which is an inline 7-byte buffer, stored in a `ValuePtr` inline.
/// 2. A 'large string', which is a shared (so reference counted) heap allocated buffer.
///
/// Both small strings and large strings expose themselves as a `&str` as much as possible, because it simplifies the API
impl ValuePtr {

    pub fn is_str(&self) -> bool {
        self.is_long_str()
    }

    /// Returns the `ValuePtr` as a string slice.
    pub fn as_str_slice(&self) -> &str {
        self.as_long_str().borrow_const().as_str()
    }

    /// Returns the `ValuePtr` as a heap-allocated, owned `String`.
    pub fn as_heap_string(&self) -> String {
        String::from(self.as_str_slice())
    }
}