use std::cell::{Cell, UnsafeCell};
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::core::NativeFunction;
use crate::util::impl_partial_ord;
use crate::vm::{Function, Iterable, RuntimeError, StructTypeImpl};
use crate::vm::value::{*};
use crate::vm::value::slice::Slice;


/// `ValuePtr` holds an arbitrary object representable by Cordy.
///
/// In order to minimize memory usage, improving cache efficiency, and reducing memory boxing, `ValuePtr` is structured as a **tagged pointer**.
/// On 64-bit systems, pointers that point to 64-bit (8-byte) aligned objects naturally have the last three bits set to zero.
/// On 32-bit systems (like web assembly), the last two bits are zero.
/// This means we can store a little bit of type information in the low bits of a pointer.
///
/// # Representation
///
/// A `ValuePtr` may be:
///
/// - An inline, 63-bit signed integer, stored with the lowest bit equal to `0`
/// - An inline constant `nil`, `true`, or `false`, `NativeFunction`, or `GetField`, all stored with the lowest three bits equal to `01`
/// - A tagged pointer, to a `Prefix<T>`, with the lowest three bits equal to `11`
///
/// In order to fully determine the type of a given `ValuePtr`, we use a secondary characteristic, which is enabled by having strict, `#[repr(C)]` struct semantics:
/// For every owned, and shared value, we can treat it as a `NonNull<Prefix>` This lets us check the `ty` field, which fully specifies the type of the value.
///
/// In order to fully interpret a `ValuePtr` which is pointing to either owned or shared memory, first, we dereference it, and check the type via `ty`.
/// Then, we are safe to reinterpret the _pointer_ into a pointer to the destination type, including the same prefix. Then, depending on the type, we can either interpret this as a `Box<Prefix<T>>` or a `&SharedPrefix<T>` as desired.
///
/// # Owned vs. Shared Memory
///
/// - **Inline** `ValuePtr`s own no memory. This makes them incredibly fast to both use on the stack, pass to operators, copy, and use. However non-inline `ValuePtr`s
/// are either **owned** or **shared**, determined by the second lowest bit of the pointer. This refers to how the memory pointed to is managed.
///
/// - **Owned** memory should be treated as if the `ValuePtr` was a `Box<Prefix>`. When creating a `.clone()` of the value, the memory needs to be copied. This is
/// used for small immutable types like `C64`, where the additional overhead and semantics of carrying around a `Rc<C64>` does not make sense.
///
/// - **Shared** memory should be treated as if the `ValuePtr` was a `Rc<RefCell<Prefix>>`. Notably, we cannot do this in practice, as that creates a layout restriction on the value being pointed to (the `ty` field of the prefix is no longer the first). So, we have to re-implement the reference counting semantics in order to fully satisfy Rust's memory safety guarantees.
pub union ValuePtr {
    /// Note that we cannot really use a `NonNull<Prefix>` here, because our pointer type has a tag in it.
    /// So every time we would try and use this, we still have to do arithmetic on the pointer to get the actual pointer.
    /// Instead use `as_mut_ptr()`
    _ptr: NonNull<Prefix<()>>,
    tag: usize,
    /// For `i64` types, we need to be able to interact with this value as if it was a `u64` - because we shift into the sign bit, etc.
    /// However, on 32-bit platforms, `sizeof(usize) != sizeof(u64)`, which means if we treat `int` and `tag` representations interchangeably, we end up with
    /// the topmost 32-bits being uninitialized memory. So whenever we store an int, we need to store it as a `long_tag` to properly set the high bits.
    long_tag: u64,
    int: i64,
    /// Used for string slices
    bytes: [u8; 8],
}


const TAG_INT: usize       = 0b______00;
const TAG_NIL: usize       = 0b__000_01;
const TAG_BOOL: usize      = 0b__001_01;
const TAG_FALSE: usize     = 0b0_001_01;
const TAG_TRUE: usize      = 0b1_001_01;
const TAG_NATIVE: usize    = 0b__010_01;
const TAG_NONE: usize      = 0b__011_01;
const TAG_FIELD: usize     = 0b__100_01;
const TAG_STR: usize       = 0b__101_01;
const TAG_PTR: usize       = 0b______11;

const MASK_INT: usize      = 0b_______1;
const MASK_NIL: usize      = 0b__111_11;
const MASK_BOOL: usize     = 0b__111_11;
const MASK_NATIVE: usize   = 0b__111_11;
const MASK_FIELD: usize    = 0b__111_11;
const MASK_STR: usize      = 0b__111_11;
const MASK_NONE: usize     = 0b__111_11;
const MASK_PTR: usize      = 0b______11;

const PTR_MASK: usize = !0b11;
const STR_LEN_MASK: usize = 0b111_000_00;
const STR_TAG_BYTES: usize = 5;
const STR_BYTES: usize = 7;

pub const MAX_INT: i64 = 0x3fff_ffff_ffff_ffffu64 as i64;
pub const MIN_INT: i64 = 0xc000_0000_0000_0000u64 as i64;


pub(super) const fn from_bool(value: bool) -> ValuePtr {
    ValuePtr { tag: if value { TAG_TRUE } else { TAG_FALSE } }
}

pub(super) const fn from_usize(value: usize) -> ValuePtr {
    debug_assert!(value <= MAX_INT as usize); // Check that it is safe to cast to `i64`
    from_i64(value as i64)
}

pub(super) const fn from_i64(value: i64) -> ValuePtr {
    ValuePtr { long_tag: TAG_INT as u64 | ((value << 1) as u64) }
}

pub(super) fn from_char(value: char) -> ValuePtr {
    // `char` encodes in at most 4 bytes, so we always store it as an inline str
    from_small_str(value.encode_utf8(&mut [0; 4]))
}

pub(super) fn from_str(str: &str) -> ValuePtr {
    if str.len() <= STR_BYTES {
        // Small strings are stored inline
        from_small_str(str)
    } else {
        // And larger strings are shared, reference counted and stored on the heap
        from_shared(SharedPrefix::new(Type::LongStr, String::from(str)))
    }
}

fn from_small_str(str: &str) -> ValuePtr {
    debug_assert!(str.len() <= STR_BYTES); // Should be asserted by the callers, but this code will blow up if not

    // The string is short enough that we can store it inline with the lowest byte taking place of the tag.
    // Create a new empty, 8-byte buffer
    let mut bytes: [u8; 8] = [0; 8];

    // Setup the tag byte - it needs to encode both the tag and the length
    // The length can be at most 7, which is 0b111
    // The tag is exactly 5 bytes long, which means we fit perfectly in a unsigned byte (8 bits)
    let tag: u8 = ((str.len() as u8) << STR_TAG_BYTES) | (TAG_STR as u8);

    // Little endian systems represent the slice as:
    //
    // [7b -------- slice ------- ][1b tag]
    // 0   1   2   3   4   5   6   7
    // high byte ----------------- low byte
    //
    // So we can obtain a pointer to 0, with length up to 7
    // The reverse is true for big-endian systems:
    //
    // [1b tag][7b ------ slice ------ ]
    //     0   1   2   3   4   5   6   7
    // low byte ---------------------- high byte
    //
    // Ultimately, this changes how we need to copy the bytes into an array, and where to set the tag
    // We need to handle this differently because we depend on the bytes being re-interpretable as a pointer later
    #[cfg(target_endian = "little")]
    {
        bytes[0] = tag;
        bytes[1..=str.len()].copy_from_slice(str.as_bytes());
    }
    #[cfg(target_endian = "big")]
    {
        bytes[..str.len()].copy_from_slice(str.as_bytes());
        bytes[7] = tag;
    }

    // Using native-endian bytes is acceptable here, because we prepared them according to the native type above
    let long_tag = u64::from_ne_bytes(bytes);

    ValuePtr { long_tag }
}

pub(super) fn from_native(value: NativeFunction) -> ValuePtr {
    ValuePtr { tag: TAG_NATIVE | ((value as usize) << 6) }
}

pub(super) fn from_field(value: u32) -> ValuePtr {
    ValuePtr { long_tag: TAG_FIELD as u64 | ((value as u64) << 6) }
}

pub(super) fn from_owned<T : OwnedValue>(value: Prefix<T>) -> ValuePtr {
    ValuePtr { tag: TAG_PTR | (Box::into_raw(Box::new(value)) as usize) }
}

pub(super) fn from_shared<T : SharedValue>(value: SharedPrefix<T>) -> ValuePtr {
    ValuePtr { tag: TAG_PTR | (Box::into_raw(Box::new(value)) as usize) }
}


impl ValuePtr {

    pub const fn nil() -> ValuePtr {
        ValuePtr { tag: TAG_NIL }
    }

    pub const fn none() -> ValuePtr {
        ValuePtr { tag: TAG_NONE }
    }

    pub const fn empty_str() -> ValuePtr {
        ValuePtr { tag: TAG_STR }
    }

    // `.as_T()` methods for inline types take a `&self` for convenience. Copying the value is the same as copying the reference.

    pub fn as_int(&self) -> i64 {
        debug_assert!(self.is_int());
        if self.is_precise_int() {
            self.as_precise_int()
        } else {
            self.is_true() as i64
        }
    }

    pub(super) fn as_short_str(&self) -> &str {
        unsafe {
            // First, check the length, which is encoded in the high three bits of the tag
            let len = (self.tag & STR_LEN_MASK) >> STR_TAG_BYTES;

            // Need to handle both endian-ness, which encode the string inline slightly differently - see comment in `from_small_str()`
            // - Using `self.bytes` is equivalent to `self.long_tag.to_ne_bytes()`, but it doesn't create data owned by this function
            // - The Rust std lib function just calls `transmute` anyway.
            // - The unchecked access is safe because we know that the buffer will only ever be UTF-8 by construction
            std::str::from_utf8_unchecked({
                #[cfg(target_endian = "little")]
                {
                    &self.bytes[1..=len]
                }
                #[cfg(target_endian = "big")]
                {
                    &self.bytes[..len]
                }
            })
        }
    }

    pub fn as_precise_int(&self) -> i64 {
        debug_assert!(self.is_precise_int());
        unsafe {
            self.int >> 1
        }
    }

    pub fn as_bool(&self) -> bool {
        debug_assert!(self.is_bool());
        self.is_true()
    }

    pub fn as_native(&self) -> NativeFunction {
        debug_assert!(self.is_native());
        unsafe { std::mem::transmute((self.tag >> 6) as u8) }
    }

    pub fn as_field(&self) -> u32 {
        debug_assert!(self.is_field());
        unsafe { (self.long_tag >> 6) as u32 }
    }

    /// Returns the `Type` of this value.
    ///
    /// **Implementation Note**
    ///
    /// Looking at the assembly of this function in compiler explorer shows some interesting insights.
    /// First, the generated assembly is _way better_ using the hardcoded functions like `is_int()` over `ty() == Type::Int`. So those will continue to exist, and should be used over `ty() == Type::Int`.
    ///
    /// Secondly, `ty() == Type::Str` vs. a more 'optimized' expression, such as `self.is_ptr() && (*self.as_ptr()).ty == Type::Str` has no significant difference in the output... _as long as_, `unreachable!()` is not used. This, is actually terrible because it forces us to consider a panic as a possible outcome of this check.
    /// I'm wary about using `unreachable_unchecked!()` because of the possibility of undefined behavior, but luckily, there is a better solution: `Type::Never`
    ///
    /// This is an elegant way of returning a real, well-defined `Type` that will never be compared equal to, and thus the compiler freely assumes the whole inline branch is false when checking i.e. an owned or shared type.
    ///
    /// Finally, these benefits obviously only happen when this entire function is inlined to a comparison with a constant `== Type::Other`, so we mark it as to always be inlined.
    #[inline(always)]
    pub fn ty(&self) -> Type {
        unsafe {
            match self.tag & MASK_PTR {
                TAG_NIL => match self.tag & MASK_BOOL {
                    TAG_NIL => Type::Nil,
                    TAG_BOOL => Type::Bool,
                    TAG_NATIVE => Type::NativeFunction,
                    TAG_NONE => Type::None,
                    TAG_FIELD => Type::GetField,
                    TAG_STR => Type::ShortStr,
                    _ => Type::Never,
                },
                TAG_PTR => (*self.as_ptr()).ty, // Check the prefix for the type
                _ => Type::Int, // Includes all remaining bit patterns with a `0` LSB
            }
        }
    }

    // Utility, and efficient, functions for checking the type of inline or pointer values.

    pub const fn is_nil(&self) -> bool { (unsafe { self.tag } & MASK_NIL) == TAG_NIL }
    pub const fn is_bool(&self) -> bool { (unsafe { self.tag } & MASK_BOOL) == TAG_BOOL }
    pub const fn is_true(&self) -> bool { (unsafe { self.tag }) == TAG_TRUE }
    pub const fn is_int(&self) -> bool { self.is_precise_int() || self.is_bool() }
    pub const fn is_precise_int(&self) -> bool { (unsafe { self.tag } & MASK_INT) == TAG_INT }
    pub const fn is_short_str(&self) -> bool { (unsafe { self.tag } & MASK_STR) == TAG_STR }
    pub const fn is_native(&self) -> bool { (unsafe { self.tag } & MASK_NATIVE) == TAG_NATIVE }
    pub const fn is_field(&self) -> bool { (unsafe { self.tag } & MASK_FIELD) == TAG_FIELD }
    pub const fn is_none(&self) -> bool { (unsafe { self.tag } & MASK_NONE) == TAG_NONE }

    pub fn is_err(&self) -> bool {
        self.ty() == Type::Error
    }

    fn is_ptr(&self) -> bool { (unsafe { self.tag } & MASK_PTR) == TAG_PTR }
    fn is_owned(&self) -> bool { self.ty().is_owned() }
    fn is_shared(&self) -> bool { self.ty().is_shared() }

    pub fn as_value_ref(&self) -> ValueRef {
        ValueRef::new(unsafe { self.tag })
    }

    /// Transmutes this `ValuePtr` into a `Box<T>`, where `T` is a type compatible with `Prefix`
    ///
    /// First `as_mut_ptr()` performs the correct pointer arithmetic to get a _real_ pointer.
    /// Then we box it using `from_raw()`, to indicate the memory is owned, and transmute it to the desired type.
    /// This is fine since the `Box<T>` types are an identical size (despite the size of `Prefix` not being the size of the result type).
    ///
    /// This hides the unsafe operations underneath, even though everything happening here is _terribly_ unsafe. But, as a result,
    /// since we consume this `ValuePtr` and return a valid representation, this is a safe API.
    pub fn as_box<T : OwnedValue>(self) -> Box<Prefix<T>> {
        debug_assert!(self.is_owned()); // Must be owned memory, to make sense converting to `Box<T>`
        unsafe {
            // Transmute self into a `Box` of the right prefix type.
            let ret = Box::from_raw(self.as_ptr() as *mut Prefix<T>);

            // Then forget the current `self`. This is a manual way of telling rust that we have fully transmuted ourselves into `ret`
            // We can't drop this here, since we technically created a copy of the same resource by calling `as_mut_ptr()`
            std::mem::forget(self);
            ret
        }
    }

    /// Transmutes this `ValuePtr` into a `&T`, where `T` is a type compatible with `Prefix`
    ///
    /// This is akin to `as_box()`, but taking in a reference, and handing one back. In that sense, it's the same safety guarantee as `.as_box()`
    /// The pointer has to be non-null, so `new_unchecked()`, and `as_mut_ptr()` are both valid.
    pub fn as_ref<T: OwnedValue>(&self) -> &T {
        debug_assert!(self.is_owned());
        unsafe {
            &(*(self.as_ptr() as *const Prefix<T>)).value
        }
    }

    pub fn as_mut_ref<T : OwnedValue>(&mut self) -> &mut T {
        debug_assert!(self.is_owned());
        unsafe {
            &mut (*(self.as_ptr() as *mut Prefix<T>)).value
        }
    }

    /// Transmutes this `ValuePtr` into a `&T`, where `T` is a type compatible with `SharedPrefix`
    pub fn as_shared_ref<T: SharedValue>(&self) -> &SharedPrefix<T> {
        debug_assert!(self.is_shared()); // Any shared pointer type can be converted to a shared prefix.
        unsafe {
            &*(self.as_ptr() as *const SharedPrefix<T>)
        }
    }

    /// Strips away the tag bits, and converts this value into a `* mut` pointer to an arbitrary `Prefix`
    unsafe fn as_ptr(&self) -> *mut Prefix<()> {
        debug_assert!(self.is_ptr());
        unsafe {
            (self.tag & PTR_MASK) as *mut Prefix<()>
        }
    }

    /// Creates a new copy of this `ValuePtr`, **pointing to the same memory!**. Whenever this is called, either the original,
    /// or the new copy **MUST** be forgotten, before being dropped.
    unsafe fn as_copy(&self) -> ValuePtr {
        ValuePtr { long_tag: self.long_tag }
    }

    /// Clones an owned `ValuePtr`, using `Box<T>` to clone the underlying memory.
    ///
    /// Create a new `Box<T>` pointing to this memory, clone it, and then forget the original.
    /// This avoids accidentally freeing the memory for this pointer.
    unsafe fn clone_owned<T: Clone + OwnedValue>(&self) -> ValuePtr {
        let copy = Box::from_raw(self.as_ptr() as *mut Prefix<T>);
        let cloned = copy.clone();
        std::mem::forget(copy);
        ValuePtr { tag: TAG_PTR | (Box::into_raw(cloned) as usize) }
    }

    /// Just increment the strong reference count, and then return a direct copy of the `ValuePtr`
    unsafe fn clone_shared<T : SharedValue>(&self) -> ValuePtr {
        self.as_shared_ref::<T>().inc_strong();
        self.as_copy()
    }

    unsafe fn drop_owned<T: OwnedValue>(&self) {
        debug_assert!(self.is_owned()); // Must be owned memory, to make sense converting to `Box<T>`
        unsafe {
            drop(Box::from_raw(self.as_ptr() as *mut Prefix<T>));
        }
    }

    unsafe fn drop_shared<T: SharedValue>(&self) {
        debug_assert!(self.is_shared()); // Any shared pointer type can be converted to a shared prefix.
        let shared: &SharedPrefix<()> = self.as_shared_ref::<()>();
        shared.dec_strong();
        if shared.refs.get() == 0 {
            unsafe {
                drop(Box::from_raw(self.as_ptr() as *mut SharedPrefix<T>));
            }
        }
    }
}


impl Default for ValuePtr {
    fn default() -> Self {
        ValuePtr::nil()
    }
}


/// Implementing `Eq` is done on a type-wide basis. We assume each pointer type implements `Eq` themselves.
/// First, we check that the types are equal, and if they are, we match on the type and check the underlying value.
impl Eq for ValuePtr {}
impl PartialEq for ValuePtr {
    fn eq(&self, other: &Self) -> bool {
        let ty: Type = self.ty();
        ty == other.ty() && match ty {
            // Inline types just need to check equality of value
            Type::Nil |
            Type::Bool |
            Type::Int |
            Type::NativeFunction => unsafe { self.tag == other.tag },
            Type::GetField |
            Type::ShortStr => unsafe { self.long_tag == other.long_tag },
            // Owned types check equality based on their ref
            Type::Complex => self.as_shared_ref::<Complex>() == other.as_shared_ref::<Complex>(),
            Type::Range => self.as_shared_ref::<Range>() == other.as_shared_ref::<Range>(),
            Type::Enumerate => self.as_shared_ref::<Enumerate>() == other.as_shared_ref::<Enumerate>(),
            Type::PartialFunction => self.as_shared_ref::<PartialFunction>() == other.as_shared_ref::<PartialFunction>(),
            Type::PartialNativeFunction => self.as_shared_ref::<PartialNativeFunction>() == other.as_shared_ref::<PartialNativeFunction>(),
            Type::Slice => self.as_shared_ref::<Slice>() == other.as_shared_ref::<Slice>(),
            Type::Error => self.as_ref::<RuntimeError>() == other.as_ref::<RuntimeError>(),
            // Shared types check equality based on the shared ref
            Type::LongStr => self.as_shared_ref::<String>() == other.as_shared_ref::<String>(),
            Type::List => self.as_shared_ref::<ListImpl>() == other.as_shared_ref::<ListImpl>(),
            Type::Set => self.as_shared_ref::<SetImpl>() == other.as_shared_ref::<SetImpl>(),
            Type::Dict => self.as_shared_ref::<DictImpl>() == other.as_shared_ref::<DictImpl>(),
            Type::Heap => self.as_shared_ref::<HeapImpl>() == other.as_shared_ref::<HeapImpl>(),
            Type::Vector => self.as_shared_ref::<VectorImpl>() == other.as_shared_ref::<VectorImpl>(),
            Type::Struct => self.as_shared_ref::<StructImpl>() == other.as_shared_ref::<StructImpl>(),
            Type::StructType => self.as_shared_ref::<StructTypeImpl>() == other.as_shared_ref::<StructTypeImpl>(),
            Type::Memoized => self.as_shared_ref::<MemoizedImpl>() == other.as_shared_ref::<MemoizedImpl>(),
            Type::Function => self.as_shared_ref::<Function>() == other.as_shared_ref::<Function>(),
            Type::Closure => self.as_shared_ref::<Closure>() == other.as_shared_ref::<Closure>(),
            // Special types that are not checked for equality
            Type::Iter | Type::None | Type::Never => false,
        }
    }
}


// In Cordy, order between different types is undefined - you can't sort `nil`, `bool` and `int`, even though they are all "int-like"
// Ordering between the same type is well defined, but some types may represent them all as equally ordered.
impl_partial_ord!(ValuePtr);
impl Ord for ValuePtr {
    fn cmp(&self, other: &Self) -> Ordering {
        let ty: Type = self.ty();
        if ty != other.ty() {
            return Ordering::Equal
        }
        match ty {
            // Inline types can directly compare the tag value. This works for all except ints and short strings
            Type::Nil |
            Type::Bool |
            Type::NativeFunction |
            Type::GetField => unsafe { self.tag.cmp(&other.tag) },
            Type::Int => self.as_precise_int().cmp(&other.as_precise_int()),
            Type::ShortStr => self.as_str_slice().cmp(other.as_str_slice()),

            // Owned types check equality based on their ref
            Type::Complex => self.as_shared_ref::<Complex>().cmp(other.as_shared_ref::<Complex>()),
            Type::Range => self.as_shared_ref::<Range>().cmp(other.as_shared_ref::<Range>()),
            Type::Enumerate => self.as_shared_ref::<Enumerate>().cmp(other.as_shared_ref::<Enumerate>()),
            // Shared types check equality based on the shared ref
            Type::LongStr => self.as_shared_ref::<String>().cmp(other.as_shared_ref::<String>()),
            Type::List => self.as_shared_ref::<ListImpl>().cmp(other.as_shared_ref::<ListImpl>()),
            Type::Set => self.as_shared_ref::<SetImpl>().cmp(other.as_shared_ref::<SetImpl>()),
            Type::Dict => self.as_shared_ref::<DictImpl>().cmp(other.as_shared_ref::<DictImpl>()),
            Type::Heap => self.as_shared_ref::<HeapImpl>().cmp(other.as_shared_ref::<HeapImpl>()),
            Type::Vector => self.as_shared_ref::<VectorImpl>().cmp(other.as_shared_ref::<VectorImpl>()),
            Type::Struct => self.as_shared_ref::<StructImpl>().cmp(other.as_shared_ref::<StructImpl>()),
            // Function-like types are not checked for ordering
            Type::StructType |
            Type::Memoized |
            Type::Function |
            Type::PartialFunction |
            Type::Closure |
            Type::PartialNativeFunction |
            Type::Slice => Ordering::Equal,
            // Special types that are not checked for ordering
            Type::Iter | Type::Error | Type::None | Type::Never => Ordering::Equal,
        }
    }
}


/// We don't implement `Copy` for `ValuePtr`, since it manages some memory which cannot be expressed with a copy.
/// For `Clone`, the behavior depends on the type of the value. Inline types are simple and just require a bit copy of the value.
///
/// - Inline Types: Copy the `ValuePtr` itself, as they have no memory to manage.
/// - Owned Types: Copy the underlying memory, using `clone_owned()`
/// - Shared Types: Increment the strong reference count, and copy the `ValuePtr` itself, using `clone_shared()`
impl Clone for ValuePtr {
    fn clone(&self) -> Self {
        unsafe {
            match self.ty() {
                // Inline types
                Type::Nil |
                Type::Bool |
                Type::Int |
                Type::ShortStr |
                Type::NativeFunction |
                Type::GetField => self.as_copy(),
                // Owned types
                Type::Complex => self.clone_shared::<Complex>(),
                Type::Range => self.clone_shared::<Range>(),
                Type::Enumerate => self.clone_shared::<Enumerate>(),
                Type::PartialFunction => self.clone_shared::<PartialFunction>(),
                Type::PartialNativeFunction => self.clone_shared::<PartialNativeFunction>(),
                Type::Slice => self.clone_shared::<Slice>(),
                Type::Iter => self.clone_shared::<Iterable>(),
                Type::Error => self.clone_owned::<RuntimeError>(),
                // Shared types
                Type::LongStr => self.clone_shared::<String>(),
                Type::List => self.clone_shared::<ListImpl>(),
                Type::Set => self.clone_shared::<SetImpl>(),
                Type::Dict => self.clone_shared::<DictImpl>(),
                Type::Heap => self.clone_shared::<HeapImpl>(),
                Type::Vector => self.clone_shared::<VectorImpl>(),
                Type::Struct => self.clone_shared::<StructImpl>(),
                Type::StructType => self.clone_shared::<StructTypeImpl>(),
                Type::Memoized => self.clone_shared::<MemoizedImpl>(),
                Type::Function => self.clone_shared::<Function>(),
                Type::Closure => self.clone_shared::<Closure>(),
                // Special types
                Type::None | Type::Never => ValuePtr::none(),
            }
        }
    }
}


/// For `Drop`, we need to once again, have specific behavior based on the type in question.
///
/// - Inline types have no drop behavior.
/// - Owned types need to drop their owned data, which is accomplished via just reinterpreting the type as a `Box<T>` and letting that drop.
/// - Shared types need to check and decrement their reference count, and potentially drop that.
impl Drop for ValuePtr {
    fn drop(&mut self) {
        unsafe {
            match self.ty() {
                Type::Nil |
                Type::Bool |
                Type::Int |
                Type::ShortStr |
                Type::NativeFunction |
                Type::GetField => {},
                // Owned types
                Type::Complex => self.drop_shared::<Complex>(),
                Type::Range => self.drop_shared::<Range>(),
                Type::Enumerate => self.drop_shared::<Enumerate>(),
                Type::PartialFunction => self.drop_shared::<PartialFunction>(),
                Type::PartialNativeFunction => self.drop_shared::<PartialNativeFunction>(),
                Type::Slice => self.drop_shared::<Slice>(),
                Type::Iter => self.drop_shared::<Iterable>(),
                Type::Error => self.drop_owned::<RuntimeError>(),
                // Shared types
                Type::LongStr => self.drop_shared::<String>(),
                Type::List => self.drop_shared::<ListImpl>(),
                Type::Set => self.drop_shared::<SetImpl>(),
                Type::Dict => self.drop_shared::<DictImpl>(),
                Type::Heap => self.drop_shared::<HeapImpl>(),
                Type::Vector => self.drop_shared::<VectorImpl>(),
                Type::Struct => self.drop_shared::<StructImpl>(),
                Type::StructType => self.drop_shared::<StructTypeImpl>(),
                Type::Memoized => self.drop_shared::<MemoizedImpl>(),
                Type::Function => self.drop_shared::<Function>(),
                Type::Closure => self.drop_shared::<Closure>(),
                Type::None | Type::Never => {}, // No drop behavior
            }
        }
    }
}


/// `Hash` just needs to call the hash methods on the underlying type.
/// Again, for inline types we can just hash the tag directly.
impl Hash for ValuePtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.ty() {
            // Inline types
            Type::Nil |
            Type::Bool |
            Type::Int |
            Type::ShortStr |
            Type::NativeFunction |
            Type::GetField => unsafe { self.tag }.hash(state),
            // Owned types
            Type::Complex => self.as_shared_ref::<Complex>().hash(state),
            Type::Enumerate => self.as_shared_ref::<Enumerate>().hash(state),
            Type::PartialFunction => self.as_shared_ref::<PartialFunction>().hash(state),
            Type::PartialNativeFunction => self.as_shared_ref::<PartialNativeFunction>().hash(state),
            Type::Slice => self.as_shared_ref::<Slice>().hash(state),
            // Shared types
            Type::Range => self.as_shared_ref::<Range>().hash(state),
            Type::LongStr => self.as_shared_ref::<String>().hash(state),
            Type::List => self.as_shared_ref::<ListImpl>().hash(state),
            Type::Set => self.as_shared_ref::<SetImpl>().hash(state),
            Type::Dict => self.as_shared_ref::<DictImpl>().hash(state),
            Type::Heap => self.as_shared_ref::<HeapImpl>().hash(state),
            Type::Vector => self.as_shared_ref::<VectorImpl>().hash(state),
            Type::Struct => self.as_shared_ref::<StructImpl>().hash(state),
            Type::StructType => self.as_shared_ref::<StructTypeImpl>().hash(state),
            Type::Memoized => self.as_shared_ref::<MemoizedImpl>().hash(state),
            Type::Function => self.as_shared_ref::<Function>().hash(state),
            Type::Closure => self.as_shared_ref::<Closure>().hash(state),
            // Special types with no hash behavior
            Type::Iter | Type::Error | Type::None | Type::Never => {},
        }
    }
}


/// `Debug` is fairly straightforward - inline types have a easy, single-line value, and all other types simply call `.fmt()` on the `as_ref()` type.
impl Debug for ValuePtr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.ty() {
            Type::Nil => write!(f, "Nil"),
            Type::Bool => Debug::fmt(&self.as_bool(), f),
            Type::Int => Debug::fmt(&self.as_int(), f),
            Type::ShortStr => Debug::fmt(&self.as_str_slice(), f),
            Type::NativeFunction => Debug::fmt(&self.as_native(), f),
            Type::GetField => f.debug_struct("GetField").field("field_index", &self.as_field()).finish(),
            // Owned types
            Type::Complex => Debug::fmt(self.as_shared_ref::<Complex>(), f),
            Type::Enumerate => Debug::fmt(self.as_shared_ref::<Enumerate>(), f),
            Type::PartialFunction => Debug::fmt(self.as_shared_ref::<PartialFunction>(), f),
            Type::PartialNativeFunction => Debug::fmt(self.as_shared_ref::<PartialNativeFunction>(), f),
            Type::Slice => Debug::fmt(self.as_shared_ref::<Slice>(), f),
            Type::Error => Debug::fmt(self.as_ref::<RuntimeError>(), f),
            // Shared types
            Type::Range => Debug::fmt(self.as_shared_ref::<Range>(), f),
            Type::LongStr => Debug::fmt(self.as_shared_ref::<String>(), f),
            Type::List => Debug::fmt(self.as_shared_ref::<ListImpl>(), f),
            Type::Set => Debug::fmt(self.as_shared_ref::<SetImpl>(), f),
            Type::Dict => Debug::fmt(self.as_shared_ref::<DictImpl>(), f),
            Type::Heap => Debug::fmt(self.as_shared_ref::<HeapImpl>(), f),
            Type::Vector => Debug::fmt(self.as_shared_ref::<VectorImpl>(), f),
            Type::Struct => Debug::fmt(self.as_shared_ref::<StructImpl>(), f),
            Type::StructType => Debug::fmt(self.as_shared_ref::<StructTypeImpl>(), f),
            Type::Memoized => Debug::fmt(self.as_shared_ref::<MemoizedImpl>(), f),
            Type::Function => Debug::fmt(self.as_shared_ref::<Function>(), f),
            Type::Closure => Debug::fmt(self.as_shared_ref::<Closure>(), f),
            // Special types with no hash behavior
            Type::Iter => write!(f, "Iter"),
            Type::None => write!(f, "None"),
            Type::Never => write!(f, "Never"),
        }
    }
}



#[repr(C)]
pub struct Prefix<T : OwnedValue> {
    ty: Type,
    pub value: T
}

impl<T : OwnedValue> Prefix<T> {
    pub fn new(ty: Type, value: T) -> Prefix<T> {
        Prefix { ty, value }
    }
}

impl<T : Clone + OwnedValue> Clone for Prefix<T> {
    fn clone(&self) -> Self {
        Prefix::new(self.ty, self.value.clone())
    }
}

impl<T : Eq + OwnedValue> Eq for Prefix<T> {}
impl<T : Eq + OwnedValue> PartialEq for Prefix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T : Eq + Ord + OwnedValue> Ord for Prefix<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T : Eq + Ord + OwnedValue> PartialOrd for Prefix<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T : Hash + OwnedValue> Hash for Prefix<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T : Debug + OwnedValue> Debug for Prefix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.value, f)
    }
}


/// `SharedPrefix` is a combination of the functionality of `Prefix`, `Rc` and `RefCell`:
///
/// - It provides a `Prefix`-compatible struct representation, meaning the `ty` field can be read from an arbitrary `ValuePtr`
/// - The memory managed by the `ValuePtr` to a shared prefix is reference counted like `Rc`
/// - It provides a (simple), safe, interior mutability concept similar to `RefCell` for types that need it.
///
/// This abstraction has a couple differences from `Rc<RefCell<T>>` that are required, and others which are optimized for Cordy:
///
/// 1. We fundamentally cannot use `Rc<RefCell<T>>` with `ValuePtr`, so the constructs have to be re-invented anyway.
/// 2. Unlike `Rc<RefCell<T>>`, we optimize for memory overhead here: `Rc<RefCell<T>>` usually has 24 bytes of overhead: `usize` strong reference count,
/// `usize` weak reference count, and `isize` borrow flag. We can improve on this, and pack with the `ty` field to only use 8 bytes total:
///     - We skimp on the number of possible references, using a `Cell<u32>` instead of `Cell<usize>`, and don't need weak references (it is simply never used in Cordy).
///     - We use a `Cell<u16>` instead of `Cell<usize>` for the borrow flag, in order to fit it into the remaining space.
///
/// Both of the above mean we can fit all the header information in 8 bytes, conveniently aligned for other pointer-sized data.
///
/// **Issues**
///
/// - We don't use a weak reference in Cordy to break cycles, mostly because there's no mechanism in which is makes sense to use. So instead, we use
/// a strategy called "not worrying about it" to deal with the potential to leak memory through cycles.
/// - Overflow... may happen on the number of borrows, or the number of references, but both are so ridiculously infeasible scenarios that we employ the same strategy.
#[repr(C)]
pub struct SharedPrefix<T : SharedValue> {
    ty: Type,
    /// A `0` indicates a mutable borrow is currently taken, a `1` indicates no borrow, and `>1` indicates the number of current immutable borrows + 1
    lock: Cell<u16>,
    /// The number of (strong) references to this value. When it reaches zero, the memory is freed.
    refs: Cell<u32>,
    /// The value itself, stored after the header information.
    /// Note that unlike `Prefix`, the value is not public as due to ref-counting and borrow checking, we only expose a `&mut T` via `borrow()` and `try_borrow()`
    value: UnsafeCell<T>,
}

/// Implementation for all (`ConstValue` and `MutValue`) `SharedPrefix<T>` types
impl<T : SharedValue> SharedPrefix<T> {
    pub fn new(ty: Type, value: T) -> SharedPrefix<T> {
        SharedPrefix {
            ty,
            lock: Cell::new(BORROW_NONE),
            refs: Cell::new(1),
            value: UnsafeCell::new(value)
        }
    }

    /// Copied from the implementation in `RefCell`. Must be implemented on `SharedPrefix<T : SharedValue>` for builtin trait implementations to reference.
    pub fn borrow(&self) -> Ref<'_, T> {
        self.try_borrow().expect("already borrowed")
    }

    pub fn try_borrow(&self) -> Option<Ref<'_, T>> {
        match self.lock.get() {
            BORROW_MUT => None, // Already borrowed,
            _ => {
                let value = unsafe { NonNull::new_unchecked(self.value.get()) };
                let lock = self.lock.get();
                self.lock.set(lock + 1);
                Some(Ref { value, lock: &self.lock })
            }
        }
    }

    /// Copied from the implementation of `Rc`. Minus the core intrinsics, as they aren't stable.
    fn inc_strong(&self) {
        let strong: u32 = self.refs.get() + 1;
        self.refs.set(strong);
    }

    fn dec_strong(&self) {
        self.refs.set(self.refs.get() - 1);
    }
}


/// `ConstValue` implementations
impl<T : ConstValue> SharedPrefix<T> {
    /// For const values, we know the underlying value is never mutable, and thus a mutable borrow is never taken.
    /// Thus, we can hand out unchecked references to the underlying value, which avoids the overhead of useless borrow checking.
    pub fn borrow_const(&self) -> &T {
        unsafe {
            NonNull::new_unchecked(self.value.get()).as_ref()
        }
    }
}

impl<T : MutValue> SharedPrefix<T> {
    /// This is explicitly unsafe, as we are handing out a unchecked, immutable reference to the underlying data.
    /// It relies on the caller knowing that we take split, or partial borrows (such as closures, having an immutable and mutable part)
    ///
    /// SAFETY: The caller must guarantee use of this function does not lead to any other mutable references being taken while this reference is held
    pub unsafe fn borrow_const_unsafe(&self) -> &T {
        NonNull::new_unchecked(self.value.get()).as_ref()
    }
}


/// In order to properly implement these traits, we need to only specialize them on `T : SharedValue`. However, we don't know a priori if these are const, or mutable types, which creates a problem for access.
/// If we had negative trait bounds, we could make `ConstValue` and `MutValue` incompatible traits, and then implement individually.
///
/// The result, is we _need_ to use runtime borrow checking for the implementation of standard library traits, since it gets declared on `SharedPrefix<T : SharedValue>`
///
/// Note that other implementations that _can_ specialize on `SharedPrefix<T : ConstValue>` can use the `borrow_const()` which does no checking. This is still safe, because we are still sure that for const types, no mutable borrows will be taken. It just represents extra work being done.
impl<T : Eq + SharedValue> Eq for SharedPrefix<T> {}
impl<T : Eq + SharedValue> PartialEq for SharedPrefix<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.borrow() == *other.borrow()
    }
}

impl<T : Eq + Ord + SharedValue> Ord for SharedPrefix<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.borrow().cmp(&other.borrow())
    }
}

impl<T : Eq + Ord + SharedValue> PartialOrd for SharedPrefix<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T : Hash + SharedValue> Hash for SharedPrefix<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().hash(state);
    }
}

impl<T : Debug + SharedValue> Debug for SharedPrefix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.borrow(), f)
    }
}

const BORROW_MUT: u16 = 0;
const BORROW_NONE: u16 = 1;

/// `MutValue` implementations. This prevents any `ConstValue` from accessing mutable borrows.
impl<T : MutValue> SharedPrefix<T> {

    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        self.try_borrow_mut().expect("already borrowed")
    }

    pub fn try_borrow_mut(&self) -> Option<RefMut<'_, T>> {
        match self.lock.get() {
            BORROW_NONE => {
                let value = unsafe { NonNull::new_unchecked(self.value.get()) };
                self.lock.set(BORROW_MUT);
                Some(RefMut { value, lock: &self.lock, marker: PhantomData })
            }
            _ => None, // Already borrowed,
        }
    }
}


/// Copy of the `Ref` implementation used in `RefCell`
pub struct Ref<'b, T: ?Sized + 'b> {
    value: NonNull<T>,
    lock: &'b Cell<u16>,
}

impl<'b, T: ?Sized + 'b> Drop for Ref<'b, T> {
    fn drop(&mut self) {
        let lock = self.lock.get();
        debug_assert!(lock > BORROW_NONE); // Should be currently immutably borrowed.
        self.lock.set(lock - 1);
    }
}

impl<T: ?Sized> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_ref() }
    }
}

/// Copy of the `RefMut` implementation used in `RefCell`
pub struct RefMut<'b, T: ?Sized + 'b> {
    value: NonNull<T>,
    lock: &'b Cell<u16>,
    // `NonNull` is covariant over `T`, so we need to reintroduce invariance.
    marker: PhantomData<&'b mut T>,
}

impl<'b, T: ?Sized + 'b> Drop for RefMut<'b, T> {
    fn drop(&mut self) {
        debug_assert_eq!(self.lock.get(), BORROW_MUT); // Should be currently mutably borrowed.
        self.lock.set(BORROW_NONE);
    }
}

impl<T: ?Sized> Deref for RefMut<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for RefMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_mut() }
    }
}


#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use crate::core::NativeFunction;
    use crate::vm::{ComplexValue, IntoValue};
    use crate::vm::value::ptr::{MAX_INT, MIN_INT, Prefix, SharedPrefix, ValuePtr};
    use crate::vm::value::Type;

    #[test]
    fn test_sizeof() {
        assert_eq!(std::mem::size_of::<ValuePtr>(), 8);
        assert_eq!(std::mem::size_of::<Prefix<()>>(), 1); // 1 byte, but usually followed by 7 bytes padding for most types, so 8 bytes in practice
        assert_eq!(std::mem::size_of::<SharedPrefix<()>>(), 8); // 1 (Type) + 1 (bool) lock + 2 padding + 4 (u32 ref count)
    }

    #[test]
    fn test_inline_nil() {
        let ptr = ValuePtr::nil();
        assert!(ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert!(!ptr.is_owned());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Nil);
        assert_eq!(format!("{:?}", ptr), "Nil");
    }

    #[test]
    fn test_inline_true() {
        let ptr = true.to_value();
        assert!(!ptr.is_nil());
        assert!(ptr.is_bool());
        assert!(ptr.is_int());
        assert!(!ptr.is_precise_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert!(!ptr.is_owned());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Bool);
        assert!(ptr.clone().as_bool());
        assert_eq!(ptr.clone().as_int(), 1);
        assert_eq!(format!("{:?}", ptr), "true");
    }

    #[test]
    fn test_inline_false() {
        let ptr = false.to_value();
        assert!(!ptr.is_nil());
        assert!(ptr.is_bool());
        assert!(ptr.is_int());
        assert!(!ptr.is_precise_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert!(!ptr.is_owned());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Bool);
        assert!(!ptr.clone().as_bool());
        assert_eq!(ptr.clone().as_int(), 0);
        assert_eq!(format!("{:?}", ptr), "false");
    }

    #[test]
    fn test_inline_int() {
        for int in -25i64..25 {
            let ptr = int.to_value();
            assert!(!ptr.is_nil());
            assert!(!ptr.is_bool());
            assert!(ptr.is_int());
            assert!(!ptr.is_native());
            assert!(!ptr.is_none());
            assert!(!ptr.is_err());
            assert!(!ptr.is_ptr());
            assert!(!ptr.is_owned());
            assert!(!ptr.is_shared());
            assert_eq!(ptr.ty(), Type::Int);
            assert_eq!(ptr.clone().as_int(), int);
            assert_eq!(ptr.clone().as_precise_int(), int);
            assert_eq!(format!("{:?}", ptr), format!("{}", int));
        }

        for int in [MIN_INT, MIN_INT + 1, MAX_INT - 1, MAX_INT] {
            let ptr = int.to_value();
            assert!(!ptr.is_nil());
            assert!(!ptr.is_bool());
            assert!(ptr.is_int());
            assert!(!ptr.is_native());
            assert!(!ptr.is_none());
            assert!(!ptr.is_err());
            assert!(!ptr.is_ptr());
            assert!(!ptr.is_owned());
            assert!(!ptr.is_shared());
            assert_eq!(ptr.ty(), Type::Int);
            assert_eq!(ptr.clone().as_int(), int);
            assert_eq!(ptr.clone().as_precise_int(), int);
            assert_eq!(format!("{:?}", ptr), format!("{}", int));
        }
    }

    #[test]
    fn test_inline_int_compare() {
        let neg1 = (-1i64).to_value();
        let one = 1i64.to_value();
        let two = 2i64.to_value();
        let three = 3i64.to_value();

        assert_eq!(neg1.cmp(&one), Ordering::Less);
        assert_eq!(one.cmp(&neg1), Ordering::Greater);
        assert_eq!(neg1.cmp(&neg1), Ordering::Equal);

        assert_eq!(one.cmp(&one), Ordering::Equal);
        assert_eq!(one.cmp(&two), Ordering::Less);
        assert_eq!(one.cmp(&three), Ordering::Less);
        assert_eq!(two.cmp(&one), Ordering::Greater);
        assert_eq!(two.cmp(&two), Ordering::Equal);
        assert_eq!(two.cmp(&three), Ordering::Less);
    }

    #[test]
    fn test_inline_int_out_of_bounds() {
        assert_eq!((MIN_INT - 1).to_value().as_int(), MAX_INT);
        assert_eq!((MAX_INT + 1).to_value().as_int(), MIN_INT);
    }

    #[test]
    fn test_inline_native_function() {
        for f in 0..NativeFunction::total() {
            let f: NativeFunction = unsafe { std::mem::transmute(f as u8) };
            let ptr = f.to_value();
            assert!(!ptr.is_nil());
            assert!(!ptr.is_bool());
            assert!(!ptr.is_int());
            assert!(ptr.is_native());
            assert!(!ptr.is_none());
            assert!(!ptr.is_err());
            assert!(!ptr.is_ptr());
            assert!(!ptr.is_owned());
            assert!(!ptr.is_shared());
            assert_eq!(ptr.ty(), Type::NativeFunction);
            assert_eq!(ptr.as_native(), f);
            assert_eq!(format!("{:?}", ptr), format!("{:?}", f));
        }
    }

    #[test]
    fn test_inline_none() {
        let ptr = ValuePtr::none();

        assert!(!ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(ptr.is_none());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert!(!ptr.is_owned());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::None);
    }

    #[test]
    fn test_owned_complex() {
        let ptr = ComplexValue::new(1, 2).to_value();

        assert!(!ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(ptr.is_ptr());
        assert!(!ptr.is_owned());
        assert!(ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Complex);
        assert_eq!(format!("{:?}", ptr), "Complex(Complex { re: 1, im: 2 })");
    }

    #[test]
    fn test_owned_complex_clone_eq() {
        let c0 = ValuePtr::nil();
        let c1 = ComplexValue::new(1, 2).to_value();
        let c2 = ComplexValue::new(1, 3).to_value();

        assert_ne!(c2, c0);
        assert_ne!(c2, c1);
        assert_eq!(c2, c2);
        assert_ne!(c1, c2.clone());
        assert_eq!(c2, c2.clone());
    }

    #[test]
    fn test_shared_str() {
        let ptr = "hello world".to_value();

        assert!(!ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(ptr.is_ptr());
        assert!(!ptr.is_owned());
        assert!(ptr.is_shared());
        assert_eq!(ptr.ty(), Type::LongStr);
        assert_eq!(format!("{:?}", ptr), String::from("\"hello world\""));
    }

    #[test]
    fn test_shared_mut_vec_eq() {
        let ptr1 = vec![].to_value();
        let ptr2 = vec![].to_value();

        assert_eq!(ptr1, ptr2);
        assert_eq!(ptr1, ptr1);
    }

    #[test]
    fn test_shared_mut_vec_clone() {
        let ptr = vec![].to_value();

        assert_eq!(ptr, ptr.clone());
        assert_eq!(ptr.clone(), ptr.clone());
    }

    #[test]
    #[should_panic]
    fn test_shared_mut_vec_two_mutable_borrow_panic() {
        let ptr = vec![].to_value();

        let _r1 = ptr.as_vector().borrow_mut();
        let _r2 = ptr.as_vector().borrow_mut();
    }

    #[test]
    #[should_panic]
    fn test_shared_mut_vec_mutable_and_normal_borrow_panic() {
        let ptr = vec![].to_value();

        let _r1 = ptr.as_vector().borrow();
        let _r2 = ptr.as_vector().borrow_mut();
    }

    #[test]
    fn test_shared_mut_vec_multiple_borrow_no_panic() {
        let ptr = vec![].to_value();

        let _r1 = ptr.as_vector().borrow();
        let _r2 = ptr.as_vector().borrow();
    }
}