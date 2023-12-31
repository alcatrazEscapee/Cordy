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


/// # ValuePtr
///
/// A `ValuePtr` is the underlying value used by Cordy. It is a **tagged pointer**, using the tag bits to store a number of inline types.
/// The pointer points to a `Prefix<T>`, which includes exact type information, reference counting and interior mutability semantics.
///
/// ### Representation
///
/// A `ValuePtr` may be:
///
/// - An inline, 63-bit signed integer, stored with the lowest bit equal to `0`
/// - An inline constant `nil`, `true`, or `false`, `NativeFunction`, or `GetField`, all stored with the lowest three bits equal to `01`
/// - A tagged pointer, to a `Prefix<T>`, with the lowest three bits equal to `11`
///
/// ### Memory Allocation
///
/// Inline `ValuePtr`s own no memory, making them easy to copy and drop. All other `ValuePtr`s function as `Rc<RefCell<T>>`, with a pointer to
/// some reference counted heap memory.
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


#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum MemoryType {
    Inline, Shared
}


const TAG_INT      : usize = 0b______00;
const TAG_NIL      : usize = 0b__000_01;
const TAG_BOOL     : usize = 0b__001_01;
const TAG_FALSE    : usize = 0b0_001_01;
const TAG_TRUE     : usize = 0b1_001_01;
const TAG_NATIVE   : usize = 0b__010_01;
const TAG_FIELD    : usize = 0b__011_01;
const TAG_STR      : usize = 0b__100_01;
//    UNUSED               = 0b__101_01
//    UNUSED               = 0b__110_01
//    UNUSED               = 0b__111_01
const TAG_PTR      : usize = 0b______11;

const MASK_INT: usize      = 0b_______1;
const MASK_NIL: usize      = 0b__111_11;
const MASK_BOOL: usize     = 0b__111_11;
const MASK_NATIVE: usize   = 0b__111_11;
const MASK_FIELD: usize    = 0b__111_11;
const MASK_STR: usize      = 0b__111_11;
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
        from_shared(Prefix::new(Type::LongStr, String::from(str)))
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

pub(super) fn from_shared<T : Value>(value: Prefix<T>) -> ValuePtr {
    ValuePtr { tag: TAG_PTR | (Box::into_raw(Box::new(value)) as usize) }
}


impl ValuePtr {

    pub const fn nil() -> ValuePtr {
        ValuePtr { tag: TAG_NIL }
    }

    pub const fn str() -> ValuePtr {
        ValuePtr { tag: TAG_STR }
    }

    // `.as_T()` methods for inline types take a `&self` for convenience. Copying the value is the same as copying the reference.

    pub fn as_int(&self) -> i64 {
        debug_assert!(self.is_int());
        match self.is_precise_int() {
            true => self.as_precise_int(),
            false => self.is_true() as i64
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
                    TAG_FIELD => Type::GetField,
                    TAG_STR => Type::ShortStr,
                    _ => Type::Never,
                },
                TAG_PTR => (*self.as_ptr::<()>()).ty, // Check the prefix for the type
                _ => Type::Int, // Includes all remaining bit patterns with a `0` LSB
            }
        }
    }

    #[inline(always)]
    fn memory_ty(&self) -> MemoryType {
        unsafe {
            match self.tag & MASK_PTR {
                TAG_PTR => MemoryType::Shared,
                _ => MemoryType::Inline,
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

    fn is_ptr(&self) -> bool { (unsafe { self.tag } & MASK_PTR) == TAG_PTR }

    /// Transmutes this `ValuePtr` into a `usize`, providing reference equality semantics by comparing the tag value
    pub(super) fn as_ptr_value(&self) -> usize {
        unsafe { self.tag }
    }

    /// Transmutes this `ValuePtr` into a `&T`, provided we have already checked the type is of the type `T`.
    ///
    /// SAFETY: The caller **must guarantee** that `self.ty()` is of the correct type associated to `T`.
    /// This is in practice ensured by the implementations of `impl_mut!()` and `impl_const!()` macros.
    pub(super) fn as_prefix<T: Value>(&self) -> &Prefix<T> {
        unsafe {
            &*self.as_ptr::<T>()
        }
    }

    /// Strips away the tag bits, and converts this value into a `* mut` pointer to an arbitrary `Prefix`
    unsafe fn as_ptr<T : Value>(&self) -> *mut Prefix<T> {
        debug_assert!(self.is_ptr());
        unsafe {
            (self.tag & PTR_MASK) as *mut Prefix<T>
        }
    }

    /// Drops the memory pointed to by this value, by reinterpreting the pointer as a `Box<Prefix<T>>`
    unsafe fn drop_value<T: Value>(&self) {
        drop(Box::from_raw(self.as_ptr::<T>()));
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
        macro_rules! eq {
            ($f:ident) => { self.$f() == other.$f() };
        }

        let ty: Type = self.ty();
        ty == other.ty() && match ty {
            Type::Nil |
            Type::Bool |
            Type::Int |
            Type::NativeFunction => unsafe { self.tag == other.tag },

            Type::GetField |
            Type::ShortStr => unsafe { self.long_tag == other.long_tag },

            Type::Complex => eq!(as_complex),
            Type::Range => eq!(as_range),
            Type::Enumerate => eq!(as_enumerate),
            Type::PartialFunction => eq!(as_partial_function),
            Type::PartialNativeFunction => eq!(as_partial_native),
            Type::Slice => eq!(as_slice),
            Type::LongStr => self.as_prefix::<String>() == other.as_prefix::<String>(),
            Type::List => eq!(as_list),
            Type::Set => eq!(as_set),
            Type::Dict => eq!(as_dict),
            Type::Heap => eq!(as_heap),
            Type::Vector => eq!(as_vector),
            Type::Struct => eq!(as_struct),
            Type::StructType => eq!(as_struct_type),
            Type::Memoized => eq!(as_memoized),
            Type::Function => eq!(as_function),
            Type::Closure => eq!(as_closure),
            Type::Error => eq!(as_err),

            Type::Iter | Type::Never => false,
        }
    }
}


// In Cordy, order between different types is undefined - you can't sort `nil`, `bool` and `int`, even though they are all "int-like"
// Ordering between the same type is well defined, but some types may represent them all as equally ordered.
impl_partial_ord!(ValuePtr);
impl Ord for ValuePtr {
    fn cmp(&self, other: &Self) -> Ordering {
        macro_rules! cmp {
            ($ty:ident, $inner:ident) => {
                if other.ty() == Type::$ty { self.as_prefix::<$inner>().cmp(&other.as_prefix::<$inner>()) } else { Ordering::Equal }
            };
        }

        let ty: Type = self.ty();
        match ty {
            // Types that convert to int compare with their integral values, then by the ordinal of their type.
            // => negative integers < nil < false < 0 < true < 1 < positive integers
            //
            // Note that `nil` is special in this case, and *does* coerce to `0`
            //
            // Complex numbers compare with their real component first, then their imaginary part
            // Note that complex numbers will always have im != 0, and thus don't need to additionally order by the type
            Type::Nil | Type::Bool | Type::Int => {
                let lhs = match ty {
                    Type::Nil => 0,
                    Type::Bool => self.is_true() as i64,
                    _ => self.as_precise_int()
                };
                let other_ty = other.ty();
                match other.ty() {
                    Type::Nil | Type::Bool | Type::Int => {
                        let rhs = match other_ty {
                            Type::Nil => 0,
                            Type::Bool => other.is_true() as i64,
                            _ => other.as_precise_int()
                        };
                        lhs.cmp(&rhs).then((ty as u8).cmp(&(other_ty as u8)))
                    },
                    Type::Complex => {
                        let rhs = other.as_complex();
                        lhs.cmp(&rhs.re).then(0.cmp(&rhs.im))
                    },
                    _ => Ordering::Equal,
                }
            },
            Type::Complex => {
                match other.ty() {
                    Type::Nil | Type::Bool | Type::Int => other.cmp(self).reverse(),
                    Type::Complex => {
                        let lhs = self.as_complex();
                        let rhs = self.as_complex();
                        lhs.re.cmp(&rhs.re).then(lhs.im.cmp(&rhs.im))
                    },
                    _ => Ordering::Equal
                }
            },

            // Strings order by length (ascending), then by natural ordering
            // Type::ShortStr < Type::LongStr, then, by definition
            Type::ShortStr | Type::LongStr => {
                match other.ty() {
                    Type::ShortStr | Type::LongStr => self.as_str_slice().cmp(other.as_str_slice()),
                    _ => Ordering::Equal
                }
            }

            // Other types first check the other type and require it is the same, and otherwise report unordered
            // If the type is the same, they check the ordering between the same types
            Type::Range => cmp!(Range, Range),
            Type::Enumerate => cmp!(Enumerate, Enumerate),
            Type::List => cmp!(List, ListImpl),
            Type::Set => cmp!(Set, SetImpl),
            Type::Dict => cmp!(Dict, DictImpl),
            Type::Heap => cmp!(Heap, HeapImpl),
            Type::Vector => cmp!(Vector, VectorImpl),
            Type::Struct => cmp!(Struct, StructImpl),

            // Any other types are not explicitly ordered
            _ => Ordering::Equal
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
        if self.memory_ty() == MemoryType::Shared {
            // Shared memory types are able to copy by incrementing the strong reference count,
            // and then just returning the same copy as inline types
            self.as_prefix::<()>().inc_strong();
        }
        unsafe {
            // Copy the raw value of the pointer
            ValuePtr { long_tag: self.long_tag }
        }
    }
}


impl Drop for ValuePtr {
    fn drop(&mut self) {
        if self.memory_ty() == MemoryType::Shared {
            // Only shared types have drop behavior
            // They all decrement the strong reference count, and if it reaches zero, performs a drop based on the type
            let shared: &Prefix<()> = self.as_prefix::<()>();

            shared.dec_strong();
            if shared.refs.get() == 0 {
                unsafe {
                    match self.ty() {
                        Type::Complex => self.drop_value::<Complex>(),
                        Type::Range => self.drop_value::<Range>(),
                        Type::Enumerate => self.drop_value::<Enumerate>(),
                        Type::PartialFunction => self.drop_value::<PartialFunction>(),
                        Type::PartialNativeFunction => self.drop_value::<PartialNativeFunction>(),
                        Type::Slice => self.drop_value::<Slice>(),
                        Type::Iter => self.drop_value::<Iterable>(),
                        Type::LongStr => self.drop_value::<String>(),
                        Type::List => self.drop_value::<ListImpl>(),
                        Type::Set => self.drop_value::<SetImpl>(),
                        Type::Dict => self.drop_value::<DictImpl>(),
                        Type::Heap => self.drop_value::<HeapImpl>(),
                        Type::Vector => self.drop_value::<VectorImpl>(),
                        Type::Struct => self.drop_value::<StructImpl>(),
                        Type::StructType => self.drop_value::<StructTypeImpl>(),
                        Type::Memoized => self.drop_value::<MemoizedImpl>(),
                        Type::Function => self.drop_value::<Function>(),
                        Type::Closure => self.drop_value::<Closure>(),
                        Type::Error => self.drop_value::<RuntimeError>(),

                        _ => {}
                    }
                }
            }
        }
    }
}


/// `Hash` just needs to call the hash methods on the underlying type.
/// Again, for inline types we can just hash the tag directly.
impl Hash for ValuePtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        macro_rules! hash {
            ($f:ident) => { self.$f().hash(state) };
        }
        match self.ty() {
            Type::Nil |
            Type::Bool |
            Type::Int |
            Type::ShortStr |
            Type::NativeFunction |
            Type::GetField => unsafe { self.tag }.hash(state),

            Type::Complex => hash!(as_complex),
            Type::Enumerate => hash!(as_enumerate),
            Type::PartialFunction => hash!(as_partial_function),
            Type::PartialNativeFunction => hash!(as_partial_native),
            Type::Slice => hash!(as_slice),
            Type::Range => hash!(as_range),
            Type::LongStr => self.as_prefix::<String>().hash(state),
            Type::List => hash!(as_list),
            Type::Set => hash!(as_set),
            Type::Dict => hash!(as_dict),
            Type::Heap => hash!(as_heap),
            Type::Vector => hash!(as_vector),
            Type::Struct => hash!(as_struct),
            Type::StructType => hash!(as_struct_type),
            Type::Memoized => hash!(as_memoized),
            Type::Function => hash!(as_function),
            Type::Closure => hash!(as_closure),

            Type::Iter | Type::Error | Type::Never => {},
        }
    }
}


/// `Debug` is fairly straightforward - inline types have a easy, single-line value, and all other types simply call `.fmt()` on the `as_ref()` type.
impl Debug for ValuePtr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        macro_rules! fmt {
            ($f:ident) => { Debug::fmt(self.$f(), f) };
        }
        match self.ty() {
            Type::Nil => write!(f, "Nil"),
            Type::Bool => Debug::fmt(&self.as_bool(), f),
            Type::Int => Debug::fmt(&self.as_int(), f),
            Type::ShortStr => Debug::fmt(&self.as_str_slice(), f),
            Type::NativeFunction => Debug::fmt(&self.as_native(), f),
            Type::GetField => f.debug_struct("GetField").field("field_index", &self.as_field()).finish(),

            Type::Complex => fmt!(as_complex),
            Type::Enumerate => fmt!(as_enumerate),
            Type::PartialFunction => fmt!(as_partial_function),
            Type::PartialNativeFunction => fmt!(as_partial_native),
            Type::Slice => fmt!(as_slice),
            Type::Range => fmt!(as_range),
            Type::LongStr => Debug::fmt(self.as_prefix::<String>(), f),
            Type::List => fmt!(as_list),
            Type::Set => fmt!(as_set),
            Type::Dict => fmt!(as_dict),
            Type::Heap => fmt!(as_heap),
            Type::Vector => fmt!(as_vector),
            Type::Struct => fmt!(as_struct),
            Type::StructType => fmt!(as_struct_type),
            Type::Memoized => fmt!(as_memoized),
            Type::Function => fmt!(as_function),
            Type::Closure => fmt!(as_closure),
            Type::Error => fmt!(as_err),

            Type::Iter => write!(f, "Iter"),
            Type::Never => write!(f, "Never"),
        }
    }
}


/// # Prefix<T>
///
/// This is the header attached to all heap-allocated values in Cordy, which provides a number of important features:
///
/// - The type information can be observed from the `ty` field, read from an arbitrary `Prefix<()>` to determine the exact type
/// - The `lock` provides `RefCell`-like, runtime interior mutability for those that need it
/// - The value is reference counted, up to a limit of `u32::MAX` references
///
/// ### Notable Differences
///
/// Unlike `Rc<RefCell<T>>`, we optimize for memory overhead here: `Rc<RefCell<T>>` usually has 24 bytes of overhead:
///
/// - `usize` strong reference count - we reduce this to a `u32`
/// - `usize` weak reference count - we don't use weak references at all
/// - `isize` borrow flag - we reduce this to a `u16`
///
/// Based on the alignment of each type, this allows the entire header to fit into 8 bytes, and is conveniently 8-byte aligned.
///
/// ### Issues
///
/// - We don't use a weak reference in Cordy to break cycles, mostly because there's no mechanism in which is makes sense to use. So instead, we use
/// a strategy called "not worrying about it" to deal with the potential to leak memory through cycles.
/// - Overflow... may happen on the number of borrows, or the number of references, but both are so ridiculously infeasible scenarios that we employ the same strategy.
///
#[repr(C)]
pub struct Prefix<T : Value> {
    ty: Type,
    /// A `0` indicates a mutable borrow is currently taken, a `1` indicates no borrow, and `>1` indicates the number of current immutable borrows + 1
    lock: Cell<u16>,
    /// The number of (strong) references to this value. When it reaches zero, the memory is freed.
    refs: Cell<u32>,
    /// The value itself, stored after the header information.
    /// Note that unlike `Prefix`, the value is not public as due to ref-counting and borrow checking, we only expose a `&mut T` via `borrow()` and `try_borrow()`
    value: UnsafeCell<T>,
}

/// **N.B.** With negative trait bounds, the `borrow()` methods could be split to require `T : Value + !ConstValue`
impl<T : Value> Prefix<T> {
    pub fn new(ty: Type, value: T) -> Prefix<T> {
        Prefix {
            ty,
            lock: Cell::new(BORROW_NONE),
            refs: Cell::new(1),
            value: UnsafeCell::new(value)
        }
    }

    /// Copied from the implementation in `RefCell`.
    ///
    /// Must be implemented on `Prefix<T : Value>` for builtin trait implementations to reference.
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
impl<T : ConstValue> Prefix<T> {
    /// For const values, we know the underlying value is never mutable, and thus a mutable borrow is never taken.
    /// Thus, we can hand out unchecked references to the underlying value, which avoids the overhead of useless borrow checking.
    pub fn borrow_const(&self) -> &T {
        unsafe {
            NonNull::new_unchecked(self.value.get()).as_ref()
        }
    }
}

impl<T : Value> Prefix<T> {
    /// This is explicitly unsafe, as we are handing out a unchecked, immutable reference to the underlying data.
    /// It relies on the caller knowing that we take split, or partial borrows (such as closures, having an immutable and mutable part)
    ///
    /// SAFETY: The caller must guarantee use of this function does not lead to any other mutable references being taken while this reference is held
    pub unsafe fn borrow_const_unsafe(&self) -> &T {
        NonNull::new_unchecked(self.value.get()).as_ref()
    }
}


/// In order to implement these traits properly, we need to specialize only on `T : Value`, which means even for `ConstValue` types, we cannot
/// use `borrow_const()`, and instead are _required_ to call `.borrow()`
///
/// **N.B.** With negative trait bounds, we could split these into an optimized variant using `.borrow_const()` for `Value + ConstValue`, and the existing
/// version for `Value + !ConstValue`
impl<T : Eq + Value> Eq for Prefix<T> {}
impl<T : Eq + Value> PartialEq for Prefix<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.borrow() == *other.borrow()
    }
}

impl<T : Eq + Ord + Value> Ord for Prefix<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.borrow().cmp(&other.borrow())
    }
}

impl<T : Eq + Ord + Value> PartialOrd for Prefix<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T : Hash + Value> Hash for Prefix<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().hash(state);
    }
}

impl<T : Debug + Value> Debug for Prefix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.borrow(), f)
    }
}

const BORROW_MUT: u16 = 0;
const BORROW_NONE: u16 = 1;

impl<T : Value> Prefix<T> {

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
    use crate::vm::value::ptr::{MAX_INT, MIN_INT, Prefix, ValuePtr};
    use crate::vm::value::Type;

    #[test]
    fn test_sizeof() {
        assert_eq!(std::mem::size_of::<ValuePtr>(), 8);
        assert_eq!(std::mem::size_of::<Prefix<()>>(), 8); // 1 (Type) + 1 (bool) lock + 2 padding + 4 (u32 ref count)
    }

    #[test]
    fn test_nil() {
        let ptr = ValuePtr::nil();
        assert!(ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert_eq!(ptr.ty(), Type::Nil);
        assert_eq!(format!("{:?}", ptr), "Nil");
    }

    #[test]
    fn test_true() {
        let ptr = true.to_value();
        assert!(!ptr.is_nil());
        assert!(ptr.is_bool());
        assert!(ptr.is_int());
        assert!(!ptr.is_precise_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert_eq!(ptr.ty(), Type::Bool);
        assert!(ptr.clone().as_bool());
        assert_eq!(ptr.clone().as_int(), 1);
        assert_eq!(format!("{:?}", ptr), "true");
    }

    #[test]
    fn test_false() {
        let ptr = false.to_value();
        assert!(!ptr.is_nil());
        assert!(ptr.is_bool());
        assert!(ptr.is_int());
        assert!(!ptr.is_precise_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert_eq!(ptr.ty(), Type::Bool);
        assert!(!ptr.clone().as_bool());
        assert_eq!(ptr.clone().as_int(), 0);
        assert_eq!(format!("{:?}", ptr), "false");
    }

    #[test]
    fn test_int() {
        for int in -25i64..25 {
            let ptr = int.to_value();
            assert!(!ptr.is_nil());
            assert!(!ptr.is_bool());
            assert!(ptr.is_int());
            assert!(!ptr.is_native());
            assert!(!ptr.is_err());
            assert!(!ptr.is_ptr());
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
            assert!(!ptr.is_err());
            assert!(!ptr.is_ptr());
            assert_eq!(ptr.ty(), Type::Int);
            assert_eq!(ptr.clone().as_int(), int);
            assert_eq!(ptr.clone().as_precise_int(), int);
            assert_eq!(format!("{:?}", ptr), format!("{}", int));
        }
    }

    #[test]
    fn test_int_compare() {
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
    fn test_int_out_of_bounds() {
        assert_eq!((MIN_INT - 1).to_value().as_int(), MAX_INT);
        assert_eq!((MAX_INT + 1).to_value().as_int(), MIN_INT);
    }

    #[test]
    fn test_native_function() {
        for f in 0..NativeFunction::total() {
            let f: NativeFunction = unsafe { std::mem::transmute(f as u8) };
            let ptr = f.to_value();
            assert!(!ptr.is_nil());
            assert!(!ptr.is_bool());
            assert!(!ptr.is_int());
            assert!(ptr.is_native());
            assert!(!ptr.is_err());
            assert!(!ptr.is_ptr());
            assert_eq!(ptr.ty(), Type::NativeFunction);
            assert_eq!(ptr.as_native(), f);
            assert_eq!(format!("{:?}", ptr), format!("{:?}", f));
        }
    }

    #[test]
    fn test_complex() {
        let ptr = ComplexValue::new(1, 2).to_value();

        assert!(!ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_err());
        assert!(ptr.is_ptr());
        assert_eq!(ptr.ty(), Type::Complex);
        assert_eq!(format!("{:?}", ptr), "Complex { re: 1, im: 2 }");
    }

    #[test]
    fn test_complex_clone_eq() {
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
    fn test_str() {
        let ptr = "hello world".to_value();

        assert!(!ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_err());
        assert!(ptr.is_ptr());
        assert_eq!(ptr.ty(), Type::LongStr);
        assert_eq!(format!("{:?}", ptr), String::from("\"hello world\""));
    }

    #[test]
    fn test_vec_eq() {
        let ptr1 = vec![].to_value();
        let ptr2 = vec![].to_value();

        assert_eq!(ptr1, ptr2);
        assert_eq!(ptr1, ptr1);
    }

    #[test]
    fn test_vec_clone() {
        let ptr = vec![].to_value();

        assert_eq!(ptr, ptr.clone());
        assert_eq!(ptr.clone(), ptr.clone());
    }

    #[test]
    #[should_panic]
    fn test_vec_two_mutable_borrow_panic() {
        let ptr = vec![].to_value();

        let _r1 = ptr.as_vector().borrow_mut();
        let _r2 = ptr.as_vector().borrow_mut();
    }

    #[test]
    #[should_panic]
    fn test_vec_mutable_and_normal_borrow_panic() {
        let ptr = vec![].to_value();

        let _r1 = ptr.as_vector().borrow();
        let _r2 = ptr.as_vector().borrow_mut();
    }

    #[test]
    fn test_vec_multiple_borrow_no_panic() {
        let ptr = vec![].to_value();

        let _r1 = ptr.as_vector().borrow();
        let _r2 = ptr.as_vector().borrow();
    }
}