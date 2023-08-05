use std::cell::{Cell, UnsafeCell};
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use num_complex::Complex;

use crate::core::NativeFunction;
use crate::util::impl_partial_ord;
use crate::vm::{FunctionImpl, IntoValue, Iterable, RuntimeError, StructTypeImpl};
use crate::vm::value::{*};



/// `ValuePtr` holds an arbitrary object representable by Cordy.
///
/// In order to minimize memory usage, improving cache efficiency, and reducing memory boxing, `ValuePtr` is structured as a **tagged pointer**.
/// On 64-bit systems, pointers that point to 64-bit (8-byte) aligned objects naturally have the last three bits set to zero.
/// This means we can store a little bit of type information in the low bits of a pointer.
///
/// # Representation
///
/// A `ValuePtr` may be:
///
/// - An inline, 63-bit signed integer, stored with the lowest bit equal to `0`
/// - An inline constant `nil`, `true`, or `false`, `NativeFunction`, or `GetField`, all stored with the lowest three bits equal to `001`
/// - An inline `Box<RuntimeError>` which is stored with the lowest three bits equal to `011`
/// - A tagged pointer to a piece of **owned** memory, including additional type information, with the lowest three bits equal to `101`
/// - A tagged pointer to a piece of **shared** memory, including additional type information, with the lowest three bits equal to `111`
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
    int: i64,
}


const TAG_INT: usize       = 0b______000;
const TAG_NIL: usize       = 0b__000_001;
const TAG_BOOL: usize      = 0b__001_001;
const TAG_FALSE: usize     = 0b0_001_001;
const TAG_TRUE: usize      = 0b1_001_001;
const TAG_NATIVE: usize    = 0b__010_001;
const TAG_NONE: usize      = 0b__011_001;
const TAG_FIELD: usize     = 0b__100_001;
const TAG_ERR: usize       = 0b______011;
const TAG_PTR: usize       = 0b______101;
const TAG_SHARED: usize    = 0b______111;

const MASK_INT: usize      = 0b________1;
const MASK_NIL: usize      = 0b__111_111;
const MASK_BOOL: usize     = 0b__111_111;
const MASK_NATIVE: usize   = 0b__111_111;
const MASK_FIELD: usize    = 0b__111_111;
const MASK_NONE: usize     = 0b__111_111;
const MASK_ERR: usize      = 0b______111;
const MASK_PTR: usize      = 0b______111;
const MASK_SHARED: usize   = 0b______111;

const PTR_MASK: usize = 0xffff_ffff_ffff_fff8;

const MAX_INT: i64 = 0x3fff_ffff_ffff_ffffu64 as i64;
const MIN_INT: i64 = 0xc000_0000_0000_0000u64 as i64;


impl ValuePtr {

    pub const fn nil() -> ValuePtr {
        ValuePtr { tag: TAG_NIL }
    }

    pub fn none() -> ValuePtr {
        ValuePtr { tag: TAG_NONE }
    }

    pub(super) fn of_bool(value: bool) -> ValuePtr {
        ValuePtr { tag: if value { TAG_TRUE } else { TAG_FALSE } }
    }

    pub(super) fn of_int(value: i64) -> ValuePtr {
        debug_assert!(MIN_INT <= value && value <= MAX_INT);
        ValuePtr { tag: TAG_INT | ((value << 1) as usize) }
    }

    pub(super) fn of_native(value: NativeFunction) -> ValuePtr {
        ValuePtr { tag: TAG_NATIVE | ((value as usize) << 6) }
    }

    pub fn of_field(field: u32) -> ValuePtr {
        ValuePtr { tag: TAG_FIELD | ((field as usize) << 6) }
    }

    pub(super) fn owned<T : OwnedValue>(value: Prefix<T>) -> ValuePtr {
        ValuePtr { tag: TAG_PTR | (Box::into_raw(Box::new(value)) as usize) }
    }

    pub(super) fn shared<T : SharedValue>(value: SharedPrefix<T>) -> ValuePtr {
        ValuePtr { tag: TAG_SHARED | (Box::into_raw(Box::new(value)) as usize) }
    }

    pub(super) fn error(err: RuntimeError) -> ValuePtr {
        ValuePtr { tag: TAG_ERR | (Box::into_raw(Box::new(err)) as usize) }
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

    pub fn as_precise_int(&self) -> i64 {
        debug_assert!(self.is_precise_int());
        unsafe {
            self.int >> 1
        }
    }

    /// If the current type is int-like, then automatically converts it to a complex number.
    pub fn as_complex(self) -> Complex<i64> {
        debug_assert!(self.ty() == Type::Bool || self.ty() == Type::Int || self.ty() == Type::Complex);
        match self.ty() {
            Type::Bool => Complex::new(self.is_true() as i64, 0),
            Type::Int => Complex::new(self.as_int(), 0),
            Type::Complex => self.as_precise_complex().value.inner,
            _ => unreachable!(),
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
        unsafe { (self.tag >> 6) as u32 }
    }

    /// Returns the `Type` of this value.
    ///
    /// **Implementation Note**
    ///
    /// Looking at the assembly of this function in compiler explorer shows some interesting insights.
    /// First, the generated assembly is _way better_ using the hardcoded functions like `is_int()` over `ty() == Type::Int`. So those will continue to exist,
    /// and should be used over `ty() == Type::Int`.
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
                    _ => Type::Never,
                },
                TAG_ERR => Type::Error,
                TAG_PTR | TAG_SHARED => (*self.as_ptr()).ty, // Check the prefix for the type
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
    pub const fn is_native(&self) -> bool { (unsafe { self.tag } & MASK_NATIVE) == TAG_NATIVE }
    pub const fn is_field(&self) -> bool { (unsafe { self.tag } & MASK_FIELD) == TAG_FIELD }
    pub const fn is_none(&self) -> bool { (unsafe { self.tag } & MASK_NONE) == TAG_NONE }
    pub const fn is_err(&self) -> bool { (unsafe { self.tag } & MASK_ERR) == TAG_ERR }

    fn is_ptr(&self) -> bool { (unsafe { self.tag } & MASK_PTR) == TAG_PTR }
    fn is_shared(&self) -> bool { (unsafe { self.tag } & MASK_SHARED) == TAG_SHARED }

    /// Transmutes this `ValuePtr` into a `Box<RuntimeError>`
    pub fn as_err(self) -> Box<RuntimeError> {
        debug_assert!(self.is_err());
        unsafe {
            let ret = Box::from_raw(self.as_ptr() as *mut RuntimeError);

            // Then forget the current `self`. This is a manual way of telling rust that we have fully transmuted ourselves into `ret`
            // We can't drop this here, since we technically created a copy of the same resource by calling `as_mut_ptr()`
            std::mem::forget(self);
            ret
        }
    }

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
        debug_assert!(self.is_ptr()); // Must be owned memory, to make sense converting to `Box<T>`
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
    pub(crate) fn as_ref<T: OwnedValue>(&self) -> &T {
        debug_assert!(self.is_ptr());
        unsafe {
            &(&*(self.as_ptr() as *const Prefix<T>)).value
        }
    }

    pub fn as_mut_ref<T : OwnedValue>(&mut self) -> &mut T {
        debug_assert!(self.is_ptr());
        unsafe {
            &mut (&mut*(self.as_ptr() as *mut Prefix<T>)).value
        }
    }

    /// Transmutes this `ValuePtr` into a `&T`, where `T` is a type compatible with `SharedPrefix`
    pub(crate) fn as_shared_ref<T: SharedValue>(&self) -> &SharedPrefix<T> {
        debug_assert!(self.is_shared()); // Any shared pointer type can be converted to a shared prefix.
        unsafe {
            &*(self.as_ptr() as *const SharedPrefix<T>)
        }
    }

    /// Strips away the tag bits, and converts this value into a `* mut` pointer to an arbitrary `Prefix`
    unsafe fn as_ptr(&self) -> *mut Prefix<()> {
        debug_assert!(self.is_ptr() || self.is_shared());
        unsafe {
            (self.tag & PTR_MASK) as *mut Prefix<()>
        }
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
        ValuePtr { tag: self.tag }
    }

    unsafe fn drop_owned<T: OwnedValue>(&self) {
        debug_assert!(self.is_ptr()); // Must be owned memory, to make sense converting to `Box<T>`
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
            Type::NativeFunction |
            Type::GetField => unsafe { self.tag == other.tag },
            // Owned types check equality based on their ref
            Type::Complex => self.as_ref::<ComplexImpl>() == other.as_ref::<ComplexImpl>(),
            Type::Range => self.as_ref::<RangeImpl>() == other.as_ref::<RangeImpl>(),
            Type::Enumerate => self.as_ref::<EnumerateImpl>() == other.as_ref::<EnumerateImpl>(),
            Type::PartialFunction => self.as_ref::<PartialFunctionImpl>() == other.as_ref::<PartialFunctionImpl>(),
            Type::PartialNativeFunction => self.as_ref::<PartialNativeFunctionImpl>() == other.as_ref::<PartialNativeFunctionImpl>(),
            Type::Slice => self.as_ref::<SliceImpl>() == other.as_ref::<SliceImpl>(),
            // Shared types check equality based on the shared ref
            Type::Str => self.as_shared_ref::<String>() == other.as_shared_ref::<String>(),
            Type::List => self.as_shared_ref::<ListImpl>() == other.as_shared_ref::<ListImpl>(),
            Type::Set => self.as_shared_ref::<SetImpl>() == other.as_shared_ref::<SetImpl>(),
            Type::Dict => self.as_shared_ref::<DictImpl>() == other.as_shared_ref::<DictImpl>(),
            Type::Heap => self.as_shared_ref::<HeapImpl>() == other.as_shared_ref::<HeapImpl>(),
            Type::Vector => self.as_shared_ref::<VectorImpl>() == other.as_shared_ref::<VectorImpl>(),
            Type::Struct => self.as_shared_ref::<StructImpl>() == other.as_shared_ref::<StructImpl>(),
            Type::StructType => self.as_shared_ref::<StructTypeImpl>() == other.as_shared_ref::<StructTypeImpl>(),
            Type::Memoized => self.as_shared_ref::<MemoizedImpl>() == other.as_shared_ref::<MemoizedImpl>(),
            Type::Function => self.as_shared_ref::<FunctionImpl>() == other.as_shared_ref::<FunctionImpl>(),
            Type::Closure => self.as_shared_ref::<ClosureImpl>() == other.as_shared_ref::<ClosureImpl>(),
            // Special types that are not checked for equality
            Type::Iter | Type::Error | Type::None | Type::Never => false,
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
            // Inline types can directly compare the tag value. This works even for nil, bool, native function, etc.
            Type::Nil |
            Type::Bool |
            Type::Int |
            Type::NativeFunction |
            Type::GetField => unsafe { self.tag.cmp(&other.tag) },
            // Owned types check equality based on their ref
            Type::Complex => self.as_ref::<ComplexImpl>().cmp(other.as_ref::<ComplexImpl>()),
            Type::Range => self.as_ref::<RangeImpl>().cmp(&other.as_ref::<RangeImpl>()),
            Type::Enumerate => self.as_ref::<EnumerateImpl>().cmp(other.as_ref::<EnumerateImpl>()),
            // Shared types check equality based on the shared ref
            Type::Str => self.as_shared_ref::<String>().cmp(&other.as_shared_ref::<String>()),
            Type::List => self.as_shared_ref::<ListImpl>().cmp(&other.as_shared_ref::<ListImpl>()),
            Type::Set => self.as_shared_ref::<SetImpl>().cmp(&other.as_shared_ref::<SetImpl>()),
            Type::Dict => self.as_shared_ref::<DictImpl>().cmp(&other.as_shared_ref::<DictImpl>()),
            Type::Heap => self.as_shared_ref::<HeapImpl>().cmp(&other.as_shared_ref::<HeapImpl>()),
            Type::Vector => self.as_shared_ref::<VectorImpl>().cmp(&other.as_shared_ref::<VectorImpl>()),
            Type::Struct => self.as_shared_ref::<StructImpl>().cmp(&other.as_shared_ref::<StructImpl>()),
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
                Type::NativeFunction |
                Type::GetField => ValuePtr { tag: self.tag },
                // Owned types
                Type::Complex => self.clone_owned::<ComplexImpl>(),
                Type::Range => self.clone_owned::<RangeImpl>(),
                Type::Enumerate => self.clone_owned::<EnumerateImpl>(),
                Type::PartialFunction => self.clone_owned::<PartialFunctionImpl>(),
                Type::PartialNativeFunction => self.clone_owned::<PartialNativeFunctionImpl>(),
                Type::Slice => self.clone_owned::<SliceImpl>(),
                Type::Iter => self.clone_owned::<Iterable>(),
                // Shared types
                Type::Str => self.clone_shared::<String>(),
                Type::List => self.clone_shared::<ListImpl>(),
                Type::Set => self.clone_shared::<SetImpl>(),
                Type::Dict => self.clone_shared::<DictImpl>(),
                Type::Heap => self.clone_shared::<HeapImpl>(),
                Type::Vector => self.clone_shared::<VectorImpl>(),
                Type::Struct => self.clone_shared::<StructImpl>(),
                Type::StructType => self.clone_shared::<StructTypeImpl>(),
                Type::Memoized => self.clone_shared::<MemoizedImpl>(),
                Type::Function => self.clone_shared::<FunctionImpl>(),
                Type::Closure => self.clone_shared::<ClosureImpl>(),
                // Special types
                Type::Error => {
                    let err = ValuePtr { tag: self.tag }.as_err();
                    let copy = err.clone().to_value();
                    std::mem::forget(err);
                    copy
                },
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
                Type::NativeFunction |
                Type::GetField => {},
                // Owned types
                Type::Complex => self.drop_owned::<ComplexImpl>(),
                Type::Range => self.drop_owned::<RangeImpl>(),
                Type::Enumerate => self.drop_owned::<EnumerateImpl>(),
                Type::PartialFunction => self.drop_owned::<PartialFunctionImpl>(),
                Type::PartialNativeFunction => self.drop_owned::<PartialNativeFunctionImpl>(),
                Type::Slice => self.drop_owned::<SliceImpl>(),
                Type::Iter => self.drop_owned::<Iterable>(),
                // Shared types
                Type::Str => self.drop_shared::<String>(),
                Type::List => self.drop_shared::<ListImpl>(),
                Type::Set => self.drop_shared::<SetImpl>(),
                Type::Dict => self.drop_shared::<DictImpl>(),
                Type::Heap => self.drop_shared::<HeapImpl>(),
                Type::Vector => self.drop_shared::<VectorImpl>(),
                Type::Struct => self.drop_shared::<StructImpl>(),
                Type::StructType => self.drop_shared::<StructTypeImpl>(),
                Type::Memoized => self.drop_shared::<MemoizedImpl>(),
                Type::Function => self.drop_shared::<FunctionImpl>(),
                Type::Closure => self.drop_shared::<ClosureImpl>(),
                // Special types
                Type::Error => {
                    // Can't call .as_err() and drop that since we only have a &self
                    drop(Box::from_raw(self.as_ptr() as *mut RuntimeError));
                },
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
            Type::NativeFunction |
            Type::GetField => unsafe { self.tag }.hash(state),
            // Owned types
            Type::Complex => self.as_ref::<ComplexImpl>().hash(state),
            Type::Range => self.as_ref::<RangeImpl>().hash(state),
            Type::Enumerate => self.as_ref::<EnumerateImpl>().hash(state),
            Type::PartialFunction => self.as_ref::<PartialFunctionImpl>().hash(state),
            Type::PartialNativeFunction => self.as_ref::<PartialNativeFunctionImpl>().hash(state),
            Type::Slice => self.as_ref::<SliceImpl>().hash(state),
            // Shared types
            Type::Str => self.as_shared_ref::<String>().hash(state),
            Type::List => self.as_shared_ref::<ListImpl>().hash(state),
            Type::Set => self.as_shared_ref::<SetImpl>().hash(state),
            Type::Dict => self.as_shared_ref::<DictImpl>().hash(state),
            Type::Heap => self.as_shared_ref::<HeapImpl>().hash(state),
            Type::Vector => self.as_shared_ref::<VectorImpl>().hash(state),
            Type::Struct => self.as_shared_ref::<StructImpl>().hash(state),
            Type::StructType => self.as_shared_ref::<StructTypeImpl>().hash(state),
            Type::Memoized => self.as_shared_ref::<MemoizedImpl>().hash(state),
            Type::Function => self.as_shared_ref::<FunctionImpl>().hash(state),
            Type::Closure => self.as_shared_ref::<ClosureImpl>().hash(state),
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
            Type::NativeFunction => Debug::fmt(&self.as_native(), f),
            Type::GetField => Debug::fmt(&self.as_field(), f),
            // Owned types
            Type::Complex => Debug::fmt(self.as_ref::<ComplexImpl>(), f),
            Type::Range => Debug::fmt(self.as_ref::<RangeImpl>(), f),
            Type::Enumerate => Debug::fmt(self.as_ref::<EnumerateImpl>(), f),
            Type::PartialFunction => Debug::fmt(self.as_ref::<PartialFunctionImpl>(), f),
            Type::PartialNativeFunction => Debug::fmt(self.as_ref::<PartialNativeFunctionImpl>(), f),
            Type::Slice => Debug::fmt(self.as_ref::<SliceImpl>(), f),
            // Shared types
            Type::Str => Debug::fmt(self.as_shared_ref::<String>(), f),
            Type::List => Debug::fmt(self.as_shared_ref::<ListImpl>(), f),
            Type::Set => Debug::fmt(self.as_shared_ref::<SetImpl>(), f),
            Type::Dict => Debug::fmt(self.as_shared_ref::<DictImpl>(), f),
            Type::Heap => Debug::fmt(self.as_shared_ref::<HeapImpl>(), f),
            Type::Vector => Debug::fmt(self.as_shared_ref::<VectorImpl>(), f),
            Type::Struct => Debug::fmt(self.as_shared_ref::<StructImpl>(), f),
            Type::StructType => Debug::fmt(self.as_shared_ref::<StructTypeImpl>(), f),
            Type::Memoized => Debug::fmt(self.as_shared_ref::<MemoizedImpl>(), f),
            Type::Function => Debug::fmt(self.as_shared_ref::<FunctionImpl>(), f),
            Type::Closure => Debug::fmt(self.as_shared_ref::<ClosureImpl>(), f),
            // Special types with no hash behavior
            Type::Iter => write!(f, "Iter"),
            Type::Error => write!(f, "Error"),
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
    pub(crate) fn prefix(ty: Type, value: T) -> Prefix<T> {
        Prefix { ty, value }
    }
}

impl<T : Clone + OwnedValue> Clone for Prefix<T> {
    fn clone(&self) -> Self {
        Prefix::prefix(self.ty, self.value.clone())
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
    pub fn prefix(ty: Type, value: T) -> SharedPrefix<T> {
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

impl SharedPrefix<ClosureImpl> {
    pub fn borrow_func(&self) -> &FunctionImpl {
        unsafe {
            NonNull::new_unchecked(self.value.get()).as_ref().func.ptr.as_function().borrow_const()
        }
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
    use num_complex::Complex;
    use crate::core::NativeFunction;
    use crate::vm::IntoValue;
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
            assert!(!ptr.is_shared());
            assert_eq!(ptr.ty(), Type::Int);
            assert_eq!(ptr.clone().as_int(), int);
            assert_eq!(ptr.clone().as_precise_int(), int);
            assert_eq!(format!("{:?}", ptr), format!("{}", int));
        }
    }

    #[test]
    #[should_panic]
    fn test_inline_int_too_small() {
        let _ = (MIN_INT - 1).to_value();
    }

    #[test]
    #[should_panic]
    fn test_inline_int_too_large() {
        let _ = (MAX_INT + 1).to_value();
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
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::None);
    }

    #[test]
    fn test_owned_complex() {
        let ptr = Complex::<i64>::new(1, 2).to_value();

        assert!(!ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(ptr.is_ptr());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Complex);
        assert_eq!(format!("{:?}", ptr), format!("{:?}", Complex::<i64>::new(1, 2)));
    }

    #[test]
    fn test_owned_complex_clone_eq() {
        let c0 = ValuePtr::nil();
        let c1 = Complex::<i64>::new(1, 2).to_value();
        let c2 = Complex::<i64>::new(1, 3).to_value();

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
        assert!(!ptr.is_ptr());
        assert!(ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Str);
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

    #[test]
    fn test_shared_const_str_can_borrow_const() {
        let ptr = "".to_value();

        let _r1 = ptr.as_str().borrow_const();
        let _r2 = ptr.as_str().borrow();
    }
}