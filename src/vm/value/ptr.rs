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
use crate::vm::RuntimeError;
use crate::vm::value::{OwnedValue, SharedValue, Type, ValueRef};



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
/// - An inline constant `nil`, `true`, or `false`, or `NativeFunction`, all stored with the lowest three bits equal to `001`
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


const TAG_INT: usize       = 0b___000;
const TAG_NIL: usize       = 0b_00001;
const TAG_BOOL: usize      = 0b_01001;
const TAG_FALSE: usize     = 0b001001;
const TAG_TRUE: usize      = 0b101001;
const TAG_NATIVE: usize    = 0b_10001;
const TAG_NONE: usize      = 0b_11001;
const TAG_ERR: usize       = 0b___011;
const TAG_PTR: usize       = 0b___101;
const TAG_SHARED: usize    = 0b___111;

const MASK_INT: usize      = 0b_____1;
const MASK_NIL: usize      = 0b_11111;
const MASK_BOOL: usize     = 0b_11111;
const MASK_TRUE: usize     = 0b111111;
const MASK_FALSE: usize    = 0b111111;
const MASK_NATIVE: usize   = 0b_11111;
const MASK_NONE: usize     = 0b_11111;
const MASK_ERR: usize      = 0b___111;
const MASK_PTR: usize      = 0b___111;
const MASK_SHARED: usize   = 0b___111;

const TY_MASK: usize   = 0b111;
const PTR_MASK: usize = 0xffff_ffff_ffff_fff8;

const MAX_INT: i64 = 0x3fff_ffff_ffff_ffffu64 as i64;
const MIN_INT: i64 = 0xc000_0000_0000_0000u64 as i64;


impl ValuePtr {

    pub fn nil() -> ValuePtr {
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
        ValuePtr { tag: TAG_NATIVE | ((value as usize) << 5) }
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
        (unsafe { self.int }) >> 1
    }

    pub fn as_int_like(&self) -> i64 {
        debug_assert!(self.ty() == Type::Bool || self.ty() == Type::Int);
        if self.is_int() {
            self.as_int()
        } else {
            self.is_true() as i64
        }
    }

    /// If the current type is int-like, then automatically converts it to a complex number.
    pub fn as_complex_like(self) -> Complex<i64> {
        debug_assert!(self.ty() == Type::Bool || self.ty() == Type::Int || self.ty() == Type::Complex);
        match self.ty() {
            Type::Bool => Complex::new(self.is_true() as i64, 0),
            Type::Int => Complex::new(self.as_int(), 0),
            Type::Complex => self.as_complex().value.inner(),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn as_option(self) -> Option<ValuePtr> {
        match self.is_none() {
            true => None,
            false => Some(self),
        }
    }

    pub fn as_bool(&self) -> bool {
        debug_assert!(self.is_bool());
        self.is_true()
    }

    pub fn as_native(&self) -> NativeFunction {
        debug_assert!(self.is_native());
        unsafe { std::mem::transmute((self.tag >> 5) as u8) }
    }

    /// Returns the `Type` of this value.
    ///
    /// In theory this should be **aggressively** inlined, especially into surrounding methods on `ValuePtr`, as it vastly simplifies some checks
    pub fn ty(&self) -> Type {
        unsafe {
            match self.tag & MASK_PTR {
                TAG_NIL => match self.tag & MASK_BOOL {
                    TAG_NIL => Type::Nil,
                    TAG_BOOL => Type::Bool,
                    TAG_NATIVE => Type::Native,
                    TAG_NONE => Type::None,
                    _ => unreachable!(),
                },
                TAG_ERR => Type::Error,
                TAG_PTR | TAG_SHARED => (*self.as_ptr()).ty, // Check the prefix for the type
                _ => Type::Int, // Includes all remaining bit patterns with a `0` LSB
            }
        }
    }

    // Utility, and efficient, functions for checking the type of inline or pointer values.

    pub fn is_nil(&self) -> bool { (unsafe { self.tag } & MASK_NIL) == TAG_NIL }
    pub fn is_bool(&self) -> bool { (unsafe { self.tag } & MASK_BOOL) == TAG_BOOL }
    pub fn is_true(&self) -> bool { (unsafe { self.tag }) == TAG_TRUE }
    pub fn is_int(&self) -> bool { (unsafe { self.tag } & MASK_INT) == TAG_INT }
    pub fn is_native(&self) -> bool { (unsafe { self.tag } & MASK_NATIVE) == TAG_NATIVE }
    pub fn is_none(&self) -> bool { (unsafe { self.tag } & MASK_NONE) == TAG_NONE }
    pub fn is_err(&self) -> bool { (unsafe { self.tag } & MASK_ERR) == TAG_ERR }

    fn is_ptr(&self) -> bool { (unsafe { self.tag } & MASK_PTR) == TAG_PTR }
    fn is_shared(&self) -> bool { (unsafe { self.tag } & MASK_SHARED) == TAG_SHARED }

    /// Returns if the value is iterable.
    pub fn is_iter(&self) -> bool {
        matches!(self.ty(), Type::Str | Type::List | Type::Set | Type::Dict | Type::Heap | Type::Vector | Type::Range | Type::Enumerate)
    }

    /// Returns if the value is function-evaluable. Note that single-element lists are not considered functions here.
    pub fn is_evaluable(&self) -> bool {
        matches!(self.ty(), Type::Function | Type::PartialFunction | Type::Native | Type::PartialNativeFunction | Type::Closure | Type::StructType | Type::Slice)
    }

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
    pub(crate) fn as_ref<T: OwnedValue>(&self) -> &Prefix<T> {
        debug_assert!(self.is_ptr() || self.is_shared()); // Any pointer type (owned or shared), can be converted to a reference on a 1-1 basis.
        unsafe {
            &*(self.as_ptr() as *const Prefix<T>)
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
            Type::Native => unsafe { self.tag == other.tag },
            // Owned types check equality based on their ref
            Type::Complex => self.as_ref::<C64>() == other.as_ref::<C64>(),
            // Shared types check equality based on the shared ref
            Type::Str => self.as_shared_ref::<String>() == other.as_shared_ref::<String>(),
            _ => unimplemented!()
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
            Type::Native => unsafe { self.tag.cmp(&other.tag) },
            Type::Complex => self.as_ref::<C64>().cmp(self.as_ref::<C64>()),
            _ => unimplemented!(),
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
                Type::Nil |
                Type::Bool |
                Type::Int |
                Type::Native => ValuePtr { tag: self.tag },
                Type::Complex => self.clone_owned::<C64>(),
                Type::Str => self.clone_shared::<String>(),
                _ => unimplemented!(),
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
                Type::Native |
                Type::None => {},
                Type::Complex => self.drop_owned::<C64>(),
                Type::Str => self.drop_shared::<String>(),
                _ => unimplemented!(),
            }
        }
    }
}


/// `Hash` just needs to call the hash methods on the underlying type.
/// Again, for inline types we can just hash the tag directly.
impl Hash for ValuePtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.ty() {
            Type::Nil |
            Type::Bool |
            Type::Int |
            Type::Native => unsafe { self.tag }.hash(state),
            Type::Complex => self.as_ref::<C64>().hash(state),
            _ => unimplemented!(),
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
            Type::Native => Debug::fmt(&self.as_native(), f),
            Type::Complex => Debug::fmt(&self.as_ref::<C64>(), f),
            Type::Str => Debug::fmt(&self.as_shared_ref::<String>(), f),
            _ => unimplemented!(),
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

const BORROW_MUT: u16 = 0;
const BORROW_NONE: u16 = 1;

impl<T : SharedValue> SharedPrefix<T> {
    pub fn prefix(ty: Type, value: T) -> SharedPrefix<T> {
        SharedPrefix {
            ty,
            lock: Cell::new(BORROW_NONE),
            refs: Cell::new(1),
            value: UnsafeCell::new(value)
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

    /// Copied from the implementation in `RefCell`
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
    use crate::core::NativeFunction;
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
        let ptr = ValuePtr::from(true);
        assert!(!ptr.is_nil());
        assert!(ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Bool);
        assert!(ptr.clone().as_bool());
        assert_eq!(ptr.clone().as_int_like(), 1);
        assert_eq!(format!("{:?}", ptr), "true");
    }

    #[test]
    fn test_inline_false() {
        let ptr = ValuePtr::from(false);
        assert!(!ptr.is_nil());
        assert!(ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(!ptr.is_ptr());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Bool);
        assert!(!ptr.clone().as_bool());
        assert_eq!(ptr.clone().as_int_like(), 0);
        assert_eq!(format!("{:?}", ptr), "false");
    }

    #[test]
    fn test_inline_int() {
        for int in -25..25 {
            let ptr = ValuePtr::from(int);
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
            assert_eq!(ptr.clone().as_int_like(), int);
            assert_eq!(format!("{:?}", ptr), format!("{}", int));
        }

        for int in [MIN_INT, MIN_INT + 1, MAX_INT - 1, MAX_INT] {
            let ptr = ValuePtr::from(int);
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
            assert_eq!(ptr.clone().as_int_like(), int);
            assert_eq!(format!("{:?}", ptr), format!("{}", int));
        }
    }

    #[test]
    #[should_panic]
    fn test_inline_int_too_small() {
        let _ = ValuePtr::from(MIN_INT - 1);
    }

    #[test]
    #[should_panic]
    fn test_inline_int_too_large() {
        let _ = ValuePtr::from(MAX_INT + 1);
    }

    #[test]
    fn test_inline_native_function() {
        for f in 0..NativeFunction::total() {
            let f: NativeFunction = unsafe { std::mem::transmute(f as u8) };
            let ptr = ValuePtr::from(f);
            assert!(!ptr.is_nil());
            assert!(!ptr.is_bool());
            assert!(!ptr.is_int());
            assert!(ptr.is_native());
            assert!(!ptr.is_none());
            assert!(!ptr.is_err());
            assert!(!ptr.is_ptr());
            assert!(!ptr.is_shared());
            assert_eq!(ptr.ty(), Type::Native);
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
    fn test_inline_none_as_option() {
        assert_eq!(ValuePtr::none().as_option(), None);
        assert_eq!(ValuePtr::nil().as_option(), Some(ValuePtr::nil()));
    }

    #[test]
    fn test_owned_complex() {
        let ptr = ValuePtr::from(C64::new(1, 2));

        assert!(!ptr.is_nil());
        assert!(!ptr.is_bool());
        assert!(!ptr.is_int());
        assert!(!ptr.is_native());
        assert!(!ptr.is_none());
        assert!(!ptr.is_err());
        assert!(ptr.is_ptr());
        assert!(!ptr.is_shared());
        assert_eq!(ptr.ty(), Type::Complex);
        assert_eq!(format!("{:?}", ptr), format!("{:?}", C64::new(1, 2)));
    }

    #[test]
    fn test_owned_complex_clone_eq() {
        let c0 = ValuePtr::nil();
        let c1 = ValuePtr::from(C64::new(1, 2));
        let c2 = ValuePtr::from(C64::new(1, 3));

        assert_ne!(c2, c0);
        assert_ne!(c2, c1);
        assert_eq!(c2, c2);
        assert_ne!(c1, c2.clone());
        assert_eq!(c2, c2.clone());
    }

    #[test]
    fn test_shared_str() {
        let ptr = ValuePtr::from(String::from("hello world"));

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
    fn test_shared_str_eq() {
        let ptr1 = ValuePtr::from(String::new());
        let ptr2 = ValuePtr::from(String::new());

        assert_eq!(ptr1, ptr2);
        assert_eq!(ptr1, ptr1);
    }

    #[test]
    fn test_shared_str_clone() {
        let ptr = ValuePtr::from(String::new());

        assert_eq!(ptr, ptr.clone());
        assert_eq!(ptr.clone(), ptr.clone());
    }

    #[test]
    #[should_panic]
    fn test_shared_str_two_mutable_borrow_panic() {
        let ptr = ValuePtr::from(String::new());

        let _r1 = ptr.as_str().borrow_mut();
        let _r2 = ptr.as_str().borrow_mut();
    }

    #[test]
    #[should_panic]
    fn test_shared_str_mutable_and_normal_borrow_panic() {
        let ptr = ValuePtr::from(String::new());

        let _r1 = ptr.as_str().borrow();
        let _r2 = ptr.as_str().borrow_mut();
    }

    #[test]
    fn test_shared_str_multiple_borrow_no_panic() {
        let ptr = ValuePtr::from(String::new());

        let _r1 = ptr.as_str().borrow();
        let _r2 = ptr.as_str().borrow();
    }
}