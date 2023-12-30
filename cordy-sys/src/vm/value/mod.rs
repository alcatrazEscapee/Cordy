use std::borrow::Cow;
use std::cell::Cell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::convert::Infallible;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::{FromIterator, FusedIterator};
use std::ops::{ControlFlow, FromResidual, Try};
use fxhash::FxBuildHasher;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use range::Enumerate;

use crate::compiler::Fields;
use crate::core;
use crate::core::{InvokeArg0, NativeFunction};
use crate::util::impl_partial_ord;
use crate::vm::error::RuntimeError;
use crate::vm::value::ptr::{RefMut, SharedPrefix};
use crate::vm::value::str::IterStr;
use crate::vm::value::range::Range;
use crate::vm::value::slice::Slice;
use crate::vm::value::complex::Complex;

pub use crate::vm::value::ptr::{MAX_INT, MIN_INT, Prefix, ValuePtr};
pub use crate::vm::value::complex::ComplexValue;
pub use crate::vm::value::func::{Closure, Function, PartialFunction, PartialNativeFunction, UpValue};

use RuntimeError::{*};

pub type ErrorResult<T> = Result<T, Box<Prefix<RuntimeError>>>;
pub type AnyResult = ErrorResult<()>;


mod ptr;
mod str;
mod func;
mod range;
mod slice;
mod complex;


/// `Type` is an enumeration of all the possible types (not including user-defined type variants such as `struct`s) possible in Cordy.
/// These are the types that are present and checked at runtime.
#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Type {
    Nil,
    Bool,
    Int,
    NativeFunction,
    GetField,
    Complex,
    ShortStr,
    LongStr,
    List,
    Set,
    Dict,
    Heap,
    Vector,
    Struct,
    StructType,
    Range,
    Enumerate,
    Slice,
    Iter,
    Memoized,
    Function,
    PartialFunction,
    PartialNativeFunction,
    Closure,
    Error,
    None, // Useful when we would otherwise hold an `Option<ValuePtr>` - this compresses the `None` state
    Never, // Optimization for type-checking code, to avoid code paths containing `unreachable!()` or similar patterns.
}

impl Type {
    fn is_owned(&self) -> bool {
        matches!(self, Type::Iter | Type::Error)
    }

    fn is_shared(&self) -> bool {
        matches!(self, Type::PartialNativeFunction | Type::PartialFunction | Type::Complex | Type::LongStr | Type::List | Type::Set | Type::Dict | Type::Heap | Type::Vector | Type::Function | Type::Closure | Type::Memoized | Type::Struct | Type::StructType | Type::Range | Type::Enumerate | Type::Slice)
    }
}


/// A trait marking the value type of an owned piece of data.
/// This means a `ValuePtr` may point to a `Prefix<T : OwnedValue>`
pub trait OwnedValue {}

/// A trait marking the value type of a shared (reference counted) piece of data.
/// This means a `ValuePtr` may point to a `SharedPrefix<T : SharedValue>`
pub trait SharedValue {}

/// A trait marking that the shared value is const (immutable), and thus can be directly accessed via `.borrow_const()`
///
/// Note that in the absence of negative type bounds (i.e. `SharedValue + !ConstValue`, this can lead to unsafe code, since `borrow_mut()` and `borrow_const()` can both be called for const types.
/// So, we split that into `MutValue`, and hope that nothing implements both `MutValue` and `SharedValue`
pub trait ConstValue : SharedValue {}

/// The counterpart to `ConstValue`, this enables `RefCell<T>` like behavior via `.borrow()` and `.borrow_mut()` for the underlying shared value.
///
/// This must **not be implemented in addition to `ConstValue`!**
pub trait MutValue : SharedValue {}


/// A specialized reference-equality version of `ValuePtr`. Unlike `ValuePtr`, this does not manage any memory, and can be created multiple times from a `ValuePtr`
///
/// This only is for tracking reference equality between two values. Note that because this is a direct copy of the _pointer_, there is no
/// guarantee that it stays valid, and converting back to a `ValuePtr` is explicitly undefined behavior.
#[derive(Debug, Eq, PartialEq)]
pub struct ValueRef {
    ptr: usize
}

impl ValueRef {
    pub(super) fn new(ptr: usize) -> ValueRef {
        ValueRef { ptr }
    }
}


/// An explicit wrapper for `Option<ValuePtr>`. Note that due to unused bit patterns, we don't need to ever wrap `ValuePtr` in an `Option<T>` (as that would waste bytes.
/// However, it is _very unclear_, and bad coding style, to always have `ValuePtr` when it's unclear if it should be optional. So, in all situations where a `ValuePtr` should be treated as a `Option<ValuePtr>`, we box it in this struct instead.
///
/// The contract here ensures that when we match against this, as an optional (which should be completely inline-able, and zero cost), we unwrap something that is not `None`
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ValueOption {
    ptr: ValuePtr
}

impl ValueOption {
    pub fn none() -> ValueOption {
        ValueOption { ptr: ValuePtr::none() }
    }

    pub fn some(ptr: ValuePtr) -> ValueOption {
        debug_assert!(!ptr.is_none()); // Should never create a `some()` from a `none()`
        ValueOption { ptr }
    }

    #[inline(always)]
    pub fn as_option(self) -> Option<ValuePtr> {
        match self.ptr.is_none() {
            true => None,
            false => Some(self.ptr)
        }
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        self.ptr.is_none()
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        !self.ptr.is_none()
    }

    /// Takes the value out of the option, leaving `None` in its place.
    #[inline]
    pub fn take(&mut self) -> ValueOption {
        std::mem::replace(self, ValueOption::none())
    }
}

impl From<Option<ValuePtr>> for ValueOption {
    fn from(value: Option<ValuePtr>) -> Self {
        match value {
            Some(ptr) => ValueOption::some(ptr),
            None => ValueOption::none(),
        }
    }
}

/// Like `Result<ValuePtr, Err>`, but again, since `ValuePtr` boxes the error inside it, this makes it explicit.
#[derive(Debug, Eq, PartialEq)]
pub struct ValueResult {
    ptr: ValuePtr
}

impl ValueResult {
    pub fn ok(ptr: ValuePtr) -> ValueResult {
        debug_assert!(!ptr.is_err()); // Should never have an error-type in `ok()`
        ValueResult { ptr }
    }

    #[cold]
    pub fn err(ptr: ValuePtr) -> ValueResult {
        debug_assert!(ptr.is_err());
        ValueResult { ptr }
    }

    pub fn new(ptr: ValuePtr) -> ValueResult {
        ValueResult { ptr }
    }

    /// Boxes this `ValueResult` into a traditional Rust `Result<T, E>`. The left will be guaranteed to contain a non-error type, and the right will be guaranteed to contain an error.
    #[inline(always)]
    pub fn as_result(self) -> ErrorResult<ValuePtr> {
        match self.ptr.is_err() {
            true => Err(self.ptr.as_err()),
            false => Ok(self.ptr)
        }
    }

    #[inline]
    pub fn is_ok(&self) -> bool {
        !self.ptr.is_err()
    }

    #[inline]
    pub fn is_err(&self) -> bool {
        self.ptr.is_err()
    }
}


/// Converts a `ErrorResult<T>` returned from a `?` expression into a `ValueResult`
impl FromResidual<ErrorResult<Infallible>> for ValueResult {
    fn from_residual(residual: ErrorResult<Infallible>) -> Self {
        match residual {
            Err(err) => ValueResult::from(err.value),
            _ => unreachable!(),
        }
    }
}

/// Converts a `Box<Prefix<RuntimeError>>` returned as the residual from a `ValueResult?` into a `ErrorResult<T>`
impl<T> FromResidual<Box<Prefix<RuntimeError>>> for ErrorResult<T> {
    fn from_residual(residual: Box<Prefix<RuntimeError>>) -> Self {
        Err(residual)
    }
}

/// Allows us to use `?` operator on `check_<T>()?` expressions, which assert that the value is of a given type, and early return if not.
/// This does, unfortunately, require nightly unstable rust, but the code clarity is worth having (and it allows us to seamlessly use `ValueResult`
/// as a zero-cost (memory layout) wise abstraction for `Result<ValuePtr, Box<Prefix<RuntimeError>>>` (which would otherwise be twice the stack size.
impl Try for ValueResult {
    type Output = ValuePtr;
    type Residual = Box<Prefix<RuntimeError>>;

    fn from_output(ptr: ValuePtr) -> ValueResult {
        ValueResult { ptr }
    }

    fn branch(self) -> ControlFlow<Box<Prefix<RuntimeError>>, ValuePtr> {
        match self.ptr.is_err() {
            true => ControlFlow::Break(self.ptr.as_err()),
            false => ControlFlow::Continue(self.ptr),
        }
    }
}

/// Associated type for `Try`
impl FromResidual for ValueResult {
    fn from_residual(residual: Box<Prefix<RuntimeError>>) -> ValueResult {
        ValueResult { ptr: ptr::from_owned(*residual) }
    }
}

impl FromIterator<ValueResult> for ErrorResult<Vec<ValuePtr>> {
    fn from_iter<T: IntoIterator<Item=ValueResult>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut vec: Vec<ValuePtr> = Vec::with_capacity(iter.size_hint().0);
        for ptr in iter {
            match ptr.as_result() {
                Ok(ptr) => vec.push(ptr),
                Err(err) => return Err(err),
            }
        }
        Ok(vec)
    }
}


impl ValuePtr {

    // Constructors

    pub fn field(field: u32) -> ValuePtr {
        ptr::from_field(field)
    }

    pub fn instance(owner: ValuePtr, values: Vec<ValuePtr>) -> ValuePtr {
        StructImpl { owner, values }.to_value()
    }

    pub fn memoized(func: ValuePtr) -> ValuePtr {
        MemoizedImpl {
            func,
            cache: HashMap::with_hasher(FxBuildHasher::default()),
        }.to_value()
    }

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn to_str(&self) -> Cow<str> { self.safe_to_str(&mut RecursionGuard::new()) }

    fn safe_to_str(&self, rc: &mut RecursionGuard) -> Cow<str> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Cow::from(self.as_str_slice()),
            Type::Function => Cow::from(self.as_function().borrow_const().name()),
            Type::PartialFunction => self.as_partial_function().borrow_const().to_str(rc),
            Type::NativeFunction => Cow::from(self.as_native().name()),
            Type::PartialNativeFunction => Cow::from(self.as_partial_native().borrow_const().to_str()),
            Type::Closure => Cow::from(self.as_closure().borrow_func().name().to_string()),
            _ => self.safe_to_repr_str(rc),
        }
    }

    /// Converts the `Value` to a representative `String`. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn to_repr_str(&self) -> Cow<str> { self.safe_to_repr_str(&mut RecursionGuard::new()) }

    fn safe_to_repr_str(&self, rc: &mut RecursionGuard) -> Cow<str> {
        macro_rules! recursive_guard {
            ($default:expr, $recursive:expr) => {{
                let ret = if rc.enter(self) { $default } else { $recursive };
                rc.leave();
                ret
            }};
        }

        fn map_join<'a, 'b, I : Iterator<Item=&'b ValuePtr>>(rc: &mut RecursionGuard, mut iter: I, prefix: char, suffix: char, empty: &'a str, sep: &str) -> Cow<'a, str> {
            // Avoids issues with `.map().join()` that create temporaries in the `map()` and then destroy them
            match iter.next() {
                None => Cow::from(empty),
                Some(first) => {
                    let (lower, _) = iter.size_hint();
                    let mut result = String::with_capacity(lower * (sep.len() + 2));
                    result.push(prefix);
                    result.push_str(&first.safe_to_repr_str(rc));
                    while let Some(next) = iter.next() {
                        result.push_str(sep);
                        result.push_str(&next.safe_to_repr_str(rc));
                    }
                    result.push(suffix);
                    Cow::from(result)
                }
            }
        }

        match self.ty() {
            Type::Nil => Cow::from("nil"),
            Type::Bool => Cow::from(if self.is_true() { "true" } else { "false" }),
            Type::Int => Cow::from(self.as_int().to_string()),
            Type::Complex => self.as_complex_ref().to_repr_str(),
            Type::ShortStr | Type::LongStr => {
                let escaped = format!("{:?}", self.as_str_slice());
                Cow::from(format!("'{}'", &escaped[1..escaped.len() - 1]))
            },

            Type::List => recursive_guard!(
                Cow::from("[...]"),
                map_join(rc, self.as_list().borrow().list.iter(), '[', ']', "[]", ", ")
            ),
            Type::Set => recursive_guard!(
                Cow::from("{...}"),
                map_join(rc, self.as_set().borrow().set.iter(), '{', '}', "{}", ", ")
            ),
            Type::Dict => recursive_guard!(
                Cow::from("{...}"),
                Cow::from(format!("{{{}}}", self.as_dict()
                    .borrow()
                    .dict
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k.safe_to_repr_str(rc), v.safe_to_repr_str(rc)))
                    .join(", ")))
            ),
            Type::Heap => recursive_guard!(
                Cow::from("[...]"),
                map_join(rc, self.as_heap().borrow().heap.iter().map(|u| &u.0), '[', ']', "[]",  ", ")
            ),
            Type::Vector => recursive_guard!(
                Cow::from("(...)"),
                map_join(rc, self.as_vector().borrow().vector.iter(), '(', ')', "()", ", ")
            ),

            Type::Struct => {
                let it = self.as_struct().borrow();
                recursive_guard!(
                    Cow::from(format!("{}(...)", it.get_owner().name)),
                    Cow::from(format!("{}({})", it.get_owner().name.as_str(), it.values.iter()
                        .zip(it.get_owner().fields.iter())
                        .map(|(v, k)| format!("{}={}", k, v.safe_to_repr_str(rc)))
                        .join(", ")))
                )
            },
            Type::StructType => Cow::from(self.as_struct_type().borrow_const().as_str()),

            Type::Range => self.as_range().borrow_const().to_repr_str(),
            Type::Enumerate => self.as_enumerate().borrow_const().to_repr_str(rc),
            Type::Slice => self.as_slice().borrow_const().to_repr_str(),

            Type::Iter => Cow::from("<synthetic> iterator"),
            Type::Memoized => Cow::from(format!("@memoize {}", self.as_memoized().borrow().func.safe_to_repr_str(rc))),

            Type::GetField => Cow::from("(->)"),

            Type::Function => Cow::from(self.as_function().borrow_const().to_repr_str()),
            Type::PartialFunction => self.as_partial_function().borrow_const().to_repr_str(rc),
            Type::NativeFunction => Cow::from(self.as_native().to_repr_str()),
            Type::PartialNativeFunction => Cow::from(self.as_partial_native().borrow_const().to_repr_str()),
            Type::Closure => Cow::from(self.as_closure().borrow_func().to_repr_str()),

            Type::Error | Type::None | Type::Never => unreachable!(),
        }
    }

    /// Returns the inner user function, either from a `Function` or `Closure` type
    pub fn as_function_or_closure(&self) -> &Function {
        match self.is_function() {
            true => self.as_function().borrow_const(),
            false => self.as_closure().borrow_func(),
        }
    }

    /// Represents the type of this `Value`. This is used for runtime error messages,
    pub fn as_type_str(&self) -> String {
        String::from(match self.ty() {
            Type::Nil => "nil",
            Type::Bool => "bool",
            Type::Int => "int",
            Type::Complex => "complex",
            Type::ShortStr | Type::LongStr => "str",
            Type::List => "list",
            Type::Set => "set",
            Type::Dict => "dict",
            Type::Heap => "heap",
            Type::Vector => "vector",
            Type::Struct => "struct",
            Type::StructType => "struct type",
            Type::Range => "range",
            Type::Enumerate => "enumerate",
            Type::Slice => "slice",
            Type::Iter => "iter",
            Type::Memoized => "memoized",
            Type::GetField => "get field",
            Type::Function => "function",
            Type::PartialFunction => "partial function",
            Type::NativeFunction => "native function",
            Type::PartialNativeFunction => "partial native function",
            Type::Closure => "closure",
            Type::Error => "error",
            Type::None => "none",
            Type::Never => "never"
        })
    }

    /// Used by `trace` disabled code, do not remove!
    pub fn as_debug_str(&self) -> String {
        format!("{}: {}", self.to_repr_str(), self.as_type_str())
    }

    pub fn to_bool(&self) -> bool {
        match self.ty() {
            Type::Nil => false,
            Type::Bool => self.as_bool(),
            Type::Int => self.as_int() != 0,
            Type::ShortStr | Type::LongStr => !self.as_str_slice().is_empty(),
            Type::List => !self.as_list().borrow().list.is_empty(),
            Type::Set => !self.as_set().borrow().set.is_empty(),
            Type::Dict => !self.as_dict().borrow().dict.is_empty(),
            Type::Heap => !self.as_heap().borrow().heap.is_empty(),
            Type::Vector => !self.as_vector().borrow().vector.is_empty(),
            Type::Range => !self.as_range().borrow_const().is_empty(),
            Type::Enumerate => self.as_enumerate().borrow_const().to_bool(),
            _ => true,
        }
    }

    /// Unwraps the value as an `iterable`, or raises a type error.
    /// For all value types except `Heap`, this is a O(1) and lazy operation. It also requires no persistent borrows of mutable types that outlast the call to `as_iter()`.
    ///
    /// Guaranteed to return either a `Error` or `Iter`
    pub fn to_iter(self) -> ErrorResult<Iterable> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Ok(Iterable::Str(self.as_str_iter())),
            Type::List | Type::Set | Type::Dict | Type::Vector => Ok(Iterable::Collection(0, self)),

            // Heaps completely unbox themselves to be iterated over
            Type::Heap => Ok(Iterable::RawVector(0, self.as_heap().borrow().heap
                .iter()
                .cloned().map(|u| u.0)
                .collect::<Vec<ValuePtr>>())),

            Type::Range => Ok(self.as_range().borrow_const().to_iter()),
            Type::Enumerate => self.as_enumerate().borrow_const().to_iter(),

            _ => TypeErrorArgMustBeIterable(self.clone()).err(),
        }
    }

    /// Unwraps the value as an `iterable`, or if it is not, yields an iterable of the single element
    /// Note that this takes a `str` to be a non-iterable primitive type, unlike `is_iter()` and `as_iter()`
    pub fn as_iter_or_unit(self) -> Iterable {
        match self.ty() {
            Type::List | Type::Set | Type::Dict | Type::Vector => Iterable::Collection(0, self),

            // Heaps completely unbox themselves to be iterated over
            Type::Heap => Iterable::RawVector(0, self.as_heap().borrow().heap
                .iter()
                .cloned()
                .map(|u| u.0)
                .collect::<Vec<ValuePtr>>()),

            _ => Iterable::Unit(ValueOption::some(self)),
        }
    }

    /// Converts this `Value` to a `ValueAsIndex`, which is a index-able object, supported for `List`, `Vector`, and `Str`
    pub fn to_index(&self) -> ErrorResult<Indexable> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Ok(Indexable::Str(self.as_str_slice())),
            Type::List => Ok(Indexable::List(self.as_list().borrow_mut())),
            Type::Vector => Ok(Indexable::Vector(self.as_vector().borrow_mut())),
            _ => TypeErrorArgMustBeIndexable(self.clone()).err()
        }
    }

    /// Converts this value into a `(ValuePTr, ValuePtr)` if possible, supported for two-element `List` and `Vector`s
    pub fn to_pair(self) -> ErrorResult<(ValuePtr, ValuePtr)> {
        match match self.ty() {
            Type::List => self.as_list().borrow().list.iter().cloned().collect_tuple(),
            Type::Vector => self.as_vector().borrow().vector.iter().cloned().collect_tuple(),
            _ => None
        } {
            Some(it) => Ok(it),
            None => ValueErrorCannotCollectIntoDict(self.clone()).err()
        }
    }

    /// Returns `None` if this value is not function evaluable.
    /// Returns `Some(nargs)` if this value is a function with the given number of minimum arguments
    pub fn min_nargs(&self) -> Option<u32> {
        match self.ty() {
            Type::Function => Some(self.as_function().borrow_const().min_args()),
            Type::PartialFunction => Some(self.as_partial_function().borrow_const().min_nargs()),
            Type::NativeFunction => Some(self.as_native().min_nargs()),
            Type::PartialNativeFunction => Some(self.as_partial_native().borrow_const().min_nargs()),
            Type::Closure => Some(self.as_closure().borrow_func().min_args()),
            Type::StructType => Some(self.as_struct_type().borrow_const().num_fields()),
            Type::Slice => Some(1),
            _ => None,
        }
    }

    /// Returns the length of this `Value`. Equivalent to the native function `len`. Raises a type error if the value does not have a length.
    pub fn len(&self) -> ErrorResult<usize> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Ok(self.as_str_slice().chars().count()),
            Type::List => Ok(self.as_list().borrow().list.len()),
            Type::Set => Ok(self.as_set().borrow().set.len()),
            Type::Dict => Ok(self.as_dict().borrow().dict.len()),
            Type::Heap => Ok(self.as_heap().borrow().heap.len()),
            Type::Vector => Ok(self.as_vector().borrow().vector.len()),
            Type::Range => Ok(self.as_range().borrow_const().len()),
            Type::Enumerate => self.as_enumerate().borrow_const().len(),
            _ => TypeErrorArgMustBeIterable(self.clone()).err()
        }
    }

    pub fn get_field(&self, fields: &Fields, constants: &Vec<ValuePtr>, field_index: u32) -> ValueResult {
        match self.ty() {
            Type::Struct => {
                let it = self.as_struct().borrow_mut();
                let owner = it.get_owner();

                if let Some(offset) = fields.get_offset(owner.instance_type, field_index) {
                    return it.get_field(offset).ok();
                }

                // Try with the constructor type - here we are accessing instance methods, as opposed to fields
                match fields.get_offset(owner.constructor_type, field_index) {
                    Some(offset) if owner.is_instance_method(offset) => {
                        // Method was an instance method, so we can access it
                        // We need to bind the self instance to the method in a partial eval
                        owner.get_method(offset, constants)
                            .to_partial(vec![self.clone()])
                            .ok()
                    }
                    _ => err_field_not_found(it.get_constructor(), fields, field_index, true, true)
                }
            }
            Type::StructType => {
                let it = self.as_struct_type().borrow_const();
                if let Some(offset) = fields.get_offset(it.constructor_type, field_index) {
                    return it.get_method(offset, constants).ok();
                }

                // Try with a field of the instance type, in which case return the get field function (partially evaluated field access)
                // This will mimic an auto-generated accessor, but everything is already implemented!
                match fields.get_offset(it.instance_type, field_index) {
                    // This is similar to `(->x)` but it checks that the field `x` exists on the type `A->x` first.
                    Some(_) => ValuePtr::field(field_index).ok(),
                    _ => err_field_not_found(it.clone().to_value(), fields, field_index, true, true)
                }
            }
            _ => err_field_not_found(self.clone(), fields, field_index, false, true)
        }
    }

    pub fn set_field(&self, fields: &Fields, field_index: u32, value: ValuePtr) -> ValueResult {
        match self.ty() {
            Type::Struct => {
                let mut it = self.as_struct().borrow_mut();
                match fields.get_offset(it.get_owner().instance_type, field_index) {
                    Some(offset) => {
                        it.set_field(offset, value.clone());
                        value.ok()
                    }
                    None => err_field_not_found(it.get_constructor(), fields, field_index, true, false)
                }
            }
            // This is just for specialization of errors
            Type::StructType => err_field_not_found(self.clone(), fields, field_index, true, false),
            _ => err_field_not_found(self.clone(), fields, field_index, false, false)
        }
    }

    /// Returns if the value is iterable.
    pub fn is_iter(&self) -> bool {
        matches!(self.ty(), Type::ShortStr | Type::LongStr | Type::List | Type::Set | Type::Dict | Type::Heap | Type::Vector | Type::Range | Type::Enumerate)
    }

    /// Returns if the value is function-evaluable. Note that single-element lists are not considered functions here.
    pub fn is_evaluable(&self) -> bool {
        matches!(self.ty(), Type::Function | Type::PartialFunction | Type::NativeFunction | Type::PartialNativeFunction | Type::Closure | Type::StructType | Type::Slice)
    }

    pub fn as_iterable_mut(&mut self) -> &mut Iterable {
        debug_assert!(self.is_iterable());
        self.as_mut_ref()
    }

    pub fn ok(self) -> ValueResult {
        ValueResult::ok(self)
    }

    pub fn check_int(self) -> ValueResult {
        match self.is_int() {
            true => self.ok(),
            false => TypeErrorArgMustBeInt(self).err(),
        }
    }

    pub fn check_str(self) -> ValueResult {
        match self.is_str() {
            true => self.ok(),
            false => TypeErrorArgMustBeStr(self).err()
        }
    }

    pub fn check_list(self) -> ValueResult {
        match self.is_list() {
            true => self.ok(),
            false => TypeErrorArgMustBeList(self).err()
        }
    }

    pub fn check_dict(self) -> ValueResult {
        match self.is_dict() {
            true => self.ok(),
            false => TypeErrorArgMustBeDict(self).err()
        }
    }
}

#[cold]
fn err_field_not_found(value: ValuePtr, fields: &Fields, field_index: u32, repr: bool, access: bool) -> ValueResult {
    TypeErrorFieldNotPresentOnValue { value, field: fields.get_name(field_index), repr, access }.err()
}

/// A type used to prevent recursive `repr()` and `str()` calls.
struct RecursionGuard(Vec<ValueRef>);

impl RecursionGuard {
    pub fn new() -> RecursionGuard { RecursionGuard(Vec::new()) }

    /// Returns `true` if the value has been seen before, triggering an early exit
    pub fn enter(&mut self, value: &ValuePtr) -> bool {
        let boxed = value.as_value_ref();
        let ret = self.0.contains(&boxed);
        self.0.push(boxed);
        ret
    }

    pub fn leave(&mut self) {
        self.0.pop().unwrap(); // `.unwrap()` is safe as we should always call `enter()` before `leave()`
    }
}


macro_rules! impl_owned_value {
    ($ty:expr, $inner:ident, $as_T:ident, $as_T_ref:ident, $is_T:ident) => {
        impl OwnedValue for $inner {}

        impl IntoValue for $inner {
            fn to_value(self) -> ValuePtr {
                ptr::from_owned(Prefix::new($ty, self))
            }
        }

        impl ValuePtr {
            pub fn $as_T(self) -> Box<Prefix<$inner>> {
                debug_assert!(self.ty() == $ty);
                self.as_box()
            }
            pub fn $as_T_ref(&self) -> &$inner {
                debug_assert!(self.ty() == $ty);
                self.as_ref()
            }
            pub fn $is_T(&self) -> bool {
                self.ty() == $ty
            }
        }
    };
}

macro_rules! impl_shared_value {
    ($ty:expr, $inner:ident, $const_or_mut:ty, $as_T:ident, $is_T:ident) => {
        impl SharedValue for $inner {}
        impl $const_or_mut for $inner {}

        impl IntoValue for $inner {
            fn to_value(self) -> ValuePtr {
                ptr::from_shared(SharedPrefix::new($ty, self))
            }
        }

        impl ValuePtr {
            pub fn $as_T(&self) -> &SharedPrefix<$inner> {
                debug_assert!(self.ty() == $ty);
                self.as_shared_ref()
            }
            pub fn $is_T(&self) -> bool {
                self.ty() == $ty
            }
        }
    };
}

impl OwnedValue for () {}
impl SharedValue for () {}

// Cannot implement for `ComplexImpl` because we need a specialized to_value() which may convert to int
//impl_owned_value!(Type::Range, Range, as_range, as_range_ref, is_range);
//impl_owned_value!(Type::Enumerate, Enumerate, as_enumerate, as_enumerate_ref, is_enumerate);
//impl_owned_value!(Type::PartialFunction, PartialFunctionImpl, as_partial_function, as_partial_function_ref, is_partial_function);
//impl_owned_value!(Type::PartialNativeFunction, PartialNativeFunction, as_partial_native, as_partial_native_ref, is_partial_native);
//impl_owned_value!(Type::Slice, Slice, as_slice, as_slice_ref, is_slice);
impl_owned_value!(Type::Iter, Iterable, as_iterable, as_iterable_ref, is_iterable);
impl_owned_value!(Type::Error, RuntimeError, as_err, as_err_ref, is_err);

impl_shared_value!(Type::PartialFunction, PartialFunction, ConstValue, as_partial_function, is_partial_function);
impl_shared_value!(Type::PartialNativeFunction, PartialNativeFunction, ConstValue, as_partial_native, is_partial_native);
impl_shared_value!(Type::Slice, Slice, ConstValue, as_slice, is_slice);
impl_shared_value!(Type::Range, Range, ConstValue, as_range, is_range);
impl_shared_value!(Type::Enumerate, Enumerate, ConstValue, as_enumerate, is_enumerate);
impl_shared_value!(Type::List, ListImpl, MutValue, as_list, is_list);
impl_shared_value!(Type::Set, SetImpl, MutValue, as_set, is_set);
impl_shared_value!(Type::Dict, DictImpl, MutValue, as_dict, is_dict);
impl_shared_value!(Type::Heap, HeapImpl, MutValue, as_heap, is_heap);
impl_shared_value!(Type::Vector, VectorImpl, MutValue, as_vector, is_vector);
impl_shared_value!(Type::Function, Function, ConstValue, as_function, is_function);
impl_shared_value!(Type::Closure, Closure, MutValue, as_closure, is_closure);
impl_shared_value!(Type::Memoized, MemoizedImpl, MutValue, as_memoized, is_memoized);
impl_shared_value!(Type::Struct, StructImpl, MutValue, as_struct, is_struct);
impl_shared_value!(Type::StructType, StructTypeImpl, ConstValue, as_struct_type, is_struct_type);


/// A trait which is responsible for converting native types into a `Value`.
/// It is preferred to boxing these types directly using `Value::Foo()`, as most types have inner complexity that needs to be managed.
pub trait IntoValue {
    fn to_value(self) -> ValuePtr;
}

macro_rules! impl_into {
    ($ty:ty, $self:ident, $ret:expr) => {
        impl IntoValue for $ty {
            fn to_value($self) -> ValuePtr {
                $ret
            }
        }
    };
}

impl_into!(ValuePtr, self, self);
impl_into!(usize, self, ptr::from_usize(self));
impl_into!(i64, self, ptr::from_i64(self));
impl_into!(bool, self, ptr::from_bool(self));
impl_into!(char, self, ptr::from_char(self));
impl_into!(&str, self, ptr::from_str(self));
impl_into!(String, self, ptr::from_str(self.as_str()));
impl_into!(Cow<'_, str>, self, ptr::from_str(&self));
impl_into!(NativeFunction, self, ptr::from_native(self));
impl_into!(VecDeque<ValuePtr>, self, ListImpl { list: self }.to_value());
impl_into!(Vec<ValuePtr>, self, VectorImpl { vector: self }.to_value());
impl_into!((ValuePtr, ValuePtr), self, vec![self.0, self.1].to_value());
impl_into!(IndexSet<ValuePtr, FxBuildHasher>, self, SetImpl { set: self }.to_value());
impl_into!(IndexMap<ValuePtr, ValuePtr, FxBuildHasher>, self, DictImpl { dict: self, default: None }.to_value());
impl_into!(BinaryHeap<Reverse<ValuePtr>>, self, HeapImpl { heap: self }.to_value());


/// A trait which is responsible for wrapping conversions from a `Iterator<Item=Value>` into `IntoValue`, which then converts to a `ValuePtr`.
pub trait IntoIterableValue {
    fn to_list(self) -> ValuePtr;
    fn to_vector(self) -> ValuePtr;
    fn to_set(self) -> ValuePtr;
    fn to_heap(self) -> ValuePtr;
}

impl<I> IntoIterableValue for I where I : Iterator<Item=ValuePtr> {
    fn to_list(self) -> ValuePtr {
        self.collect::<VecDeque<ValuePtr>>().to_value()
    }

    fn to_vector(self) -> ValuePtr {
        self.collect::<Vec<ValuePtr>>().to_value()
    }

    fn to_set(self) -> ValuePtr {
        self.collect::<IndexSet<ValuePtr, FxBuildHasher>>().to_value()
    }

    fn to_heap(self) -> ValuePtr {
        self.map(Reverse).collect::<BinaryHeap<Reverse<ValuePtr>>>().to_value()
    }
}

/// A trait which is responsible for wrapping conversions from an `Iterator<Item=(ValuePtr, ValuePtr)>` into a `dict()`
pub trait IntoDictValue {
    fn to_dict(self) -> ValuePtr;
}

impl<I> IntoDictValue for I where I : Iterator<Item=(ValuePtr, ValuePtr)> {
    fn to_dict(self) -> ValuePtr {
        self.collect::<IndexMap<ValuePtr, ValuePtr, FxBuildHasher>>().to_value()
    }
}


#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ListImpl {
    pub list: VecDeque<ValuePtr>
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorImpl {
    pub vector: Vec<ValuePtr>
}

#[derive(Debug, PartialEq, Eq)]
pub struct SetImpl {
    pub set: IndexSet<ValuePtr, FxBuildHasher>
}

impl PartialOrd for SetImpl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SetImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        for (l, r) in self.set.iter().zip(other.set.iter()) {
            match l.cmp(r) {
                Ordering::Equal => {},
                ord => return ord,
            }
        }
        self.set.len().cmp(&other.set.len())
    }
}

/// `set()` is one object which can enter into a recursive hash situation:
/// ```cordy
/// let x = set()
/// x.push(x)
/// ```
///
/// This will take a mutable borrow on `x`, in the implementation of `push`, but then need to compute the hash of `x` to insert it into the set.
/// It can also apply to nested structures, as long as any recursive entry is formed.
///
/// The resolution is twofold:
///
/// - We don't implement `Hash` for `SetImpl`, instead implementing for `Mut<SetImpl>`, as before unboxing we need to do a borrow check
/// - If the borrow check fails, we set a global flag that we've entered this pathological case, which is checked by `ArrayStore` before yielding back to user code
///
/// Note this also applies to `DictImpl` / `dict()`, although only when used as a key.
impl Hash for SharedPrefix<SetImpl> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.try_borrow() {
            Some(it) => {
                for v in &it.set {
                    v.hash(state)
                }
            },
            None => FLAG_RECURSIVE_HASH.with(|cell| cell.set(true)),
        }
    }
}


// Support for `set` and `dict` recursive hash exceptions
thread_local! {
    static FLAG_RECURSIVE_HASH: Cell<bool> = Cell::new(false);
}

/// Returns `Err` if a recursive hash error occurred, `Ok` otherwise.
#[inline]
pub fn guard_recursive_hash<T, F : FnOnce() -> T>(f: F) -> Result<(), ()> {
    FLAG_RECURSIVE_HASH.with(|cell| cell.set(false));
    f();
    if FLAG_RECURSIVE_HASH.with(|cell| cell.get()) { Err(()) } else { Ok(()) }
}


#[derive(Debug, Clone)]
pub struct DictImpl {
    pub dict: IndexMap<ValuePtr, ValuePtr, FxBuildHasher>,
    pub default: Option<InvokeArg0>
}

impl Eq for DictImpl {}
impl PartialEq<Self> for DictImpl { fn eq(&self, other: &Self) -> bool { self.dict == other.dict } }
impl PartialOrd for DictImpl { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }

impl Ord for DictImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        for (l, r) in self.dict.keys().zip(other.dict.keys()) {
            match l.cmp(r) {
                Ordering::Equal => {},
                ord => return ord,
            }
        }
        self.dict.len().cmp(&other.dict.len())
    }
}

/// See justification for the unique `Hash` implementation on `SetImpl`
impl Hash for SharedPrefix<DictImpl> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.try_borrow() {
            Some(it) => {
                for v in &it.dict {
                    v.hash(state)
                }
            },
            None => FLAG_RECURSIVE_HASH.with(|cell| cell.set(true))
        }
    }
}


/// As `BinaryHeap` is missing `Eq`, `PartialEq`, and `Hash` implementations
/// We also wrap values in `Reverse` as we want to expose a min-heap by default
#[derive(Debug, Clone)]
pub struct HeapImpl {
    pub heap: BinaryHeap<Reverse<ValuePtr>>
}

impl Eq for HeapImpl {}
impl PartialEq<Self> for HeapImpl {
    fn eq(&self, other: &Self) -> bool {
        self.heap.len() == other.heap.len() && self.heap.iter().zip(other.heap.iter()).all(|(x, y)| x == y)
    }
}

// Heap ordering is, much like the heap itself, just based on the lowest (top) value of the heap.
// Empty heaps will return `None`, and this is implicit less than `Some`. So empty heap < non-empty heap
impl_partial_ord!(HeapImpl);
impl Ord for HeapImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        self.heap.peek().cmp(&other.heap.peek())
    }
}

impl Hash for HeapImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.heap {
            v.hash(state)
        }
    }
}

/// The implementation type for an instance of a struct.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StructImpl {
    /// `owner` must be a pointer to a `StructType` and will be dereferenced as `as_struct_type()` without checking.
    owner: ValuePtr,
    values: Vec<ValuePtr>,
}

impl StructImpl {

    /// Returns `true` if the current struct instance is of the provided constructor type.
    pub fn is_instance_of(&self, other: &StructTypeImpl) -> bool {
        self.get_owner().instance_type == other.instance_type
    }

    /// Returns a cloned (owned) copy of the constructor of this struct type.
    /// This is equivalent to the `typeof self` operator
    pub fn get_constructor(&self) -> ValuePtr {
        self.owner.clone()
    }

    /// Returns the owner (constructor type) as a reference to the `StructTypeImpl`
    pub fn get_owner(&self) -> &StructTypeImpl {
        self.owner.as_struct_type().borrow_const()
    }

    fn get_field(&self, field_offset: usize) -> ValuePtr {
        self.values[field_offset].clone()
    }

    fn set_field(&mut self, field_offset: usize, value: ValuePtr) {
        self.values[field_offset] = value;
    }
}

// Struct ordering is based on fields, like a vector
impl_partial_ord!(StructImpl);
impl Ord for StructImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        self.values.cmp(&other.values)
    }
}


/// The `Value` type for a struct constructor. It is a single instance, immutable object which only holds metadata about the struct itself.
#[derive(Debug, Clone, Eq)]
pub struct StructTypeImpl {
    name: String,
    fields: Vec<String>,
    /// Methods are references to constant indices, and so accessing a method involves going (type, field) -> offset -> constant -> `ValuePtr`
    methods: Vec<Method>,

    /// The `u32` type index of the instances created by this owner / constructor object.
    /// This is the type used to reference fields.
    instance_type: u32,
    /// The `u32` type index of this owner / constructor object.
    /// This is the type used to reference methods.
    constructor_type: u32,

    /// Flag that specifies this constructor object is a **module**. This affects a few properties:
    /// - Modules are not invokable as functions, and raise an error upon doing so
    /// - Modules canonical string representation is `module X` whereas structs return themselves as `struct X(... fields ...)`
    module: bool,
}

impl StructTypeImpl {
    pub fn new(name: String, fields: Vec<String>, methods: Vec<Method>, instance_type: u32, constructor_type: u32, module: bool) -> StructTypeImpl {
        StructTypeImpl { name, fields, methods, instance_type, constructor_type, module }
    }

    pub fn num_fields(&self) -> u32 {
        self.fields.len() as u32
    }

    pub fn is_module(&self) -> bool {
        self.module
    }

    /// Returns the method associated with this constructor / owner type.
    fn get_method(&self, method_offset: usize, constants: &Vec<ValuePtr>) -> ValuePtr {
        constants[self.methods[method_offset].function_id() as usize].clone()
    }

    /// Returns `true` if the method at the given offset is an instance / self method.
    fn is_instance_method(&self, method_offset: usize) -> bool {
        self.methods[method_offset].instance()
    }

    /// Returns the canonical representation of a struct/module in Cordy form, i.e. `Foo(a, b, c)`
    pub fn as_str(&self) -> String {
        match self.module {
            true => format!("module {}", self.name),
            false => format!("struct {}({})", self.name, self.fields.join(", "))
        }
    }
}

impl PartialEq for StructTypeImpl {
    fn eq(&self, other: &Self) -> bool {
        self.instance_type == other.instance_type
    }
}

impl Hash for StructTypeImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.instance_type.hash(state);
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Method(u32, bool);

impl Method {
    pub fn new(function_id: u32, instance: bool) -> Method {
        Method(function_id, instance)
    }

    /// The index into the constants array of the corresponding function.
    pub fn function_id(&self) -> u32 { self.0 }

    /// If `true`, this method is a instance method with a leading `self` parameter.
    pub fn instance(&self) -> bool { self.1 }
}


/// ### Iterator Type
///
/// Iterators are complex to model within the type system of Rust, with the restrictions imposed by Cordy:
///
/// - Rust `Iterator` methods access `Mut` values which enforce that their iterator only lives as long as the borrow from `Ref<'a, T>`. This is unusable as our iterators, i.e. in `for` loops, need to live on the stack.
/// - In native code, the borrow on the inner value must last for as long as the loop is ran, which means native functions like `map` essentially acquire a lock on their value, preventing mutation. For example the following code:
///
/// ```cordy
/// let a = [1, 2, 3]
/// a . map(fn(i) -> if len(a) < 4 then a.push(4) else nil) . print
/// ```
///
/// The outer `map` needs to break back into user code, but semantically, it cannot do so as it has a borrow on `a`.
/// We solve this problem by having a manner of 'stateless iterator'. An iterator is simply a unstable pointer into the structure, i.e. a `usize`, along with a *not-borrowed* reference to the value it is iterating over.
/// This can then be iterated over as a `Iterator<usize>`, and obtain the inner value *only by taking a borrow during `next()`*.
///
/// Almost all applications of this iterator will want to `.clone()` the returned values, i.e. because they need to be placed somewhere on the stack, so we function as a cloning iterator that provides ownership of `Value`s to the source.
///
/// Finally, this iterator is as lazy as it can be, and efficient as possible with the aforementioned restrictions. Most Cordy types support O(1) index-by-ordinal, and we use `IndexMap` and `IndexSet` for this exact purpose. The exceptions are `Heap` (which gets unboxed completely into a `Vector` before iterating), and `Str` (more on this later). This makes the following code:
///
/// ```cordy
/// for a in x { break }
/// ```
///
/// is O(n) where `x` is a  `Heap` type, as it is desugared into `for a in vector(x) { break }`, but O(1) for all other types, as expected.
///
/// #### String Iterators
///
/// Rust's `Chars` iterator has a lifetime - explicitly tied to the lifetime of the string. As it requires that while we iterate over the string, it is not modified.
/// In our case we can make those same requirements explicit - `String`s are immutable, and still immutable once they are handed over to an iterator.
///
/// To do this explicitly, we need a tiny bit of unsafe Rust, in particular, to hold a reference to a `String` and it's own `Chars` iterator in the same struct. Thus, we must meet the following requirement:
///
/// SAFETY: The `String` field of `Str` **cannot be modified**.
///
/// ---
///
/// This makes string iteration with early exiting, `O(1)` upfront, and reduces the constant factor of boxing each `char` into a `Value::Str`.
#[derive(Debug, Clone)]
pub enum Iterable {
    Str(IterStr),
    Unit(ValueOption),
    Collection(usize, ValuePtr),
    RawVector(usize, Vec<ValuePtr>),
    Range(i64, Range),
    Enumerate(usize, Box<Iterable>),
}

impl Iterable {

    /// Returns the original length of the iterable - not the amount of elements remaining.
    pub fn len(&self) -> usize {
        match &self {
            Iterable::Str(it) => it.count(),
            Iterable::Unit(it) => it.is_some() as usize,
            Iterable::Collection(_, it) => it.len().unwrap(), // `.unwrap()` is safe because we only construct this with collection types
            Iterable::RawVector(_, it) => it.len(),
            Iterable::Range(_, it) => it.len(),
            Iterable::Enumerate(_, it) => it.len(),
        }
    }

    pub fn reverse(self) -> IterableRev {
        let len: usize = self.len();
        match self {
            Iterable::Range(_, it) => IterableRev(it.reverse().to_iter()),
            Iterable::Collection(_, it) => IterableRev(Iterable::Collection(len, it)),
            Iterable::RawVector(_, it) => IterableRev(Iterable::RawVector(len, it)),
            Iterable::Enumerate(_, it) => IterableRev(Iterable::Enumerate(len, Box::new(it.reverse().0))),
            it => IterableRev(it)
        }
    }
}


/// A simple wrapper around reverse iteration
/// As most of our iterators are weirdly stateful, we can't support simple reverse iteration via `next_back()`
/// Instead, we wrap them in this type, by calling `Iterable.reverse()`. This then supports iteration in reverse.
#[derive(Debug)]
pub struct IterableRev(Iterable);

impl IterableRev {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Iterable {
    /// Returns the next element from a collection-like `ValuePtr` acting as an iterable
    fn get(ptr: &ValuePtr, index: usize) -> Option<ValuePtr> {
        match ptr.ty() {
            Type::List => ptr.as_list().borrow().list.get(index).cloned(),
            Type::Set => ptr.as_set().borrow().set.get_index(index).cloned(),
            Type::Dict => ptr.as_dict().borrow().dict.get_index(index).map(|(l, r)| (l.clone(), r.clone()).to_value()),
            Type::Vector => ptr.as_vector().borrow().vector.get(index).cloned(),
            _ => unreachable!(),
        }
    }
}


impl Iterator for Iterable {
    type Item = ValuePtr;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iterable::Str(it) => it.next().map(|u| u.to_value()),
            Iterable::Unit(it) => it.take().as_option(),
            Iterable::Collection(index, it) => {
                let ret = Iterable::get(it, *index);
                *index += 1;
                ret
            },
            Iterable::RawVector(index, it) => {
                let ret = it.get(*index).cloned();
                *index += 1;
                ret
            },
            Iterable::Range(it, range) => range.next(it),
            Iterable::Enumerate(index, it) => {
                let ret = (*it).next().map(|u| (index.to_value(), u).to_value());
                *index += 1;
                ret
            },
        }
    }
}

impl Iterator for IterableRev {
    type Item = ValuePtr;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            Iterable::Str(it) => it.next_back().map(|u| u.to_value()),
            Iterable::Unit(it) => it.take().as_option(),
            Iterable::Collection(index, it) => {
                if *index == 0 {
                    return None
                }
                *index -= 1;
                Iterable::get(it, *index)
            }
            Iterable::RawVector(index, it) => {
                if *index == 0 {
                    return None
                }
                *index -= 1;
                it.get(*index).cloned()
            }
            Iterable::Range(it, range) => range.next(it),
            Iterable::Enumerate(index, it) => {
                let ret = (*it).next().map(|u| (u, index.to_value()).to_value());
                *index += 1;
                ret
            },
        }
    }
}

impl FusedIterator for Iterable {}
impl FusedIterator for IterableRev {}


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct MemoizedImpl {
    pub func: ValuePtr,
    pub cache: HashMap<Vec<ValuePtr>, ValuePtr, FxBuildHasher>
}

impl Hash for MemoizedImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


pub enum Indexable<'a> {
    Str(&'a str),
    List(RefMut<'a, ListImpl>),
    Vector(RefMut<'a, VectorImpl>),
}

impl<'a> Indexable<'a> {

    pub fn len(&self) -> usize {
        match self {
            Indexable::Str(it) => it.len(),
            Indexable::List(it) => it.list.len(),
            Indexable::Vector(it) => it.vector.len(),
        }
    }

    /// Takes a convertable-to-int value, representing a bounded index in `[-len, len)`, and converts to a real index in `[0, len)`, or raises an error.
    pub fn check_index(&self, value: ValuePtr) -> ErrorResult<usize> {
        let index: i64 = value.check_int()?.as_int();
        let len: usize = self.len();
        let raw: usize = core::to_index(len as i64, index) as usize;
        if raw < len {
            Ok(raw)
        } else {
            ValueErrorIndexOutOfBounds(index, len).err()
        }
    }

    pub fn get_index(&self, index: usize) -> ValuePtr {
        match self {
            Indexable::Str(it) => it.chars().nth(index).unwrap().to_value(),
            Indexable::List(it) => it.list[index].clone(),
            Indexable::Vector(it) => it.vector[index].clone(),
        }
    }

    /// Setting indexes only works for immutable collections - so not strings
    pub fn set_index(&mut self, index: usize, value: ValuePtr) -> AnyResult {
        match self {
            Indexable::Str(it) => TypeErrorArgMustBeIndexable(it.to_value()).err(),
            Indexable::List(it) => {
                it.list[index] = value;
                Ok(())
            },
            Indexable::Vector(it) => {
                it.vector[index] = value;
                Ok(())
            },
        }
    }
}


#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LiteralType {
    List, Vector, Set, Dict
}


pub enum Literal {
    List(VecDeque<ValuePtr>),
    Vector(Vec<ValuePtr>),
    Set(IndexSet<ValuePtr, FxBuildHasher>),
    Dict(IndexMap<ValuePtr, ValuePtr, FxBuildHasher>),
}

impl Literal {
    pub fn new(op: LiteralType, size_hint: u32) -> Literal {
        match op {
            LiteralType::List => Literal::List(VecDeque::with_capacity(size_hint as usize)),
            LiteralType::Vector => Literal::Vector(Vec::with_capacity(size_hint as usize)),
            LiteralType::Set => Literal::Set(IndexSet::with_capacity_and_hasher(size_hint as usize, FxBuildHasher::default())),
            LiteralType::Dict => Literal::Dict(IndexMap::with_capacity_and_hasher(size_hint as usize, FxBuildHasher::default())),
        }
    }

    pub fn accumulate<I : Iterator<Item=ValuePtr>>(&mut self, mut iter: I) {
        match self {
            Literal::List(it) => for value in iter { it.push_back(value); },
            Literal::Vector(it) => for value in iter { it.push(value); },
            Literal::Set(it) => for value in iter { it.insert(value); }
            Literal::Dict(it) => while let Some(key) = iter.next() {
                let value = iter.next().unwrap();
                it.insert(key, value);
            },
        };
    }

    pub fn unroll<I : Iterator<Item=ValuePtr>>(&mut self, iter: I) -> AnyResult {
        match self {
            Literal::Dict(it) => for value in iter {
                let (key, value) = value.to_pair()?;
                it.insert(key, value);
            },
            _ => self.accumulate(iter),
        };
        Ok(())
    }
}

impl IntoValue for Literal {
    fn to_value(self) -> ValuePtr {
        match self {
            Literal::List(it) => it.to_value(),
            Literal::Vector(it) => it.to_value(),
            Literal::Set(it) => it.to_value(),
            Literal::Dict(it) => it.to_value(),
        }
    }
}


#[cfg(test)]
mod test {
    use crate::vm::{ValueOption, ValuePtr, ValueResult};
    use crate::vm::error::RuntimeError;
    use crate::vm::value::{IntoIterableValue, IntoValue};

    #[test]
    fn test_layout() {
        // Should be no size overhead, since both error and none states are already represented by `ValuePtr`
        assert_eq!(std::mem::size_of::<ValueResult>(), std::mem::size_of::<ValuePtr>());
        assert_eq!(std::mem::size_of::<ValueOption>(), std::mem::size_of::<ValuePtr>());
    }

    #[test]
    fn test_value_ref_is_ref_equality() {
        #[inline]
        fn list_of(vec: Vec<i64>) -> ValuePtr {
            vec.into_iter().map(|u| u.to_value()).to_list()
        }

        let ptr1 = list_of(vec![1, 2, 3]);
        let ptr2 = list_of(vec![123]);
        let ptr3 = list_of(vec![1, 2, 3]);
        let ptr4 = ptr1.clone(); // [1, 2, 3]

        assert_eq!(ptr1, ptr1);
        assert_ne!(ptr1, ptr2);
        assert_eq!(ptr1, ptr3); // Same value, different reference
        assert_eq!(ptr1, ptr4);

        assert_eq!(ptr1.as_value_ref(), ptr1.as_value_ref());
        assert_ne!(ptr1.as_value_ref(), ptr2.as_value_ref());
        assert_ne!(ptr1.as_value_ref(), ptr3.as_value_ref()); // Same value, different reference
        assert_eq!(ptr1.as_value_ref(), ptr4.as_value_ref());
    }

    #[test]
    fn test_value_option() {
        let some = ValueOption::some(ValuePtr::nil());
        let none = ValueOption::none();

        assert!(some.is_some());
        assert!(none.is_none());

        assert_eq!(some.as_option(), Some(ValuePtr::nil()));
        assert_eq!(none.as_option(), None);
    }

    #[test]
    #[should_panic]
    fn test_value_option_some_of_none() {
        let _ = ValueOption::some(ValuePtr::none());
    }

    #[test]
    fn test_value_result() {
        let ok = ValuePtr::nil().ok();
        let err = RuntimeError::RuntimeExit.err::<ValueResult>();

        assert!(ok.is_ok());
        assert!(err.is_err());

        assert_eq!(ok.as_result(), Ok(ValuePtr::nil()));
        assert_eq!(err.as_result(), RuntimeError::RuntimeExit.err())
    }

    #[test]
    #[should_panic]
    fn test_value_result_ok_of_err() {
        let _ = ValueResult::ok(RuntimeError::RuntimeExit.to_value());
    }
}
