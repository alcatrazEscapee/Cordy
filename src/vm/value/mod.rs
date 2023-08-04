use std::borrow::Borrow;
use std::cell::{Cell, Ref, RefMut};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::rc::Rc;
use std::str::Chars;
use fxhash::FxBuildHasher;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;

use crate::compiler::Fields;
use crate::core;
use crate::core::{InvokeArg0, NativeFunction, PartialArgument};
use crate::vm::error::RuntimeError;
use crate::vm::value::ptr::{Prefix, SharedPrefix};
use crate::util::impl_partial_ord;

pub use crate::vm::value::ptr::ValuePtr;

use Value::{*};
use RuntimeError::{*};


pub mod checks {
    use crate::vm::{RuntimeError, ValuePtr};
    use crate::vm::value::Type;

    macro_rules! check {
        ($result:expr) => {
            match $result {
                Ok(ptr) => ptr,
                Err(err) => return err
            }
        };
        ($ty:ident, $check:expr) => {{
            if checks::type_check(Type::$ty)(&$check) {
                return ValueResult::err(checks::type_error(Type::$ty)($check));
            } else {
                $check
            }
        }};
        ($check:expr, $err:expr) => {
            if !$check {
                return $err
            }
        };
    }

    pub(crate) use check;

    #[inline(always)]
    pub fn catch<T, E>(err: &mut Option<E>, f: impl FnOnce() -> Result<T, E>, default: T) -> T {
        match f() {
            Ok(e) => e,
            Err(e) => {
                *err = Some(e);
                default
            }
        }
    }

    #[inline(always)]
    pub fn join<T, E>(result: T, err: Option<E>) -> Result<T, E> {
        match err {
            Some(e) => Err(e),
            None => Ok(result)
        }
    }

    #[inline(always)]
    pub const fn type_check(ty: Type) -> fn(&ValuePtr) -> bool {
        match ty {
            Type::Int => ValuePtr::is_int,
            Type::Str => ValuePtr::is_str,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub const fn type_error(ty: Type) -> fn(ValuePtr) -> RuntimeError {
        match ty {
            Type::Int => RuntimeError::TypeErrorArgMustBeInt,
            Type::Str => RuntimeError::TypeErrorArgMustBeStr,
            _ => unreachable!(),
        }
    }
}


mod ptr;


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
    Str,
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
        std::mem::replace(&mut self, ValueOption::none())
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
pub struct ValueResult {
    ptr: ValuePtr
}

impl ValueResult {
    pub fn ok(ptr: ValuePtr) -> ValueResult {
        debug_assert!(!ptr.is_err()); // Should never have an error-type in `ok()`
        ValueResult { ptr }
    }

    #[cold]
    pub fn err(err: RuntimeError) -> ValueResult {
        ValueResult { ptr: err.to_value() }
    }

    pub fn new(ptr: ValuePtr) -> ValueResult {
        ValueResult { ptr }
    }

    /// Boxes this `ValueResult` into a traditional Rust `Result<T, E>`. The left will be guaranteed to contain a non-error type, and the right will be guaranteed to contain an error.
    #[inline(always)]
    pub fn as_result(self) -> Result<ValuePtr, Box<RuntimeError>> {
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

    #[cold]
    pub fn to_err(self) -> RuntimeError {
        *self.ptr.as_err()
    }
}

impl From<Result<ValuePtr, Box<RuntimeError>>> for ValueResult {
    fn from(value: Result<ValuePtr, Box<RuntimeError>>) -> Self {
        match value {
            Ok(ptr) => ValueResult::ok(ptr),
            Err(err) => ValueResult::err(*err)
        }
    }
}

/// Like `ValueOption` or `ValueResult`, this is a type indicating that the underlying `ValuePtr` has a specific form. In this case, it **must** be a `Type::Function` or `Type::Closure`, and provides methods to unbox the underlying `FunctionImpl`
pub struct ValueFunction {
    ptr: ValuePtr
}

impl ValueFunction {
    pub fn new(ptr: ValuePtr) -> ValueFunction {
        debug_assert!(ptr.is_function() || ptr.is_closure());
        ValueFunction { ptr }
    }

    /// Returns a reference to the function of this pointer.
    pub fn get(&self) -> &SharedPrefix<FunctionImpl> {
        self.ptr.get_function()
    }
}

/// Like `ValueOption` or `ValueResult`, but indicates via type saftey, that the underlying `ValuePtr` is a `StructType`
pub struct ValueStructType {
    ptr: ValuePtr
}

impl ValueStructType {
    pub fn new(ptr: ValuePtr) -> ValueStructType {
        debug_assert!(ptr.is_struct_type());
        ValueStructType { ptr }
    }

    pub fn get(&self) -> &StructTypeImpl {
        self.ptr.as_struct_type()
    }
}

pub type C64 = num_complex::Complex<i64>;


/// The runtime sum type used by the virtual machine
/// All `Value` type objects must be cloneable, and so mutable objects must be reference counted to ensure memory safety
/// todo: remove
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum Value {
    // Primitive (Immutable) Types
    Nil,
    Bool(bool),
    Int(i64),
    _Complex(Box<C64>),
    Str(Rc<String>),

    // Reference (Mutable) Types
    List(Rc<VecDeque<Value>>),
    Set(Rc<SetImpl>),
    Dict(Rc<DictImpl>),
    Heap(Rc<HeapImpl>), // `List` functions as both Array + Deque, but that makes it un-viable for a heap. So, we have a dedicated heap structure
    Vector(Rc<Vec<Value>>), // `Vector` is a fixed-size list (in theory, not in practice), that most operations will behave elementwise on

    /// A mutable instance of a struct - basically a named tuple.
    Struct(Rc<StructImpl>),
    /// The constructor / single type instance of a struct. This can be invoked to create new instances.
    StructType(Rc<StructTypeImpl>),

    // Iterator Types (Immutable)
    Range(Box<RangeImpl>),

    /// ### Enumerate Type
    ///
    /// This is the type used by the native function `enumerate(...)`. It does not have any additional functionality and is just a wrapper around an internal `Value`.
    ///
    /// Note that `enumerate()` object needs to be stateless, hence wrapping a `Value`, and not an `IteratorImpl`. When a `enumerate()` is iterated through, i.e. `is_iter()` is invoked on it, the internal value will be converted to the respective iterator at that time.
    Enumerate(Box<Value>),

    /// The type of a native slice literal. Immutable, and holds the slice values. Raises an error on construction if the arguments are not convertable to int.
    Slice(Box<SliceImpl>),

    /// Synthetic Iterator Type - Mutable, but not aliasable.
    /// This will never be user-code-accessible, as it will only be on the stack as a synthetic variable, or in native code.
    Iter(Box<Iterable>),

    /// Synthetic Memoized Function.
    Memoized(Box<MemoizedImpl>),

    /// A unique type for a partially evaluated `->` operator, i.e. `(->some_field)`
    /// The parameter is a field index. The only use of this type is as a function, where it shortcuts to a `GetField` operation.
    GetField(u32),

    // Functions
    Function(Rc<FunctionImpl>),
    PartialFunction(Box<PartialFunctionImpl>),
    NativeFunction(NativeFunction),
    PartialNativeFunction(NativeFunction, Box<PartialArgument>),
    Closure(Box<ClosureImpl>),
}


impl ValuePtr {

    // Constructors

    pub fn partial(func: ValuePtr, args: Vec<ValuePtr>) -> ValuePtr {
        PartialFunctionImpl { func: ValueFunction::new(func), args }.to_value()
    }

    pub fn closure(func: ValuePtr) -> ValuePtr {
        func.to_closure()
    }

    pub fn instance(type_impl: ValueStructType, values: Vec<ValuePtr>) -> ValuePtr {
        StructImpl {
            type_index: type_impl.get().type_index,
            type_impl,
            values,
        }.to_value()
    }

    pub fn memoized(func: ValuePtr) -> ValuePtr {
        MemoizedImpl {
            func,
            cache: HashMap::with_hasher(FxBuildHasher::default()),
        }.to_value()
    }

    /// Creates a new `Range()` value from a given set of integer parameters.
    /// Raises an error if `step == 0`
    ///
    /// Note: this implementation internally replaces all empty range values with the single `range(0, 0, 0)` instance. This means that `range(1, 2, -1) . str` will have to handle this case as it will not be representative.
    pub fn range(start: i64, stop: i64, step: i64) -> ValueResult {
        if step == 0 {
            ValueResult::err(ValueErrorStepCannotBeZero)
        } else if (stop > start && step > 0) || (stop < start && step < 0) { // Non-empty range
            ValueResult::ok(RangeImpl { start, stop, step }.to_value())
        } else { // Empty range
            ValueResult::ok(RangeImpl { start: 0, stop: 0, step: 0 }.to_value())
        }
    }

    pub fn slice(arg1: ValuePtr, arg2: ValuePtr, arg3: ValuePtr) -> ValueResult {
        checks::check!(arg1.is_int_like_or_nil(), TypeErrorArgMustBeInt(arg1));
        checks::check!(arg2.is_int_like_or_nil(), TypeErrorArgMustBeInt(arg2));
        checks::check!(arg3.is_int_like_or_nil(), TypeErrorArgMustBeInt(arg3));

        ValueResult::ok(SliceImpl { arg1, arg2, arg3 }.to_value())
    }

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn to_str(&self) -> String { self.safe_to_str(&mut RecursionGuard::new()) }

    fn safe_to_str(&self, rc: &mut RecursionGuard) -> String {
        match self.ty() {
            Type::Str => self.as_str().as_ref().to_owned(),
            Type::Function => self.as_function().as_ref().name.clone(),
            Type::PartialFunction => self.as_partial_function().as_ref().func.ptr.safe_to_str(rc),
            Type::NativeFunction => self.as_native().name().to_string(),
            Type::PartialNativeFunction => self.as_partial_native_ref().value.func.name().to_string(),
            Type::Closure => self.as_closure().as_ref().func.get().as_ref().name.to_owned(),
            _ => self.safe_to_repr_str(rc),
        }
    }

    /// Converts the `Value` to a representative `String. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn to_repr_str(&self) -> String { self.safe_to_repr_str(&mut RecursionGuard::new()) }

    fn safe_to_repr_str(&self, rc: &mut RecursionGuard) -> String {
        macro_rules! recursive_guard {
            ($default:expr, $recursive:expr) => {{
                let ret = if rc.enter(self) { $default } else { $recursive };
                rc.leave();
                ret
            }};
        }

        match self.ty() {
            Type::Nil => String::from("nil"),
            Type::Bool => self.as_bool().to_string(),
            Type::Int => self.as_int().to_string(),
            Type::Complex => {
                let c = &self.as_complex_ref().inner;
                if c.re == 0 {
                    format!("{}i", c.im)
                } else {
                    format!("{} + {}i", c.re, c.im)
                }
            },
            Type::Str => {
                let escaped = format!("{:?}", self.as_str().as_ref());
                format!("'{}'", &escaped[1..escaped.len() - 1])
            },

            Type::List => recursive_guard!(
                String::from("[...]"),
                format!("[{}]", self.as_list().borrow().list.iter()
                    .map(|t| t.safe_to_repr_str(rc))
                    .join(", "))
            ),
            Type::Set => recursive_guard!(
                String::from("{...}"),
                format!("{{{}}}", self.as_set().borrow().set.iter()
                    .map(|t| t.safe_to_repr_str(rc))
                    .join(", "))
            ),
            Type::Dict => recursive_guard!(
                String::from("{...}"),
                format!("{{{}}}", self.as_dict().borrow().dict.iter()
                    .map(|(k, v)| format!("{}: {}", k.safe_to_repr_str(rc), v.safe_to_repr_str(rc)))
                    .join(", "))
            ),
            Type::Heap => recursive_guard!(
                String::from("[...]"),
                format!("[{}]", self.as_heap().borrow().heap.iter()
                    .map(|t| t.0.safe_to_repr_str(rc))
                    .join(", "))
            ),
            Type::Vector => recursive_guard!(
                String::from("(...)"),
                format!("({})", self.as_vector().borrow().vector.iter()
                    .map(|t| t.safe_to_repr_str(rc))
                    .join(", "))
            ),

            Type::Struct => {
                let it = self.as_struct().borrow();
                recursive_guard!(
                    format!("{}(...)", it.type_impl.get().name),
                    format!("{}({})", it.type_impl.get().name.as_str(), it.values.iter()
                        .zip(it.type_impl.get().field_names.iter())
                        .map(|(v, k)| format!("{}={}", k, v.safe_to_repr_str(rc)))
                        .join(", "))
                )
            },
            Type::StructType => {
                let it = self.as_struct_type().as_ref();
                format!("struct {}({})", it.name.clone(), it.field_names.join(", "))
            },

            Type::Range => {
                let r = self.as_range_ref();
                if r.step == 0 {
                    String::from("range(empty)")
                } else {
                    format!("range({}, {}, {})", r.start, r.stop, r.step)
                }
            },
            Type::Enumerate => format!("enumerate({})", self.as_enumerate_ref().inner.safe_to_repr_str(rc)),
            Type::Slice => {
                #[inline]
                fn to_str(i: ValuePtr) -> String {
                    if i.is_nil() {
                        String::new()
                    } else {
                        i.as_int_like().to_string()
                    }
                }

                let it = self.as_slice_ref();
                match it.arg3.is_nil() {
                    false => format!("[{}:{}:{}]", to_str(it.arg1), to_str(it.arg2), it.arg3.as_int_like()),
                    true => format!("[{}:{}]", to_str(it.arg1), to_str(it.arg2)),
                }
            },

            Type::Iter => String::from("<synthetic> iterator"),
            Type::Memoized => format!("@memoize {}", self.as_memoized().borrow().func.safe_to_repr_str(rc)),

            Type::GetField => String::from("(->)"),

            Type::Function => self.as_function().as_ref().repr(),
            Type::PartialFunction => self.as_partial_function().as_ref().func.safe_to_repr_str(rc),
            Type::NativeFunction => self.as_native().repr(),
            Type::PartialNativeFunction => self.as_partial_native_ref().func.repr(),
            Type::Closure => self.as_closure().as_ref().func.get().as_ref().repr(),

            Type::Error | Type::None | Type::Never => unreachable!(),
        }
    }

    /// Returns the inner user function, either from a `Function` or `Closure` type
    pub fn get_function(&self) -> &SharedPrefix<FunctionImpl> {
        match self.is_function() {
            true => self.as_function(),
            false => self.as_closure().as_ref().func.get(),
        }
    }

    /// Converts a `Function` into a new empty `Closure`
    pub fn to_closure(self) -> ValuePtr {
        debug_assert!(self.is_function());
        ClosureImpl { func: ValueFunction::new(self), environment: Vec::new() }.to_value()
    }

    /// Represents the type of this `Value`. This is used for runtime error messages,
    pub fn as_type_str(&self) -> String {
        String::from(match self.ty() {
            Type::Nil => "nil",
            Type::Bool => "bool",
            Type::Int => "int",
            Type::Complex => "complex",
            Type::Str => "str",
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
            Type::Str => !self.as_str().as_ref().is_empty(),
            Type::List => !self.as_list().borrow().list.is_empty(),
            Type::Set => !self.as_set().borrow().set.is_empty(),
            Type::Dict => !self.as_dict().borrow().dict.is_empty(),
            Type::Heap => !self.as_heap().borrow().heap.is_empty(),
            Type::Vector => !self.as_vector().borrow().vector.is_empty(),
            Type::Range => !self.as_range_ref().is_empty(),
            Type::Enumerate => self.as_enumerate_ref().inner.to_bool(),
            Type::Iter | Type::Memoized => panic!("{:?} is a synthetic type should not have as_bool() invoked on it", self),
            _ => true,
        }
    }

    /// Unwraps the value as an `iterable`, or raises a type error.
    /// For all value types except `Heap`, this is a O(1) and lazy operation. It also requires no persistent borrows of mutable types that outlast the call to `as_iter()`.
    ///
    /// Guaranteed to return either a `Error` or `Iter`
    pub fn as_iter(self) -> Result<Iterable, Box<RuntimeError>> {
        match self.ty() {
            Type::Str => {
                let string: String = self.as_str().as_ref().clone();
                let chars: Chars<'static> = unsafe {
                    std::mem::transmute(string.chars())
                };
                ValueResult::ok(Iterable::Str(string, chars).to_value())
            },
            Type::List | Type::Set | Type::Dict | Type::Vector => ValueResult::ok(Iterable::Collection(0, self).to_value()),

            // Heaps completely unbox themselves to be iterated over
            Type::Heap => ValueResult::ok(Iterable::RawVector(0, self.as_heap().borrow().heap
                .iter()
                .cloned().map(|u| u.0)
                .collect::<Vec<ValuePtr>>())
                .to_value()),

            Type::Range => {
                let it = self.as_range_ref();
                ValueResult::ok(Iterable::Range(it.start, (**it).clone()).to_value())
            },
            Type::Enumerate => Iterable::Enumerate(0, Box::new(self.as_enumerate_ref().inner.as_iter()?)),

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

            Type::Range => {
                let it = self.as_range();
                Iterable::Range(it.value.start, it.value)
            },
            Type::Enumerate => Iterable::Enumerate(0, Box::new(self.as_enumerate().value.inner.as_iter_or_unit())),

            _ => Iterable::Unit(ValueOption::some(self)),
        }
    }

    /// Converts this `Value` to a `ValueAsIndex`, which is a index-able object, supported for `List`, `Vector`, and `Str`
    pub fn as_index(&self) -> Result<Indexable, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(Indexable::Str(it)),
            List(it) => Ok(Indexable::List(it.unbox_mut())),
            Vector(it) => Ok(Indexable::Vector(it.unbox_mut())),
            _ => TypeErrorArgMustBeIndexable(self.clone()).err()
        }
    }

    /// Converts this `Value` to a `ValueAsSlice`, which is a builder for slice-like structures, supported for `List` and `Str`
    pub fn to_slice(&self) -> Result<Sliceable, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(Sliceable::Str(it, String::new())),
            List(it) => Ok(Sliceable::List(it.unbox(), VecDeque::new())),
            Vector(it) => Ok(Sliceable::Vector(it.unbox(), Vec::new())),
            _ => TypeErrorArgMustBeSliceable(self.clone()).err()
        }
    }

    /// Converts this `Value` into a `(Value, Value)` if possible, supported for two-element `List` and `Vector`s
    pub fn as_pair(&self) -> Result<(Value, Value), Box<RuntimeError>> {
        match match self {
            List(it) => it.unbox().iter().cloned().collect_tuple(),
            Vector(it) => it.unbox().iter().cloned().collect_tuple(),
            _ => None
        } {
            Some(it) => Ok(it),
            None => ValueErrorCannotCollectIntoDict(self.clone()).err()
        }
    }

    /// Returns `None` if this value is not function evaluable.
    /// Returns `Some(nargs)` if this value is a function with the given number of minimum arguments
    pub fn min_nargs(&self) -> Option<u32> {
        match self {
            Function(it) => Some(it.min_args()),
            PartialFunction(it) => Some(it.func.as_function().min_args() - it.args.len() as u32),
            NativeFunction(it) => Some(it.min_nargs()),
            PartialNativeFunction(_, it) => Some(it.min_nargs()),
            Closure(it) => Some(it.func.min_args()),
            StructType(it) => Some(it.field_names.len() as u32),
            Slice(_) => Some(1),
            _ => None,
        }
    }

    /// Returns the length of this `Value`. Equivalent to the native function `len`. Raises a type error if the value does not have a lenth.
    pub fn len(&self) -> Result<usize, Box<RuntimeError>> {
        match self.ty() {
            Type::Str => Ok(self.as_str().as_ref().chars().count()),
            Type::List => Ok(self.as_list().borrow().list.len()),
            Type::Set => Ok(self.as_set().borrow().set.len()),
            Type::Dict => Ok(self.as_dict().borrow().dict.len()),
            Type::Heap => Ok(self.as_heap().borrow().heap.len()),
            Type::Vector => Ok(self.as_vector().borrow().vector.len()),
            Type::Range => Ok(self.as_range_ref().len()),
            Type::Enumerate => self.as_enumerate_ref().len(),
            _ => TypeErrorArgMustBeIterable(self.clone()).err()
        }
    }

    pub fn get_field(self, fields: &Fields, field_index: u32) -> ValueResult {
        match self.ty() {
            Type::Struct => {
                let mut it = self.as_struct().borrow_mut();
                match fields.get_field_offset(it.type_index, field_index) {
                    Some(field_offset) => ValueResult::ok(it.get_field(field_offset)),
                    None => ValueResult::err(TypeErrorFieldNotPresentOnValue(it.type_impl.ptr.clone(), fields.get_field_name(field_index), true))
                }
            },
            _ => ValueResult::err(TypeErrorFieldNotPresentOnValue(self, fields.get_field_name(field_index), false))
        }
    }

    pub fn set_field(self, fields: &Fields, field_index: u32, value: ValuePtr) -> ValueResult {
        match self.ty() {
            Type::Struct => {
                let mut it = self.as_struct().borrow_mut();
                match fields.get_field_offset(it.type_index, field_index) {
                    Some(field_offset) => {
                        it.set_field(field_offset, value.clone());
                        ValueResult::ok(value)
                    },
                    None => ValueResult::err(TypeErrorFieldNotPresentOnValue(it.type_impl.ptr.clone(), fields.get_field_name(field_index), true))
                }
            },
            _ => ValueResult::err(TypeErrorFieldNotPresentOnValue(self, fields.get_field_name(field_index), false))
        }
    }

    /// Returns if the value is iterable.
    pub fn is_iter(&self) -> bool {
        matches!(self.ty(), Type::Str | Type::List | Type::Set | Type::Dict | Type::Heap | Type::Vector | Type::Range | Type::Enumerate)
    }

    /// Returns if the value is function-evaluable. Note that single-element lists are not considered functions here.
    pub fn is_evaluable(&self) -> bool {
        matches!(self.ty(), Type::Function | Type::PartialFunction | Type::Native | Type::PartialNativeFunction | Type::Closure | Type::StructType | Type::Slice)
    }
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
                ValuePtr::owned(Prefix::prefix($ty, self))
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
                ValuePtr::shared(SharedPrefix::prefix($ty, self))
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
impl_owned_value!(Type::Range, RangeImpl, as_range, as_range_ref, is_range);
impl_owned_value!(Type::Enumerate, EnumerateImpl, as_enumerate, as_enumerate_ref, is_enumerate);
impl_owned_value!(Type::PartialNativeFunction, PartialNativeFunctionImpl, as_partial_native, as_partial_native_ref, is_partial_native);
impl_owned_value!(Type::Slice, SliceImpl, as_slice, as_slice_ref, is_slice);
impl_owned_value!(Type::Iter, Iterable, as_iterable, as_iterable_ref, is_iterable);

impl_shared_value!(Type::Str, String, ConstValue, as_str, is_str);
impl_shared_value!(Type::List, ListImpl, MutValue, as_list, is_list);
impl_shared_value!(Type::Set, SetImpl, MutValue, as_set, is_set);
impl_shared_value!(Type::Dict, DictImpl, MutValue, as_dict, is_dict);
impl_shared_value!(Type::Heap, HeapImpl, MutValue, as_heap, is_heap);
impl_shared_value!(Type::Vector, VectorImpl, MutValue, as_vector, is_vector);
impl_shared_value!(Type::Function, FunctionImpl, ConstValue, as_function, is_function);
impl_shared_value!(Type::PartialFunction, PartialFunctionImpl, ConstValue, as_partial_function, is_partial_function);
impl_shared_value!(Type::Closure, ClosureImpl, ConstValue, as_closure, is_closure);
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
impl_into!(usize, self, self as i64);
impl_into!(i64, self, ValuePtr::of_int(self));
impl_into!(num_complex::Complex<i64>, self, ComplexImpl { inner: self }.to_value());
impl_into!(ComplexImpl, self, if self.inner.im == 0 {
    ValuePtr::of_int(self.inner.re)
} else {
    ValuePtr::owned(Prefix::prefix(Type::Complex, self))
});
impl_into!(bool, self, ValuePtr::of_bool(self));
impl_into!(char, self, String::from(self));
impl_into!(&str, self, String::from(self));
impl_into!(NativeFunction, self, ValuePtr::of_native(self));
impl_into!(VecDeque<ValuePtr>, self, ListImpl { list: self }.to_value());
impl_into!(Vec<ValuePtr>, self, VectorImpl { vector: self }.to_value());
impl_into!((ValuePtr, ValuePtr), self, vec![self.0, self.1].to_value());
impl_into!(IndexSet<ValuePtr, FxBuildHasher>, self, SetImpl { set: self }.to_value());
impl_into!(IndexMap<ValuePtr, ValuePtr, FxBuildHasher>, self, DictImpl { dict: self, default: None }.to_value());
impl_into!(BinaryHeap<Reverse<ValuePtr>>, self, HeapImpl { heap: self }.to_value());
impl_into!(RuntimeError, self, ValuePtr::error(self));
impl_into!(Sliceable<'_>, self, match self {
    Sliceable::Str(_, it) => it.to_value(),
    Sliceable::List(_, it) => it.to_value(),
    Sliceable::Vector(_, it) => it.to_value(),
});


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


#[derive(Debug, Eq, PartialEq, Hash)]
pub struct ComplexImpl {
    pub inner: num_complex::Complex<i64>,
}

impl OwnedValue for ComplexImpl {}

impl ValuePtr {
    pub fn as_complex(self) -> Box<Prefix<ComplexImpl>> {
        debug_assert!(self.ty() == Type::Complex);
        self.as_box()
    }

    pub fn as_complex_ref(&self) -> &ComplexImpl {
        debug_assert!(self.ty() == Type::Complex);
        self.as_ref()
    }

    pub fn is_complex(&self) -> bool {
        self.ty() == Type::Complex
    }
}

impl_partial_ord!(ComplexImpl);
impl Ord for ComplexImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.re.cmp(&other.inner.re)
            .then(self.inner.im.cmp(&other.inner.im))
    }
}



#[derive(Eq, PartialEq, Debug, Clone)]
pub struct FunctionImpl {
    pub head: usize, // Pointer to the first opcode of the function's execution
    pub tail: usize, // Pointer to the final `Return` opcode.

    name: String, // The name of the function, useful to show in stack traces
    args: Vec<String>, // Names of the arguments
    default_args: Vec<usize>, // Jump offsets for each default argument
    var_arg: bool, // If the last argument in this function is variadic
}

impl FunctionImpl {
    pub fn new(head: usize, tail: usize, name: String, args: Vec<String>, default_args: Vec<usize>, var_arg: bool) -> FunctionImpl {
        FunctionImpl { head, tail, name, args, default_args, var_arg, }
    }

    /// The minimum number of required arguments, inclusive.
    pub fn min_args(&self) -> u32 {
        (self.args.len() - self.default_args.len()) as u32
    }

    /// The maximum number of required arguments, inclusive.
    pub fn max_args(&self) -> u32 {
        self.args.len() as u32
    }

    pub fn in_range(&self, nargs: u32) -> bool {
        self.min_args() <= nargs && (self.var_arg || nargs <= self.max_args())
    }

    /// Returns the number of variadic arguments that need to be collected, before invoking the function, if needed.
    pub fn num_var_args(&self, nargs: u32) -> Option<u32> {
        if self.var_arg && nargs >= self.max_args() {
            Some(nargs + 1 - self.max_args())
        } else {
            None
        }
    }

    /// Returns the jump offset of the function
    /// For typical functions, this is just the `head`, however when default arguments are present, or not, this is offset by the default argument offsets.
    /// The `nargs` must be legal (between `[min_args(), max_args()]`
    pub fn jump_offset(&self, nargs: u32) -> usize {
        self.head + if nargs == self.min_args() {
            0
        } else if self.var_arg && nargs >= self.max_args() {
            *self.default_args.last().unwrap()
        } else {
            self.default_args[(nargs - self.min_args() - 1) as usize]
        }
    }

    pub fn repr(&self) -> String {
        format!("fn {}({})", self.name, self.args.join(", "))
    }
}

impl Hash for FunctionImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct PartialFunctionImpl {
    pub func: ValueFunction,
    pub args: Vec<ValuePtr>,
}

impl Hash for PartialFunctionImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct PartialNativeFunctionImpl {
    pub func: NativeFunction,
    pub partial: PartialArgument,
}


/// A closure is a combination of a function, and a set of `environment` variables.
/// These variables are references either to locals in the enclosing function, or captured variables from the enclosing function itself.
///
/// A closure also provides *interior mutability* for it's captured upvalues, allowing them to be modified even from the surrounding function.
/// Unlike with other mutable `ValuePtr` types, this does so using `Rc<Cell<ValuePtr>>`. The reason being:
///
/// - A `Mut` cannot be unboxed without creating a borrow, which introduces lifetime restrictions. It also cannot be mutably unboxed without creating a write lock. With a closure, we need to be free to unbox the environment straight onto the stack, so this is off the table.
/// - The closure's inner value can be thought of as immutable. As `ValuePtr` is immutable, and clone-able, so can the contents of `Cell`. We can then unbox this completely - take a reference to the `Rc`, and call `get()` to unbox the current value of the cell, onto the stack.
///
/// This has one problem, which is we can't call `.get()` unless the cell is `Copy`, which `ValuePtr` isn't, and can't be, because `Mut` can't be copy due to the presence of `Rc`... Fortunately, this is just an API limitation, and we can unbox the cell in other ways.
///
/// Note we cannot derive most functions, as that also requires `Cell<ValuePtr>` to be `Copy`, due to convoluted trait requirements.
#[derive(Clone)]
pub struct ClosureImpl {
    func: ValueFunction,
    environment: Vec<Rc<Cell<UpValue>>>,
}

impl ClosureImpl {
    pub fn push(&mut self, value: Rc<Cell<UpValue>>) {
        self.environment.push(value);
    }

    /// Returns the current environment value for the upvalue index `index.
    pub fn get(&self, index: usize) -> Rc<Cell<UpValue>> {
        self.environment[index].clone()
    }
}

#[derive(Clone)]
pub enum UpValue {
    Open(usize),
    Closed(Value)
}

/// Implement `Default` to have access to `.take()`
impl Default for UpValue {
    fn default() -> Self {
        UpValue::Open(0)
    }
}

impl PartialEq for ClosureImpl {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}

impl Eq for ClosureImpl {}

impl Debug for ClosureImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.func, f)
    }
}

impl Hash for ClosureImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
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
        match self.try_unbox() {
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
        match self.try_unbox() {
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
    pub heap: BinaryHeap<Reverse<Value>>
}

impl PartialEq<Self> for HeapImpl {
    fn eq(&self, other: &Self) -> bool {
        self.heap.len() == other.heap.len() && self.heap.iter().zip(other.heap.iter()).all(|(x, y)| x == y)
    }
}

impl Eq for HeapImpl {}

impl Hash for HeapImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.heap {
            v.hash(state)
        }
    }
}

/// The `Value` type for a instance of a struct.
/// It holds the `type_index` for easy access, but also the `type_impl`, in order to access fields such as the struct name or field names, when converting to a string.
#[derive(Debug, Clone)]
pub struct StructImpl {
    pub type_index: u32,
    pub type_impl: ValueStructType,
    values: Vec<ValuePtr>,
}

impl StructImpl {
    fn get_field(&mut self, field_offset: usize) -> ValuePtr {
        self.values[field_offset].clone()
    }

    fn set_field(&mut self, field_offset: usize, value: ValuePtr) {
        self.values[field_offset] = value;
    }
}

impl PartialEq<Self> for StructImpl {
    fn eq(&self, other: &Self) -> bool {
        self.type_index == other.type_index && self.values == other.values
    }
}

impl Eq for StructImpl {}

impl Hash for StructImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_index.hash(state);
        self.values.hash(state);
    }
}

/// The `Value` type for a struct constructor. It is a single instance, immutable object which only holds metadata about the struct itself.
#[derive(Debug, Clone, Eq)]
pub struct StructTypeImpl {
    pub name: String,
    pub field_names: Vec<String>,

    pub type_index: u32,
}

impl StructTypeImpl {
    pub fn new(name: String, field_names: Vec<String>, type_index: u32) -> StructTypeImpl {
        StructTypeImpl { name, field_names, type_index }
    }

    pub fn as_str(&self) -> String {
        format!("struct {}({})", self.name, self.field_names.join(", "))
    }
}

impl PartialEq for StructTypeImpl {
    fn eq(&self, other: &Self) -> bool {
        self.type_index == other.type_index
    }
}

impl Hash for StructTypeImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_index.hash(state);
    }
}


/// ### Range Type
///
/// This is the internal lazy type used by the native function `range(...)`. For non-empty ranges, `step` must be non-zero.
/// For an empty range, this will store the `step` as `0` - in this case the `start` and `stop` values should be ignored
/// Note that depending on the relation of `start`, `stop` and the sign of `step`, this may represent an empty range.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct RangeImpl {
    start: i64,
    stop: i64,
    step: i64,
}

impl RangeImpl {
    /// Used by `operator in`, to check if a value is in this range.
    pub fn contains(&self, value: i64) -> bool {
        match self.step.cmp(&0) {
            Ordering::Equal => false,
            Ordering::Greater => value >= self.start && value < self.stop && (value - self.start) % self.step == 0,
            Ordering::Less => value <= self.start && value > self.stop && (self.start - value) % self.step == 0
        }
    }

    /// Reverses the range, so that iteration advances from the end to the start
    /// Note this is not as simple as just swapping `start` and `stop`, due to non-unit step sizes.
    pub fn reverse(self) -> RangeImpl {
        match self.step.cmp(&0) {
            Ordering::Equal => self,
            Ordering::Greater => RangeImpl { start: self.start + self.len() as i64 * self.step, stop: self.start + 1, step: -self.step },
            Ordering::Less => RangeImpl { start: self.start + self.len() as i64 * self.step, stop: self.start - 1, step: -self.step }
        }
    }

    /// Advances the `Range`, based on the external `current` value.
    /// The `current` value is the one that will be returned, and internally advanced to the next value.
    fn next(&self, current: &mut i64) -> Option<Value> {
        if *current == self.stop || self.step == 0 {
            None
        } else if self.step > 0 {
            let ret = *current;
            *current += self.step;
            if *current > self.stop {
                *current = self.stop;
            }
            Some(Int(ret))
        } else {
            let ret = *current;
            *current += self.step;
            if *current < self.stop {
                *current = self.stop;
            }
            Some(Int(ret))
        }
    }

    fn len(&self) -> usize {
        // Since this type ensures that the range is non-empty, we can do simple checked arithmetic
        if self.step == 0 { 0 } else { (self.start.abs_diff(self.stop) / self.step.unsigned_abs()) as usize }
    }

    fn is_empty(&self) -> bool {
        self.step == 0
    }
}


#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct EnumerateImpl {
    pub inner: ValuePtr
}


/// All arguments must either be `nil` (which will be treated as `None`), or an int-like type.
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Clone)]
pub struct SliceImpl {
    arg1: ValuePtr,
    arg2: ValuePtr,
    arg3: ValuePtr,
}

impl SliceImpl {
    pub fn apply(self, arg: ValuePtr) -> ValueResult {
        core::literal_slice(arg, self.arg1, self.arg2, self.arg3)
    }
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
    Str(String, Chars<'static>),
    Unit(ValueOption),
    Collection(usize, ValuePtr),
    RawVector(usize, Vec<ValuePtr>),
    Range(i64, RangeImpl),
    Enumerate(usize, Box<Iterable>),
}

impl Iterable {

    /// Returns the original length of the iterable - not the amount of elements remaining.
    pub fn len(&self) -> usize {
        match &self {
            Iterable::Str(it, _) => it.chars().count(),
            Iterable::Unit(it) => it.is_some() as usize,
            Iterable::Collection(_, it) => it.len(),
            Iterable::RawVector(_, it) => it.len(),
            Iterable::Range(_, it) => it.len(),
            Iterable::Enumerate(_, it) => it.len(),
        }
    }

    pub fn reverse(self) -> IterableRev {
        match self {
            Iterable::Range(_, it) => {
                let range = it.reverse();
                IterableRev(Iterable::Range(range.start, range))
            },
            Iterable::Collection(_, it) => IterableRev(Iterable::Collection(self.len(), it)),
            Iterable::RawVector(_, it) => IterableRev(Iterable::RawVector(self.len(), it)),
            Iterable::Enumerate(_, it) => IterableRev(Iterable::Enumerate(0, Box::new(it.reverse().0))),
            it => IterableRev(it)
        }
    }
}


/// A simple wrapper around reverse iteration
/// As most of our iterators are weirdly stateful, we can't support simple reverse iteration via `next_back()`
/// Instead, we wrap them in this type, by calling `Iterable.reverse()`. This then supports iteration in reverse.
pub struct IterableRev(Iterable);

impl IterableRev {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Iterable {
    /// Returns the next element from a collection-like `ValuePtr` acting as an iterable
    fn get(ptr: & mut ValuePtr, index: usize) -> Option<ValuePtr> {
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
            Iterable::Str(_, chars) => chars.next().map(|u| u.to_value()),
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
                let ret = (*it).next().map(|u| (u, index.to_value()).to_value());
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
            Iterable::Str(_, chars) => chars.next_back().map(|u| u.to_value()),
            Iterable::Unit(it) => it.take(),
            Iterable::Collection(index, it) => {
                if index == 0 {
                    return None
                }
                let ret = Iterable::get(it, *index);
                *index -= 1;
                ret
            }
            Iterable::RawVector(index, it) => {
                if index == 0 {
                    return None
                }
                let ret = it.get(*index);
                *index -= 1;
                ret
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
    Str(&'a Rc<String>),
    List(RefMut<'a, VecDeque<ValuePtr>>),
    Vector(RefMut<'a, Vec<ValuePtr>>),
}

impl<'a> Indexable<'a> {

    pub fn len(&self) -> usize {
        match self {
            Indexable::Str(it) => it.len(),
            Indexable::List(it) => it.len(),
            Indexable::Vector(it) => it.len(),
        }
    }

    /// Takes a convertable-to-int value, representing a bounded index in `[-len, len)`, and converts to a real index in `[0, len)`, or raises an error.
    pub fn check_index(&self, value: Value) -> Result<usize, Box<RuntimeError>> {
        let index: i64 = value.as_int()?;
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
            Indexable::List(it) => it[index].clone(),
            Indexable::Vector(it) => it[index].clone(),
        }
    }

    /// Setting indexes only works for immutable collections - so not strings
    pub fn set_index(&mut self, index: usize, value: ValuePtr) -> ValueResult {
        match self {
            Indexable::Str(it) => TypeErrorArgMustBeIndexable((*it).clone().to_value()).err(),
            Indexable::List(it) => { it[index] = value; Ok(()) },
            Indexable::Vector(it) => { it[index] = value; Ok(()) },
        }
    }
}


pub enum Sliceable<'a> {
    Str(&'a Rc<String>, String),
    List(Ref<'a, VecDeque<Value>>, VecDeque<Value>),
    Vector(Ref<'a, Vec<Value>>, Vec<Value>),
}

impl<'a> Sliceable<'a> {

    pub fn len(&self) -> usize {
        match self {
            Sliceable::Str(it, _) => it.len(),
            Sliceable::List(it, _) => it.len(),
            Sliceable::Vector(it, _) => it.len(),
        }
    }

    pub fn accept(&mut self, index: i64) {
        if index >= 0 && index < self.len() as i64 {
            let index = index as usize;
            match self {
                Sliceable::Str(src, dest) => dest.push(src.chars().nth(index).unwrap()),
                Sliceable::List(src, dest) => dest.push_back(src[index].clone()),
                Sliceable::Vector(src, dest) => dest.push(src[index].clone()),
            }
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LiteralType {
    List, Vector, Set, Dict
}


pub enum Literal {
    List(VecDeque<Value>),
    Vector(Vec<Value>),
    Set(IndexSet<Value, FxBuildHasher>),
    Dict(IndexMap<Value, Value, FxBuildHasher>),
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

    pub fn accumulate<I : Iterator<Item=Value>>(&mut self, mut iter: I) {
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

    pub fn unroll<I : Iterator<Item=Value>>(&mut self, iter: I) -> Result<(), Box<RuntimeError>> {
        match self {
            Literal::Dict(it) => for value in iter {
                let (key, value) = value.as_pair()?;
                it.insert(key, value);
            },
            _ => self.accumulate(iter),
        };
        Ok(())
    }
}

impl IntoValue for Literal {
    fn to_value(self) -> Value {
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
    use std::collections::VecDeque;
    use std::rc::Rc;
    use fxhash::FxBuildHasher;
    use indexmap::{IndexMap, IndexSet};
    use crate::core::{NativeFunction, PartialArgument};
    use crate::vm::error::RuntimeError;
    use crate::vm::value::{FunctionImpl, IntoIterableValue, IntoValue, Value};
    use crate::vm::{ValueOption, ValuePtr, ValueResult};

    #[test]
    fn test_layout() {
        // Should be no size overhead, since both error and none states are already represented by `ValuePtr`
        assert_eq!(std::mem::size_of::<ValueResult>(), std::mem::size_of::<ValuePtr>());
        assert_eq!(std::mem::size_of::<ValueOption>(), std::mem::size_of::<ValuePtr>());
    }

    #[test]
    fn test_value_ref_is_ref_equality() {
        let ptr1 = vec![1, 2, 3].into_iter().to_list();
        let ptr2 = vec![123].into_iter().to_list();
        let ptr3 = vec![1, 2, 3].into_iter().to_list();
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
        let ok = ValueResult::ok(ValuePtr::nil());
        let err = ValueResult::err(RuntimeError::RuntimeExit);

        assert!(ok.is_ok());
        assert!(err.is_err());

        assert_eq!(ok.as_result(), Ok(ValuePtr::nil()));
        assert_eq!(err.as_result(), Err(Box::new(RuntimeError::RuntimeExit)))
    }

    #[test]
    #[should_panic]
    fn test_value_result_ok_of_err() {
        let _ = ValueResult::ok(RuntimeError::RuntimeExit.to_value());
    }
}
