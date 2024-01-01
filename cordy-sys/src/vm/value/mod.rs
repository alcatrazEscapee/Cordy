use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use fxhash::FxBuildHasher;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use range::Enumerate;

use crate::compiler::Fields;
use crate::core;
use crate::core::NativeFunction;
use crate::util::impl_partial_ord;
use crate::vm::error::RuntimeError;
use crate::vm::value::ptr::{RefMut, Prefix};
use crate::vm::value::str::IterStr;
use crate::vm::value::range::Range;
use crate::vm::value::slice::Slice;
use crate::vm::value::number::{Complex, Rational};
use crate::vm::value::collections::{Dict, DictType, Heap, HeapType, List, ListType, Set, SetType, Vector, VectorType};

pub use crate::vm::value::ptr::{MAX_INT, MIN_INT, ValuePtr};
pub use crate::vm::value::number::ComplexType;
pub use crate::vm::value::func::{Closure, Function, PartialFunction, PartialNativeFunction, UpValue};
pub use crate::vm::value::error::{AnyResult, ErrorPtr, ErrorResult, ValueResult};

use RuntimeError::{*};


mod ptr;
mod str;
mod func;
mod range;
mod slice;
mod error;
mod number;
mod collections;


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
    ShortStr,
    LongStr,
    Complex,
    Rational,
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
    Memoized,
    Function,
    PartialFunction,
    PartialNativeFunction,
    Closure,
    Iter,
    Error,
    Never, // Optimization for type-checking code, to avoid code paths containing `unreachable!()` or similar patterns.
}

impl Type {
    fn name(self) -> &'static str {
        use Type::{*};
        match self {
            Nil => "nil",
            Bool => "bool",
            Int => "int",
            Complex => "complex",
            Rational => "rational",
            ShortStr | LongStr => "str",
            List => "list",
            Set => "set",
            Dict => "dict",
            Heap => "heap",
            Vector => "vector",
            Struct => "struct",
            StructType => "struct type",
            Range => "range",
            Enumerate => "enumerate",
            Slice => "slice",
            Iter => "iter",
            Memoized => "memoized",
            GetField => "get field",
            Function => "function",
            PartialFunction => "partial function",
            NativeFunction => "native function",
            PartialNativeFunction => "partial native function",
            Closure => "closure",
            Error => "error",
            Never => "never"
        }
    }

    /// Returns `true` if this type is considered **iterable**, aka `(typeof T) is iterable` would return `true`.
    fn is_iterable(self) -> bool {
        use Type::{*};
        match self {
            ShortStr | LongStr | List | Set | Dict | Heap | Vector | Range | Enumerate => true,
            _ => false
        }
    }

    /// Returns `true` if this type is considered **function-evaluable**, aka if `(typeof T) is function` would return `true`.
    ///
    /// **N.B.** This is not identical to if `T()` is legal, as some types are not considered functions, but are still evaluable,
    /// such as single element lists (as `list` in general as a type, is not function evaluable).
    fn is_evaluable(self) -> bool {
        use Type::{*};
        match self {
            Function | PartialFunction | NativeFunction | PartialNativeFunction | Closure | StructType | Slice => true,
            _ => false
        }
    }
}


/// A trait which marks types as values which may be contained by a `Prefix<T>` pointer.
///
/// All values pointed to in this manner are memory managed by `ValuePtr`, with reference counting for clone/drop,
/// and provide interior mutability via `borrow()`, `borrow_mut()`
pub trait Value {}

/// Before we know the type of an object, we must interpret it as a `Prefix<()>`, thus `()` needs to be a `Value`
impl Value for () {}

/// A trait marking that a certain type is constant or immutable. This then implies:
///
/// - The `borrow()` and `borrow_mut()` values must **never** be called for this type.
/// - The `borrow_const()` is provided, which does an unchecked, immutable borrow.
///
/// This is enforced by the implementation of `impl_const!()`, which notably never hands out a reference to the underlying `Prefix<T>`,
/// which ensures that callers can not call `borrow()` or `borrow_mut()`, both of which are invalid.
///
/// **N.B.** With negative trait bounds, we could refine `borrow()` and `borrow_mut()` to be `Value + !ConstValue`, preventing the above issue
/// at compile time, instead of relying on a contract.
pub trait ConstValue : Value {}


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
    pub fn to_str(&self) -> Cow<str> {
        self.safe_to_str(&mut RecursionGuard::new())
    }

    fn safe_to_str(&self, rc: &mut RecursionGuard) -> Cow<str> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Cow::from(self.as_str_slice()),
            Type::Function => self.as_function().to_str(),
            Type::PartialFunction => self.as_partial_function().to_str(rc),
            Type::NativeFunction => Cow::from(self.as_native().name()),
            Type::PartialNativeFunction => Cow::from(self.as_partial_native().to_str()),
            Type::Closure => self.as_closure().borrow_func().to_str(),
            _ => self.safe_to_repr_str(rc),
        }
    }

    /// Converts the `Value` to a representative `String`. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn to_repr_str(&self) -> Cow<str> {
        self.safe_to_repr_str(&mut RecursionGuard::new())
    }

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
            Type::Complex => Complex::to_repr_str(self.as_complex()),
            Type::Rational => Rational::to_repr_str(&self.as_rational()),
            Type::ShortStr | Type::LongStr => {
                let escaped = format!("{:?}", self.as_str_slice());
                Cow::from(format!("'{}'", &escaped[1..escaped.len() - 1]))
            },

            Type::List => recursive_guard!(
                Cow::from("[...]"),
                map_join(rc, self.as_list().borrow().iter(), '[', ']', "[]", ", ")
            ),
            Type::Set => recursive_guard!(
                Cow::from("{...}"),
                map_join(rc, self.as_set().borrow().iter(), '{', '}', "{}", ", ")
            ),
            Type::Dict => recursive_guard!(
                Cow::from("{...}"),
                Cow::from(format!("{{{}}}", self.as_dict()
                    .borrow()
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k.safe_to_repr_str(rc), v.safe_to_repr_str(rc)))
                    .join(", ")))
            ),
            Type::Heap => recursive_guard!(
                Cow::from("[...]"),
                map_join(rc, self.as_heap().borrow().iter().map(|u| &u.0), '[', ']', "[]",  ", ")
            ),
            Type::Vector => recursive_guard!(
                Cow::from("(...)"),
                map_join(rc, self.as_vector().borrow().iter(), '(', ')', "()", ", ")
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
            Type::StructType => Cow::from(self.as_struct_type().as_str()),

            Type::Range => self.as_range().to_repr_str(),
            Type::Enumerate => self.as_enumerate().to_repr_str(rc),
            Type::Slice => self.as_slice().to_repr_str(),

            Type::Iter => Cow::from("synthetic iterator"),
            Type::Memoized => Cow::from(format!("@memoize {}", self.as_memoized().borrow().func.safe_to_repr_str(rc))),

            Type::GetField => Cow::from("(->)"),

            Type::Function => Cow::from(self.as_function().to_repr_str()),
            Type::PartialFunction => self.as_partial_function().to_repr_str(rc),
            Type::NativeFunction => Cow::from(self.as_native().to_repr_str()),
            Type::PartialNativeFunction => Cow::from(self.as_partial_native().to_repr_str()),
            Type::Closure => Cow::from(self.as_closure().borrow_func().to_repr_str()),

            Type::Error | Type::Never => unreachable!(),
        }
    }

    /// Returns the inner user function, either from a `Function` or `Closure` type
    pub fn as_function_or_closure(&self) -> &Function {
        match self.is_function() {
            true => self.as_function(),
            false => self.as_closure().borrow_func(),
        }
    }

    /// Represents the type of this `Value`. This is used for runtime error messages,
    pub fn as_type_str(&self) -> &'static str {
        self.ty().name()
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
            Type::List => !self.as_list().borrow().is_empty(),
            Type::Set => !self.as_set().borrow().is_empty(),
            Type::Dict => !self.as_dict().borrow().is_empty(),
            Type::Heap => !self.as_heap().borrow().is_empty(),
            Type::Vector => !self.as_vector().borrow().is_empty(),
            Type::Range => !self.as_range().is_empty(),
            Type::Enumerate => self.as_enumerate().to_bool(),
            _ => true,
        }
    }

    /// Unwraps the value as an `iterable`, or raises a type error.
    /// For all value types except `Heap`, this is a O(1) and lazy operation. It also requires no persistent borrows of mutable types that outlast the call to `as_iter()`.
    pub fn to_iter(self) -> ErrorResult<Iterable> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Ok(Iterable::Str(self.as_str_iter())),
            Type::List | Type::Set | Type::Dict | Type::Vector => Ok(Iterable::Collection(0, self)),

            // Heaps completely unbox themselves to be iterated over
            Type::Heap => Ok(Iterable::RawVector(0, self.as_heap()
                .borrow()
                .iter()
                .cloned().map(|u| u.0)
                .collect::<Vec<ValuePtr>>())),

            Type::Range => Ok(self.as_range().to_iter()),
            Type::Enumerate => self.as_enumerate().to_iter(),

            _ => TypeErrorArgMustBeIterable(self.clone()).err(),
        }
    }

    /// Converts this value to a _sequence_, which is exclusively used by string formatting.
    ///
    /// - Lists, vectors, are treated as iterables and produce an ordered set of values.
    ///   This is intended to provide future support for numbered string formatting arguments.
    /// - Any other value is treated as a single argument for string formatting.
    ///
    /// Returns an `Iterable` for now, as it's the easiest mechanism - lacking numbered format arguments - to provide for arbitrary values.
    pub fn as_sequence(self) -> Iterable {
        match self.ty() {
            Type::List | Type::Vector => Iterable::Collection(0, self),
            _ => Iterable::Unit(Some(self)),
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

    /// Converts this value into a `(ValuePtr, ValuePtr)` if possible, supported for two-element `List` and `Vector`s
    pub fn to_pair(self) -> ErrorResult<(ValuePtr, ValuePtr)> {
        match match self.ty() {
            Type::List => self.as_list().borrow().iter().cloned().collect_tuple(),
            Type::Vector => self.as_vector().borrow().iter().cloned().collect_tuple(),
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
            Type::Function => Some(self.as_function().min_args()),
            Type::PartialFunction => Some(self.as_partial_function().min_nargs()),
            Type::NativeFunction => Some(self.as_native().min_nargs()),
            Type::PartialNativeFunction => Some(self.as_partial_native().min_nargs()),
            Type::Closure => Some(self.as_closure().borrow_func().min_args()),
            Type::StructType => Some(self.as_struct_type().num_fields()),
            Type::Slice => Some(1),
            _ => None,
        }
    }

    /// Returns the length of this `Value`. Equivalent to the native function `len`. Raises a type error if the value does not have a length.
    pub fn len(&self) -> ErrorResult<usize> {
        match self.ty() {
            Type::ShortStr | Type::LongStr => Ok(self.as_str_slice().chars().count()),
            Type::List => Ok(self.as_list().borrow().len()),
            Type::Set => Ok(self.as_set().borrow().len()),
            Type::Dict => Ok(self.as_dict().borrow().len()),
            Type::Heap => Ok(self.as_heap().borrow().len()),
            Type::Vector => Ok(self.as_vector().borrow().len()),
            Type::Range => Ok(self.as_range().len()),
            Type::Enumerate => self.as_enumerate().len(),
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
                let it = self.as_struct_type();
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

    pub fn is_iterable(&self) -> bool {
        self.ty().is_iterable()
    }

    pub fn is_evaluable(&self) -> bool {
        self.ty().is_evaluable()
    }

    #[inline(always)]
    pub fn ok(self) -> ValueResult {
        ValueResult::ok(self)
    }

    #[inline(always)]
    pub fn as_int_checked(self) -> ErrorResult<i64> {
        match self.is_int() {
            true => Ok(self.as_int()),
            false => TypeErrorArgMustBeInt(self).err()
        }
    }

    #[inline(always)]
    pub fn as_str_checked(&self) -> ErrorResult<&str> {
        match self.is_str() {
            true => Ok(self.as_str_slice()),
            false => TypeErrorArgMustBeStr(self.clone()).err()
        }
    }

    #[inline(always)]
    pub fn as_list_checked(&self) -> ErrorResult<&Prefix<List>> {
        match self.is_list() {
            true => Ok(self.as_list()),
            false => TypeErrorArgMustBeList(self.clone()).err()
        }
    }

    #[inline(always)]
    pub fn as_dict_checked(&self) -> ErrorResult<&Prefix<Dict>> {
        match self.is_dict() {
            true => Ok(self.as_dict()),
            false => TypeErrorArgMustBeDict(self.clone()).err()
        }
    }
}

#[cold]
fn err_field_not_found(value: ValuePtr, fields: &Fields, field_index: u32, repr: bool, access: bool) -> ValueResult {
    TypeErrorFieldNotPresentOnValue { value, field: fields.get_name(field_index), repr, access }.err()
}


/// A type used to prevent recursive `repr()` and `str()` calls.
///
/// This stores a pointer representation of the object in question, to properly prevent recursive structures, while providing easy equality checks.
struct RecursionGuard(Vec<usize>);

impl RecursionGuard {
    pub fn new() -> RecursionGuard { RecursionGuard(Vec::new()) }

    /// Returns `true` if the value has been seen before, triggering an early exit
    pub fn enter(&mut self, value: &ValuePtr) -> bool {
        let boxed = value.as_ptr_value();
        let ret = self.0.contains(&boxed);
        self.0.push(boxed);
        ret
    }

    pub fn leave(&mut self) {
        self.0.pop().unwrap(); // `.unwrap()` is safe as we should always call `enter()` before `leave()`
    }
}


macro_rules! impl_mut {
    ($ty:expr, $inner:ident, $as_T:ident, $is_T:ident) => {
        impl Value for $inner {}

        impl IntoValue for $inner {
            fn to_value(self) -> ValuePtr {
                ptr::from_shared(Prefix::new($ty, self))
            }
        }

        impl ValuePtr {
            pub fn $as_T(&self) -> &Prefix<$inner> {
                debug_assert!(self.ty() == $ty);
                self.as_prefix()
            }
            pub fn $is_T(&self) -> bool {
                self.ty() == $ty
            }
        }
    };
}

macro_rules! impl_const {
    ($ty:expr, $inner:ident, $as_T:ident, $is_T:ident) => {
        impl Value for $inner {}
        impl ConstValue for $inner {}

        impl IntoValue for $inner {
            fn to_value(self) -> ValuePtr {
                ptr::from_shared(Prefix::new($ty, self))
            }
        }

        impl ValuePtr {
            pub fn $as_T(&self) -> &$inner {
                debug_assert!(self.ty() == $ty);
                self.as_prefix::<$inner>().borrow_const()
            }
            pub fn $is_T(&self) -> bool {
                self.ty() == $ty
            }
        }
    };
}


impl_const!(Type::Function, Function, as_function, is_function);
impl_const!(Type::PartialFunction, PartialFunction, as_partial_function, is_partial_function);
impl_const!(Type::PartialNativeFunction, PartialNativeFunction, as_partial_native, is_partial_native);
impl_const!(Type::Slice, Slice, as_slice, is_slice);
impl_const!(Type::Range, Range, as_range, is_range);
impl_const!(Type::Enumerate, Enumerate, as_enumerate, is_enumerate);
impl_const!(Type::StructType, StructTypeImpl, as_struct_type, is_struct_type);
impl_const!(Type::Error, RuntimeError, as_err, is_err);

impl_mut!(Type::List, List, as_list, is_list);
impl_mut!(Type::Set, Set, as_set, is_set);
impl_mut!(Type::Dict, Dict, as_dict, is_dict);
impl_mut!(Type::Heap, Heap, as_heap, is_heap);
impl_mut!(Type::Vector, Vector, as_vector, is_vector);
impl_mut!(Type::Closure, Closure, as_closure, is_closure);
impl_mut!(Type::Memoized, MemoizedImpl, as_memoized, is_memoized);
impl_mut!(Type::Struct, StructImpl, as_struct, is_struct);
impl_mut!(Type::Iter, Iterable, as_synthetic_iterable, is_synthetic_iterable);


/// A trait which converts instances of Rust types into a Cordy `ValuePtr`
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

pub(crate) use impl_into;

impl_into!(usize, self, ptr::from_i64(self as i64));
impl_into!(i64, self, ptr::from_i64(self));
impl_into!(bool, self, ptr::from_bool(self));
impl_into!(char, self, ptr::from_char(self));
impl_into!(&str, self, ptr::from_str(self));
impl_into!(String, self, ptr::from_str(self.as_str()));
impl_into!(Cow<'_, str>, self, ptr::from_str(&self));
impl_into!(NativeFunction, self, ptr::from_native(self));
impl_into!(ListType, self, List::new(self).to_value());
impl_into!(VectorType, self, Vector::new(self).to_value());
impl_into!((ValuePtr, ValuePtr), self, vec![self.0, self.1].to_value());
impl_into!(SetType, self, Set::new(self).to_value());
impl_into!(DictType, self, Dict::new(self).to_value());
impl_into!(HeapType, self, Heap::new(self).to_value());


/// A trait which is responsible for wrapping conversions from a `Iterator<Item=ValuePtr>` into `IntoValue`, which then converts to a `ValuePtr`.
pub trait IntoIterableValue {
    fn to_list(self) -> ValuePtr;
    fn to_vector(self) -> ValuePtr;
    fn to_set(self) -> ValuePtr;
    fn to_heap(self) -> ValuePtr;
}

impl<I> IntoIterableValue for I where I : Iterator<Item=ValuePtr> {
    fn to_list(self) -> ValuePtr {
        self.collect::<ListType>().to_value()
    }

    fn to_vector(self) -> ValuePtr {
        self.collect::<VectorType>().to_value()
    }

    fn to_set(self) -> ValuePtr {
        self.collect::<SetType>().to_value()
    }

    fn to_heap(self) -> ValuePtr {
        self.map(Reverse).collect::<HeapType>().to_value()
    }
}

/// A trait which is responsible for wrapping conversions from an `Iterator<Item=(ValuePtr, ValuePtr)>` into a `dict()`
pub trait IntoDictValue {
    fn to_dict(self) -> ValuePtr;
}

impl<I> IntoDictValue for I where I : Iterator<Item=(ValuePtr, ValuePtr)> {
    fn to_dict(self) -> ValuePtr {
        self.collect::<DictType>().to_value()
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
        self.owner.as_struct_type()
    }

    fn get_field(&self, field_offset: usize) -> ValuePtr {
        self.values[field_offset].clone()
    }

    fn set_field(&mut self, field_offset: usize, value: ValuePtr) {
        self.values[field_offset] = value;
    }

    pub fn fields(&self) -> impl Iterator<Item=&ValuePtr> {
        self.values.iter()
    }

    pub fn fields_mut(&mut self) -> &mut Vec<ValuePtr> {
        &mut self.values
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
    ///
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
    Unit(Option<ValuePtr>),
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
            Type::List => ptr.as_list().borrow().get(index).cloned(),
            Type::Set => ptr.as_set().borrow().get_index(index).cloned(),
            Type::Dict => ptr.as_dict().borrow().get_index(index).map(|(l, r)| (l.clone(), r.clone()).to_value()),
            Type::Vector => ptr.as_vector().borrow().get(index).cloned(),
            _ => unreachable!(),
        }
    }
}


impl Iterator for Iterable {
    type Item = ValuePtr;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iterable::Str(it) => it.next().map(|u| u.to_value()),
            Iterable::Unit(it) => it.take(),
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
            Iterable::Unit(it) => it.take(),
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
    List(RefMut<'a, List>),
    Vector(RefMut<'a, Vector>),
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
    pub fn check_index(&self, value: ValuePtr) -> ErrorResult<usize> {
        let index: i64 = value.as_int_checked()?;
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
    pub fn set_index(&mut self, index: usize, value: ValuePtr) -> AnyResult {
        match self {
            Indexable::Str(it) => TypeErrorArgMustBeIndexable(it.to_value()).err(),
            Indexable::List(it) => {
                it[index] = value;
                Ok(())
            },
            Indexable::Vector(it) => {
                it[index] = value;
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
    use crate::vm::{ValuePtr, ValueResult};
    use crate::vm::error::RuntimeError;
    use crate::vm::value::{IntoIterableValue, IntoValue};

    #[test]
    fn test_layout() {
        // Should be no size overhead, since both error and none states are already represented by `ValuePtr`
        assert_eq!(std::mem::size_of::<ValueResult>(), std::mem::size_of::<ValuePtr>());
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

        assert_eq!(ptr1.as_ptr_value(), ptr1.as_ptr_value());
        assert_ne!(ptr1.as_ptr_value(), ptr2.as_ptr_value());
        assert_ne!(ptr1.as_ptr_value(), ptr3.as_ptr_value()); // Same value, different reference
        assert_eq!(ptr1.as_ptr_value(), ptr4.as_ptr_value());
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
