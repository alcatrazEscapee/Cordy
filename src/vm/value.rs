use std::borrow::Borrow;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::mem;
use std::rc::Rc;
use std::str::Chars;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;

use crate::compiler::Fields;
use crate::util::RecursionGuard;
use crate::core;
use crate::core::NativeFunction;
use crate::vm::ValueResult;
use crate::vm::error::RuntimeError;

use Value::{*};
use RuntimeError::{*};


pub type C64 = num_complex::Complex<i64>;


/// The runtime sum type used by the virtual machine
/// All `Value` type objects must be cloneable, and so mutable objects must be reference counted to ensure memory safety
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum Value {
    // Primitive (Immutable) Types
    Nil,
    Bool(bool),
    Int(i64),
    Complex(Box<C64>),
    Str(Rc<String>),

    // Reference (Mutable) Types
    List(Mut<VecDeque<Value>>),
    Set(Mut<SetImpl>),
    Dict(Mut<DictImpl>),
    Heap(Mut<HeapImpl>), // `List` functions as both Array + Deque, but that makes it un-viable for a heap. So, we have a dedicated heap structure
    Vector(Mut<Vec<Value>>), // `Vector` is a fixed-size list (in theory, not in practice), that most operations will behave elementwise on

    /// A mutable instance of a struct - basically a named tuple.
    Struct(Mut<StructImpl>),
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

    /// Synthetic Memoized Function Argument
    /// This is a argument which is partially evaluated to the result of a `Memoized` stdlib function.
    /// It will only ever be present as the first partial argument for a `SyntheticMemoizedFunction` native function.
    Memoized(Box<MemoizedImpl>),

    /// A unique type for a partially evaluated `->` operator, i.e. `(->some_field)`
    /// The parameter is a field index. The only use of this type is as a function, where it shortcuts to a `GetField` operation.
    GetField(u32),

    // Functions
    Function(Rc<FunctionImpl>),
    PartialFunction(Box<PartialFunctionImpl>),
    NativeFunction(NativeFunction),
    PartialNativeFunction(NativeFunction, Box<Vec<Value>>),
    Closure(Box<ClosureImpl>),
}


impl Value {

    // Constructors

    pub fn partial(func: Value, args: Vec<Value>) -> Value { PartialFunction(Box::new(PartialFunctionImpl { func, args }))}
    pub fn closure(func: Rc<FunctionImpl>) -> Value { Closure(Box::new(ClosureImpl { func, environment: Vec::new() })) }
    pub fn instance(type_impl: Rc<StructTypeImpl>, values: Vec<Value>) -> Value { Struct(Mut::new(StructImpl { type_index: type_impl.type_index, type_impl, values }))}

    /// Creates a memoized `PartialNativeFunction`, which wraps the provided function `func`, as a memoized function.
    pub fn memoized(func: Value) -> Value {
        PartialNativeFunction(NativeFunction::SyntheticMemoizedFunction, Box::new(vec![Memoized(Box::new(MemoizedImpl::new(func)))]))
    }

    /// Creates a new `Range()` value from a given set of integer parameters.
    /// Raises an error if `step == 0`
    ///
    /// Note: this implementation internally replaces all empty range values with the single `range(0, 0, 0)` instance. This means that `range(1, 2, -1) . str` will have to handle this case as it will not be representative.
    pub fn range(start: i64, stop: i64, step: i64) -> ValueResult {
        if step == 0 {
            ValueErrorStepCannotBeZero.err()
        } else if (stop > start && step > 0) || (stop < start && step < 0) {
            Ok(Range(Box::new(RangeImpl { start, stop, step }))) // Non-empty range
        } else {
            Ok(Range(Box::new(RangeImpl { start: 0, stop: 0, step: 0 }))) // Empty range
        }
    }

    pub fn slice(arg1: Value, arg2: Value, arg3: Option<Value>) -> ValueResult {
        Ok(Slice(Box::new(SliceImpl {
            arg1: arg1.as_int_or()?,
            arg2: arg2.as_int_or()?,
            arg3: arg3.unwrap_or(Nil).as_int_or()?
        })))
    }

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn to_str(self: &Self) -> String { self.safe_to_str(&mut RecursionGuard::new()) }

    fn safe_to_str(self: &Self, rc: &mut RecursionGuard) -> String {
        match self {
            Str(s) => (**s).to_owned(),
            Function(f) => f.name.clone(),
            PartialFunction(f) => f.func.safe_to_str(rc),
            NativeFunction(b) => String::from(b.name()),
            PartialNativeFunction(b, _) => String::from(b.name()),
            Closure(c) => (*c).func.as_ref().name.clone(),
            _ => self.safe_to_repr_str(rc),
        }
    }

    /// Converts the `Value` to a representative `String. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn to_repr_str(self: &Self) -> String { self.safe_to_repr_str(&mut RecursionGuard::new()) }
    
    fn safe_to_repr_str(self: &Self, rc: &mut RecursionGuard) -> String {
        macro_rules! recursive_guard {
            ($default:expr, $recursive:expr) => {{
                let ret = if rc.enter(self) { $default } else { $recursive };
                rc.leave();
                ret
            }};
        }

        #[inline]
        fn to_str(i: Option<i64>) -> String {
            i.map(|u| u.to_string()).unwrap_or(String::new())
        }

        match self {
            Nil => String::from("nil"),
            Bool(b) => b.to_string(),
            Int(i) => i.to_string(),
            Complex(c) => if c.re == 0 {
                format!("{}i", c.im)
            } else {
                format!("{} + {}i", c.re, c.im)
            },
            Str(s) => {
                let escaped = format!("{:?}", s);
                format!("'{}'", &escaped[1..escaped.len() - 1])
            },

            List(v) => recursive_guard!(
                String::from("[...]"),
                format!("[{}]", v.unbox().iter()
                    .map(|t| t.safe_to_repr_str(rc))
                    .join(", "))
            ),
            Set(v) => recursive_guard!(
                String::from("{...}"),
                format!("{{{}}}", v.unbox().set.iter()
                    .map(|t| t.safe_to_repr_str(rc))
                    .join(", "))
            ),
            Dict(v) => recursive_guard!(
                String::from("{...}"),
                format!("{{{}}}", v.unbox().dict.iter()
                    .map(|(k, v)| format!("{}: {}", k.safe_to_repr_str(rc), v.safe_to_repr_str(rc)))
                    .join(", "))
            ),
            Heap(v) => recursive_guard!(
                String::from("[...]"),
                format!("[{}]", v.unbox().heap.iter()
                    .map(|t| t.0.safe_to_repr_str(rc))
                    .join(", "))
            ),
            Vector(v) => recursive_guard!(
                String::from("(...)"),
                format!("({})", v.unbox().iter()
                    .map(|t| t.safe_to_repr_str(rc))
                    .join(", "))
            ),

            Struct(it) => {
                let it = it.unbox();
                recursive_guard!(
                    format!("{}(...)", it.type_impl.name),
                    format!("{}({})", it.type_impl.name.as_str(), it.values.iter()
                        .zip(it.type_impl.field_names.iter())
                        .map(|(v, k)| format!("{}={}", k, v.safe_to_repr_str(rc)))
                        .join(", "))
                )
            },
            StructType(it) => format!("struct {}({})", it.name.clone(), it.field_names.join(", ")),

            Range(r) => if r.step == 0 { String::from("range(empty)") } else { format!("range({}, {}, {})", r.start, r.stop, r.step) },
            Enumerate(v) => format!("enumerate({})", v.safe_to_repr_str(rc)),
            Slice(v) => match &v.arg3 {
                Some(arg3) => format!("[{}:{}:{}]", to_str(v.arg1), to_str(v.arg2), arg3.to_string()),
                None => format!("[{}:{}]", to_str(v.arg1), to_str(v.arg2)),
            }

            Iter(_) => String::from("<synthetic> iterator"),
            Memoized(_) => String::from("<synthetic> memoized"),

            GetField(_) => String::from("(->)"),

            Function(f) => (*f).as_ref().borrow().as_str(),
            PartialFunction(f) => (*f).as_ref().borrow().func.safe_to_repr_str(rc),
            NativeFunction(b) => format!("fn {}({})", b.name(), b.args()),
            PartialNativeFunction(b, _) => format!("fn {}({})", b.name(), b.args()),
            Closure(c) => (*c).func.as_ref().borrow().as_str(),
        }
    }

    /// Represents the type of this `Value`. This is used for runtime error messages,
    pub fn as_type_str(self: &Self) -> String {
        String::from(match self {
            Nil => "nil",
            Bool(_) => "bool",
            Int(_) => "int",
            Complex(_) => "complex",
            Str(_) => "str",
            List(_) => "list",
            Set(_) => "set",
            Dict(_) => "dict",
            Heap(_) => "heap",
            Vector(_) => "vector",
            Struct(_) => "struct",
            StructType(_) => "struct type",
            Range(_) => "range",
            Enumerate(_) => "enumerate",
            Slice(_) => "slice",
            Iter(_) => "iter",
            Memoized(_) => "memoized",
            GetField(_) => "get field",
            Function(_) => "function",
            PartialFunction(_) => "partial function",
            NativeFunction(_) => "native function",
            PartialNativeFunction(_, _) => "partial native function",
            Closure(_) => "closure",
        })
    }

    /// Used by `trace` disabled code, do not remove!
    pub fn as_debug_str(self: &Self) -> String {
        format!("{}: {}", self.to_repr_str(), self.as_type_str())
    }

    pub fn as_bool(self: &Self) -> bool {
        match self {
            Nil => false,
            Bool(it) => *it,
            Int(it) => *it != 0,
            Complex(_) => false, // complex will never be zero
            Str(it) => !it.is_empty(),
            List(it) => !it.unbox().is_empty(),
            Set(it) => !it.unbox().set.is_empty(),
            Dict(it) => !it.unbox().dict.is_empty(),
            Heap(it) => !it.unbox().heap.is_empty(),
            Range(it) => !it.is_empty(),
            Enumerate(it) => (**it).as_bool(),
            Iter(_) | Memoized(_) => panic!("{:?} is a synthetic type should not have as_bool() invoked on it", self),
            Vector(v) => v.unbox().is_empty(),
            _ => true,
        }
    }

    /// Unwraps the value as an `int`, or raises a type error
    pub fn as_int(self: &Self) -> Result<i64, Box<RuntimeError>> {
        match self {
            Bool(b) => Ok(*b as i64),
            Int(i) => Ok(*i),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    #[inline]
    pub fn as_int_unchecked(self: &Self) -> i64 {
        match self {
            Bool(b) => *b as i64,
            Int(i) => *i,
            _ => 0,
        }
    }

    /// Unwraps the value as a `complex`, or raises a type error.
    pub fn as_complex(self: &Self) -> Result<C64, Box<RuntimeError>> {
        match self {
            Bool(b) => Ok(C64::new(*b as i64, 0)),
            Int(i) => Ok(C64::new(*i, 0)),
            Complex(c) => Ok(*c.clone()),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    #[inline]
    pub fn as_complex_unchecked(self: &Self) -> C64 {
        match self {
            Bool(b) => C64::new(*b as i64, 0),
            Int(i) => C64::new(*i, 0),
            Complex(c) => *c.clone(),
            _ => C64::new(0, 0),
        }
    }

    /// Like `as_int()` but returns an `Option<i64>`, and converts `nil` to `None`
    pub fn as_int_or(self: &Self) -> Result<Option<i64>, Box<RuntimeError>> {
        match self {
            Nil => Ok(None),
            Int(i) => Ok(Some(*i)),
            Bool(b) => Ok(Some(if *b { 1 } else { 0 })),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    /// Unwraps the value as a `str`, or raises a type error
    pub fn as_str(self: &Self) -> Result<&String, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(it),
            v => TypeErrorArgMustBeStr(v.clone()).err()
        }
    }

    /// Unwraps the value as an `iterable`, or raises a type error.
    /// For all value types except `Heap`, this is a O(1) and lazy operation. It also requires no persistent borrows of mutable types that outlast the call to `as_iter()`.
    pub fn as_iter(self: &Self) -> Result<Iterable, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(Iterable::str((**it).clone())),
            List(it) => Ok(Iterable::Collection(0, CollectionIterable::List(it.clone()))),
            Set(it) => Ok(Iterable::Collection(0, CollectionIterable::Set(it.clone()))),
            Dict(it) => Ok(Iterable::Collection(0, CollectionIterable::Dict(it.clone()))),
            Vector(it) => Ok(Iterable::Collection(0, CollectionIterable::Vector(it.clone()))),

            Heap(it) => {
                // Heaps completely unbox themselves to be iterated over
                let vec = it.unbox().heap.iter().cloned().map(|u| u.0).collect::<Vec<Value>>();
                Ok(Iterable::Collection(0, CollectionIterable::RawVector(vec)))
            },

            Range(it) => Ok(Iterable::Range(it.start, (**it).clone())),
            Enumerate(it) => Ok(Iterable::Enumerate(0, Box::new((**it).as_iter()?))),

            _ => TypeErrorArgMustBeIterable(self.clone()).err(),
        }
    }

    /// Unwraps the value as an `iterable`, or if it is not, yields an iterable of the single element
    /// Note that this takes a `str` to be a non-iterable primitive type, unlike `is_iter()` and `as_iter()`
    pub fn as_iter_or_unit(self: &Self) -> Iterable {
        match self {
            List(it) => Iterable::Collection(0, CollectionIterable::List(it.clone())),
            Set(it) => Iterable::Collection(0, CollectionIterable::Set(it.clone())),
            Dict(it) => Iterable::Collection(0, CollectionIterable::Dict(it.clone())),
            Vector(it) => Iterable::Collection(0, CollectionIterable::Vector(it.clone())),

            Heap(it) => {
                // Heaps completely unbox themselves to be iterated over
                let vec = it.unbox().heap.iter().cloned().map(|u| u.0).collect::<Vec<Value>>();
                Iterable::Collection(0, CollectionIterable::RawVector(vec))
            },

            Range(it) => Iterable::Range(it.start, (**it).clone()),
            Enumerate(it) => Iterable::Enumerate(0, Box::new((**it).as_iter_or_unit())),

            it => Iterable::Unit(Some(it.clone())),
        }
    }

    /// Converts this `Value` to a `ValueAsIndex`, which is a index-able object, supported for `List`, `Vector`, and `Str`
    pub fn as_index(self: &Self) -> Result<Indexable, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(Indexable::Str(it)),
            List(it) => Ok(Indexable::List(it.unbox_mut())),
            Vector(it) => Ok(Indexable::Vector(it.unbox_mut())),
            _ => TypeErrorArgMustBeIndexable(self.clone()).err()
        }
    }

    /// Converts this `Value` to a `ValueAsSlice`, which is a builder for slice-like structures, supported for `List` and `Str`
    pub fn as_slice(self: &Self) -> Result<Sliceable, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(Sliceable::Str(it, String::new())),
            List(it) => Ok(Sliceable::List(it.unbox(), VecDeque::new())),
            Vector(it) => Ok(Sliceable::Vector(it.unbox(), Vec::new())),
            _ => TypeErrorArgMustBeSliceable(self.clone()).err()
        }
    }

    /// Converts this `Value` into a `(Value, Value)` if possible, supported for two-element `List` and `Vector`s
    pub fn as_pair(self: &Self) -> Result<(Value, Value), Box<RuntimeError>> {
        match match self {
            List(it) => it.unbox().iter().cloned().collect_tuple(),
            Vector(it) => it.unbox().iter().cloned().collect_tuple(),
            _ => None
        } {
            Some(it) => Ok(it),
            None => ValueErrorCannotCollectIntoDict(self.clone()).err()
        }
    }

    /// Returns the internal `FunctionImpl` of this value.
    /// Must only be called on a `Function` or `Closure`, will panic otherwise.
    pub fn unbox_func(self: &Self) -> &Rc<FunctionImpl> {
        match self {
            Function(f) => f,
            Closure(c) => &c.func,
            _ => panic!("Tried to unwrap a {:?} as a function", self),
        }
    }

    /// Returns `None` if this value is not a function
    /// Returns `Some(None)` if this value is a function with an unknown number of arguments
    /// Returns `Some(Some(nargs))` if this value is a function with a known number of arguments
    pub fn unbox_func_args(self: &Self) -> Option<Option<u32>> {
        match self {
            Function(it) => Some(Some(it.min_args())),
            PartialFunction(it) => Some(Some(it.func.unbox_func().min_args() - it.args.len() as u32)),
            NativeFunction(it) => Some(it.nargs()),
            PartialNativeFunction(it, args) => Some(it.nargs().map(|u| u - args.len() as u32)),
            Closure(it) => Some(Some(it.func.min_args())),
            StructType(it) => Some(Some(it.field_names.len() as u32)),
            Slice(_) => Some(Some(1)),
            _ => None,
        }
    }

    /// Returns the length of this `Value`. Equivalent to the native function `len`. Raises a type error if the value does not have a lenth.
    pub fn len(self: &Self) -> Result<usize, Box<RuntimeError>> {
        match &self {
            Str(it) => Ok(it.chars().count()),
            List(it) => Ok(it.unbox().len()),
            Set(it) => Ok(it.unbox().set.len()),
            Dict(it) => Ok(it.unbox().dict.len()),
            Heap(it) => Ok(it.unbox().heap.len()),
            Vector(it) => Ok(it.unbox().len()),
            Range(it) => Ok(it.len()),
            Enumerate(it) => it.len(),
            _ => TypeErrorArgMustBeIterable(self.clone()).err()
        }
    }

    pub fn get_field(self: Self, fields: &Fields, field_index: u32) -> ValueResult {
        match self {
            Struct(it) => {
                let mut it = it.unbox_mut();
                match fields.get_field_offset(it.type_index, field_index) {
                    Some(field_offset) => Ok(it.get_field(field_offset)),
                    None => TypeErrorFieldNotPresentOnValue(StructType(it.type_impl.clone()), fields.get_field_name(field_index), true).err()
                }
            },
            _ => TypeErrorFieldNotPresentOnValue(self, fields.get_field_name(field_index), false).err()
        }
    }

    pub fn set_field(self: Self, fields: &Fields, field_index: u32, value: Value) -> ValueResult {
        match self {
            Struct(it) => {
                let mut it = it.unbox_mut();
                match fields.get_field_offset(it.type_index, field_index) {
                    Some(field_offset) => {
                        it.set_field(field_offset, value.clone());
                        Ok(value)
                    },
                    None => TypeErrorFieldNotPresentOnValue(StructType(it.type_impl.clone()), fields.get_field_name(field_index), true).err()
                }
            },
            _ => TypeErrorFieldNotPresentOnValue(self, fields.get_field_name(field_index), false).err()
        }
    }

    pub fn is_bool(self: &Self) -> bool { match self { Bool(_) => true, _ => false } }
    pub fn is_int(self: &Self) -> bool { match self { Bool(_) | Int(_) => true, _ => false } }
    pub fn is_complex(self: &Self) -> bool { match self { Bool(_) | Int(_) | Complex(_) => true, _ => false } }
    pub fn is_str(self: &Self) -> bool { match self { Str(_) => true, _ => false } }

    pub fn is_list(self: &Self) -> bool { match self { List(_) => true, _ => false } }
    pub fn is_set(self: &Self) -> bool { match self { Set(_) => true, _ => false } }
    pub fn is_dict(self: &Self) -> bool { match self { Dict(_) => true, _ => false } }
    pub fn is_vector(self: &Self) -> bool { match self { Vector(_) => true, _ => false } }

    pub fn is_iter(self: &Self) -> bool {
        match self {
            Str(_) | List(_) | Set(_) | Dict(_) | Heap(_) | Vector(_) | Range(_) | Enumerate(_) => true,
            _ => false
        }
    }

    /// Returns if the `Value` is function-evaluable.
    /// Note that single-element lists are not considered functions here.
    pub fn is_function(self: &Self) -> bool {
        match self {
            Function(_) | PartialFunction(_) | NativeFunction(_) | PartialNativeFunction(_, _) | Closure(_) | StructType(_) | Slice(_) => true,
            _ => false
        }
    }

    pub fn ptr_eq(self: &Self, other: &Value) -> bool {
        match (&self, &other) {
            (List(l), List(r)) => l.ptr_eq(r),
            (Set(l), Set(r)) => l.ptr_eq(r),
            (Dict(l), Dict(r)) => l.ptr_eq(r),
            (Heap(l), Heap(r)) => l.ptr_eq(r),
            (Vector(l), Vector(r)) => l.ptr_eq(r),
            (Struct(l), Struct(r)) => l.ptr_eq(r),
            _ if mem::discriminant(self) != mem::discriminant(other) => false,
            _ => panic!("Value::ptr_eq() should only be called on boxed mutable pointer types"),
        }
    }
}

/// Implement Ord and PartialOrd explicitly, to derive implementations for each individual type.
/// All types are explicitly ordered because we need it in order to call `sort()` and I don't see why not otherwise.
impl PartialOrd<Self> for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Bool(l), Bool(r)) => (*l as i32).cmp(&(*r as i32)),
            (Int(l), Int(r)) => l.cmp(r),
            (Str(l), Str(r)) => l.cmp(r),
            (List(l), List(r)) => {
                let ls = (*l).unbox();
                let rs = (*r).unbox();
                ls.cmp(&rs)
            },
            (Vector(l), Vector(r)) => {
                let ls = (*l).unbox();
                let rs = (*r).unbox();
                ls.cmp(&rs)
            }
            (Heap(l), Heap(r)) => {
                let ls = (*l).unbox();
                let rs = (*r).unbox();
                ls.heap.iter().cmp(&rs.heap)
            }
            // Un-order-able things are defined as equal ordering
            (_, _) => Ordering::Equal,
        }
    }
}



/// A trait which is responsible for converting native types into a `Value`.
/// It is preferred to boxing these types directly using `Value::Foo()`, as most types have inner complexity that needs to be managed.
pub trait IntoValue {
    fn to_value(self) -> Value;
}

impl IntoValue for Value { fn to_value(self) -> Value { self } }
impl IntoValue for bool { fn to_value(self) -> Value { Bool(self) } }
impl IntoValue for i64 { fn to_value(self) -> Value { Int(self) } }
impl IntoValue for char { fn to_value(self) -> Value { Str(Rc::new(String::from(self))) } }
impl<'a> IntoValue for &'a str { fn to_value(self) -> Value { Str(Rc::new(String::from(self))) } }
impl IntoValue for NativeFunction { fn to_value(self) -> Value { NativeFunction(self) } }
impl IntoValue for String { fn to_value(self) -> Value { Str(Rc::new(self)) } }
impl IntoValue for VecDeque<Value> { fn to_value(self) -> Value { List(Mut::new(self)) } }
impl IntoValue for Vec<Value> { fn to_value(self) -> Value { Vector(Mut::new(self)) } }
impl IntoValue for IndexSet<Value> { fn to_value(self) -> Value { Set(Mut::new(SetImpl { set: self })) } }
impl IntoValue for IndexMap<Value, Value> { fn to_value(self) -> Value { Dict(Mut::new(DictImpl { dict: self, default: None })) } }
impl IntoValue for BinaryHeap<Reverse<Value>> { fn to_value(self) -> Value { Heap(Mut::new(HeapImpl { heap: self }))} }
impl IntoValue for FunctionImpl { fn to_value(self) -> Value { Function(Rc::new(self)) }}
impl IntoValue for StructTypeImpl { fn to_value(self) -> Value { StructType(Rc::new(self)) } }

impl IntoValue for C64 {
    fn to_value(self) -> Value {
        if self.im == 0 { Int(self.re) } else { Complex(Box::new(self)) }
    }
}

impl<'a> IntoValue for Sliceable<'a> {
    fn to_value(self: Self) -> Value {
        match self {
            Sliceable::Str(_, it) => it.to_value(),
            Sliceable::List(_, it) => it.to_value(),
            Sliceable::Vector(_, it) => it.to_value(),
        }
    }
}

pub trait IntoValueResult {
    fn to_value(self) -> ValueResult;
}

impl<T : IntoValue> IntoValueResult for Result<T, Box<RuntimeError>> {
    fn to_value(self) -> ValueResult {
        self.map(|u| u.to_value())
    }
}


/// A trait which is responsible for wrapping conversions from a `Iterator<Item=Value>` into `IntoValue`, which then converts to a `Value`.
pub trait IntoIterableValue {
    fn to_list(self) -> Value;
    fn to_vector(self) -> Value;
    fn to_set(self) -> Value;
    fn to_heap(self) -> Value;
}

impl<I> IntoIterableValue for I where I : Iterator<Item=Value> {
    fn to_list(self) -> Value { self.collect::<VecDeque<Value>>().to_value() }
    fn to_vector(self) -> Value { self.collect::<Vec<Value>>().to_value() }
    fn to_set(self) -> Value { self.collect::<IndexSet<Value>>().to_value() }
    fn to_heap(self) -> Value { self.map(|u| Reverse(u)).collect::<BinaryHeap<Reverse<Value>>>().to_value() }
}

/// A trait which is responsible for wrapping conversions from an `Iterator<Item=(Value, Value)>` into a `Value::Dict`
pub trait IntoDictValue {
    fn to_dict(self) -> Value;
}

impl<I> IntoDictValue for I where I : Iterator<Item=(Value, Value)> {
    fn to_dict(self) -> Value {
        self.collect::<IndexMap<Value, Value>>().to_value()
    }
}



/// `Mut<T>` is a wrapper around internally mutable types. It implements the required traits for `Value` through it's inner type.
/// Note that it also implements `Hash`, even though the internal type is mutable. This is required to satisfy rust's type system.
/// Mutating values stored in a hash backed structure is legal, from a language point of view, but will just invoke undefined behavior.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Mut<T : Eq + PartialEq + Debug + Clone>(Rc<RefCell<T>>);

impl<T : Eq + PartialEq + Debug + Clone> Mut<T> {

    pub fn new(value: T) -> Mut<T> {
        Mut(Rc::new(RefCell::new(value)))
    }

    /// Unbox the `Mut<T>`, obtaining a borrow on the contents.
    /// Note that while semantically in Rust, this is a non-unique borrow, and we can treat it as such while in native code, we **cannot** yield into user code while this borrow is active.
    pub fn unbox(&self) -> Ref<T> {
        (*self.0).borrow()
    }

    /// Unbox the `Mut<T>`, obtaining a mutable and unique borrow on the contents.
    pub fn unbox_mut(&self) -> RefMut<T> {
        (*self.0).borrow_mut()
    }

    /// Attempt to unbox the `Mut<T>`, obtaining a borrow on the contents if it not already borrowed - otherwise return `None`
    pub fn try_unbox(&self) -> Option<Ref<T>> {
        (*self.0).try_borrow().ok()
    }

    /// Returns `true` if the two inner instances are part of the same object.
    pub fn ptr_eq(&self, other: &Mut<T>) -> bool {
        return Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<T : Eq + PartialEq + Debug + Clone + Hash> Hash for Mut<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (*self).unbox().hash(state)
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
    pub fn min_args(self: &Self) -> u32 {
        (self.args.len() - self.default_args.len()) as u32
    }

    /// The maximum number of required arguments, inclusive.
    pub fn max_args(self: &Self) -> u32 {
        self.args.len() as u32
    }

    pub fn in_range(self: &Self, nargs: u32) -> bool {
        self.min_args() <= nargs && (self.var_arg || nargs <= self.max_args())
    }

    /// Returns the number of variadic arguments that need to be collected, before invoking the function, if needed.
    pub fn num_var_args(self: &Self, nargs: u32) -> Option<u32> {
        if self.var_arg && nargs >= self.max_args() {
            Some(nargs + 1 - self.max_args())
        } else {
            None
        }
    }

    /// Returns the jump offset of the function
    /// For typical functions, this is just the `head`, however when default arguments are present, or not, this is offset by the default argument offsets.
    /// The `nargs` must be legal (between `[min_args(), max_args()]`
    pub fn jump_offset(self: &Self, nargs: u32) -> usize {
        self.head + if nargs == self.min_args() {
            0
        } else if self.var_arg && nargs >= self.max_args() {
            *self.default_args.last().unwrap()
        } else {
            self.default_args[(nargs - self.min_args() - 1) as usize]
        }
    }

    pub fn as_str(self: &Self) -> String {
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
    /// The `Value` must be either a `Function` or `Closure`
    pub func: Value,
    pub args: Vec<Value>,
}

impl Hash for PartialFunctionImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


/// A closure is a combination of a function, and a set of `environment` variables.
/// These variables are references either to locals in the enclosing function, or captured variables from the enclosing function itself.
///
/// A closure also provides *interior mutability* for it's captured upvalues, allowing them to be modified even from the surrounding function.
/// Unlike with other mutable `Value` types, this does so using `Rc<Cell<Value>>`. The reason being:
///
/// - A `Mut` cannot be unboxed without creating a borrow, which introduces lifetime restrictions. It also cannot be mutably unboxed without creating a write lock. With a closure, we need to be free to unbox the environment straight onto the stack, so this is off the table.
/// - The closure's inner value can be thought of as immutable. As `Value` is immutable, and clone-able, so can the contents of `Cell`. We can then unbox this completely - take a reference to the `Rc`, and call `get()` to unbox the current value of the cell, onto the stack.
///
/// This has one problem, which is we can't call `.get()` unless the cell is `Copy`, which `Value` isn't, and can't be, because `Mut` can't be copy due to the presence of `Rc`... Fortunately, this is just an API limitation, and we can unbox the cell in other ways.
///
/// Note we cannot derive most functions, as that also requires `Cell<Value>` to be `Copy`, due to convoluted trait requirements.
#[derive(Clone)]
pub struct ClosureImpl {
    func: Rc<FunctionImpl>,
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

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SetImpl {
    pub set: IndexSet<Value>
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
impl Hash for Mut<SetImpl> {
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


/// Boxes a `IndexMap<Value, Value>`, along with an optional default value
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct DictImpl {
    pub dict: IndexMap<Value, Value>,
    pub default: Option<Value>
}

impl PartialOrd for DictImpl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

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
impl Hash for Mut<DictImpl> {
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
    pub type_impl: Rc<StructTypeImpl>,
    values: Vec<Value>,
}

impl StructImpl {
    fn get_field(self: &mut Self, field_offset: usize) -> Value {
        self.values[field_offset].clone()
    }

    fn set_field(self: &mut Self, field_offset: usize, value: Value) {
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

    pub fn as_str(self: &Self) -> String {
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
        if self.step == 0 {
            false
        } else if self.step > 0 {
            value >= self.start && value < self.stop && (value - self.start) % self.step == 0
        } else {
            value <= self.start && value > self.stop && (self.start - value) % self.step == 0
        }
    }

    /// Reverses the range, so that iteration advances from the end to the start
    /// Note this is not as simple as just swapping `start` and `stop`, due to non-unit step sizes.
    pub fn reverse(self) -> RangeImpl {
        if self.step == 0 {
            self
        } else if self.step > 0 {
            RangeImpl { start: self.start + self.len() as i64 * self.step, stop: self.start + 1, step: -self.step }
        } else {
            RangeImpl { start: self.start + self.len() as i64 * self.step, stop: self.start - 1, step: -self.step }
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


#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Clone)]
pub struct SliceImpl {
    arg1: Option<i64>,
    arg2: Option<i64>,
    arg3: Option<i64>
}

impl SliceImpl {
    pub fn apply(self: Self, arg: Value) -> ValueResult {
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
    Unit(Option<Value>),
    Collection(usize, CollectionIterable),
    Range(i64, RangeImpl),
    Enumerate(usize, Box<Iterable>),
}

impl Iterable {
    fn str(string: String) -> Iterable {
        let chars: Chars<'static> = unsafe { mem::transmute(string.chars()) };
        Iterable::Str(string, chars)
    }

    /// Returns the original length of the iterable - not the amount of elements remaining.
    pub fn len(&self) -> usize {
        match &self {
            Iterable::Str(it, _) => it.chars().count(),
            Iterable::Unit(it) => it.is_some() as usize,
            Iterable::Collection(_, it) => it.len(),
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


impl Iterator for Iterable {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iterable::Str(_, chars) => chars.next().map(|u| u.to_value()),
            Iterable::Unit(it) => it.take(),
            Iterable::Collection(index, it) => {
                let ret = it.current(*index);
                *index += 1;
                ret
            }
            Iterable::Range(it, range) => range.next(it),
            Iterable::Enumerate(index, it) => {
                let ret = (*it).next().map(|u| vec![Int(*index as i64), u].to_value());
                *index += 1;
                ret
            },
        }
    }
}

impl Iterator for IterableRev {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            Iterable::Str(_, chars) => chars.next_back().map(|u| u.to_value()),
            Iterable::Unit(it) => it.take(),
            Iterable::Collection(index, it) => {
                if *index < it.len() {
                    let ret = it.current(it.len() - 1 - *index);
                    *index += 1;
                    ret
                } else {
                    None
                }
            }
            Iterable::Range(it, range) => range.next(it),
            Iterable::Enumerate(index, it) => {
                let ret = (*it).next().map(|u| {
                    let vec = vec![Int(*index as i64), u];
                    vec.to_value()
                });
                *index += 1;
                ret
            },
        }
    }
}

impl FusedIterator for Iterable {}
impl FusedIterator for IterableRev {}

// Instead of deriving these, assert that they panic because it should never happen.
impl PartialEq for Iterable { fn eq(&self, _: &Self) -> bool { panic!("Iter() is a synthetic type and should not be =='d"); } }
impl Eq for Iterable {}
impl PartialOrd for Iterable { fn partial_cmp(&self, _: &Self) -> Option<Ordering> { panic!("Iter() is a synthetic type and should not be compared"); } }
impl Ord for Iterable { fn cmp(&self, _: &Self) -> Ordering { panic!("Iter() is a synthetic type and should not be compared"); } }
impl Hash for Iterable { fn hash<H: Hasher>(&self, _: &mut H) { panic!("Iter() is a synthetic type and should not be hashed"); } }


/// A single type for all collection iterators that are indexable by `usize`. Exposes a single common method `current()` which returns the value at the current index, or `None` if the index is longer than the length of the collection.
#[derive(Debug, Clone)]
pub enum CollectionIterable {
    List(Mut<VecDeque<Value>>),
    Set(Mut<SetImpl>),
    Dict(Mut<DictImpl>),
    Vector(Mut<Vec<Value>>),
    RawVector(Vec<Value>),
}

impl CollectionIterable {

    fn len(&self) -> usize {
        match self {
            CollectionIterable::List(it) => it.unbox().len(),
            CollectionIterable::Set(it) => it.unbox().set.len(),
            CollectionIterable::Dict(it) => it.unbox().dict.len(),
            CollectionIterable::Vector(it) => it.unbox().len(),
            CollectionIterable::RawVector(it) => it.len(),
        }
    }

    fn current(&self, index: usize) -> Option<Value> {
        match self {
            CollectionIterable::List(it) => it.unbox().get(index).cloned(),
            CollectionIterable::Set(it) => it.unbox().set.get_index(index).cloned(),
            CollectionIterable::Dict(it) => it.unbox().dict.get_index(index).map(|(l, r)| {
                let vec = vec![l.clone(), r.clone()];
                vec.to_value()
            }),
            CollectionIterable::Vector(it) => it.unbox().get(index).cloned(),
            CollectionIterable::RawVector(it) => it.get(index).cloned(),
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct MemoizedImpl {
    pub func: Value,
    pub cache: Mut<HashMap<Vec<Value>, Value>>
}

impl MemoizedImpl {
    pub fn new(func: Value) -> MemoizedImpl {
        MemoizedImpl { func, cache: Mut::new(HashMap::new()) }
    }
}

impl Hash for MemoizedImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


pub enum Indexable<'a> {
    Str(&'a Rc<String>),
    List(RefMut<'a, VecDeque<Value>>),
    Vector(RefMut<'a, Vec<Value>>),
}

impl<'a> Indexable<'a> {

    pub fn len(self: &Self) -> usize {
        match self {
            Indexable::Str(it) => it.len(),
            Indexable::List(it) => it.len(),
            Indexable::Vector(it) => it.len(),
        }
    }

    /// Takes a convertable-to-int value, representing a bounded index in `[-len, len)`, and converts to a real index in `[0, len)`, or raises an error.
    pub fn check_index(self: &Self, value: Value) -> Result<usize, Box<RuntimeError>> {
        let index: i64 = value.as_int()?;
        let len: usize = self.len();
        let raw: usize = core::to_index(len as i64, index) as usize;
        if raw < len {
            Ok(raw)
        } else {
            ValueErrorIndexOutOfBounds(index, len).err()
        }
    }

    pub fn get_index(self: &Self, index: usize) -> Value {
        match self {
            Indexable::Str(it) => it.chars().nth(index).unwrap().to_value(),
            Indexable::List(it) => (&it[index]).clone(),
            Indexable::Vector(it) => (&it[index]).clone(),
        }
    }

    /// Setting indexes only works for immutable collections - so not strings
    pub fn set_index(self: &mut Self, index: usize, value: Value) -> Result<(), Box<RuntimeError>> {
        match self {
            Indexable::Str(it) => TypeErrorArgMustBeIndexable(Str((*it).clone())).err(),
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

    pub fn len(self: &Self) -> usize {
        match self {
            Sliceable::Str(it, _) => it.len(),
            Sliceable::List(it, _) => it.len(),
            Sliceable::Vector(it, _) => it.len(),
        }
    }

    pub fn accept(self: &mut Self, index: i64) {
        if index >= 0 && index < self.len() as i64 {
            let index = index as usize;
            match self {
                Sliceable::Str(src, dest) => dest.push(src.chars().nth(index).unwrap()),
                Sliceable::List(src, dest) => dest.push_back((&src[index]).clone()),
                Sliceable::Vector(src, dest) => dest.push((&src[index]).clone()),
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
    Set(IndexSet<Value>),
    Dict(IndexMap<Value, Value>),
}

impl Literal {
    pub fn new(op: LiteralType, size_hint: u32) -> Literal {
        match op {
            LiteralType::List => Literal::List(VecDeque::with_capacity(size_hint as usize)),
            LiteralType::Vector => Literal::Vector(Vec::with_capacity(size_hint as usize)),
            LiteralType::Set => Literal::Set(IndexSet::with_capacity(size_hint as usize)),
            LiteralType::Dict => Literal::Dict(IndexMap::with_capacity(size_hint as usize)),
        }
    }

    pub fn accumulate<I : Iterator<Item=Value>>(self: &mut Self, mut iter: I) {
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

    pub fn unroll<I : Iterator<Item=Value>>(self: &mut Self, iter: I) -> Result<(), Box<RuntimeError>> {
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
    use indexmap::{IndexMap, IndexSet};
    use crate::core::NativeFunction;
    use crate::vm::error::RuntimeError;
    use crate::vm::value::{FunctionImpl, IntoIterableValue, IntoValue, Value};

    #[test] fn test_layout() { assert_eq!(16, std::mem::size_of::<Value>()); }
    #[test] fn test_result_box_layout() { assert_eq!(16, std::mem::size_of::<Result<Value, Box<RuntimeError>>>()); }

    #[test]
    fn test_consistency() {
        for v in all_values() {
            assert_eq!(v.is_iter(), v.as_iter().is_ok(), "is_iter() and as_iter() not consistent for {}", v.as_type_str());
            assert_eq!(v.is_iter(), v.len().is_ok(), "is_iter() and len() not consistent for {}", v.as_type_str());
            assert_eq!(v.is_function(), v.unbox_func_args().is_some(), "is_function() and as_function_args() not consistent for {}", v.as_type_str());

            if v.as_index().is_ok() {
                assert!(v.len().is_ok(), "as_index() and len() not consistent for {}", v.as_type_str());
            }
        }
    }

    fn all_values() -> Vec<Value> {
        let function = Rc::new(FunctionImpl::new(0, 0, String::new(), vec![], vec![], false));
        vec![
            Value::Nil,
            Value::Bool(true),
            Value::Int(1),
            VecDeque::new().to_value(),
            IndexSet::new().to_value(),
            IndexMap::new().to_value(),
            std::iter::empty().to_heap(),
            vec![].to_value(),
            Value::range(0, 1, 1).unwrap(),
            Value::Enumerate(Box::new(vec![].to_value())),
            Value::slice(Value::Nil, Value::Int(3), None).unwrap(),
            Value::Function(function.clone()),
            Value::partial(Value::Function(function.clone()), vec![]),
            Value::NativeFunction(NativeFunction::Print),
            Value::PartialNativeFunction(NativeFunction::Print, Box::new(vec![])),
            Value::closure(function.clone())
        ]
    }
}
