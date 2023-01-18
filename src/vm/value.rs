use std::borrow::Borrow;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::Rc;
use std::str::Chars;
use indexmap::{IndexMap, IndexSet};

use crate::stdlib::NativeFunction;
use crate::vm::error::RuntimeError;

use Value::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


/// The runtime sum type used by the virtual machine
/// All `Value` type objects must be cloneable, and so mutable objects must be reference counted to ensure memory safety
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum Value {
    // Primitive (Immutable) Types
    Nil,
    Bool(bool),
    Int(i64),
    Str(Box<String>),

    // Reference (Mutable) Types
    List(Mut<VecDeque<Value>>),
    Set(Mut<SetImpl>),
    Dict(Mut<DictImpl>),
    Heap(Mut<HeapImpl>), // `List` functions as both Array + Deque, but that makes it un-viable for a heap. So, we have a dedicated heap structure
    Vector(Mut<Vec<Value>>), // `Vector` is a fixed-size list (in theory, not in practice), that most operations will behave elementwise on

    // Iterator Types (Immutable)
    Range(Box<RangeImpl>),

    /// ### Enumerate Type
    ///
    /// This is the type used by the native function `enumerate(...)`. It does not have any additional functionality and is just a wrapper around an internal `Value`.
    ///
    /// Note that `enumerate()` object needs to be stateless, hence wrapping a `Value`, and not an `IteratorImpl`. When a `enumerate()` is iterated through, i.e. `is_iter()` is invoked on it, the internal value will be converted to the respective iterator at that time.
    Enumerate(Box<Value>),

    /// Synthetic Iterator Type - Mutable, but not aliasable.
    /// This will never be user-code-accessible, as it will only be on the stack as a synthetic variable, or in native code.
    Iter(Box<Iterable>),

    // Functions
    Function(Rc<FunctionImpl>),
    PartialFunction(Box<PartialFunctionImpl>),
    NativeFunction(NativeFunction),
    PartialNativeFunction(NativeFunction, Box<Vec<Box<Value>>>),
    Closure(Box<ClosureImpl>),
}


impl Value {

    // Constructors
    pub fn iter_list(iter: impl Iterator<Item=Value>) -> Value { Value::list(iter.collect()) }
    pub fn iter_set(iter: impl Iterator<Item=Value>) -> Value { Value::set(iter.collect()) }
    pub fn iter_dict(iter: impl Iterator<Item=(Value, Value)>) -> Value { Value::dict(iter.collect()) }
    pub fn iter_heap(iter: impl Iterator<Item=Value>) -> Value { Heap(Mut::new(HeapImpl::new(iter.map(|t| Reverse(t)).collect::<BinaryHeap<Reverse<Value>>>()))) }
    pub fn iter_vector(iter: impl Iterator<Item=Value>) -> Value { Value::vector(iter.collect()) }

    pub fn str(c: char) -> Value { Str(Box::new(String::from(c))) }
    pub fn list(vec: VecDeque<Value>) -> Value { List(Mut::new(vec.into_iter().collect::<VecDeque<Value>>())) }
    pub fn set(set: IndexSet<Value>) -> Value { Set(Mut::new(SetImpl::new(set))) }
    pub fn dict(dict: IndexMap<Value, Value>) -> Value { Dict(Mut::new(DictImpl::new(dict))) }
    pub fn heap(heap: BinaryHeap<Reverse<Value>>) -> Value { Heap(Mut::new(HeapImpl::new(heap))) }
    pub fn vector(vec: Vec<Value>) -> Value { Vector(Mut::new(vec)) }

    pub fn partial(func: Value, args: Vec<Value>) -> Value { PartialFunction(Box::new(PartialFunctionImpl { func, args: args.into_iter().map(|v| Box::new(v)).collect() }))}

    pub fn closure(func: Rc<FunctionImpl>) -> Value { Closure(Box::new(ClosureImpl { func, environment: Vec::new() })) }

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn to_str(self: &Self) -> String {
        match self {
            Str(s) => *s.clone(),
            Function(f) => f.name.clone(),
            PartialFunction(f) => f.func.to_str(),
            NativeFunction(b) => String::from(b.name()),
            PartialNativeFunction(b, _) => String::from(b.name()),
            _ => self.to_repr_str(),
        }
    }

    /// Converts the `Value` to a representative `String. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn to_repr_str(self: &Self) -> String {
        match self {
            Nil => String::from("nil"),
            Bool(b) => b.to_string(),
            Int(i) => i.to_string(),
            Str(s) => {
                let escaped = format!("{:?}", s);
                format!("'{}'", &escaped[1..escaped.len() - 1])
            },

            List(v) => format!("[{}]", v.unbox().iter().map(|t| t.to_repr_str()).collect::<Vec<String>>().join(", ")),
            Set(v) => format!("{{{}}}", v.unbox().set.iter().map(|t| t.to_repr_str()).collect::<Vec<String>>().join(", ")),
            Dict(v) => format!("{{{}}}", v.unbox().dict.iter().map(|(k, v)| format!("{}: {}", k.to_repr_str(), v.to_repr_str())).collect::<Vec<String>>().join(", ")),
            Heap(v) => format!("[{}]", v.unbox().heap.iter().map(|t| t.0.to_repr_str()).collect::<Vec<String>>().join(", ")),
            Vector(v) => format!("({})", v.unbox().iter().map(|t| t.to_repr_str()).collect::<Vec<String>>().join(", ")),

            Range(r) => if r.step == 0 { String::from("range(empty)") } else { format!("range({}, {}, {})", r.start, r.stop, r.step) },
            Enumerate(v) => format!("enumerate({})", v.to_repr_str()),

            Iter(_) => format!("iterator"),

            Function(f) => (*f).as_ref().borrow().as_str(),
            PartialFunction(f) => (*f).as_ref().borrow().func.to_str(),
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
            Str(_) => "str",
            List(_) => "list",
            Set(_) => "set",
            Dict(_) => "dict",
            Heap(_) => "heap",
            Vector(_) => "vector",
            Range(_) => "range",
            Enumerate(_) => "enumerate",
            Iter(_) => "iterator",
            Function(_) => "function",
            PartialFunction(_) => "partial function",
            NativeFunction(_) => "native function",
            PartialNativeFunction(_, _) => "partial native function",
            Closure(_) => "closure",
        })
    }

    pub fn as_debug_str(self: &Self) -> String {
        format!("{}: {}", self.to_repr_str(), self.as_type_str())
    }

    pub fn as_bool(self: &Self) -> bool {
        match self {
            Nil => false,
            Bool(it) => *it,
            Int(it) => *it != 0,
            Str(it) => !it.is_empty(),
            List(it) => !it.unbox().is_empty(),
            Set(it) => !it.unbox().set.is_empty(),
            Dict(it) => !it.unbox().dict.is_empty(),
            Heap(it) => !it.unbox().heap.is_empty(),
            Range(it) => !it.is_empty(),
            Enumerate(it) => (**it).as_bool(),
            Iter(_) => panic!("Iter() is a synthetic type and should not have as_bool() invoked on it"),
            Vector(v) => v.unbox().is_empty(),
            Function(_) | PartialFunction(_) | NativeFunction(_) | PartialNativeFunction(_, _) | Closure(_) => true,
        }
    }

    /// Unwraps the value as an `int`, or raises a type error
    pub fn as_int(self: &Self) -> Result<i64, Box<RuntimeError>> {
        match self {
            Int(i) => Ok(*i),
            Bool(b) => Ok(if *b { 1 } else { 0 }),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    /// Like `as_int()` but converts `nil` to the provided default value
    pub fn as_int_or(self: &Self, def: i64) -> Result<i64, Box<RuntimeError>> {
        match self {
            Nil => Ok(def),
            Int(i) => Ok(*i),
            Bool(b) => Ok(if *b { 1 } else { 0 }),
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
            List(it) => Ok(Indexable::List(it.unbox())),
            Vector(it) => Ok(Indexable::Vector(it.unbox())),
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
    pub fn unbox_func_args(self: &Self) -> Option<Option<u8>> {
        match self {
            Function(it) => Some(Some(it.nargs)),
            PartialFunction(it) => Some(Some(it.func.unbox_func().nargs - it.args.len() as u8)),
            NativeFunction(it) => Some(it.nargs()),
            PartialNativeFunction(it, args) => Some(it.nargs().map(|u| u - args.len() as u8)),
            Closure(it) => Some(Some(it.func.nargs)),
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

    pub fn is_nil(self: &Self) -> bool { match self { Nil => true, _ => false } }
    pub fn is_bool(self: &Self) -> bool { match self { Bool(_) => true, _ => false } }
    pub fn is_int(self: &Self) -> bool { match self { Int(_) => true, _ => false } }
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

    pub fn is_function(self: &Self) -> bool {
        match self {
            Function(_) | PartialFunction(_) | NativeFunction(_) | PartialNativeFunction(_, _) | Closure(_) => true,
            _ => false
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


/// `Mut<T>` is a wrapper around internally mutable types. It implements the required traits for `Value` through it's inner type.
/// Note that it also implements `Hash`, even though the internal type is mutable. This is required to satisfy rust's type system.
/// Mutating values stored in a hash backed structure is legal, from a language point of view, but will just invoke undefined behavior.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Mut<T : Eq + PartialEq + Debug + Clone + Hash>(Rc<RefCell<T>>);

impl<T : Eq + PartialEq + Debug + Clone + Hash> Mut<T> {

    pub fn new(value: T) -> Mut<T> {
        Mut(Rc::new(RefCell::new(value)))
    }

    pub fn unbox(&self) -> Ref<T> {
        (*self.0).borrow()
    }

    pub fn unbox_mut(&self) -> RefMut<T> {
        (*self.0).borrow_mut()
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
    pub nargs: u8, // The number of arguments the function takes

    name: String, // The name of the function, useful to show in stack traces
    args: Vec<String>, // Names of the arguments
}

impl FunctionImpl {
    pub fn new(head: usize, name: String, args: Vec<String>) -> FunctionImpl {
        FunctionImpl {
            head,
            tail: head + 1,
            nargs: args.len() as u8,
            name,
            args
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
    pub args: Vec<Box<Value>>,
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

impl SetImpl {
    fn new(set: IndexSet<Value>) -> SetImpl { SetImpl { set } }
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

impl Hash for SetImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.set {
            v.hash(state)
        }
    }
}


/// Boxes a `IndexMap<Value, Value>`, along with an optional default value
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct DictImpl {
    pub dict: IndexMap<Value, Value>,
    pub default: Option<Value>
}

impl DictImpl {
    fn new(dict: IndexMap<Value, Value>) -> DictImpl { DictImpl { dict, default: None }}
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

impl Hash for DictImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.dict {
            v.hash(state)
        }
    }
}


/// As `BinaryHeap` is missing `Eq`, `PartialEq`, and `Hash` implementations
/// We also wrap values in `Reverse` as we want to expose a min-heap by default
#[derive(Debug, Clone)]
pub struct HeapImpl {
    pub heap: BinaryHeap<Reverse<Value>>
}

impl HeapImpl {
    pub fn new(heap: BinaryHeap<Reverse<Value>>) -> HeapImpl {
        HeapImpl { heap }
    }
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
    /// Creates a new `Range()` value from a given set of integer parameters.
    /// Raises an error if `step == 0`
    ///
    /// Note: this implementation internally replaces all empty range values with the single `range(0, 0, 0)` instance. This means that `range(1, 2, -1) . str` will have to handle this case as it will not be representative.
    pub fn new(start: i64, stop: i64, step: i64) -> ValueResult {
        if step == 0 {
            ValueErrorStepCannotBeZero.err()
        } else if (stop > start && step > 0) || (stop < start && step < 0) {
            Ok(Range(Box::new(RangeImpl { start, stop, step }))) // Non-empty range
        } else {
            Ok(Range(Box::new(RangeImpl { start: 0, stop: 0, step: 0 }))) // Empty range
        }
    }

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




/// ### Iterator Type
///
/// Iterators are complex to model within the type system of Rust, with the restrictions imposed by Cordy:
///
/// - Rust `Iterator` methods access `Mut` values which enforce that their iterator only lives as long as the borrow from `Ref<'a, T>`. This is unusable as our iterators, i.e. in `for` loops, need to live on the stack.
/// - In native code, the borrow on the inner value must last for as long as the loop is ran, which means native functions like `map` essentially acquire a lock on their value, preventing mutation. For example the following code:
///
/// ```
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
/// ```
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
            Iterable::Unit(it) => if it.is_some() { 1 } else { 0 },
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
            Iterable::Str(_, chars) => chars.next().map(|u| Value::str(u)),
            Iterable::Unit(it) => it.take(),
            Iterable::Collection(index, it) => {
                let ret = it.current(*index);
                *index += 1;
                ret
            }
            Iterable::Range(it, range) => range.next(it),
            Iterable::Enumerate(index, it) => {
                let ret = (*it).next().map(|u| Value::vector(vec![Int(*index as i64), u]));
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
            Iterable::Str(_, chars) => chars.next_back().map(|u| Value::str(u)),
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
                let ret = (*it).next().map(|u| Value::vector(vec![Int(*index as i64), u]));
                *index += 1;
                ret
            },
        }
    }
}

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
            CollectionIterable::Dict(it) => it.unbox().dict.get_index(index).map(|(l, r)| Value::vector(vec![l.clone(), r.clone()])),
            CollectionIterable::Vector(it) => it.unbox().get(index).cloned(),
            CollectionIterable::RawVector(it) => it.get(index).cloned(),
        }
    }
}


pub enum Indexable<'a> {
    Str(&'a Box<String>),
    List(Ref<'a, VecDeque<Value>>),
    Vector(Ref<'a, Vec<Value>>),
}

impl<'a> Indexable<'a> {

    pub fn len(self: &Self) -> usize {
        match self {
            Indexable::Str(it) => it.len(),
            Indexable::List(it) => it.len(),
            Indexable::Vector(it) => it.len(),
        }
    }

    pub fn get_index(self: &Self, index: usize) -> Value {
        match self {
            Indexable::Str(it) => Value::str(it.chars().nth(index).unwrap()),
            Indexable::List(it) => (&it[index]).clone(),
            Indexable::Vector(it) => (&it[index]).clone(),
        }
    }
}


pub enum Sliceable<'a> {
    Str(&'a Box<String>, String),
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

    pub fn to_value(self: Self) -> Value {
        match self {
            Sliceable::Str(_, it) => Str(Box::new(it)),
            Sliceable::List(_, it) => List(Mut::new(it)),
            Sliceable::Vector(_, it) => Vector(Mut::new(it)),
        }
    }
}


#[cfg(test)]
mod test {
    use std::collections::VecDeque;
    use std::rc::Rc;
    use indexmap::{IndexMap, IndexSet};
    use crate::stdlib::NativeFunction;
    use crate::vm::error::RuntimeError;
    use crate::vm::value::{FunctionImpl, RangeImpl, Value};

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
        let rc = Rc::new(FunctionImpl::new(0, String::new(), vec![]));
        vec![
            Value::Nil,
            Value::Bool(true),
            Value::Int(1),
            Value::list(VecDeque::new()),
            Value::set(IndexSet::new()),
            Value::dict(IndexMap::new()),
            Value::iter_heap(std::iter::empty()),
            Value::vector(vec![]),
            RangeImpl::new(0, 1, 1).unwrap(),
            Value::Enumerate(Box::new(Value::vector(vec![]))),
            Value::Function(rc.clone()),
            Value::partial(Value::Function(rc.clone()), vec![]),
            Value::NativeFunction(NativeFunction::Print),
            Value::PartialNativeFunction(NativeFunction::Print, Box::new(vec![])),
            Value::closure(rc.clone())
        ]
    }
}
