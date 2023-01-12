use std::borrow::Borrow;
use std::cell::{Ref, RefCell, RefMut};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use hashlink::{LinkedHashMap, LinkedHashSet};

use crate::stdlib;
use crate::stdlib::StdBinding;
use crate::vm::error::RuntimeError;

use Value::{*};
use RuntimeError::{TypeErrorArgMustBeIterable, TypeErrorArgMustBeIndexable, TypeErrorArgMustBeSliceable, TypeErrorArgMustBeInt, TypeErrorArgMustBeStr};


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
    // Memory is not really managed at all and cycles are entirely possible.
    // In future, a GC'd system could be implemented with `Mut` only owning weak references, and the GC owning the only strong reference.
    // But that comes at even more performance overhead because we'd need to still obey rust reference counting semantics.
    List(Mut<VecDeque<Value>>),
    Set(Mut<LinkedHashSet<Value>>),
    Dict(Mut<DictImpl>),
    Heap(Mut<HeapImpl>), // `List` functions as both Array + Deque, but that makes it un-viable for a heap. So, we have a dedicated heap structure
    Vector(Mut<Vec<Value>>), // `Vector` is a fixed-size list (in theory, not in practice), that most operations will behave elementwise on

    // Functions
    Function(Rc<FunctionImpl>),
    PartialFunction(Box<PartialFunctionImpl>),
    NativeFunction(StdBinding),
    PartialNativeFunction(StdBinding, Box<Vec<Box<Value>>>),
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
    pub fn set(set: LinkedHashSet<Value>) -> Value { Set(Mut::new(set)) }
    pub fn dict(dict: LinkedHashMap<Value, Value>) -> Value { Dict(Mut::new(DictImpl::new(dict))) }
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
            NativeFunction(b) => String::from(stdlib::lookup_name(*b)),
            PartialNativeFunction(b, _) => String::from(stdlib::lookup_name(*b)),
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
            Set(v) => format!("{{{}}}", v.unbox().iter().map(|t| t.to_repr_str()).collect::<Vec<String>>().join(", ")),
            Dict(v) => format!("{{{}}}", v.unbox().dict.iter().map(|(k, v)| format!("{}: {}", k.to_repr_str(), v.to_repr_str())).collect::<Vec<String>>().join(", ")),
            Heap(v) => format!("[{}]", v.unbox().heap.iter().map(|t| t.0.to_repr_str()).collect::<Vec<String>>().join(", ")),
            Vector(v) => format!("({})", v.unbox().iter().map(|t| t.to_repr_str()).collect::<Vec<String>>().join(", ")),
            Function(f) => (*f).as_ref().borrow().as_str(),
            PartialFunction(f) => (*f).as_ref().borrow().func.to_str(),
            NativeFunction(b) => format!("fn {}()", stdlib::lookup_name(*b)),
            PartialNativeFunction(b, _) => format!("fn {}()", stdlib::lookup_name(*b)),
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
            Bool(b) => *b,
            Int(i) => *i != 0,
            Str(s) => !s.is_empty(),
            List(l) => !l.unbox().is_empty(),
            Set(v) => !v.unbox().is_empty(),
            Dict(v) => !v.unbox().dict.is_empty(),
            Heap(v) => !v.unbox().heap.is_empty(),
            Vector(v) => v.unbox().is_empty(),
            Function(_) | PartialFunction(_) | NativeFunction(_) | PartialNativeFunction(_, _) | Closure(_) => true,
        }
    }

    pub fn as_int(self: &Self) -> Result<i64, Box<RuntimeError>> {
        match self {
            Int(i) => Ok(*i),
            Bool(b) => Ok(if *b { 1 } else { 0 }),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    /// Like `as_int()` but converts `Nil` to the provided default value
    pub fn as_int_or(self: &Self, def: i64) -> Result<i64, Box<RuntimeError>> {
        match self {
            Nil => Ok(def),
            Int(i) => Ok(*i),
            Bool(b) => Ok(if *b { 1 } else { 0 }),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    pub fn as_str(self: &Self) -> Result<&String, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(it),
            v => TypeErrorArgMustBeStr(v.clone()).err()
        }
    }

    pub fn as_iter(self: &Self) -> Result<ValueIntoIter, Box<RuntimeError>> {
        match self {
            Str(it) => {
                let chars: Vec<Value> = it.chars()
                    .map(|c| Value::str(c))
                    .collect::<Vec<Value>>();
                Ok(ValueIntoIter::Str(chars))
            }
            List(it) => Ok(ValueIntoIter::List(it.unbox())),
            Set(it) => Ok(ValueIntoIter::Set(it.unbox())),
            Dict(it) => Ok(ValueIntoIter::Dict(it.unbox())),
            Heap(it) => Ok(ValueIntoIter::Heap(it.unbox())),
            Vector(it) => Ok(ValueIntoIter::Vector(it.unbox())),
            _ => TypeErrorArgMustBeIterable(self.clone()).err(),
        }
    }

    /// Converts this `Value` to a `ValueAsIndex`, which is a index-able object, supported for `List`, `Vector`, and `Str`
    pub fn to_index(self: &Self) -> Result<ValueAsIndex, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(ValueAsIndex::Str(it)),
            List(it) => Ok(ValueAsIndex::List(it.unbox())),
            Vector(it) => Ok(ValueAsIndex::Vector(it.unbox())),
            _ => TypeErrorArgMustBeIndexable(self.clone()).err()
        }
    }

    /// Converts this `Value` to a `ValueAsSlice`, which is a builder for slice-like structures, supported for `List` and `Str`
    pub fn to_slice(self: &Self) -> Result<ValueAsSlice, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(ValueAsSlice::Str(it, String::new())),
            List(it) => Ok(ValueAsSlice::List(it.unbox(), VecDeque::new())),
            Vector(it) => Ok(ValueAsSlice::Vector(it.unbox(), Vec::new())),
            _ => TypeErrorArgMustBeSliceable(self.clone()).err()
        }
    }

    /// Returns the internal `FunctionImpl` of this value.
    /// Must only be called on a `Function` or `Closure`, will panic otherwise.
    pub fn function_impl(self: &Self) -> &Rc<FunctionImpl> {
        match self {
            Function(f) => f,
            Closure(c) => &c.func,
            _ => panic!("Tried to unwrap a {:?} as a function", self),
        }
    }

    /// Returns `None` if this value is not a function
    /// Returns `Some(None)` if this value is a function with an unknown number of arguments
    /// Returns `Some(Some(nargs))` if this value is a function with a known number of arguments
    pub fn as_function_args(self: &Self) -> Option<Option<u8>> {
        match self {
            Function(it) => Some(Some(it.nargs)),
            PartialFunction(it) => Some(Some(it.func.function_impl().nargs - it.args.len() as u8)),
            NativeFunction(it) => Some(it.nargs()),
            PartialNativeFunction(it, args) => Some(it.nargs().map(|u| u - args.len() as u8)),
            Closure(it) => Some(Some(it.func.nargs)),
            _ => None,
        }
    }

    pub fn len(self: &Self) -> Result<usize, Box<RuntimeError>> {
        match &self {
            Str(it) => Ok(it.chars().count()),
            List(it) => Ok(it.unbox().len()),
            Set(it) => Ok(it.unbox().len()),
            Dict(it) => Ok(it.unbox().dict.len()),
            Heap(it) => Ok(it.unbox().heap.len()),
            Vector(it) => Ok(it.unbox().len()),
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
    pub fn is_vector(self: &Self) -> bool { match self { Vector(_) => true, _ => false }}

    pub fn is_iter(self: &Self) -> bool {
        match self {
            Str(_) | List(_) | Set(_) | Dict(_) | Heap(_) | Vector(_) => true,
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


/// Wrapper type to allow mutable types to hash
/// Yes, mutable types in hash maps is dangerous but this is a language about shooting yourself in the foot
/// So why would we not allow this?
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Mut<T : Eq + PartialEq + Debug + Clone + Hash> {
    value: Rc<RefCell<T>>
}

impl<T : Eq + PartialEq + Debug + Clone + Hash> Mut<T> {

    pub fn new(value: T) -> Mut<T> {
        Mut { value: Rc::new(RefCell::new(value)) }
    }

    pub fn unbox(self: &Self) -> Ref<T> {
        (*self.value).borrow()
    }

    pub fn unbox_mut(self: &Self) -> RefMut<T> {
        (*self.value).borrow_mut()
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
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ClosureImpl {
    pub func: Rc<FunctionImpl>,
    pub environment: Vec<Value>,
}

impl Hash for ClosureImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


/// Boxes a `LinkedHashMap<Value, Value>`, along with an optional default value
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Hash)]
pub struct DictImpl {
    pub dict: LinkedHashMap<Value, Value>,
    pub default: Option<Value>
}

impl DictImpl {
    fn new(dict: LinkedHashMap<Value, Value>) -> DictImpl { DictImpl { dict, default: None }}
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

// Escapes iterators all having different concrete types
pub enum ValueIter<'a> {
    Str(std::slice::Iter<'a, Value>),
    List(std::collections::vec_deque::Iter<'a, Value>),
    Set(hashlink::linked_hash_set::Iter<'a, Value>),
    Dict(hashlink::linked_hash_map::Keys<'a, Value, Value>),
    Heap(std::collections::binary_heap::Iter<'a, Reverse<Value>>),
    Vector(std::slice::Iter<'a, Value>),
}

impl<'a> Iterator for ValueIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ValueIter::Str(it) => it.next(),
            ValueIter::List(it) => it.next(),
            ValueIter::Set(it) => it.next(),
            ValueIter::Dict(it) => it.next(),
            ValueIter::Heap(it) => it.next().map(|u| &u.0),
            ValueIter::Vector(it) => it.next(),
        }
    }
}

impl<'a> DoubleEndedIterator for ValueIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            ValueIter::Str(it) => it.next_back(),
            ValueIter::List(it) => it.next_back(),
            ValueIter::Set(it) => it.next_back(),
            ValueIter::Dict(it) => it.next_back(),
            ValueIter::Heap(it) => it.next_back().map(|u| &u.0),
            ValueIter::Vector(it) => it.next_back(),
        }
    }
}

// Escapes the interior mutability pattern
pub enum ValueIntoIter<'a> {
    Str(Vec<Value>),
    List(Ref<'a, VecDeque<Value>>),
    Set(Ref<'a, LinkedHashSet<Value>>),
    Dict(Ref<'a, DictImpl>),
    Heap(Ref<'a, HeapImpl>),
    Vector(Ref<'a, Vec<Value>>),
}

impl<'a> ValueIntoIter<'a> {
    pub fn len(self: &Self) -> usize {
        match self {
            ValueIntoIter::Str(it) => it.len(),
            ValueIntoIter::List(it) => it.len(),
            ValueIntoIter::Set(it) => it.len(),
            ValueIntoIter::Dict(it) => it.dict.len(),
            ValueIntoIter::Heap(it) => it.heap.len(),
            ValueIntoIter::Vector(it) => it.len(),
        }
    }
}

impl<'b: 'a, 'a> IntoIterator for &'b ValueIntoIter<'a> {
    type Item = &'a Value;
    type IntoIter = ValueIter<'a>;

    fn into_iter(self) -> ValueIter<'a> {
        match self {
            ValueIntoIter::Str(it) => ValueIter::Str(it.into_iter()),
            ValueIntoIter::List(it) => ValueIter::List(it.iter()),
            ValueIntoIter::Set(it) => ValueIter::Set(it.iter()),
            ValueIntoIter::Dict(it) => ValueIter::Dict(it.dict.keys()),
            ValueIntoIter::Heap(it) => ValueIter::Heap(it.heap.iter()),
            ValueIntoIter::Vector(it) => ValueIter::Vector(it.iter()),
        }
    }
}


pub enum ValueAsIndex<'a> {
    Str(&'a Box<String>),
    List(Ref<'a, VecDeque<Value>>),
    Vector(Ref<'a, Vec<Value>>),
}

impl<'a> ValueAsIndex<'a> {

    pub fn len(self: &Self) -> usize {
        match self {
            ValueAsIndex::Str(it) => it.len(),
            ValueAsIndex::List(it) => it.len(),
            ValueAsIndex::Vector(it) => it.len(),
        }
    }

    pub fn get_index(self: &Self, index: usize) -> Value {
        match self {
            ValueAsIndex::Str(it) => Value::str(it.chars().nth(index).unwrap()),
            ValueAsIndex::List(it) => (&it[index]).clone(),
            ValueAsIndex::Vector(it) => (&it[index]).clone(),
        }
    }
}


pub enum ValueAsSlice<'a> {
    Str(&'a Box<String>, String),
    List(Ref<'a, VecDeque<Value>>, VecDeque<Value>),
    Vector(Ref<'a, Vec<Value>>, Vec<Value>),
}

impl<'a> ValueAsSlice<'a> {

    pub fn len(self: &Self) -> usize {
        match self {
            ValueAsSlice::Str(it, _) => it.len(),
            ValueAsSlice::List(it, _) => it.len(),
            ValueAsSlice::Vector(it, _) => it.len(),
        }
    }

    pub fn accept(self: &mut Self, index: i64) {
        if index >= 0 && index < self.len() as i64 {
            let index = index as usize;
            match self {
                ValueAsSlice::Str(src, dest) => dest.push(src.chars().nth(index).unwrap()),
                ValueAsSlice::List(src, dest) => dest.push_back((&src[index]).clone()),
                ValueAsSlice::Vector(src, dest) => dest.push((&src[index]).clone()),
            }
        }
    }

    pub fn to_value(self: Self) -> Value {
        match self {
            ValueAsSlice::Str(_, it) => Str(Box::new(it)),
            ValueAsSlice::List(_, it) => List(Mut::new(it)),
            ValueAsSlice::Vector(_, it) => Vector(Mut::new(it)),
        }
    }
}


#[cfg(test)]
mod test {
    use std::collections::VecDeque;
    use std::rc::Rc;
    use hashlink::{LinkedHashMap, LinkedHashSet};
    use crate::stdlib::StdBinding;
    use crate::vm::error::RuntimeError;
    use crate::vm::value::{FunctionImpl, Value};

    #[test] fn test_layout() { assert_eq!(16, std::mem::size_of::<Value>()); }
    #[test] fn test_result_box_layout() { assert_eq!(16, std::mem::size_of::<Result<Value, Box<RuntimeError>>>()); }

    #[test]
    fn test_consistency() {
        for v in all_values() {
            assert_eq!(v.is_iter(), v.as_iter().is_ok(), "is_iter() and as_iter() not consistent for {}", v.as_type_str());
            assert_eq!(v.is_iter(), v.len().is_ok(), "is_iter() and len() not consistent for {}", v.as_type_str());
            assert_eq!(v.is_function(), v.as_function_args().is_some(), "is_function() and as_function_args() not consistent for {}", v.as_type_str());

            if v.to_index().is_ok() {
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
            Value::set(LinkedHashSet::new()),
            Value::dict(LinkedHashMap::new()),
            Value::iter_heap(std::iter::empty()),
            Value::vector(vec![]),
            Value::Function(rc.clone()),
            Value::partial(Value::Function(rc.clone()), vec![]),
            Value::NativeFunction(StdBinding::Void),
            Value::PartialNativeFunction(StdBinding::Void, Box::new(vec![])),
            Value::closure(rc.clone())
        ]
    }
}
