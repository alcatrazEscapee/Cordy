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
use RuntimeError::{TypeErrorArgMustBeIterable, TypeErrorArgMustBeSliceable, TypeErrorArgMustBeInt};


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
    Dict(Mut<LinkedHashMap<Value, Value>>),
    Heap(Mut<ValueHeap>), // `List` functions as both Array + Deque, but that makes it un-viable for a heap. So, we have a dedicated heap structure

    // Functions
    Function(Rc<FunctionImpl>),
    PartialFunction(Box<PartialFunctionImpl>),
    NativeFunction(StdBinding),
    PartialNativeFunction(StdBinding, Box<Vec<Box<Value>>>),
    Closure(Box<ClosureImpl>),
}


impl Value {

    // Constructors
    pub fn iter_list(vec: impl Iterator<Item=Value>) -> Value { List(Mut::new(vec.collect::<VecDeque<Value>>())) }
    pub fn iter_set(vec: impl Iterator<Item=Value>) -> Value { Set(Mut::new(vec.collect::<LinkedHashSet<Value>>())) }
    pub fn iter_heap(vec: impl Iterator<Item=Value>) -> Value { Heap(Mut::new(ValueHeap::new(vec.map(|t| Reverse(t)).collect::<BinaryHeap<Reverse<Value>>>()))) }

    pub fn str(c: char) -> Value { Str(Box::new(String::from(c))) }
    pub fn list(vec: Vec<Value>) -> Value { List(Mut::new(vec.into_iter().collect::<VecDeque<Value>>())) }
    pub fn set(set: LinkedHashSet<Value>) -> Value { Set(Mut::new(set)) }
    pub fn dict(dict: LinkedHashMap<Value, Value>) -> Value { Dict(Mut::new(dict)) }

    pub fn partial(func: Value, args: Vec<Value>) -> Value { PartialFunction(Box::new(PartialFunctionImpl { func, args: args.into_iter().map(|v| Box::new(v)).collect() }))}

    pub fn closure(func: Rc<FunctionImpl>) -> Value { Closure(Box::new(ClosureImpl { func, environment: Vec::new() })) }

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn as_str(self: &Self) -> String {
        match self {
            Str(s) => *s.clone(),
            Function(f) => f.name.clone(),
            PartialFunction(f) => f.func.as_str(),
            NativeFunction(b) => String::from(stdlib::lookup_name(*b)),
            PartialNativeFunction(b, _) => String::from(stdlib::lookup_name(*b)),
            _ => self.as_repr_str(),
        }
    }

    /// Converts the `Value` to a representative `String. This is equivalent to the stdlib function `repr()`, and meant to be an inverse of `eval()`
    pub fn as_repr_str(self: &Self) -> String {
        match self {
            Nil => String::from("nil"),
            Bool(b) => b.to_string(),
            Int(i) => i.to_string(),
            Str(s) => {
                let escaped = format!("{:?}", s);
                format!("'{}'", &escaped[1..escaped.len() - 1])
            },
            List(v) => format!("[{}]", v.unbox().iter().map(|t| t.as_repr_str()).collect::<Vec<String>>().join(", ")),
            Set(v) => format!("{{{}}}", v.unbox().iter().map(|t| t.as_repr_str()).collect::<Vec<String>>().join(", ")),
            Dict(v) => format!("{{{}}}", v.unbox().iter().map(|(k, v)| format!("{}: {}", k.as_repr_str(), v.as_repr_str())).collect::<Vec<String>>().join(", ")),
            Heap(v) => format!("[{}]", v.unbox().heap.iter().map(|t| t.0.as_repr_str()).collect::<Vec<String>>().join(", ")),
            Function(f) => (*f).as_ref().borrow().as_str(),
            PartialFunction(f) => (*f).as_ref().borrow().func.as_str(),
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
            Function(_) => "function",
            PartialFunction(_) => "partial function",
            NativeFunction(_) => "native function",
            PartialNativeFunction(_, _) => "partial native function",
            Closure(_) => "closure",
        })
    }

    #[cfg(any(trace_interpreter = "on", trace_interpreter_stack = "on"))]
    pub fn as_debug_str(self: &Self) -> String {
        format!("{}: {}", self.as_repr_str(), self.as_type_str())
    }

    pub fn as_bool(self: &Self) -> bool {
        match self {
            Nil => false,
            Bool(b) => *b,
            Int(i) => *i != 0,
            Str(s) => !s.is_empty(),
            List(l) => !l.unbox().is_empty(),
            Set(v) => !v.unbox().is_empty(),
            Dict(v) => !v.unbox().is_empty(),
            Heap(v) => !v.unbox().heap.is_empty(),
            Function(_) | PartialFunction(_) | NativeFunction(_) | PartialNativeFunction(_, _) | Closure(_) => true,
        }
    }

    pub fn as_int(self: &Self) -> Result<i64, Box<RuntimeError>> {
        match self {
            Int(i) => Ok(*i),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    /// Like `as_int()` but converts `Nil` to the provided default value
    pub fn as_int_or(self: &Self, def: i64) -> Result<i64, Box<RuntimeError>> {
        match self {
            Nil => Ok(def),
            Int(i) => Ok(*i),
            _ => TypeErrorArgMustBeInt(self.clone()).err(),
        }
    }

    pub fn as_iter(self: &Self) -> Result<ValueIntoIter, Box<RuntimeError>> {
        match self {
            Str(s) => {
                let chars: Vec<Value> = s.chars()
                    .map(|c| Value::str(c))
                    .collect::<Vec<Value>>();
                Ok(ValueIntoIter::Str(chars))
            },
            List(l) => Ok(ValueIntoIter::List(l.unbox())),
            Set(s) => Ok(ValueIntoIter::Set(s.unbox())),
            Dict(d) => Ok(ValueIntoIter::Dict(d.unbox())),
            Heap(v) => Ok(ValueIntoIter::Heap(v.unbox())),
            _ => TypeErrorArgMustBeIterable(self.clone()).err(),
        }
    }

    /// Converts this `Value` to a `ValueAsSlice`, which is a builder for slice-like structures, supported for `List` and `Str`
    pub fn to_slice(self: &Self) -> Result<ValueAsSlice, Box<RuntimeError>> {
        match self {
            Str(it) => Ok(ValueAsSlice::Str(it, String::new())),
            List(it) => Ok(ValueAsSlice::List(it.unbox(), VecDeque::new())),
            _ => TypeErrorArgMustBeSliceable(self.clone()).err()
        }
    }

    /// Returns the internal `FunctionImpl` of this value.
    /// Must only be called on a `Function` or `Closure`, will panic otherwise
    pub fn as_function(self: &Self) -> &Rc<FunctionImpl> {
        match self {
            Function(f) => f,
            Closure(c) => &c.func,
            _ => panic!("Tried to unwrap a {:?} as a function", self),
        }
    }

    pub fn len(self: &Self) -> Result<usize, Box<RuntimeError>> {
        match &self {
            Str(it) => Ok(it.len()),
            List(it) => Ok(it.unbox().len()),
            Set(it) => Ok(it.unbox().len()),
            Dict(it) => Ok(it.unbox().len()),
            Heap(it) => Ok(it.unbox().heap.len()),
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

    pub fn is_iter(self: &Self) -> bool {
        match self {
            Str(_) | List(_) | Set(_) | Dict(_) | Heap(_) => true,
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

impl PartialFunctionImpl {

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


/// As `BinaryHeap` is missing `Eq`, `PartialEq`, and `Hash` implementations
/// We also wrap values in `Reverse` as we want to expose a min-heap by default
#[derive(Debug, Clone)]
pub struct ValueHeap {
    pub heap: BinaryHeap<Reverse<Value>>
}

impl ValueHeap {
    pub fn new(heap: BinaryHeap<Reverse<Value>>) -> ValueHeap {
        ValueHeap { heap }
    }
}

impl PartialEq<Self> for ValueHeap {
    fn eq(&self, other: &Self) -> bool {
        self.heap.len() == other.heap.len() && self.heap.iter().zip(other.heap.iter()).all(|(x, y)| x == y)
    }
}

impl Eq for ValueHeap {}

impl Hash for ValueHeap {
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
        }
    }
}

// Escapes the interior mutability pattern
pub enum ValueIntoIter<'a> {
    Str(Vec<Value>),
    List(Ref<'a, VecDeque<Value>>),
    Set(Ref<'a, LinkedHashSet<Value>>),
    Dict(Ref<'a, LinkedHashMap<Value, Value>>),
    Heap(Ref<'a, ValueHeap>)
}

impl<'b: 'a, 'a> IntoIterator for &'b ValueIntoIter<'a> {
    type Item = &'a Value;
    type IntoIter = ValueIter<'a>;

    fn into_iter(self) -> ValueIter<'a> {
        match self {
            ValueIntoIter::Str(it) => ValueIter::Str(it.into_iter()),
            ValueIntoIter::List(it) => ValueIter::List(it.iter()),
            ValueIntoIter::Set(it) => ValueIter::Set(it.iter()),
            ValueIntoIter::Dict(it) => ValueIter::Dict(it.keys()),
            ValueIntoIter::Heap(it) => ValueIter::Heap(it.heap.iter()),
        }
    }
}


pub enum ValueAsSlice<'a> {
    Str(&'a Box<String>, String),
    List(Ref<'a, VecDeque<Value>>, VecDeque<Value>),
}

impl<'a> ValueAsSlice<'a> {
    pub fn len(self: &Self) -> usize {
        match self {
            ValueAsSlice::Str(it, _) => it.len(),
            ValueAsSlice::List(it, _) => it.len(),
        }
    }

    pub fn accept(self: &mut Self, index: i64) {
        if index >= 0 && index < self.len() as i64 {
            let index = index as usize;
            match self {
                ValueAsSlice::Str(src, dest) => dest.push(src.chars().nth(index).unwrap()),
                ValueAsSlice::List(src, dest) => dest.push_back((&src[index]).clone()),
            }
        }
    }

    pub fn to_value(self: Self) -> Value {
        match self {
            ValueAsSlice::Str(_, it) => Str(Box::new(it)),
            ValueAsSlice::List(_, it) => List(Mut::new(it)),
        }
    }
}


#[cfg(test)]
mod test {
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
        }
    }

    fn all_values() -> Vec<Value> {
        let rc = Rc::new(FunctionImpl::new(0, String::new(), vec![]));
        vec![Value::Nil, Value::Bool(true), Value::Int(1), Value::list(vec![]), Value::set(LinkedHashSet::new()), Value::dict(LinkedHashMap::new()), Value::iter_heap(std::iter::empty()), Value::Function(rc.clone()), Value::partial(Value::Function(rc.clone()), vec![]), Value::NativeFunction(StdBinding::Void), Value::PartialNativeFunction(StdBinding::Void, Box::new(vec![])), Value::closure(rc.clone())]
    }
}
