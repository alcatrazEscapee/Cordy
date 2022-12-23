use std::borrow::Borrow;
use std::cell::{Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use hashlink::{LinkedHashMap, LinkedHashSet};

use crate::stdlib;
use crate::stdlib::StdBinding;

use Value::{*};


/// The runtime sum type used by the virtual machine
/// All `Value` type objects must be cloneable, and so mutable objects must be reference counted to ensure memory safety
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum Value {
    // Primitive (Immutable) Types
    Nil,
    Bool(bool),
    Int(i64),
    Str(Box<String>),
    Function(Rc<FunctionImpl>),
    PartialFunction(Box<PartialFunctionImpl>),

    // Reference (Mutable) Types
    // Memory is not really managed at all and cycles are entirely possible.
    // In future, a GC'd system could be implemented with `Mut` only owning weak references, and the GC owning the only strong reference.
    // But that comes at even more performance overhead because we'd need to still obey rust reference counting semantics.
    List(Mut<VecDeque<Value>>),
    Set(Mut<LinkedHashSet<Value>>),
    Dict(Mut<LinkedHashMap<Value, Value>>),

    // Functions
    Binding(StdBinding),
    PartialBinding(StdBinding, Box<Vec<Box<Value>>>),
}


impl Value {

    // Constructors
    pub fn iter(vec: impl Iterator<Item=Value>) -> Value { List(Mut::new(vec.collect::<VecDeque<Value>>())) }
    pub fn list(vec: Vec<Value>) -> Value { List(Mut::new(vec.into_iter().collect::<VecDeque<Value>>())) }
    pub fn set(set: LinkedHashSet<Value>) -> Value { Set(Mut::new(set)) }
    pub fn dict(dict: LinkedHashMap<Value, Value>) -> Value { Dict(Mut::new(dict)) }
    pub fn dequeue(vec: VecDeque<Value>) -> Value { List(Mut::new(vec)) }
    pub fn partial1(func: Rc<FunctionImpl>, arg: Value) -> Value { PartialFunction(Box::new(PartialFunctionImpl { func, args: vec![Box::new(arg)] }))}
    pub fn partial(func: Rc<FunctionImpl>, args: Vec<Value>) -> Value { PartialFunction(Box::new(PartialFunctionImpl { func, args: args.into_iter().map(|v| Box::new(v)).collect() }))}

    /// Converts the `Value` to a `String`. This is equivalent to the stdlib function `str()`
    pub fn as_str(self: &Self) -> String {
        match self {
            Str(s) => *s.clone(),
            Function(f) => f.name.clone(),
            PartialFunction(f) => f.func.name.clone(),
            Binding(b) => String::from(stdlib::lookup_name(*b)),
            PartialBinding(b, _) => String::from(stdlib::lookup_name(*b)),
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
            List(v) => format!("[{}]", v.borrow().iter().map(|t| t.as_repr_str()).collect::<Vec<String>>().join(", ")),
            Set(v) => format!("{{{}}}", v.borrow().iter().map(|t| t.as_repr_str()).collect::<Vec<String>>().join(", ")),
            Dict(v) => format!("{{{}}}", v.borrow().iter().map(|(k, v)| format!("{}: {}", k.as_repr_str(), v.as_repr_str())).collect::<Vec<String>>().join(", ")),
            Function(f) => {
                let f = (*f).as_ref().borrow();
                format!("fn {}({})", f.name, f.args.join(", "))
            },
            PartialFunction(f) => {
                let f = (*f).as_ref().borrow();
                format!("fn {}({})", f.func.name, f.func.args.join(", "))
            },
            Binding(b) => format!("fn {}()", stdlib::lookup_name(*b)),
            PartialBinding(b, _) => format!("fn {}()", stdlib::lookup_name(*b)),
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
            Function(_) => "function",
            PartialFunction(_) => "partial function",
            Binding(_) => "native function",
            PartialBinding(_, _) => "partial native function",
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
            List(l) => !l.borrow().is_empty(),
            Set(v) => !v.borrow().is_empty(),
            Dict(v) => !v.borrow().is_empty(),
            Function(_) => true,
            PartialFunction(_) => true,
            Binding(_) => true,
            PartialBinding(_, _) => true,
        }
    }

    pub fn is_nil(self: &Self) -> bool {
        match self {
            Nil => true,
            _ => false
        }
    }

    pub fn is_bool(self: &Self) -> bool {
        match self {
            Bool(_) => true,
            _ => false
        }
    }

    pub fn is_int(self: &Self) -> bool {
        match self {
            Int(_) => true,
            _ => false
        }
    }

    pub fn is_str(self: &Self) -> bool {
        match self {
            Str(_) => true,
            _ => false
        }
    }

    pub fn is_function(self: &Self) -> bool {
        match self {
            Function(_) | PartialFunction(_) | Binding(_) | PartialBinding(_, _) => true,
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
                let ls = (*l).borrow();
                let rs = (*r).borrow();
                ls.cmp(&rs)
            },
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

    pub fn borrow(self: &Self) -> Ref<T> {
        (*self.value).borrow()
    }

    pub fn borrow_mut(self: &Self) -> RefMut<T> {
        (*self.value).borrow_mut()
    }
}

impl<T : Eq + PartialEq + Debug + Clone + Hash> Hash for Mut<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (*self).borrow().hash(state)
    }
}


#[derive(Eq, PartialEq, Debug, Clone)]
pub struct FunctionImpl {
    pub head: usize, // Pointer to the first opcode of the function's execution
    pub nargs: u8, // The number of arguments the function takes
    name: String, // The name of the function, useful to show in stack traces
    args: Vec<String>, // Names of the arguments
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct PartialFunctionImpl {
    pub func: Rc<FunctionImpl>,
    pub args: Vec<Box<Value>>,
}


impl FunctionImpl {
    pub fn new(head: usize, name: String, args: Vec<String>) -> FunctionImpl {
        FunctionImpl {
            head,
            nargs: args.len() as u8,
            name,
            args
        }
    }
}

impl Hash for FunctionImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl Hash for PartialFunctionImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


#[cfg(test)]
mod test {
    use crate::vm::error::RuntimeError;
    use crate::vm::value::Value;

    #[test] fn test_layout() { assert_eq!(16, std::mem::size_of::<Value>()); }
    #[test] fn test_result_box_layout() { assert_eq!(16, std::mem::size_of::<Result<Value, Box<RuntimeError>>>()); }
}
