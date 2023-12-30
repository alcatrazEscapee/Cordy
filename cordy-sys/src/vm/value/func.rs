use std::borrow::Cow;
use std::cell::Cell;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::core::{NativeFunction, PartialArgument};
use crate::vm::{IntoValue, ValuePtr};
use crate::vm::value::ptr::SharedPrefix;
use crate::vm::value::RecursionGuard;


/// # Functions
///
/// Functions are implemented by a number of separate types.
///
/// - `Function` describes a user function, along with calling conventions
/// - `PartialFunction` is a partially evaluated user function, with a pointer to the `Function`
/// - `NativeFunction` is an inline type used to identify a native function
/// - `PartialNativeFunction` is a partially evaluated native function, with a special-case partial argument
///
/// `Function` is a shared, immutable value. It is constructed by the parser during teardown, and only ever referenced as
/// a constant value through the VM's `constants` array.
///
#[derive(Eq, PartialEq, Hash, Debug, Clone)]
pub struct Function {
    /// Pointer to the first opcode of the function. Used in combination with `tail` to identify the extent of this function in stack traces.
    head: usize,
    /// Pointer to the final `Return` opcode in the function
    tail: usize,

    /// The user-defined name of the function.
    /// This and the arguments are used in stack traces and for debugging purposes.
    name: String,
    /// Names of each of the arguments.
    args: Vec<String>,

    /// Jump offsets for each default argument
    default_args: Vec<usize>,

    /// If `true`, the last argument in this function is variadic
    variadic: bool,
    /// If `true`, this function is a native FFI function
    native: bool,
    /// If `true`, this function is an instance function on a struct, and takes `self` as a first parameter.
    is_self: bool,
}

impl Function {
    pub fn new(head: usize, tail: usize, name: String, args: Vec<String>, default_args: Vec<usize>, variadic: bool, native: bool, is_self: bool) -> Function {
        Function { head, tail, name, args, default_args, variadic, native, is_self }
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
        self.min_args() <= nargs && (self.variadic || nargs <= self.max_args())
    }

    /// Returns the number of variadic arguments that need to be collected, before invoking the function, if needed.
    pub fn num_var_args(&self, nargs: u32) -> Option<u32> {
        if self.variadic && nargs >= self.max_args() {
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
        } else if self.variadic && nargs >= self.max_args() {
            *self.default_args.last().unwrap()
        } else {
            self.default_args[(nargs - self.min_args() - 1) as usize]
        }
    }

    /// Returns `true` if this function contains within it, the given IP pointer.
    ///
    /// Functions are emitted serially, so function bodies will never contain other function bodies.
    /// Thus it is sufficient to just check the bounds of the `head` and `tail`
    pub fn contains_ip(&self, ip: usize) -> bool {
        self.head <= ip && ip <= self.tail
    }

    /// Returns the function name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the disassembly representation for a `Function` opcode featuring this function, which also includes a reference to the `head` and `tail` of this function.
    pub fn to_asm_str(&self) -> String {
        format!("Function({} -> L[{}, {}])", self.to_repr_str(), self.head, self.tail)
    }

    /// Returns the `repr` representation of this function, including `native? fn <name>(<args>...)`
    pub fn to_repr_str(&self) -> String {
        format!("{}fn {}({})", if self.native { "native " } else { "" }, self.name, self.args.join(", "))
    }
}


/// # Partial Functions
///
/// `PartialFunction` represents a partially evaluated _user_ function.
/// It references a `func: ValuePtr`, which **must** be either a `Type::Function` or `Type::Closure`
///
/// Partial functions are shared, immutable values. They can be re-evaluated, which requires copying the current function,
/// along with the arguments, to create a new partial function.
#[derive(Debug, Clone)]
pub struct PartialFunction {
    func: ValuePtr,
    args: Vec<ValuePtr>,
}

impl PartialFunction {
    fn new(func: ValuePtr, args: Vec<ValuePtr>) -> PartialFunction {
        PartialFunction { func, args }
    }

    /// Returns a new `PartialFunction` with additional arguments.
    pub fn with(&self, args: impl Iterator<Item=ValuePtr>) -> PartialFunction {
        let mut new_args = self.args.clone();
        for arg in args {
            new_args.push(arg);
        }
        PartialFunction::new(self.func.clone(), new_args)
    }

    /// Consumes this partial function, returning a pair of the function, and any arguments
    pub fn consume(&self) -> (ValuePtr, impl Iterator<Item=ValuePtr> + '_) {
        (self.func.clone(), self.args.iter().cloned())
    }

    pub(super) fn to_str(&self, rc: &mut RecursionGuard) -> Cow<str> {
        self.func.safe_to_str(rc)
    }

    pub(super) fn to_repr_str(&self, rc: &mut RecursionGuard) -> Cow<str> {
        self.func.safe_to_repr_str(rc)
    }

    pub fn as_function(&self) -> &Function {
        self.func.as_function_or_closure()
    }

    /// Returns the minimum number of arguments required to evaluate this function, accounting for partial arguments
    pub fn min_nargs(&self) -> u32 {
        self.as_function().min_args() - self.nargs()
    }

    /// Returns the number of partial arguments currently held by this partial function
    pub fn nargs(&self) -> u32 {
        self.args.len() as u32
    }
}

impl ValuePtr {
    /// When provided either a user function or closure, returns the function partially evaluated with the given arguments.
    pub fn to_partial(self, args: Vec<ValuePtr>) -> ValuePtr {
        debug_assert!(self.is_closure() || self.is_function());
        PartialFunction::new(self, args).to_value()
    }
}

impl Eq for PartialFunction {}
impl PartialEq<Self> for PartialFunction {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}

impl Hash for PartialFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


/// # Closures
///
/// A closure is a combination of a function, and a set of `environment` variables.
/// These variables are references either to locals in the enclosing function, or captured variables from the enclosing function itself.
///
/// A closure also provides *interior mutability* for it's captured upvalues, allowing them to be modified even from the surrounding function.
/// Unlike with other mutable `ValuePtr` types, this does so using `Rc<Cell<ValuePtr>>`. The reason being:
///
/// - A `Rc<RefCell<>>` cannot be unboxed without creating a borrow, which introduces lifetime restrictions. It also cannot be mutably unboxed without creating a write lock. With a closure, we need to be free to unbox the environment straight onto the stack, so this is off the table.
/// - The closure's inner value can be thought of as immutable. As `ValuePtr` is immutable, and clone-able, so can the contents of `Cell`. We can then unbox this completely - take a reference to the `Rc`, and call `get()` to unbox the current value of the cell, onto the stack.
///
/// This has one problem, which is we can't call `.get()` unless the cell is `Copy`, which `ValuePtr` isn't, and can't be, because `Rc<RefCell<>>` can't be copy due to the presence of `Rc`... Fortunately, this is just an API limitation, and we can unbox the cell in other ways.
///
/// Note we cannot derive most functions, as that also requires `Cell<ValuePtr>` to be `Copy`, due to convoluted trait requirements.
#[derive(Clone)]
pub struct Closure {
    /// This function must be **never modified**, as we hand out special, non-counted immutable references via `borrow_func()`.
    /// This **must** be a `Type::Function`
    ///
    /// In this sense, closures are partially immutable (`func`), `ConstValue`-like, and partially mutable, `MutValue`-like, via `environment`
    func: ValuePtr,
    environment: Vec<Rc<Cell<UpValue>>>,
}

impl Closure {
    fn new(func: ValuePtr) -> Closure {
        Closure { func, environment: Vec::new() }
    }

    pub fn push(&mut self, value: Rc<Cell<UpValue>>) {
        self.environment.push(value);
    }

    /// Returns the current environment value for the upvalue index `index.
    pub fn get(&self, index: usize) -> Rc<Cell<UpValue>> {
        self.environment[index].clone()
    }
}

impl SharedPrefix<Closure> {
    pub fn borrow_func(&self) -> &Function {
        unsafe {
            // SAFETY: We only hand out immutable references to `self.func`, and only ever mutate `self.environment`
            self.borrow_const_unsafe().func.as_function().borrow_const()
        }
    }
}

impl ValuePtr {
    /// Converts a `Function` into a new empty `Closure` with no environment
    pub fn to_closure(self) -> ValuePtr {
        debug_assert!(self.is_function());
        Closure::new(self).to_value()
    }
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}

impl Eq for Closure {}

impl Debug for Closure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.func, f)
    }
}

impl Hash for Closure {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state)
    }
}


#[derive(Clone)]
pub enum UpValue {
    Open(usize),
    Closed(ValuePtr)
}

/// Implement `Default` to have access to `.take()`
impl Default for UpValue {
    fn default() -> Self {
        UpValue::Open(0)
    }
}


#[derive(Debug, Clone)]
pub struct PartialNativeFunction {
    pub func: NativeFunction,
    pub partial: PartialArgument,
}

impl Eq for PartialNativeFunction {}
impl PartialEq<Self> for PartialNativeFunction {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}


impl Hash for PartialNativeFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
    }
}