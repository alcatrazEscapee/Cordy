use std::cell::Cell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use fxhash::FxBuildHasher;
use indexmap::{IndexMap, IndexSet};

use crate::core::InvokeArg0;
use crate::util::{impl_deref, impl_partial_ord};
use crate::vm::value::ptr::Prefix;
use crate::vm::ValuePtr;


type ListType = VecDeque<ValuePtr>;
type VectorType = Vec<ValuePtr>;
type SetType = IndexSet<ValuePtr, FxBuildHasher>;
type DictType = IndexMap<ValuePtr, ValuePtr, FxBuildHasher>;
type HeapType = BinaryHeap<Reverse<ValuePtr>>;

impl_deref!(List, ListType, list);
impl_deref!(Vector, VectorType, vector);
impl_deref!(Set, SetType, set);
impl_deref!(Dict, DictType, dict);
impl_deref!(Heap, HeapType, heap);


/// # List
///
/// List is implemented as a `VecDeque` to provide O(1) `push_front` and `push` operations.
/// It derives all the required traits as `VecDeque` implements them all naturally.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct List {
    list: ListType
}

impl List {
    pub(super) fn new(list: ListType) -> List {
        List { list }
    }
}


/// # Vector
///
/// A vector in Cordy is a fixed-length low-cost list (as such, implemented with a `Vec<T>`), which behaves elementwise w.r.t most operations.
/// Vectors are then as useful for simple tuple types, and as elementwise types such as coordinates.
///
/// It derives all the required traits as `Vec` implements them all naturally.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vector {
    vector: VectorType
}

impl Vector {
    pub(super) fn new(vector: VectorType) -> Vector {
        Vector { vector }
    }
}


/// # Set
///
/// A set is a unique, hash-based, insertion-ordered collection of elements. All three properties are provided by `IndexSet`, based on Python's `set`
///
/// Equality and ordering are straightforward, while hash needs to prevent the creation of recursive structures and thus uses a thread-local
/// to prevent a recursive hash from computing - as that takes a borrow on the set during the mutable borrow of an insertion.
#[derive(Debug, PartialEq, Eq)]
pub struct Set {
    set: SetType
}

impl Set {
    pub(super) fn new(set: SetType) -> Set {
        Set { set }
    }
}

impl_partial_ord!(Set);
impl Ord for Set {
    fn cmp(&self, other: &Self) -> Ordering {
        for (l, r) in self.iter().zip(other.iter()) {
            match l.cmp(r) {
                Ordering::Equal => {},
                ord => return ord,
            }
        }
        self.len().cmp(&other.len())
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
/// - We don't implement `Hash` for `Set`, instead implementing for `Prefix<Set>`, as before unboxing we need to do a borrow check
/// - If the borrow check fails, we set a global flag that we've entered this pathological case, which is checked by `guard_recursive_hash()` before yielding back to user code
///
/// Note this also applies to `dict()`, although only when used as a key.
impl Hash for Prefix<Set> {
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


/// # Dict
///
/// A hash-based, insertion-order mapping of keys to values. All properties are provided by `IndexMap`, which is based on Python's `dict`.
///
/// Like with `set`, we prevent recursive dictionary construction using `guard_recursive_hash`, when a dictionary is used as it's own key
#[derive(Debug, Clone)]
pub struct Dict {
    dict: DictType,
    default: Option<InvokeArg0>
}

impl Dict {
    pub(super) fn new(dict: DictType) -> Dict {
        Dict { dict, default: None }
    }

    pub fn set_default(&mut self, default: InvokeArg0) {
        self.default = Some(default)
    }

    pub fn get_default(&self) -> &Option<InvokeArg0> {
        &self.default
    }
}

impl Eq for Dict {}
impl PartialEq<Self> for Dict { fn eq(&self, other: &Self) -> bool { self.dict == other.dict } }
impl PartialOrd for Dict { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }

impl Ord for Dict {
    fn cmp(&self, other: &Self) -> Ordering {
        for (l, r) in self.keys().zip(other.keys()) {
            match l.cmp(r) {
                Ordering::Equal => {},
                ord => return ord,
            }
        }
        self.len().cmp(&other.len())
    }
}

/// See justification for the unique `Hash` implementation on `Set`
impl Hash for Prefix<Dict> {
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


/// # Heap
///
/// `heap()` is a min-heap collection type.
///
/// As `BinaryHeap` is missing `Eq`, `PartialEq`, and `Hash` implementations
/// We also wrap values in `Reverse` as we want to expose a min-heap by default
#[derive(Debug, Clone)]
pub struct Heap {
    heap: HeapType
}

impl Heap {
    pub(super) fn new(heap: HeapType) -> Heap {
        Heap { heap }
    }
}

impl Eq for Heap {}
impl PartialEq<Self> for Heap {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

// Heap ordering is, much like the heap itself, just based on the lowest (top) value of the heap.
// Empty heaps will return `None`, and this is implicit less than `Some`. So empty heap < non-empty heap
impl_partial_ord!(Heap);
impl Ord for Heap {
    fn cmp(&self, other: &Self) -> Ordering {
        self.peek().cmp(&other.peek())
    }
}

impl Hash for Heap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.heap {
            v.hash(state)
        }
    }
}