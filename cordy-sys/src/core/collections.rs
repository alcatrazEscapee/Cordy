use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::collections::hash_map::Entry;
use fxhash::FxBuildHasher;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;

use crate::util;
use crate::core::{InvokeArg0, InvokeArg1, InvokeArg2};
use crate::vm::{AnyResult, ErrorResult, IntoDictValue, IntoIterableValue, IntoValue, Iterable, RuntimeError, Type, ValuePtr, ValueResult, VirtualInterface};

use RuntimeError::{*};


pub fn get_index<VM : VirtualInterface>(vm: &mut VM, target: &ValuePtr, index: ValuePtr) -> ValueResult {
    if target.is_dict() {
        return get_dict_index(vm, target, index);
    }

    let indexable = target.to_index()?;
    let index: usize = indexable.check_index(index)?;

    indexable.get_index(index).ok()
}

fn get_dict_index<VM : VirtualInterface>(vm: &mut VM, dict: &ValuePtr, key: ValuePtr) -> ValueResult {
    // Dict objects have their own overload of indexing to mean key-value lookups, that doesn't fit with ValueAsIndex (as it doesn't take integer keys, always)
    // The handling for this is a bit convoluted due to `clone()` issues, and possible cases of default / no default / functional default

    // Initially unbox (non mutable) to clone out the default value.
    // If the default is a function, we can't have a reference out of the dict while we're accessing the default.

    let dict = dict.as_dict();
    let default_factory: InvokeArg0;
    {
        let dict = dict.borrow();
        match &dict.get_default() {
            Some(default) => match dict.get(&key) {
                Some(existing_value) => return existing_value.clone().ok(),
                None => {
                    // We need to insert, so fallthrough as we need to drop the borrow on `dict`
                    default_factory = default.clone();
                },
            },
            None => return match dict.get(&key) {
                Some(existing_value) => existing_value.clone().ok(),
                None => ValueErrorKeyNotPresent(key).err()
            },
        }
    }

    // Invoke the new value supplier - this might modify the dict
    // We go through the `.entry()` API again in this case
    let new_value: ValuePtr = default_factory.invoke(vm)?;
    let mut dict = dict.borrow_mut();

    dict.entry(key)
        .or_insert(new_value)
        .clone()
        .ok()
}

pub fn set_index(target: &ValuePtr, index: ValuePtr, value: ValuePtr) -> AnyResult {
    if target.is_dict() {
        target.as_dict()
            .borrow_mut()
            .insert_checked(index, value)
    } else {
        let mut indexable = target.to_index()?;
        let index: usize = indexable.check_index(index)?;

        indexable.set_index(index, value)
    }
}


/// Performs a slice operation on `target`, given the operands `[low:high:step]`
///
/// Note that each operand can be interpreted as either an `int` type or `nil`
pub fn get_slice(target: &ValuePtr, low: ValuePtr, high: ValuePtr, step: ValuePtr) -> ValueResult {

    #[inline]
    fn unwrap_or(ptr: ValuePtr, default: i64) -> ErrorResult<i64> {
        if ptr.is_int() {
            Ok(ptr.as_int())
        } else if ptr.is_nil() {
            Ok(default)
        } else {
            TypeErrorArgMustBeInt(ptr).err()
        }
    }

    let mut slice = target.to_slice()?;
    let length: i64 = slice.len() as i64;

    let step: i64 = unwrap_or(step, 1)?;
    if step == 0 {
        return ValueErrorStepCannotBeZero.err()
    }

    let low: i64 = unwrap_or(low, if step > 0 { 0 } else { -1 })?;
    let high: i64 = unwrap_or(high, if step > 0 { length } else { -length - 1 })?;

    let abs_start: i64 = to_index(length, low);
    let abs_stop: i64 = to_index(length, high);
    let abs_step: usize = step.unsigned_abs() as usize;

    if step > 0 {
        for i in (abs_start..abs_stop).step_by(abs_step) {
            slice.accept(i)
        }
    } else {
        // Don't use `step_by` here, as it doesn't get optimized correctly with `rev_range()`
        for i in rev_range(abs_start, abs_stop, abs_step as i64) {
            slice.accept(i)
        }
    }

    slice.to_value().ok()
}


#[inline(always)]
pub fn to_index(len: i64, pos_or_neg: i64) -> i64 {
    if pos_or_neg >= 0 {
        pos_or_neg
    } else {
        len + pos_or_neg
    }
}

#[inline(always)]
fn rev_range(start_high_inclusive: i64, stop_low_exclusive: i64, step_size: i64) -> impl Iterator<Item = i64> {
    let mut start: i64 = start_high_inclusive;
    let end: i64 = stop_low_exclusive;
    std::iter::from_fn(move || {
        if start <= end {
            None
        } else {
            start -= step_size;
            Some(start + step_size)
        }
    })
}


// ===== Library Functions ===== //


pub fn sum(args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    let mut sum: i64 = 0;
    for v in args {
        sum += v.as_int_checked()?;
    }
    sum.to_value().ok()
}

pub fn min(args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    non_empty(args.min())
}

pub fn min_by<VM: VirtualInterface>(vm: &mut VM, by: ValuePtr, args: ValuePtr) -> ValueResult {
    let iter = args.to_iter()?;
    match by.min_nargs() {
        Some(2) => {
            let by: InvokeArg2 = InvokeArg2::from(by)?;
            let mut err = None;
            let ret = iter.min_by(|a, b|
                util::catch(&mut err, ||
                    Ok(by.invoke((*a).clone(), (*b).clone(), vm)?.as_int_checked()?.cmp(&0)), Ordering::Equal));
            util::join(non_empty(ret)?, err)
        },
        Some(1) => {
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            let mut err = None;
            let ret = iter.min_by_key(|u|
                util::catch(&mut err, ||
                    by.invoke((*u).clone(), vm).as_result(), ValuePtr::nil()));
            util::join(non_empty(ret)?, err)
        },
        Some(_) => TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => TypeErrorArgMustBeFunction(by).err(),
    }
}

pub fn max(args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    non_empty(args.max())
}

pub fn max_by<VM: VirtualInterface>(vm: &mut VM, by: ValuePtr, args: ValuePtr) -> ValueResult {
    let iter = args.to_iter()?;
    match by.min_nargs() {
        Some(2) => {
            let by: InvokeArg2 = InvokeArg2::from(by)?;
            let mut err = None;
            let ret = iter.max_by(|a, b|
                util::catch(&mut err, ||
                    Ok(by.invoke((*a).clone(), (*b).clone(), vm)?.as_int_checked()?.cmp(&0)), Ordering::Equal));
            util::join(non_empty(ret)?, err)
        },
        Some(1) => {
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            let mut err = None;
            let ret = iter.max_by_key(|u|
                util::catch(&mut err, ||
                    by.invoke((*u).clone(), vm).as_result(), ValuePtr::nil()));
            util::join(non_empty(ret)?, err)
        },
        Some(_) => TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => TypeErrorArgMustBeFunction(by).err(),
    }
}


pub fn sort(args: impl Iterator<Item=ValuePtr>) -> ValuePtr {
    let mut sorted: Vec<ValuePtr> = args.collect::<Vec<ValuePtr>>();
    sorted.sort_unstable();
    sorted.into_iter().to_list()
}

pub fn sort_by<VM : VirtualInterface>(vm: &mut VM, by: ValuePtr, args: ValuePtr) -> ValueResult {
    let mut sorted: Vec<ValuePtr> = args.to_iter()?.collect::<Vec<ValuePtr>>();
    match by.min_nargs() {
        Some(2) => {
            let by: InvokeArg2 = InvokeArg2::from(by)?;
            let mut err = None;
            sorted.sort_unstable_by(|a, b|
                util::catch(&mut err, ||
                    Ok(by.invoke(a.clone(), b.clone(), vm)?.as_int_checked()?.cmp(&0)), Ordering::Equal));
            if let Some(err) = err {
                return ValueResult::from(err);
            }
        },
        Some(1) => {
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            let mut err = None;
            sorted.sort_unstable_by_key(|a|
                util::catch(&mut err, ||
                    by.invoke(a.clone(), vm).as_result(), ValuePtr::nil()));
            if let Some(err) = err {
                return ValueResult::from(err);
            }
        },
        Some(_) => return TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => return TypeErrorArgMustBeFunction(by).err(),
    }
    sorted.into_iter().to_list().ok()
}

#[inline]
fn non_empty(it: Option<ValuePtr>) -> ValueResult {
    match it {
        Some(v) => v.ok(),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}


pub fn group_by<VM : VirtualInterface>(vm: &mut VM, by: ValuePtr, args: ValuePtr) -> ValueResult {
    let iter = args.to_iter()?;
    match by.is_int() {
        true => {
            // `group_by(n, iter) will return a list of vectors of `n` values each. Last value will have whatever, instead of raising an error
            let i = by.as_int();
            if i <= 0 {
                return ValueErrorValueMustBePositive(i).err()
            }
            let size: usize = i as usize;
            let mut groups: VecDeque<ValuePtr> = VecDeque::with_capacity(1 + iter.len() / size); // Accurate guess
            let mut group: Vec<ValuePtr> = Vec::with_capacity(size);
            for value in iter {
                group.push(value);
                if group.len() == size {
                    groups.push_back(group.to_value());
                    group = Vec::with_capacity(size);
                }
            }
            if !group.is_empty() {
                groups.push_back(group.to_value());
            }
            groups.to_value().ok()
        },
        _ => {
            // Otherwise, we assume this is a group_by(f), in which case we assume the function to be a item -> key, and create a dictionary of keys -> vector of values
            // For capacity, we guess that we're halving. That seems to be a reasonable compromise between overestimating, and optimal values.
            let size = iter.len();
            let mut groups: IndexMap<ValuePtr, ValuePtr, FxBuildHasher> = IndexMap::with_capacity_and_hasher(size / 2, FxBuildHasher::default());
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            for value in iter {
                let key = by.invoke(value.clone(), vm)?;
                groups.entry(key)
                    .or_insert_with(|| Vec::with_capacity(size / 4).to_value()) // Rough guess
                    .as_vector() // This is safe because we should only have vectors in the map
                    .borrow_mut()
                    .push(value);
            }
            groups.to_value().ok()
        }
    }
}

pub fn reverse(args: impl Iterator<Item=ValuePtr>) -> ValuePtr {
    let mut vec = args.collect::<Vec<ValuePtr>>();
    vec.reverse();
    vec.into_iter().to_list()
}

pub fn permutations(n: ValuePtr, args: ValuePtr) -> ValueResult {
    let n = n.as_int_checked()?;
    if n <= 0 {
        return ValueErrorValueMustBeNonNegative(n).err();
    }
    args.to_iter()?
        .permutations(n as usize)
        .map(|u| u.to_value())
        .to_list()
        .ok()
}

pub fn combinations(n: ValuePtr, args: ValuePtr) -> ValueResult {
    let n = n.as_int_checked()?;
    if n <= 0 {
        return ValueErrorValueMustBeNonNegative(n).err();
    }
    args.to_iter()?
        .combinations(n as usize)
        .map(|u| u.to_value())
        .to_list()
        .ok()
}

pub fn any<VM : VirtualInterface>(vm: &mut VM, f: ValuePtr, args: ValuePtr) -> ValueResult {
    predicate(vm, f, args, true)
}

pub fn all<VM: VirtualInterface>(vm: &mut VM, f: ValuePtr, args: ValuePtr) -> ValueResult {
    predicate(vm, f, args, false)
}

/// Iterates `args`, checking each element with the predicate `f`, until one returns `is_any`, then returns `is_any`. Otherwise returns `!is_any`
///
/// With `is_any = true`, this behaves like `any()`, with it `false`, it behaves like `all()`
fn predicate<VM : VirtualInterface>(vm: &mut VM, f: ValuePtr, args: ValuePtr, is_any: bool) -> ValueResult {
    let f: InvokeArg1 = InvokeArg1::from(f)?;
    for r in args.to_iter()? {
        if f.invoke(r, vm)?.to_bool() == is_any {
            return is_any.to_value().ok()
        }
    }
    (!is_any).to_value().ok()
}


pub fn map<VM: VirtualInterface>(vm: &mut VM, f: ValuePtr, args: ValuePtr) -> ValueResult {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<ValuePtr> = VecDeque::with_capacity(len);
    let f: InvokeArg1 = InvokeArg1::from(f)?;
    for r in args.to_iter()? {
        acc.push_back(f.invoke(r, vm)?);
    }
    acc.to_value().ok()
}

pub fn filter<VM: VirtualInterface>(vm: &mut VM, f: ValuePtr, args: ValuePtr) -> ValueResult {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<ValuePtr> = VecDeque::with_capacity(len);
    let f: InvokeArg1 = InvokeArg1::from(f)?;
    for r in args.to_iter()? {
        let ret = f.invoke(r.clone(), vm)?;
        if ret.to_bool() {
            acc.push_back(r);
        }
    }
    acc.to_value().ok()
}

pub fn flat_map<VM>(vm: &mut VM, f: Option<ValuePtr>, args: ValuePtr) -> ValueResult where VM : VirtualInterface {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<ValuePtr> = VecDeque::with_capacity(len);
    let f: Option<InvokeArg1> = match f {
        Some(f) => Some(InvokeArg1::from(f)?),
        None => None,
    };
    for r in args.to_iter()? {
        let elem = match &f {
            Some(l) => l.invoke(r, vm)?,
            None => r
        };
        for e in elem.to_iter()? {
            acc.push_back(e);
        }
    }
    acc.to_value().ok()
}

pub fn zip(args: impl Iterator<Item=ValuePtr>) -> ValueResult {
    let mut iters = args
        .map(|v| v.to_iter())
        .collect::<ErrorResult<Vec<Iterable>>>()?;
    if iters.is_empty() {
        return ValueErrorValueMustBeNonEmpty.err()
    }
    let size: usize = iters.iter()
        .map(|u| u.len())
        .min()
        .unwrap_or(0);
    let mut acc = VecDeque::with_capacity(size);
    loop {
        let mut vec = Vec::with_capacity(iters.len());
        for iter in &mut iters {
            match iter.next() {
                Some(it) => vec.push(it),
                None => return acc.to_value().ok(),
            }
        }
        acc.push_back(vec.to_value());
    }
}

pub fn reduce<VM: VirtualInterface>(vm: &mut VM, f: ValuePtr, args: ValuePtr) -> ValueResult {
    let mut iter = args.to_iter()?;
    let mut acc: ValuePtr = match iter.next() {
        Some(v) => v,
        None => return ValueErrorValueMustBeNonEmpty.err()
    };

    let f: InvokeArg2 = InvokeArg2::from(f)?;
    for r in iter {
        acc = f.invoke(acc, r, vm)?;
    }
    acc.ok()
}

pub fn peek(target: ValuePtr) -> ValueResult {
    match match target.ty() {
        Type::List => target.as_list().borrow().front().cloned(),
        Type::Set => target.as_set().borrow().first().cloned(),
        Type::Dict => target.as_dict().borrow().first().map(|(l, r)| (l.clone(), r.clone()).to_value()),
        Type::Heap => target.as_heap().borrow().peek().map(|u| u.clone().0),
        Type::Vector => target.as_vector().borrow().first().cloned(),
        _ => return TypeErrorArgMustBeIterable(target).err(),
    } {
        Some(v) => v.ok(),
        None => ValueErrorValueMustBeNonEmpty.err(),
    }
}

pub fn pop(target: ValuePtr) -> ValueResult {
    match match target.ty() {
        Type::List => target.as_list().borrow_mut().pop_back(),
        Type::Set => target.as_set().borrow_mut().pop(),
        Type::Dict => target.as_dict().borrow_mut().pop().map(|u| u.to_value()),
        Type::Heap => target.as_heap().borrow_mut().pop().map(|t| t.0),
        _ => return TypeErrorArgMustBeIterable(target).err()
    } {
        Some(v) => v.ok(),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn pop_front(target: ValuePtr) -> ValueResult {
    match target.as_list_checked()?.borrow_mut().pop_front() {
        Some(v) => v.ok(),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn push(value: ValuePtr, target: ValuePtr) -> ValueResult {
    match target.ty() {
        Type::List => target.as_list().borrow_mut().push_back(value),
        Type::Set => match target.as_set().borrow_mut().insert_checked(value) {
            Ok(_) => {},
            Err(e) => return ValueResult::from(e),
        }
        Type::Heap => target.as_heap().borrow_mut().push(Reverse(value)),
        _ => return TypeErrorArgMustBeIterable(target).err()
    }
    target.ok()
}

pub fn push_front(value: ValuePtr, target: ValuePtr) -> ValueResult {
    target.as_list_checked()?.borrow_mut().push_front(value);
    target.ok()
}

pub fn insert(index: ValuePtr, value: ValuePtr, target: ValuePtr) -> ValueResult {
    match target.ty() {
        Type::List => {
            {
                let mut it = target.as_list().borrow_mut();
                let index = index.as_int_checked()?;
                let len = it.len();
                if 0 <= index && index < len as i64 {
                    it.insert(index as usize, value);
                } else if index == len as i64 {
                    it.push_back(value);
                } else {
                    return ValueErrorIndexOutOfBounds(index, len).err()
                }
            }
            target.ok()
        },
        Type::Dict => {
            match target.as_dict().borrow_mut().insert_checked(index, value) {
                Ok(_) => {},
                Err(e) => return ValueResult::from(e)
            }
            target.ok()
        }
        _ => TypeErrorArgMustBeIndexable(target).err()
    }
}

pub fn remove(needle: ValuePtr, target: ValuePtr) -> ValueResult {
    match target.ty() {
        Type::List => {
            let mut it = target.as_list().borrow_mut();
            let index = needle.as_int_checked()?;
            let len = it.len();
            if 0 <= index && index < len as i64 {
                it.remove(index as usize)
                    .unwrap() // .unwrap() is safe, as we pre-checked the index
                    .ok()
            } else {
                ValueErrorIndexOutOfBounds(index, len).err()
            }
        },
        // Other operations on `dict`, `set`, generally preserve ordering, so this should do, despite O(n) cost
        // `remove()` is not a particularly common op (`in` or access) is, so this is a fair tradeoff to make
        //
        // For reference, Python >=3.7 makes the same guarantee, see https://docs.python.org/3/reference/datamodel.html, 3.2.7.1
        Type::Set => target.as_set().borrow_mut().shift_remove(&needle).to_value().ok(),
        Type::Dict => target.as_dict().borrow_mut().shift_remove(&needle).is_some().to_value().ok(),
        _ => TypeErrorArgMustBeIterable(target).err(),
    }
}

pub fn clear(target: ValuePtr) -> ValueResult {
    match target.ty() {
        Type::List => target.as_list().borrow_mut().clear(),
        Type::Set => target.as_set().borrow_mut().clear(),
        Type::Dict => target.as_dict().borrow_mut().clear(),
        Type::Heap => target.as_heap().borrow_mut().clear(),
        _ => return TypeErrorArgMustBeIterable(target).err(),
    }
    target.ok()
}


pub fn collect_into_dict(iter: impl Iterator<Item=ValuePtr>) -> ValueResult {
    iter.map(|t| t.to_pair())
        .collect::<ErrorResult<Vec<(ValuePtr, ValuePtr)>>>()?
        .into_iter()
        .to_dict()
        .ok()
}

pub fn dict_set_default(def: ValuePtr, target: ValuePtr) -> ValueResult {
    target.as_dict_checked()?
        .borrow_mut()
        .set_default(match def.is_evaluable() {
            true => InvokeArg0::from(def)?,
            false => InvokeArg0::Noop(def) // Treat single argument defaults still as a function, which is optimized to just copy its value
        });
    target.ok()
}

pub fn dict_keys(target: ValuePtr) -> ValueResult {
    target.as_dict_checked()?
        .borrow()
        .keys()
        .cloned()
        .to_set()
        .ok()
}

pub fn dict_values(target: ValuePtr) -> ValueResult {
    target.as_dict_checked()?
        .borrow()
        .values()
        .cloned()
        .to_list()
        .ok()
}

pub fn left_find<VM: VirtualInterface>(vm: &mut VM, finder: ValuePtr, args: ValuePtr, return_index: bool) -> ValueResult {
    // Supports both find index (`index_of`), and find position (`find`)
    // For predicates, we use the same `enumerate()`, but then either return index, or value
    // For index with value, we use `.position()`
    // For value with value, we just use `.find()`
    let mut iter = args.to_iter()?;
    if finder.is_evaluable() {
        let finder: InvokeArg1 = InvokeArg1::from(finder)?;
        for (i, v) in iter.enumerate() {
            let ret = finder.invoke(v.clone(), vm)?;
            if ret.to_bool() {
                return if return_index { (i as i64).to_value() } else { v }.ok()
            }
        }
        if return_index { (-1i64).to_value() } else { ValuePtr::nil() }.ok()
    } else if return_index {
        match iter.position(|v| v == finder) {
            Some(i) => i as i64,
            None => -1
        }.to_value().ok()
    } else {
        iter.find(|v| v == &finder).unwrap_or(ValuePtr::nil()).ok()
    }
}

pub fn right_find<VM: VirtualInterface>(vm: &mut VM, finder: ValuePtr, args: ValuePtr, return_index: bool) -> ValueResult {
    // Identical to the above except we use `.reverse()`, and subtract the index from `len`
    let mut iter = args.to_iter()?.reverse();
    let len = iter.len();
    if finder.is_evaluable() {
        let finder: InvokeArg1 = InvokeArg1::from(finder)?;
        for (i, v) in iter.enumerate() {
            let ret = finder.invoke(v.clone(), vm)?;
            if ret.to_bool() {
                return if return_index { ((len - 1 - i) as i64).to_value() } else { v }.ok()
            }
        }
        if return_index { (-1i64).to_value() } else { ValuePtr::nil() }.ok()
    } else if return_index {
        match iter.position(|v| v == finder) {
            Some(i) => (len - 1 - i) as i64,
            None => -1
        }.to_value().ok()
    } else {
        iter.find(|v| v == &finder).unwrap_or(ValuePtr::nil()).ok()
    }
}

pub fn create_memoized(f: ValuePtr) -> ValueResult {
    match f.is_evaluable() {
        true => ValuePtr::memoized(f).ok(),
        false => TypeErrorArgMustBeFunction(f).err()
    }
}

pub fn set_union(other: ValuePtr, this: ValuePtr) -> ValueResult {
    match this.ty() {
        Type::Set => {
            // this.union(other) := add everything from other to this
            let mut set = this.as_set().borrow_mut();
            for e in other.to_iter()? {
                set.insert(e);
            }
            drop(set);
            this.ok()
        },
        _ => TypeErrorArgMustBeSet(this).err()
    }
}

pub fn set_intersect(other: ValuePtr, this: ValuePtr) -> ValueResult {
    match this.ty() {
        Type::Set => {
            // this.intersect(other) := only keep elements of this that are also in other
            // Since we have just an iterator, we need to compute a set of `other`, then repeatedly check `contains()` for each element in `this`
            let mut set = this.as_set().borrow_mut();
            let other = other.to_iter()?.collect::<IndexSet<ValuePtr, FxBuildHasher>>();
            set.retain(|e| other.contains(e));
            drop(set);
            this.ok()
        },
        _ => TypeErrorArgMustBeSet(this).err()
    }
}

pub fn set_difference(other: ValuePtr, this: ValuePtr) -> ValueResult {
    match this.ty() {
        Type::Set => {
            // this.difference(other) := remove everything from this that is in other
            let mut set = this.as_set().borrow_mut();
            for e in other.to_iter()? {
                set.remove(&e);
            }
            drop(set);
            this.ok()
        },
        _ => TypeErrorArgMustBeSet(this).err()
    }
}

pub fn counter(it: ValuePtr) -> ValueResult {
    let iter = it.to_iter()?;
    let mut counter: IndexMap<ValuePtr, ValuePtr, FxBuildHasher> = IndexMap::with_capacity_and_hasher(iter.len(), FxBuildHasher::default());
    for elem in iter {
        counter.entry(elem)
            .and_modify(|e| {
                *e = (e.as_precise_int() + 1).to_value()
            })
            .or_insert(1i64.to_value());
    }
    counter.to_value().ok()
}

/// Copies a value, recursively, and preserving identity relationships.
/// The complexity of this method is in properly handling a number of edge cases:
///
/// - Recursive structures (i.e. a list that contains itself)
/// - Objects with the same identity (i.e. a list containing seven references to the same list)
/// - Deep nested objects (i.e. lists containing vectors containing structs containing fields, etc.)
pub fn copy(it: ValuePtr) -> ValuePtr {
    copy_recursive(&it, &mut HashMap::with_hasher(FxBuildHasher::default()))
}

fn copy_recursive(it: &ValuePtr, objects: &mut HashMap<usize, ValuePtr, FxBuildHasher>) -> ValuePtr {
    use Type::{*};

    macro_rules! copy_collection {
        ($as_T:ident, $insert:ident, $default:expr) => {
            copy_collection!($as_T, $insert, $default, copy_recursive, ValuePtr)
        };
        ($as_T:ident, $insert:ident, $default:expr, $copy:ident, $elem_ty:ty) => {
            match objects.entry(it.as_ptr_value()) {
                Entry::Occupied(e) => e.get().clone(), // If the object already exists, copy it directly
                Entry::Vacant(e) => {
                    // Otherwise,
                    // 1. insert a empty 'dummy' list into the `object` map, to preserve the identity
                    // 2. call copy_recursive() on all children of the original collection, saving them in a temporary `Vec<ValuePtr``
                    // 3. insert all children in order into the copied collection
                    e.insert($default.to_value());

                    let elements: Vec<$elem_ty> = it.$as_T()
                        .borrow()
                        .iter()
                        .map(|e| $copy(&e, objects))
                        .collect();
                    let ptr = objects.get_mut(&it.as_ptr_value())
                        .unwrap(); // This should always exist in the map, since we only insert and just inserted this above

                    let mut copy = ptr.$as_T().borrow_mut();
                    for e in elements {
                        copy.$insert(e);
                    }

                    ptr.clone()
                }
            }
        };
    }

    match it.ty() {
        // These types do not contain any further types, and have no concept of identity
        // We can freely clone them without worrying about other references to the object
        //
        // - Range, partial functions, etc. are all immutable, and thus have no identity, so we can re-use the same pointer
        // - Slice contains `ValuePtr`s, but they are ensured to be `int` or `nil`, so we can copy directly
        // - Closures have a mutable environment, but that is not considered within *reach* of `copy()`
        Nil | Bool | Int | NativeFunction | GetField | ShortStr | LongStr | Complex | StructType | Range | Slice | Function | PartialFunction | PartialNativeFunction | Closure | Memoized
            => it.clone(),

        // Collections all have identity, and are mutable, so we need to clone their contents, but remember the collection in the objects map
        // If the collection already exists in the objects map, we use that instead
        List => copy_collection!(as_list, push_back, VecDeque::new()),
        Vector => copy_collection!(as_vector, push, Vec::new()),
        Set => copy_collection!(as_set, insert, IndexSet::with_hasher(FxBuildHasher::default())),
        Heap => copy_collection!(as_heap, push, BinaryHeap::new(), copy_recursive_rev, Reverse<ValuePtr>),

        // Dict is a bit different, and needs a custom implementation, as we need to copy *both* keys and values
        Dict => match objects.entry(it.as_ptr_value()) {
            Entry::Occupied(e) => e.get().clone(), // If the object already exists, copy it directly
            Entry::Vacant(e) => {
                // Otherwise,
                // 1. insert a empty 'dummy' list into the `object` map, to preserve the identity
                // 2. call copy_recursive() on all children of the original collection, saving them in a temporary `Vec<ValuePtr``
                // 3. insert all children in order into the copied collection
                e.insert(IndexMap::with_hasher(FxBuildHasher::default()).to_value());

                let elements: Vec<(ValuePtr, ValuePtr)> = it.as_dict()
                    .borrow()
                    .iter()
                    .map(|(key, value)| (copy_recursive(key, objects), copy_recursive(value, objects)))
                    .collect();
                let ptr = objects.get_mut(&it.as_ptr_value())
                    .unwrap(); // This should always exist in the map, since we only insert and just inserted this above

                let mut copy = ptr.as_dict().borrow_mut();
                for (key, value) in elements {
                    copy.insert(key, value);
                }

                ptr.clone()
            }
        },

        // Structs are mutable, and have identity, and need to copy much like collection types.
        Struct => match objects.entry(it.as_ptr_value()) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => {
                let owner = it.as_struct().borrow().get_constructor();

                e.insert(ValuePtr::instance(owner, vec![]));

                let fields: Vec<ValuePtr> = it.as_struct()
                    .borrow()
                    .fields()
                    .map(|f| copy_recursive(f, objects))
                    .collect();
                let ptr = objects.get_mut(&it.as_ptr_value())
                    .unwrap(); // This should always exist in the map, since we only insert and just inserted this above

                let mut copy = ptr.as_struct().borrow_mut();
                for f in fields {
                    copy.fields_mut().push(f)
                }

                ptr.clone()
            }
        },

        // Enumerate is immutable, provided it has a reference to the correct underlying value
        // Thus, we just need to recursively call copy() for the inner value, but can create separate instances of the enumerate itself
        Enumerate => ValuePtr::enumerate(copy_recursive(it.as_enumerate().inner(), objects)),

        Iter | Error | Never => panic!("should not copy a synthetic type")
    }
}

fn copy_recursive_rev(it: &Reverse<ValuePtr>, objects: &mut HashMap<usize, ValuePtr, FxBuildHasher>) -> Reverse<ValuePtr> {
    Reverse(copy_recursive(&it.0, objects)) // To support the `Reverse` boxing that heaps do
}