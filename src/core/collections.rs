use std::cmp::{Ordering, Reverse};
use std::collections::VecDeque;
use fxhash::FxBuildHasher;
use indexmap::IndexMap;

use itertools::Itertools;

use crate::{util, vm};
use crate::core::{InvokeArg0, InvokeArg1, InvokeArg2};
use crate::vm::{IntoDictValue, IntoIterableValue, IntoValue, Iterable, RuntimeError, Value, ValueResult, VirtualInterface};

use RuntimeError::{*};
use Value::{*};


pub fn get_index<VM>(vm: &mut VM, target: Value, index: Value) -> ValueResult where VM : VirtualInterface {
    if target.is_dict() {
        return get_dict_index(vm, target, index);
    }

    let indexable = target.as_index()?;
    let index: usize = indexable.check_index(index)?;

    Ok(indexable.get_index(index))
}

fn get_dict_index<VM>(vm: &mut VM, dict: Value, key: Value) -> ValueResult where VM : VirtualInterface {
    // Dict objects have their own overload of indexing to mean key-value lookups, that doesn't fit with ValueAsIndex (as it doesn't take integer keys, always)
    // The handling for this is a bit convoluted due to `clone()` issues, and possible cases of default / no default / functional default

    // Initially unbox (non mutable) to clone out the default value.
    // If the default is a function, we can't have a reference out of the dict while we're accessing the default.

    let dict = match dict { Dict(it) => it, _ => panic!() };
    let default_factory: InvokeArg0;
    {
        let dict = dict.unbox();
        match &dict.default {
            Some(default) => match dict.dict.get(&key) {
                Some(existing_value) => return Ok(existing_value.clone()),
                None => {
                    // We need to insert, so fallthrough as we need to drop the borrow on `dict`
                    default_factory = default.clone();
                },
            },
            None => return match dict.dict.get(&key) {
                Some(existing_value) => Ok(existing_value.clone()),
                None => ValueErrorKeyNotPresent(key).err()
            },
        }
    }

    // Invoke the new value supplier - this might modify the dict
    // We go through the `.entry()` API again in this case
    let new_value: Value = default_factory.invoke(vm)?;
    let mut dict = dict.unbox_mut();
    Ok(dict.dict.entry(key).or_insert(new_value).clone())
}

pub fn set_index(target: &Value, index: Value, value: Value) -> Result<(), Box<RuntimeError>> {

    if let Dict(it) = target {
        match vm::guard_recursive_hash(|| it.unbox_mut().dict.insert(index, value)) {
            Err(_) => ValueErrorRecursiveHash(target.clone()).err(),
            Ok(_) => Ok(())
        }
    } else {
        let mut indexable = target.as_index()?;
        let index: usize = indexable.check_index(index)?;

        indexable.set_index(index, value)
    }
}


pub fn list_slice(slice: Value, low: Value, high: Value, step: Value) -> ValueResult {
    literal_slice(slice, low.as_int_or()?, high.as_int_or()?, step.as_int_or()?)
}

pub fn literal_slice(target: Value, low: Option<i64>, high: Option<i64>, step: Option<i64>) -> ValueResult {
    let mut slice = target.as_slice()?;
    let length: i64 = slice.len() as i64;

    let step: i64 = step.unwrap_or(1);
    if step == 0 {
        return ValueErrorStepCannotBeZero.err()
    }

    let low: i64 = low.unwrap_or(if step > 0 { 0 } else { -1 });
    let high: i64 = high.unwrap_or(if step > 0 { length } else { -length - 1 });

    let abs_start: i64 = to_index(length, low);
    let abs_stop: i64 = to_index(length, high);
    let abs_step: usize = step.unsigned_abs() as usize;

    if step > 0 {
        for i in (abs_start..abs_stop).step_by(abs_step) {
            slice.accept(i)
        }
    } else {
        for i in rev_range(abs_start, abs_stop).step_by(abs_step) {
            slice.accept(i)
        }
    }

    Ok(slice.to_value())
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
fn rev_range(start_high_inclusive: i64, stop_low_exclusive: i64) -> impl Iterator<Item = i64> {
    let mut start: i64 = start_high_inclusive;
    let end: i64 = stop_low_exclusive;
    std::iter::from_fn(move || {
        if start <= end {
            None
        } else {
            start -= 1;
            Some(start + 1)
        }
    })
}


// ===== Library Functions ===== //


pub fn sum(args: impl Iterator<Item=Value>) -> ValueResult {
    let mut sum: i64 = 0;
    for v in args {
        sum += v.as_int()?;
    }
    Ok(Int(sum))
}

pub fn min(args: impl Iterator<Item=Value>) -> ValueResult {
    non_empty(args.min())
}

pub fn min_by<VM>(vm: &mut VM, by: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let iter = args.as_iter()?;
    match by.min_nargs() {
        Some(2) => {
            let by: InvokeArg2 = InvokeArg2::from(by)?;
            let mut err: Option<Box<RuntimeError>> = None;
            let ret = iter.min_by(|a, b|
                util::yield_result(&mut err, ||
                    Ok(by.invoke((*a).clone(), (*b).clone(), vm)?.as_int()?.cmp(&0)), Ordering::Equal));
            non_empty(util::join_result(ret, err)?)
        },
        Some(1) => {
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            let mut err = None;
            let ret = iter.min_by_key(|u|
                util::yield_result(&mut err, ||
                    by.invoke((*u).clone(), vm), Nil));
            non_empty(util::join_result(ret, err)?)
        },
        Some(_) => TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => TypeErrorArgMustBeFunction(by).err(),
    }
}

pub fn max(args: impl Iterator<Item=Value>) -> ValueResult {
    non_empty(args.max())
}

pub fn max_by<VM>(vm: &mut VM, by: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let iter = args.as_iter()?;
    match by.min_nargs() {
        Some(2) => {
            let by: InvokeArg2 = InvokeArg2::from(by)?;
            let mut err: Option<Box<RuntimeError>> = None;
            let ret = iter.max_by(|a, b|
                util::yield_result(&mut err, ||
                    Ok(by.invoke((*a).clone(), (*b).clone(), vm)?.as_int()?.cmp(&0)), Ordering::Equal));
            non_empty(util::join_result(ret, err)?)
        },
        Some(1) => {
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            let mut err = None;
            let ret = iter.max_by_key(|u|
                util::yield_result(&mut err, ||
                    by.invoke((*u).clone(), vm), Nil));
            non_empty(util::join_result(ret, err)?)
        },
        Some(_) => TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => TypeErrorArgMustBeFunction(by).err(),
    }
}


pub fn sort(args: impl Iterator<Item=Value>) -> ValueResult {
    let mut sorted: Vec<Value> = args.collect::<Vec<Value>>();
    sorted.sort_unstable();
    Ok(sorted.into_iter().to_list())
}

pub fn sort_by<VM : VirtualInterface>(vm: &mut VM, by: Value, args: Value) -> ValueResult {
    let mut sorted: Vec<Value> = args.as_iter()?.collect::<Vec<Value>>();
    match by.min_nargs() {
        Some(2) => {
            let by: InvokeArg2 = InvokeArg2::from(by)?;
            let mut err: Option<Box<RuntimeError>> = None;
            sorted.sort_unstable_by(|a, b|
                util::yield_result(&mut err, ||
                    Ok(by.invoke(a.clone(), b.clone(), vm)?.as_int()?.cmp(&0)), Ordering::Equal));
            util::join_result((), err)?
        },
        Some(1) => {
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            let mut err: Option<Box<RuntimeError>> = None;
            sorted.sort_unstable_by_key(|a|
                util::yield_result(&mut err, ||
                    by.invoke(a.clone(), vm), Nil));
            util::join_result((), err)?
        },
        Some(_) => return TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => return TypeErrorArgMustBeFunction(by).err(),
    }
    Ok(sorted.into_iter().to_list())
}

#[inline]
fn non_empty(it: Option<Value>) -> ValueResult {
    match it {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}


pub fn group_by<VM : VirtualInterface>(vm: &mut VM, by: Value, args: Value) -> ValueResult {
    let iter = args.as_iter()?;
    Ok(match by {
        Int(i) => {
            // `group_by(n, iter) will return a list of vectors of `n` values each. Last value will have whatever, instead of raising an error
            if i <= 0 {
                return ValueErrorValueMustBePositive(i).err()
            }
            let size: usize = i as usize;
            let mut groups: VecDeque<Value> = VecDeque::with_capacity(1 + iter.len() / size); // Accurate guess
            let mut group: Vec<Value> = Vec::with_capacity(size);
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
            groups.to_value()
        },
        _ => {
            // Otherwise, we assume this is a group_by(f), in which case we assume the function to be a item -> key, and create a dictionary of keys -> vector of values
            // For capacity, we guess that we're halving. That seems to be a reasonable compromise between overestimating, and optimal values.
            let mut groups: IndexMap<Value, Value, FxBuildHasher> = IndexMap::with_capacity_and_hasher(iter.len() / 2, FxBuildHasher::default());
            let by: InvokeArg1 = InvokeArg1::from(by)?;
            for value in iter {
                let key = by.invoke(value.clone(), vm)?;
                match groups.entry(key)
                    .or_insert_with(|| Vec::new().to_value()) {
                    Vector(it) => it.unbox_mut().push(value),
                    _ => panic!("Expected only vectors"),
                }
            }
            groups.to_value()
        }
    })
}

pub fn reverse(args: impl Iterator<Item=Value>) -> ValueResult {
    let mut vec = args.collect::<Vec<Value>>();
    vec.reverse();
    Ok(vec.into_iter().to_list())
}

pub fn permutations(n: Value, args: Value) -> ValueResult {
    let n = n.as_int()?;
    if n <= 0 {
        return ValueErrorValueMustBeNonNegative(n).err();
    }
    Ok(args.as_iter()?.permutations(n as usize).map(|u| u.to_value()).to_list())
}

pub fn combinations(n: Value, args: Value) -> ValueResult {
    let n = n.as_int()?;
    if n <= 0 {
        return ValueErrorValueMustBeNonNegative(n).err();
    }
    Ok(args.as_iter()?.combinations(n as usize).map(|u| u.to_value()).to_list())
}

pub fn any<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let f: InvokeArg1 = InvokeArg1::from(f)?;
    for r in args.as_iter()? {
        if f.invoke(r, vm)?.as_bool() {
            return Ok(Bool(true))
        }
    }
    Ok(Bool(false))
}

pub fn all<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let f: InvokeArg1 = InvokeArg1::from(f)?;
    for r in args.as_iter()? {
        if !f.invoke(r, vm)?.as_bool() {
            return Ok(Bool(false))
        }
    }
    Ok(Bool(true))
}


pub fn map<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    let f: InvokeArg1 = InvokeArg1::from(f)?;
    for r in args.as_iter()? {
        acc.push_back(f.invoke(r, vm)?);
    }
    Ok(acc.to_value())
}

pub fn filter<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    let f: InvokeArg1 = InvokeArg1::from(f)?;
    for r in args.as_iter()? {
        let ret = f.invoke(r.clone(), vm)?;
        if ret.as_bool() {
            acc.push_back(r);
        }
    }
    Ok(acc.to_value())
}

pub fn flat_map<VM>(vm: &mut VM, f: Option<Value>, args: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    let f: Option<InvokeArg1> = match f {
        Some(f) => Some(InvokeArg1::from(f)?),
        None => None,
    };
    for r in args.as_iter()? {
        let elem = match &f {
            Some(l) => l.invoke(r, vm)?,
            None => r
        };
        for e in elem.as_iter()? {
            acc.push_back(e);
        }
    }
    Ok(acc.to_value())
}

pub fn zip(args: impl Iterator<Item=Value>) -> ValueResult {
    let mut iters = args
        .map(|v| v.as_iter())
        .collect::<Result<Vec<Iterable>, Box<RuntimeError>>>()?;
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
                None => return Ok(acc.to_value()),
            }
        }
        acc.push_back(vec.to_value());
    }
}

pub fn reduce<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let mut iter = args.as_iter()?;
    let mut acc: Value = match iter.next() {
        Some(v) => v,
        None => return ValueErrorValueMustBeNonEmpty.err()
    };

    let f: InvokeArg2 = InvokeArg2::from(f)?;
    for r in iter {
        acc = f.invoke(acc, r, vm)?;
    }
    Ok(acc)
}

pub fn peek(target: Value) -> ValueResult {
    match match &target {
        List(v) => v.unbox().front().cloned(),
        Set(v) => v.unbox().set.first().cloned(),
        Dict(v) => v.unbox().dict.first().map(|u| vec![u.0.clone(), u.1.clone()].to_value()),
        Heap(v) => v.unbox().heap.peek().map(|u| u.clone().0),
        Vector(v) => v.unbox().first().cloned(),
        _ => return TypeErrorArgMustBeIterable(target).err(),
    } {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err(),
    }
}

pub fn pop(target: Value) -> ValueResult {
    match match &target {
        List(v) => v.unbox_mut().pop_back(),
        Set(v) => v.unbox_mut().set.pop(),
        Dict(v) => v.unbox_mut().dict.pop().map(|u| vec![u.0, u.1].to_value()),
        Heap(v) => v.unbox_mut().heap.pop().map(|t| t.0),
        _ => return TypeErrorArgMustBeIterable(target).err()
    } {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn pop_front(target: Value) -> ValueResult {
    match match &target {
        List(v) => v.unbox_mut().pop_front(),
        _ => return TypeErrorArgMustBeIterable(target).err()
    } {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn push(value: Value, target: Value) -> ValueResult {
    match &target {
        List(v) => { v.unbox_mut().push_back(value); Ok(target) }
        Set(v) => match vm::guard_recursive_hash(|| v.unbox_mut().set.insert(value)) {
            Err(_) => ValueErrorRecursiveHash(target).err(),
            Ok(_) => Ok(target)
        }
        Heap(v) => { v.unbox_mut().heap.push(Reverse(value)); Ok(target) }
        _ => TypeErrorArgMustBeIterable(target).err()
    }
}

pub fn push_front(value: Value, target: Value) -> ValueResult {
    match &target {
        List(v) => { v.unbox_mut().push_front(value); Ok(target) }
        _ => TypeErrorArgMustBeIterable(target).err()
    }
}

pub fn insert(index: Value, value: Value, target: Value) -> ValueResult {
    match &target {
        List(v) => {
            let index = index.as_int()?;
            let len = v.unbox().len();
            if 0 <= index && index < v.unbox().len() as i64 {
                v.unbox_mut().insert(index as usize, value);
                Ok(target)
            } else if index == len as i64 {
                v.unbox_mut().push_back(value);
                Ok(target)
            } else {
                ValueErrorIndexOutOfBounds(index, len).err()
            }
        },
        Dict(v) => match vm::guard_recursive_hash(|| v.unbox_mut().dict.insert(index, value)) {
            Err(_) => ValueErrorRecursiveHash(target).err(),
            Ok(_) => Ok(target)
        },
        _ => TypeErrorArgMustBeIndexable(target).err()
    }
}

pub fn remove(needle: Value, target: Value) -> ValueResult {
    match &target {
        List(v) => {
            let index = needle.as_int()?;
            let len = v.unbox().len();
            if 0 <= index && index < v.unbox().len() as i64 {
                Ok(v.unbox_mut().remove(index as usize).unwrap()) // .unwrap() is safe, as we pre-checked the index
            } else {
                ValueErrorIndexOutOfBounds(index, len).err()
            }
        },
        Set(v) => Ok(Bool(v.unbox_mut().set.remove(&needle))),
        Dict(v) => Ok(Bool(v.unbox_mut().dict.remove(&needle).is_some())),
        _ => TypeErrorArgMustBeIterable(target).err(),
    }
}

pub fn clear(target: Value) -> ValueResult {
    match &target {
        List(v) => { v.unbox_mut().clear(); Ok(target) },
        Set(v) => { v.unbox_mut().set.clear(); Ok(target) },
        Dict(v) => { v.unbox_mut().dict.clear(); Ok(target) },
        Heap(v) => { v.unbox_mut().heap.clear(); Ok(target) },
        _ => TypeErrorArgMustBeIterable(target).err(),
    }
}


pub fn collect_into_dict(iter: impl Iterator<Item=Value>) -> ValueResult {
    Ok(iter.map(|t| t.as_pair())
        .collect::<Result<Vec<(Value, Value)>, Box<RuntimeError>>>()?
        .into_iter()
        .to_dict())
}

pub fn dict_set_default(def: Value, target: Value) -> ValueResult {
    match target {
        Dict(it) => {
            it.unbox_mut().default = Some(if def.is_function() {
                InvokeArg0::from(def)?
            } else {
                InvokeArg0::Noop(def) // Treat single argument defaults still as a function, which is optimized to just copy its value
            });
            Ok(Dict(it))
        },
        a2 => TypeErrorArgMustBeDict(a2).err()
    }
}

pub fn dict_keys(target: Value) -> ValueResult {
    match target {
        Dict(it) => Ok(it.unbox().dict.keys().cloned().to_set()),
        a1 => TypeErrorArgMustBeDict(a1).err()
    }
}

pub fn dict_values(target: Value) -> ValueResult {
    match target {
        Dict(it) => Ok(it.unbox().dict.values().cloned().to_list()),
        a1 => TypeErrorArgMustBeDict(a1).err()
    }
}

pub fn left_find<VM>(vm: &mut VM, finder: Value, args: Value, return_index: bool) -> ValueResult where VM : VirtualInterface {
    // Supports both find index (`index_of`), and find position (`find`)
    // For predicates, we use the same `enumerate()`, but then either return index, or value
    // For index with value, we use `.position()`
    // For value with value, we just use `.find()`
    let mut iter = args.as_iter()?;
    if finder.is_function() {
        let finder: InvokeArg1 = InvokeArg1::from(finder)?;
        for (i, v) in iter.enumerate() {
            let ret = finder.invoke(v.clone(), vm)?;
            if ret.as_bool() {
                return Ok(if return_index { Int(i as i64) } else { v })
            }
        }
        Ok(if return_index { Int(-1) } else { Nil })
    } else if return_index {
        Ok(Int(match iter.position(|v| v == finder) {
            Some(i) => i as i64,
            None => -1
        }))
    } else {
        Ok(iter.find(|v| v == &finder).unwrap_or(Nil))
    }
}

pub fn right_find<VM>(vm: &mut VM, finder: Value, args: Value, return_index: bool) -> ValueResult where VM : VirtualInterface {
    // Identical to the above except we use `.reverse()`, and subtract the index from `len`
    let mut iter = args.as_iter()?.reverse();
    let len = iter.len();
    if finder.is_function() {
        let finder: InvokeArg1 = InvokeArg1::from(finder)?;
        for (i, v) in iter.enumerate() {
            let ret = finder.invoke(v.clone(), vm)?;
            if ret.as_bool() {
                return Ok(if return_index { Int((len - 1 - i) as i64) } else { v })
            }
        }
        Ok(if return_index { Int(-1) } else { Nil })
    } else if return_index {
        Ok(Int(match iter.position(|v| v == finder) {
            Some(i) => (len - 1 - i) as i64,
            None => -1
        }))
    } else {
        Ok(iter.find(|v| v == &finder).unwrap_or(Nil))
    }
}

pub fn create_memoized(f: Value) -> ValueResult {
    match &f.is_function() {
        true => Ok(Value::memoized(f)),
        false => TypeErrorArgMustBeFunction(f).err()
    }
}