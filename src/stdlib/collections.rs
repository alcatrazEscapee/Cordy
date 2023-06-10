use std::cmp::{Ordering, Reverse};
use std::collections::VecDeque;

use itertools::Itertools;

use crate::{misc, vm};
use crate::vm::{IntoDictValue, IntoIterableValue, IntoValue, Iterable, RuntimeError, Value, VirtualInterface};

use RuntimeError::{*};
use Value::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


/// Converts a `i64` index into a bounds-checked, `usize` index which can safely be used to index (via `Value.to_index().get_index()`) a `Value`
pub fn get_checked_index(len: usize, rhs: i64) -> Result<usize, Box<RuntimeError>> {
    let index: usize = to_index(len as i64, rhs) as usize;
    if index < len {
        Ok(index)
    } else {
        ValueErrorIndexOutOfBounds(rhs, len).err()
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
fn to_index(len: i64, pos_or_neg: i64) -> i64 {
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
    match args.min() {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn min_by<VM>(vm: &mut VM, by: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let iter = args.as_iter()?;
    match by.unbox_func_args() {
        Some(Some(2)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            let ret = iter
                .min_by(|a, b| misc::yield_result(&mut err, || {
                    let cmp = vm.invoke_func2(by.clone(), (*a).clone(), (*b).clone())?.as_int()?;
                    cmp_to_ord(cmp)
                }, Ordering::Equal));

            non_empty(misc::join_result(ret, err)?)
        },
        Some(Some(1)) => {
            let mut err = None;
            let ret = iter
                .min_by_key(|u| misc::yield_result(&mut err, || vm.invoke_func1(by.clone(), (*u).clone()), Nil));

            non_empty(misc::join_result(ret, err)?)
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
    match by.unbox_func_args() {
        Some(Some(2)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            let ret = iter
                .max_by(|a, b| misc::yield_result(&mut err, || {
                    let cmp = vm.invoke_func2(by.clone(), (*a).clone(), (*b).clone())?.as_int()?;
                    cmp_to_ord(cmp)
                }, Ordering::Equal));

            non_empty(misc::join_result(ret, err)?)
        },
        Some(Some(1)) => {
            let mut err = None;
            let ret = iter
                .max_by_key(|u| misc::yield_result(&mut err, || vm.invoke_func1(by.clone(), (*u).clone()), Nil));

            non_empty(misc::join_result(ret, err)?)
        },
        Some(_) => TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => TypeErrorArgMustBeFunction(by).err(),
    }
}

#[inline(always)]
fn non_empty(it: Option<Value>) -> ValueResult {
    match it {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

#[inline(always)]
fn cmp_to_ord<E>(i: i64) -> Result<Ordering, E> {
    Ok(if i == 0 {
        Ordering::Equal
    } else if i > 0 {
        Ordering::Greater
    } else {
        Ordering::Less
    })
}


pub fn sort(args: impl Iterator<Item=Value>) -> ValueResult {
    let mut sorted: Vec<Value> = args.collect::<Vec<Value>>();
    sorted.sort_unstable();
    Ok(sorted.into_iter().to_list())
}

pub fn sort_by<VM>(vm: &mut VM, by: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let mut sorted: Vec<Value> = args.as_iter()?.collect::<Vec<Value>>();
    match by.unbox_func_args() {
        Some(Some(2)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            sorted.sort_unstable_by(|a, b| misc::yield_result(&mut err, || {
                let cmp = vm.invoke_func2(by.clone(), a.clone(), b.clone())?.as_int()?;
                cmp_to_ord(cmp)
            }, Ordering::Equal));
            misc::join_result((), err)?
        },
        Some(Some(1)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            sorted.sort_unstable_by_key(|a| misc::yield_result(&mut err, || vm.invoke_func1(by.clone(), a.clone()), Nil));
            misc::join_result((), err)?
        },
        Some(_) => return TypeErrorArgMustBeCmpOrKeyFunction(by).err(),
        None => return TypeErrorArgMustBeFunction(by).err(),
    }
    Ok(sorted.into_iter().to_list())
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
    for r in args.as_iter()? {
        if vm.invoke_func1(f.clone(), r)?.as_bool() {
            return Ok(Bool(true))
        }
    }
    Ok(Bool(false))
}

pub fn all<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    for r in args.as_iter()? {
        if !vm.invoke_func1(f.clone(), r)?.as_bool() {
            return Ok(Bool(false))
        }
    }
    Ok(Bool(true))
}


pub fn map<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    for r in args.as_iter()? {
        acc.push_back(vm.invoke_func1(f.clone(), r)?);
    }
    Ok(acc.to_value())
}

pub fn filter<VM>(vm: &mut VM, f: Value, args: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    for r in args.as_iter()? {
        let ret = vm.invoke_func1(f.clone(), r.clone())?;
        if ret.as_bool() {
            acc.push_back(r);
        }
    }
    Ok(acc.to_value())
}

pub fn flat_map<VM>(vm: &mut VM, f: Option<Value>, args: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = args.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    for r in args.as_iter()? {
        let elem = match &f {
            Some(l) => vm.invoke_func1(l.clone(), r)?,
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
    let mut acc = VecDeque::new();
    loop {
        let mut vec = Vec::new();
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

    for r in iter {
        acc = vm.invoke_func2(f.clone(), acc, r)?;
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
            true => ValueErrorRecursiveHash(target).err(),
            false => Ok(target)
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
                ValueErrorIndexOutOfBounds(index as i64, len).err()
            }
        },
        Dict(v) => match vm::guard_recursive_hash(|| v.unbox_mut().dict.insert(index, value)) {
            true => ValueErrorRecursiveHash(target).err(),
            false => Ok(target)
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
                ValueErrorIndexOutOfBounds(index as i64, len).err()
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
    Ok(iter.map(|t| {
        let index = t.as_index()?;
        if index.len() == 2 {
            Ok((index.get_index(0), index.get_index(1)))
        } else {
            ValueErrorCannotCollectIntoDict(t.clone()).err()
        }
    })
        .collect::<Result<Vec<(Value, Value)>, Box<RuntimeError>>>()?
        .into_iter()
        .to_dict())
}

pub fn dict_set_default(def: Value, target: Value) -> ValueResult {
    match target {
        Dict(it) => {
            it.unbox_mut().default = Some(def);
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
        for (i, v) in iter.enumerate() {
            let ret = vm.invoke_func1(finder.clone(), v.clone())?;
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
        for (i, v) in iter.enumerate() {
            let ret = vm.invoke_func1(finder.clone(), v.clone())?;
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

pub fn invoke_memoized<VM>(vm: &mut VM, memoized: Value, args: Vec<Value>) -> ValueResult where VM : VirtualInterface {
    let memoized = match memoized {
        Memoized(it) => it,
        _ => panic!("Missing partial argument for `Memoize`")
    };

    // We cannot use the `.entry()` API, as that requires we mutably borrow the cache during the call to `vm.invoke_func()`
    // We only lookup by key once (in the cached case), and twice (in the uncached case)
    {
        let cache = memoized.cache.unbox();
        match cache.get(&args) {
            Some(ret) => return Ok(ret.clone()),
            None => {}
        }
    } // cache falls out of scope, and thus is no longer borrowed

    let value: Value = vm.invoke_func(memoized.func.clone(), &args)?;

    // The above computation might've entered a value into the cache - so we have to go through `.entry()` again
    return Ok(memoized.cache.unbox_mut()
        .entry(args)
        .or_insert(value)
        .clone());
}