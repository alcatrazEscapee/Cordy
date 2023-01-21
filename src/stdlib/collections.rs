use std::cmp::{Ordering, Reverse};
use std::collections::VecDeque;
use itertools::Itertools;

use crate::vm::error::RuntimeError;
use crate::vm::value::{Iterable, MemoizedImpl, Value};
use crate::vm::VirtualInterface;

use RuntimeError::{*};
use Value::{*};
use crate::misc;
use crate::stdlib::NativeFunction::{SyntheticMemoizedFunction};

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


pub fn list_slice(a1: Value, a2: Value, a3: Value, a4: Value) -> ValueResult {

    let mut slice = a1.as_slice()?;
    let length: i64 = slice.len() as i64;

    let step: i64 = a4.as_int_or(1)?;
    if step == 0 {
        return ValueErrorStepCannotBeZero.err()
    }

    let low: i64 = a2.as_int_or(if step > 0 { 0 } else { -1 })?;
    let high: i64 = a3.as_int_or(if step > 0 { length } else { -length - 1 })?;

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

pub fn min_by<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let iter = a2.as_iter()?;
    match a1.unbox_func_args() {
        Some(Some(2)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            let ret = iter
                .min_by(|a, b| misc::yield_result(&mut err, || {
                    let cmp = vm.invoke_func2(a1.clone(), (*a).clone(), (*b).clone())?.as_int()?;
                    cmp_to_ord(cmp)
                }, Ordering::Equal));

            non_empty(misc::join_result(ret, err)?)
        },
        Some(Some(1)) => {
            let mut err = None;
            let ret = iter
                .min_by_key(|u| misc::yield_result(&mut err, || vm.invoke_func1(a1.clone(), (*u).clone()), Nil));

            non_empty(misc::join_result(ret, err)?)
        },
        Some(_) => TypeErrorArgMustBeCmpOrKeyFunction(a1).err(),
        None => TypeErrorArgMustBeFunction(a1).err(),
    }
}

pub fn max(args: impl Iterator<Item=Value>) -> ValueResult {
    non_empty(args.max())
}

pub fn max_by<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let iter = a2.as_iter()?;
    match a1.unbox_func_args() {
        Some(Some(2)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            let ret = iter
                .max_by(|a, b| misc::yield_result(&mut err, || {
                    let cmp = vm.invoke_func2(a1.clone(), (*a).clone(), (*b).clone())?.as_int()?;
                    cmp_to_ord(cmp)
                }, Ordering::Equal));

            non_empty(misc::join_result(ret, err)?)
        },
        Some(Some(1)) => {
            let mut err = None;
            let ret = iter
                .max_by_key(|u| misc::yield_result(&mut err, || vm.invoke_func1(a1.clone(), (*u).clone()), Nil));

            non_empty(misc::join_result(ret, err)?)
        },
        Some(_) => TypeErrorArgMustBeCmpOrKeyFunction(a1).err(),
        None => TypeErrorArgMustBeFunction(a1).err(),
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
    Ok(Value::iter_list(sorted.into_iter()))
}

pub fn sort_by<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let mut sorted: Vec<Value> = a2.as_iter()?.collect::<Vec<Value>>();
    match a1.unbox_func_args() {
        Some(Some(2)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            sorted.sort_unstable_by(|a, b| misc::yield_result(&mut err, || {
                let cmp = vm.invoke_func2(a1.clone(), a.clone(), b.clone())?.as_int()?;
                cmp_to_ord(cmp)
            }, Ordering::Equal));
            misc::join_result((), err)?
        },
        Some(Some(1)) => {
            let mut err: Option<Box<RuntimeError>> = None;
            sorted.sort_unstable_by_key(|a| misc::yield_result(&mut err, || vm.invoke_func1(a1.clone(), a.clone()), Nil));
            misc::join_result((), err)?
        },
        Some(_) => return TypeErrorArgMustBeCmpOrKeyFunction(a1).err(),
        None => return TypeErrorArgMustBeFunction(a1).err(),
    }
    Ok(Value::iter_list(sorted.into_iter()))
}

pub fn reverse(args: impl Iterator<Item=Value>) -> ValueResult {
    let mut vec = args.collect::<Vec<Value>>();
    vec.reverse();
    Ok(Value::iter_list(vec.into_iter()))
}

pub fn permutations(a1: Value, a2: Value) -> ValueResult {
    let n = a1.as_int()?;
    if n <= 0 {
        return ValueErrorValueMustBeNonNegative(n).err();
    }
    Ok(Value::iter_list(a2.as_iter()?.permutations(n as usize).map(|u| Value::vector(u))))
}

pub fn combinations(a1: Value, a2: Value) -> ValueResult {
    let n = a1.as_int()?;
    if n <= 0 {
        return ValueErrorValueMustBeNonNegative(n).err();
    }
    Ok(Value::iter_list(a2.as_iter()?.combinations(n as usize).map(|u| Value::vector(u))))
}

pub fn any<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    for r in a2.as_iter()? {
        if vm.invoke_func1(a1.clone(), r)?.as_bool() {
            return Ok(Bool(true))
        }
    }
    Ok(Bool(false))
}

pub fn all<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    for r in a2.as_iter()? {
        if !vm.invoke_func1(a1.clone(), r)?.as_bool() {
            return Ok(Bool(false))
        }
    }
    Ok(Bool(true))
}


pub fn map<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = a2.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    for r in a2.as_iter()? {
        acc.push_back(vm.invoke_func1(a1.clone(), r)?);
    }
    Ok(Value::list(acc))
}

pub fn filter<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = a2.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    for r in a2.as_iter()? {
        let ret = vm.invoke_func1(a1.clone(), r.clone())?;
        if ret.as_bool() {
            acc.push_back(r);
        }
    }
    Ok(Value::list(acc))
}

pub fn flat_map<VM>(vm: &mut VM, a1: Option<Value>, a2: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = a2.len().unwrap_or(0);
    let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
    for r in a2.as_iter()? {
        let elem = match &a1 {
            Some(l) => vm.invoke_func1(l.clone(), r)?,
            None => r
        };
        for e in elem.as_iter()? {
            acc.push_back(e);
        }
    }
    Ok(Value::list(acc))
}

pub fn zip(a1: impl Iterator<Item=Value>) -> ValueResult {
    let mut iters = a1
        .map(|v| v.as_iter())
        .collect::<Result<Vec<Iterable>, Box<RuntimeError>>>()?;
    let mut acc = VecDeque::new();
    loop {
        let mut vec = Vec::new();
        for iter in &mut iters {
            match iter.next() {
                Some(it) => vec.push(it),
                None => return Ok(Value::list(acc)),
            }
        }
        acc.push_back(Value::vector(vec));
    }
}

pub fn reduce<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let mut iter = a2.as_iter()?;
    let mut acc: Value = match iter.next() {
        Some(v) => v,
        None => return ValueErrorValueMustBeNonEmpty.err()
    };

    for r in iter {
        acc = vm.invoke_func2(a1.clone(), acc, r)?;
    }
    Ok(acc)
}

pub fn peek(a1: Value) -> ValueResult {
    match match &a1 {
        List(v) => v.unbox().front().cloned(),
        Set(v) => v.unbox().set.first().cloned(),
        Dict(v) => v.unbox().dict.first().map(|u| Value::vector(vec![u.0.clone(), u.1.clone()])),
        Heap(v) => v.unbox().heap.peek().map(|u| u.clone().0),
        Vector(v) => v.unbox().first().cloned(),
        _ => return TypeErrorArgMustBeIterable(a1).err(),
    } {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err(),
    }
}

pub fn pop(a1: Value) -> ValueResult {
    match match &a1 {
        List(v) => v.unbox_mut().pop_back(),
        Set(v) => v.unbox_mut().set.pop(),
        Dict(v) => v.unbox_mut().dict.pop().map(|u| Value::vector(vec![u.0, u.1])),
        Heap(v) => v.unbox_mut().heap.pop().map(|t| t.0),
        _ => return TypeErrorArgMustBeIterable(a1).err()
    } {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn pop_front(a1: Value) -> ValueResult {
    match match &a1 {
        List(v) => v.unbox_mut().pop_front(),
        _ => return TypeErrorArgMustBeIterable(a1).err()
    } {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn push(a1: Value, a2: Value) -> ValueResult {
    match &a2 {
        List(v) => { v.unbox_mut().push_back(a1); Ok(a2) }
        Set(v) => { v.unbox_mut().set.insert(a1); Ok(a2) }
        Heap(v) => { v.unbox_mut().heap.push(Reverse(a1)); Ok(a2) }
        _ => TypeErrorArgMustBeIterable(a2).err()
    }
}

pub fn push_front(a1: Value, a2: Value) -> ValueResult {
    match &a2 {
        List(v) => { v.unbox_mut().push_front(a1); Ok(a2) }
        _ => TypeErrorArgMustBeIterable(a2).err()
    }
}

pub fn insert(a1: Value, a2: Value, a3: Value) -> ValueResult {
    match &a3 {
        List(v) => {
            let index = a1.as_int()?;
            let len = v.unbox().len();
            if 0 <= index && index < v.unbox().len() as i64 {
                v.unbox_mut().insert(index as usize, a2);
                Ok(a3)
            } else if index == len as i64 {
                v.unbox_mut().push_back(a2);
                Ok(a3)
            } else {
                ValueErrorIndexOutOfBounds(index as i64, len).err()
            }
        },
        Dict(v) => { v.unbox_mut().dict.insert(a1, a2); Ok(a3) },
        _ => TypeErrorArgMustBeIndexable(a3).err()
    }
}

pub fn remove(a1: Value, a2: Value) -> ValueResult {
    match &a2 {
        List(v) => {
            let index = a1.as_int()?;
            let len = v.unbox().len();
            if 0 <= index && index < v.unbox().len() as i64 {
                Ok(v.unbox_mut().remove(index as usize).unwrap()) // .unwrap() is safe, as we pre-checked the index
            } else {
                ValueErrorIndexOutOfBounds(index as i64, len).err()
            }
        },
        Set(v) => Ok(Bool(v.unbox_mut().set.remove(&a1))),
        Dict(v) => Ok(Bool(v.unbox_mut().dict.remove(&a1).is_some())),
        _ => TypeErrorArgMustBeIterable(a2).err(),
    }
}

pub fn clear(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => { v.unbox_mut().clear(); Ok(a1) },
        Set(v) => { v.unbox_mut().set.clear(); Ok(a1) },
        Dict(v) => { v.unbox_mut().dict.clear(); Ok(a1) },
        Heap(v) => { v.unbox_mut().heap.clear(); Ok(a1) },
        _ => TypeErrorArgMustBeIterable(a1).err(),
    }
}


pub fn collect_into_dict(iter: impl Iterator<Item=Value>) -> ValueResult {
    Ok(Value::iter_dict(iter.map(|t| {
        let index = t.as_index()?;
        if index.len() == 2 {
            Ok((index.get_index(0), index.get_index(1)))
        } else {
            ValueErrorCannotCollectIntoDict(t.clone()).err()
        }
    }).collect::<Result<Vec<(Value, Value)>, Box<RuntimeError>>>()?.into_iter()))
}

pub fn dict_set_default(a1: Value, a2: Value) -> ValueResult {
    match a2 {
        Dict(it) => {
            it.unbox_mut().default = Some(a1);
            Ok(Dict(it))
        },
        a2 => TypeErrorArgMustBeDict(a2).err()
    }
}

pub fn dict_keys(a1: Value) -> ValueResult {
    match a1 {
        Dict(it) => Ok(Value::iter_set(it.unbox().dict.keys().cloned())),
        a1 => TypeErrorArgMustBeDict(a1).err()
    }
}

pub fn dict_values(a1: Value) -> ValueResult {
    match a1 {
        Dict(it) => Ok(Value::iter_list(it.unbox().dict.values().cloned())),
        a1 => TypeErrorArgMustBeDict(a1).err()
    }
}

pub fn left_find<VM>(vm: &mut VM, a1: Value, a2: Value, return_index: bool) -> ValueResult where VM : VirtualInterface {
    // Supports both find index (`index_of`), and find position (`find`)
    // For predicates, we use the same `enumerate()`, but then either return index, or value
    // For index with value, we use `.position()`
    // For value with value, we just use `.find()`
    let mut iter = a2.as_iter()?;
    if a1.is_function() {
        for (i, v) in iter.enumerate() {
            let ret = vm.invoke_func1(a1.clone(), v.clone())?;
            if ret.as_bool() {
                return Ok(if return_index { Int(i as i64) } else { v })
            }
        }
        Ok(if return_index { Int(-1) } else { Nil })
    } else if return_index {
        Ok(Int(match iter.position(|v| v == a1) {
            Some(i) => i as i64,
            None => -1
        }))
    } else {
        Ok(iter.find(|v| v == &a1).unwrap_or(Nil))
    }
}

pub fn right_find<VM>(vm: &mut VM, a1: Value, a2: Value, return_index: bool) -> ValueResult where VM : VirtualInterface {
    // Identical to the above except we use `.reverse()`, and subtract the index from `len`
    let mut iter = a2.as_iter()?.reverse();
    let len = iter.len();
    if a1.is_function() {
        for (i, v) in iter.enumerate() {
            let ret = vm.invoke_func1(a1.clone(), v.clone())?;
            if ret.as_bool() {
                return Ok(if return_index { Int((len - 1 - i) as i64) } else { v })
            }
        }
        Ok(if return_index { Int(-1) } else { Nil })
    } else if return_index {
        Ok(Int(match iter.position(|v| v == a1) {
            Some(i) => (len - 1 - i) as i64,
            None => -1
        }))
    } else {
        Ok(iter.find(|v| v == &a1).unwrap_or(Nil))
    }
}

pub fn create_memoized(f: Value) -> ValueResult {
    match &f.is_function() {
        true => Ok(PartialNativeFunction(SyntheticMemoizedFunction, Box::new(vec![Memoized(Box::new(MemoizedImpl::new(f)))]))),
        false => TypeErrorArgMustBeFunction(f).err()
    }
}

pub fn invoke_memoized<VM>(vm: &mut VM, a1: Value, args: Vec<Value>) -> ValueResult where VM : VirtualInterface {
    let memoized = match a1 {
        Memoized(it) => it,
        _ => panic!("Missing partial argument for `Memoize`")
    };

    let mut cache = memoized.cache.unbox_mut();
    let mut err: Option<Box<RuntimeError>> = None;
    let ret = cache.entry(args)
        .or_insert_with_key(|args| misc::yield_result(&mut err, || vm.invoke_func(memoized.func.clone(), args), Nil));

    misc::join_result(ret.clone(), err)
}