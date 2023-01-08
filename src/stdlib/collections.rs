use std::cmp::Reverse;
use std::collections::VecDeque;
use hashlink::LinkedHashMap;

use crate::vm::error::RuntimeError;
use crate::vm::value::{Mut, Value, ValueIntoIter};
use crate::vm::VirtualInterface;

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

pub fn list_set_index(list_ref: Mut<VecDeque<Value>>, rhs: i64, value: Value) -> Result<(), Box<RuntimeError>> {
    let mut list = list_ref.unbox_mut();
    let index: usize = to_index(list.len() as i64, rhs) as usize;
    if index < list.len() {
        list[index] = value;
        Ok(())
    } else {
        ValueErrorIndexOutOfBounds(rhs, list.len()).err()
    }
}

pub fn list_slice(a1: Value, a2: Value, a3: Value, a4: Value) -> ValueResult {

    let mut slice = a1.to_slice()?;
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

pub fn range_1(a1: Value) -> ValueResult {
    range_3(Int(0), a1, Int(1))
}

pub fn range_2(a1: Value, a2: Value) -> ValueResult {
    range_3(a1, a2, Int(1))
}

pub fn range_3(a1: Value, a2: Value, a3: Value) -> ValueResult {
    let low: i64 = a1.as_int_or(0)?;
    let high: i64 = a2.as_int()?;
    let step: i64 = a3.as_int_or(1)?;

    if step == 0 {
        ValueErrorStepCannotBeZero.err()
    } else if step > 0 {
        Ok(Value::iter_list((low..high).step_by(step as usize).map(|i| Int(i))))
    } else {
        Ok(Value::iter_list(rev_range(low, high).step_by(-step as usize).map(|i| Int(i))))
    }
}

pub fn enumerate(a1: Value) -> ValueResult {
    match a1.as_iter() {
        Ok(it) => Ok(Value::iter_list((&it).into_iter().cloned().enumerate().map(|(i, v)| Value::vector(vec![Int(i as i64), v])))),
        Err(e) => Err(e)
    }
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

pub fn sum<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    let mut sum: i64 = 0;
    for v in args {
        match v {
            Int(i) => sum += i,
            _ => return TypeErrorArgMustBeInt(v.clone()).err(),
        }
    }
    Ok(Int(sum))
}

pub fn max<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    match args.max() {
        Some(v) => Ok(v.clone()),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn min<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    match args.min() {
        Some(v) => Ok(v.clone()),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn sorted<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    let mut sorted: Vec<Value> = args.cloned().collect::<Vec<Value>>();
    sorted.sort_unstable();
    Ok(Value::iter_list(sorted.into_iter()))
}

pub fn reversed<'a>(args: impl DoubleEndedIterator<Item=&'a Value>) -> ValueResult {
    Ok(Value::iter_list(args.rev().cloned()))
}


pub fn map<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = a2.len().unwrap_or(0);
    match (a1, a2.as_iter()) {
        (l, Ok(rs)) => {
            let rs = (&rs).into_iter();
            let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
            for r in rs {
                vm.push(l.clone());
                vm.push(r.clone());
                let f = vm.invoke_func_eval(1)?;
                vm.run_after_invoke(f)?;
                acc.push_back(vm.pop());
            }
            Ok(Value::list(acc))
        },
        (_, Err(e)) => return Err(e),
    }
}

pub fn filter<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = a2.len().unwrap_or(0);
    match (a1, a2.as_iter()) {
        (l, Ok(rs)) => {
            let rs = (&rs).into_iter();
            let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
            for r in rs {
                vm.push(l.clone());
                vm.push(r.clone());
                let f = vm.invoke_func_eval(1)?;
                vm.run_after_invoke(f)?;
                if vm.pop().as_bool() {
                    acc.push_back(r.clone());
                }
            }
            Ok(Value::list(acc))
        },
        (_, Err(e)) => Err(e),
    }
}

pub fn flat_map<VM>(vm: &mut VM, a1: Option<Value>, a2: Value) -> ValueResult where VM : VirtualInterface {
    let len: usize = a2.len().unwrap_or(0);
    match (a1, a2.as_iter()) {
        (l, Ok(rs)) => {
            let rs = (&rs).into_iter();
            let mut acc: VecDeque<Value> = VecDeque::with_capacity(len);
            for r in rs {
                let elem = if let Some(l) = &l {
                    vm.push(l.clone());
                    vm.push(r.clone());
                    let f = vm.invoke_func_eval(1)?;
                    vm.run_after_invoke(f)?;
                    vm.pop()
                } else {
                    r.clone()
                };
                for e in elem.as_iter()?.into_iter() {
                    acc.push_back(e.clone());
                }
            }
            Ok(Value::list(acc))
        },
        (_, Err(e)) => Err(e),
    }
}

pub fn zip<'a>(a1: impl Iterator<Item=&'a Value>) -> ValueResult {
    // This is convoluted due to rust borrow semantics
    // We cannot zip a vector of iterators, rust doesn't support anything of the sort
    // And due to `Value`'s annoying `into_iter()` only functioning for `&'b ValueIntoIter`, it does not work properly if we were to do something like the following
    // https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=90c1cc7997177df305f549da39a9aaef
    let iters = a1
        .map(|v| v.as_iter())
        .collect::<Result<Vec<ValueIntoIter>, Box<RuntimeError>>>()?;
    let len: usize = iters.len();
    let mut iters = iters.iter();
    let mut acc: VecDeque<Vec<Value>> = iters.next()
        .unwrap()
        .into_iter()
        .map(|u| {
            let mut v = Vec::with_capacity(len);
            v.push(u.clone());
            v
        })
        .collect::<VecDeque<Vec<Value>>>();
    let mut min_len: usize = acc.len();
    for iter in iters {
        let mut i: usize = 0;
        for it in iter.into_iter() {
            if i >= min_len {
                break
            }
            acc[i].push(it.clone());
            i += 1;
        }
        if i < min_len {
            min_len = i;
        }
    }
    Ok(Value::iter_list(acc.into_iter().take(min_len).map(|u| Value::vector(u))))
}

pub fn reduce<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    match (a1, a2.as_iter()) {
        (l, Ok(rs)) => {
            let mut iter = (&rs).into_iter().cloned();
            let mut acc: Value = match iter.next() {
                Some(v) => v,
                None => return ValueErrorValueMustBeNonEmpty.err()
            };

            for r in iter {
                vm.push(l.clone()); // Function
                vm.push(acc); // Accumulator (arg1)
                vm.push(r); // Value (arg2)
                let f = vm.invoke_func_eval(2)?;
                vm.run_after_invoke(f)?;
                acc = vm.pop();
            }
            Ok(acc)
        },
        (_, Err(e)) => Err(e),
    }
}

pub fn pop(a1: Value) -> ValueResult {
    match match &a1 {
        List(v) => v.unbox_mut().pop_back(),
        Set(v) => v.unbox_mut().pop_back(),
        Dict(v) => v.unbox_mut().dict.pop_back().map(|(k, _)| k),
        Heap(v) => v.unbox_mut().heap.pop().map(|t| t.0),
        _ => return TypeErrorArgMustBeIterable(a1).err()
    } {
        Some(v) => Ok(v),
        None => ValueErrorValueMustBeNonEmpty.err()
    }
}

pub fn push(a1: Value, a2: Value) -> ValueResult {
    match &a2 {
        List(v) => { v.unbox_mut().push_back(a1); Ok(a2) }
        Set(v) => { v.unbox_mut().insert(a1); Ok(a2) }
        Heap(v) => { v.unbox_mut().heap.push(Reverse(a1)); Ok(a2) }
        _ => TypeErrorArgMustBeIterable(a2).err()
    }
}

pub fn last(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => Ok(v.unbox_mut().back().cloned().unwrap_or(Nil)),
        Set(v) => Ok(v.unbox_mut().back().cloned().unwrap_or(Nil)),
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn head(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => Ok(v.unbox().front().cloned().unwrap_or(Nil)),
        Set(v) => Ok(v.unbox().front().cloned().unwrap_or(Nil)),
        Heap(v) => Ok(v.unbox().heap.peek().cloned().map(|t| t.0).unwrap_or(Nil)),
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn init(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => {
            let v = v.unbox();
            let mut iter = v.iter();
            iter.next_back();
            Ok(Value::iter_list(iter.cloned()))
        },
        Set(v) => {
            let v = v.unbox();
            let mut iter = v.iter();
            iter.next_back();
            Ok(Value::iter_set(iter.cloned()))
        },
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn tail(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => Ok(Value::iter_list(v.unbox().iter().skip(1).cloned())),
        Set(v) => Ok(Value::iter_set(v.unbox().iter().skip(1).cloned())),
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn collect_into_dict(iter: impl Iterator<Item=Value>) -> ValueResult {
    Ok(Value::dict(iter.map(|t| {
        let index = t.to_index()?;
        if index.len() == 2 {
            Ok((index.get_index(0), index.get_index(1)))
        } else {
            ValueErrorCannotCollectIntoDict(t.clone()).err()
        }
    }).collect::<Result<Vec<(Value, Value)>, Box<RuntimeError>>>()?
        .into_iter()
        .collect::<LinkedHashMap<Value, Value>>()))
}

pub fn dict_set_default(a1: Value, a2: Value) -> ValueResult {
    if let Dict(it) = a2 {
        it.unbox_mut().default = Some(a1);
        Ok(Dict(it))
    } else {
        TypeErrorArgMustBeDict(a2).err()
    }
}
