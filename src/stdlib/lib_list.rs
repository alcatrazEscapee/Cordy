use std::collections::VecDeque;

use crate::vm::error::RuntimeError;
use crate::vm::value::{Mut, Value};
use crate::vm::VirtualInterface;

use RuntimeError::{*};
use Value::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


macro_rules! slice_arg {
    ($a:ident, $n:expr, $def:expr) => {
        match $a {
            Value::Int(x) => x,
            Value::Nil => $def,
            t => return TypeErrorSliceArgMustBeInt($n, t).err(),
        }
    };
}


pub fn list_index(list_ref: Mut<VecDeque<Value>>, r: i64) -> ValueResult {
    let list = list_ref.borrow();
    let index: usize = if r < 0 { (list.len() as i64 + r) as usize } else { r as usize };
    if index < list.len() {
        Ok(list[index].clone())
    } else {
        IndexOutOfBounds(r, list.len()).err()
    }
}

pub fn list_slice(a1: Value, a2: Value, a3: Value, a4: Value) -> ValueResult {

    let list_ref= match a1 {
        List(ls) => ls,
        t => return TypeErrorCannotSlice(t).err(),
    };
    let list = list_ref.borrow();
    let length: i64 = list.len() as i64;

    let step: i64 = slice_arg!(a4, "step", 1);
    if step == 0 {
        return SliceStepZero.err()
    }

    let low: i64 = slice_arg!(a2, "low", if step > 0 { 0 } else { -1 });
    let high: i64 = slice_arg!(a3, "high", if step > 0 { 0 } else { -length - 1 });

    let abs_start: i64 = to_index(length, low);
    let abs_step: usize = step.unsigned_abs() as usize;

    return Ok(if step > 0 {
        let abs_stop: i64 = to_index(length, high - 1);

        Value::iter((abs_start..=abs_stop).step_by(abs_step)
            .filter_map(|i| safe_get(&list, i))
            .cloned())
    } else {
        let abs_stop: i64 = to_index(length, high);

        Value::iter(rev_range(abs_start, abs_stop).step_by(abs_step)
            .filter_map(|i| safe_get(&list, i))
            .cloned())
    })
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
fn safe_get(list: &VecDeque<Value>, index: i64) -> Option<&Value> {
    if index < 0 {
        None
    } else {
        list.get(index as usize)
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

macro_rules! declare_varargs_iter {
    ($varargs:ident, $iter:ident) => {
        pub fn $varargs(arg: Value) -> ValueResult {
            match arg {
                List(v) => $iter(v.borrow().iter()),
                Set(v) => $iter(v.borrow().iter()),
                Dict(v) => $iter(v.borrow().keys()),
                e => TypeErrorArgMustBeIterable(e).err(),
            }
        }
    };
}

declare_varargs_iter!(sum_varargs, sum_iter);
declare_varargs_iter!(min_varargs, min_iter);
declare_varargs_iter!(max_varargs, max_iter);
declare_varargs_iter!(sorted_varargs, sorted_iter);
declare_varargs_iter!(reversed_varargs, reversed_iter);


pub fn sum_iter<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    let mut sum: i64 = 0;
    for v in args {
        match v {
            Int(i) => sum += *i,
            _ => return TypeErrorArgMustBeInt(v.clone()).err(),
        }
    }
    Ok(Int(sum))
}

pub fn max_iter<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    match args.max() {
        Some(v) => Ok(v.clone()),
        None => ValueErrorMaxArgMustBeNonEmptySequence.err()
    }
}

pub fn min_iter<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    match args.min() {
        Some(v) => Ok(v.clone()),
        None => ValueErrorMaxArgMustBeNonEmptySequence.err()
    }
}

pub fn sorted_iter<'a>(args: impl Iterator<Item=&'a Value>) -> ValueResult {
    let mut sorted: Vec<Value> = args.cloned().collect::<Vec<Value>>();
    sorted.sort_unstable();
    Ok(Value::list(sorted))
}

pub fn reversed_iter<'a>(args: impl DoubleEndedIterator<Item=&'a Value>) -> ValueResult {
    Ok(Value::list(args
        .cloned()
        .rev()
        .collect::<Vec<Value>>()))
}



pub fn map<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    match (a1, a2) {
        (l, List(rs)) => {
            let rs = rs.borrow();
            let mut acc: Vec<Value> = Vec::with_capacity(rs.len());
            for r in rs.iter() {
                vm.push(r.clone());
                vm.push(l.clone());
                let f = match vm.invoke_func_compose() {
                    Err(e) => return e.err(),
                    Ok(f) => f
                };
                vm.run_after_invoke(f)?;
                acc.push(vm.pop());
            }
            Ok(Value::list(acc))
        },
        (_, r) => return TypeErrorArgMustBeIterable(r).err(),
    }
}

pub fn filter<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    match (a1, a2) {
        (l, List(rs)) => {
            let rs = rs.borrow();
            let mut acc: Vec<Value> = Vec::with_capacity(rs.len());
            for r in rs.iter() {
                vm.push(r.clone());
                vm.push(l.clone());
                let f = match vm.invoke_func_compose() {
                    Err(e) => return e.err(),
                    Ok(f) => f
                };
                vm.run_after_invoke(f)?;
                if vm.pop().as_bool() {
                    acc.push(r.clone());
                }
            }
            Ok(Value::list(acc))
        },
        (_, r) => return TypeErrorArgMustBeIterable(r).err(),
    }
}

pub fn reduce<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    match (a1, a2) {
        (l, List(rs)) => {
            let rs = rs.borrow();
            let mut iter = rs.iter().cloned();
            let mut acc: Value = match iter.next() {
                Some(v) => v,
                None => return ValueErrorReduceArgMustBeNonEmptySequence.err()
            };

            for r in iter {
                vm.push(l.clone()); // Function
                vm.push(acc); // Accumulator (arg1)
                vm.push(r); // Value (arg2)
                let f = match vm.invoke_func_eval(2) {
                    Err(e) => return e.err(),
                    Ok(f) => f
                };
                vm.run_after_invoke(f)?;
                acc = vm.pop();
            }
            Ok(acc)
        },
        (_, r) => return TypeErrorArgMustBeIterable(r).err(),
    }
}

pub fn pop(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => { v.borrow_mut().pop_back(); Ok(a1) },
        Set(v) => { v.borrow_mut().pop_back(); Ok(a1) },
        Dict(v) => { v.borrow_mut().pop_back(); Ok(a1) },
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn push(a1: Value, a2: Value) -> ValueResult {
    match &a2 {
        List(v) => { v.borrow_mut().push_back(a1); Ok(a2) }
        Set(v) => { v.borrow_mut().insert(a1); Ok(a2) }
        _ => TypeErrorArgMustBeIterable(a2).err()
    }
}

pub fn last(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => Ok(v.borrow_mut().back().cloned().unwrap_or(Nil)),
        Set(v) => Ok(v.borrow_mut().back().cloned().unwrap_or(Nil)),
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn head(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => Ok(v.borrow_mut().front().cloned().unwrap_or(Nil)),
        Set(v) => Ok(v.borrow_mut().front().cloned().unwrap_or(Nil)),
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn init(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => {
            let v = v.borrow();
            let mut iter = v.iter();
            iter.next_back();
            Ok(Value::iter(iter.cloned()))
        },
        Set(v) => {
            let v = v.borrow();
            let mut iter = v.iter();
            iter.next_back();
            Ok(Value::iter(iter.cloned()))
        },
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}

pub fn tail(a1: Value) -> ValueResult {
    match &a1 {
        List(v) => Ok(Value::iter(v.borrow().iter().skip(1).cloned())),
        Set(v) => Ok(Value::iter(v.borrow().iter().skip(1).cloned())),
        _ => TypeErrorArgMustBeIterable(a1).err()
    }
}
