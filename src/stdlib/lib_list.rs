use std::cell::RefCell;
use std::rc::Rc;

use crate::vm::error::RuntimeError;
use crate::vm::value::Value;
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


pub fn list_index(list_ref: Rc<RefCell<Vec<Value>>>, r: i64) -> ValueResult {
    let list = (*list_ref).borrow();
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
    let list = (*list_ref).borrow();
    let length: i64 = list.len() as i64;

    let step: i64 = slice_arg!(a4, "step", 1);
    if step == 0 {
        return SliceStepZero.err()
    }

    let low: i64 = slice_arg!(a2, "low", if step > 0 { 0 } else { -1 });
    let high: i64 = slice_arg!(a3, "high", if step > 0 { 0 } else { -length - 1 });

    let abs_start: i64 = to_index(length, low);
    let abs_step: usize = step.unsigned_abs() as usize;

    return Ok(Value::list(if step > 0 {
        let abs_stop: i64 = to_index(length, high - 1);

        (abs_start..=abs_stop).step_by(abs_step)
            .filter_map(|i| safe_get(&list, i))
            .cloned()
            .collect::<Vec<Value>>()
    } else {
        let abs_stop: i64 = to_index(length, high);

        rev_range(abs_start, abs_stop).step_by(abs_step)
            .filter_map(|i| safe_get(&list, i))
            .cloned()
            .collect::<Vec<Value>>()
    }))
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
fn safe_get(list: &Vec<Value>, index: i64) -> Option<&Value> {
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
    ($iter:ident, $list:ident) => {
        pub fn $iter(arg: Value) -> ValueResult {
            match arg {
                List(ls) => $list((*ls).borrow().as_ref()),
                e => TypeErrorArgMustBeIterable(e).err(),
            }
        }
    };
}

declare_varargs_iter!(sum_iter, sum_list);
declare_varargs_iter!(min_iter, min_list);
declare_varargs_iter!(max_iter, max_list);

//declare_varargs_iter!(filter_iter, filter_list);
//declare_varargs_iter!(map_iter, map_list);
//declare_varargs_iter!(reduce_iter, reduce_list);

pub fn sum_list(args: &Vec<Value>) -> ValueResult {
    let mut sum: i64 = 0;
    for v in args {
        match v {
            Int(i) => sum += *i,
            _ => return TypeErrorArgMustBeInt(v.clone()).err(),
        }
    }
    Ok(Int(sum))
}

pub fn max_list(args: &Vec<Value>) -> ValueResult {
    let mut iter = args.iter();
    let mut max = match iter.next() {
        Some(t) => t,
        None => return ValueErrorMaxArgMustBeNonEmptySequence.err()
    };
    for v in iter {
        if max.is_less_than(v)? {
            max = v;
        }
    }
    Ok(max.clone())
}

pub fn min_list(args: &Vec<Value>) -> ValueResult {
    let mut iter = args.iter();
    let mut min = match iter.next() {
        Some(t) => t,
        None => return ValueErrorMinArgMustBeNonEmptySequence.err()
    };
    for v in iter {
        if v.is_less_than(min)? {
            min = v;
        }
    }
    Ok(min.clone())
}

pub fn map<VM>(vm: &mut VM, a1: Value, a2: Value) -> ValueResult where VM : VirtualInterface {
    match (a1, a2) {
        (l, List(rs)) => {
            let rs = (*rs).borrow();
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
