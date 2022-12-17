use std::cell::RefCell;
use std::rc::Rc;
use crate::vm::RuntimeErrorType;
use crate::vm::value::Value;

use RuntimeErrorType::{*};
use Value::{*};


macro_rules! slice_arg {
    ($a:ident, $n:expr, $def:expr) => {
        match $a {
            Value::Int(x) => x,
            Value::Nil => $def,
            t => return Err(TypeErrorSliceArgMustBeInt($n, t)),
        }
    };
}


pub fn list_index(l: Rc<RefCell<Vec<Value>>>, r: i64) -> Result<Value, RuntimeErrorType> {
    let list = (*l).borrow();
    let index: usize = if r < 0 { (list.len() as i64 + r) as usize } else { r as usize };
    if 0 <= index && index < list.len() {
        Ok(list[index].clone())
    } else {
        Err(IndexOutOfBounds(r, list.len()))
    }
}

pub fn list_slice(a1: Value, a2: Value, a3: Value, a4: Value) -> Result<Value, RuntimeErrorType> {

    let list_ref= match a1 {
        List(ls) => ls,
        t => return Err(TypeErrorCannotSlice(t)),
    };
    let list = (*list_ref).borrow();
    let length: i64 = list.len() as i64;

    let step: i64 = slice_arg!(a4, "step", 1);
    if step == 0 {
        return Err(SliceStepZero)
    }

    let low: i64 = slice_arg!(a2, "low", if step > 0 { 0 } else { -1 });
    let high: i64 = slice_arg!(a3, "high", if step > 0 { 0 } else { -length - 1 });


    dbg!(low, high);

    let abs_step: usize = step.unsigned_abs() as usize;
    return if step > 0 {
        let abs_start: i64 = to_index(length, low);
        let abs_stop: i64 = to_index(length, high - 1);

        let slice: Vec<Value> = (abs_start..=abs_stop).step_by(abs_step)
            .filter_map(|i| safe_get(&list, i))
            .cloned()
            .collect::<Vec<Value>>();

        Ok(Value::list(slice))
    } else {
        let abs_start: i64 = to_index(length, low);
        let abs_stop: i64 = to_index(length, high);

        dbg!(abs_start, abs_stop);
        dbg!(rev_range(abs_start, abs_stop).collect::<Vec<i64>>());

        let slice: Vec<Value> = rev_range(abs_start, abs_stop).step_by(abs_step)
            .filter_map(|i| safe_get(&list, i))
            .cloned()
            .collect::<Vec<Value>>();

        Ok(Value::list(slice))
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
fn safe_get(list: &Vec<Value>, index: i64) -> Option<&Value> {
    if index < 0 {
        None
    } else {
        list.get(index as usize)
    }
}

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