use std::borrow::Cow;
use std::cmp::Ordering;

use crate::vm::{ErrorResult, IntoValue, Iterable, RuntimeError, ValuePtr, ValueResult};
use crate::vm::value::RecursionGuard;


/// # Range
///
/// This is the value returned by the native function `range(...)`. It is a lazily-evaluated iterable, which produces integer values.
///
/// `Range` is considered immutable, and shared.
///
/// ### Empty Ranges
///
/// Note that empty ranges are represented by `step` equal to `0`, in which case, both `start` and `stop` should be ignored.
///
/// This is the internal lazy type used by the native function `range(...)`. For non-empty ranges, `step` must be non-zero.
/// For an empty range, this will store the `step` as `0` - in this case the `start` and `stop` values should be ignored
/// Note that depending on the relation of `start`, `stop` and the sign of `step`, this may represent an empty range.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Range {
    start: i64,
    stop: i64,
    step: i64,
}

impl ValuePtr {
    pub fn range(start: i64, stop: i64, step: i64) -> ValueResult {
        Range::from(start, stop, step)
    }
}

impl Range {
    /// Creates a new `range()` from the given parameters. Raises an error if `step` is zero, and converts all empty ranges to `range(0, 0, 0)` internally.
    fn from(start: i64, stop: i64, step: i64) -> ValueResult {
        if step == 0 {
            RuntimeError::ValueErrorStepCannotBeZero.err()
        } else if (stop > start && step > 0) || (stop < start && step < 0) {
            // Non-empty range
            Range::new(start, stop, step).to_value().ok()
        } else {
            // Empty range
            Range::new(0, 0, 0).to_value().ok()
        }
    }

    const fn new(start: i64, stop: i64, step: i64) -> Range {
        Range { start, stop, step }
    }

    /// Check if a value is contained within a given range. This actually checks if the value is produced by this range
    pub fn contains(&self, value: i64) -> bool {
        match self.step.cmp(&0) {
            Ordering::Equal => false,
            Ordering::Greater => value >= self.start && value < self.stop && (value - self.start) % self.step == 0,
            Ordering::Less => value <= self.start && value > self.stop && (self.start - value) % self.step == 0
        }
    }

    /// Reverses the range, so that iteration advances from the end to the start
    pub(super) fn reverse(self) -> Range {
        match self.step.cmp(&0) {
            Ordering::Equal => self,
            Ordering::Greater => Range::new(
                self.start + self.len() as i64 * self.step,
                self.start + 1,
                -self.step
            ),
            Ordering::Less => Range::new(
                self.start + self.len() as i64 * self.step,
                self.start - 1,
                -self.step
            )
        }
    }

    /// Advances the `Range`, based on the external `current` value.
    /// The `current` value is the one that will be returned, and internally advanced to the next value.
    pub(super) fn next(&self, current: &mut i64) -> Option<ValuePtr> {
        if *current == self.stop || self.step == 0 {
            None
        } else if self.step > 0 {
            let ret = *current;
            *current += self.step;
            if *current > self.stop {
                *current = self.stop;
            }
            Some(ret.to_value())
        } else {
            let ret = *current;
            *current += self.step;
            if *current < self.stop {
                *current = self.stop;
            }
            Some(ret.to_value())
        }
    }

    pub(super) fn to_repr_str(&self) -> Cow<str> {
        match self.step {
            0 => Cow::from("range(empty)"),
            _ => Cow::from(format!("range({}, {}, {})", self.start, self.stop, self.step))
        }
    }

    pub(super) fn to_iter(&self) -> Iterable {
        Iterable::Range(self.start, self.clone())
    }

    pub(super) fn len(&self) -> usize {
        // Since this type ensures that the range is non-empty, we can do simple checked arithmetic
        match self.step {
            0 => 0,
            _ => (self.start.abs_diff(self.stop) / self.step.unsigned_abs()) as usize
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.step == 0
    }
}


/// # Enumerate
///
/// This is the value returned by the native function `enumerate()`. It is a lazily-evaluated wrapper around the underlying iterable.
///
/// `Enumerate` is considered immutable, and shared.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Enumerate {
    inner: ValuePtr
}

impl Enumerate {
    pub fn inner(&self) -> &ValuePtr {
        &self.inner
    }
}

impl ValuePtr {
    pub fn enumerate(inner: ValuePtr) -> ValuePtr {
        Enumerate::new(inner)
    }
}

impl Enumerate {
    fn new(inner: ValuePtr) -> ValuePtr {
        Enumerate { inner }.to_value()
    }

    pub(super) fn to_iter(&self) -> ErrorResult<Iterable> {
        Ok(Iterable::Enumerate(0, Box::new(self.inner.clone().to_iter()?)))
    }

    pub(super) fn to_repr_str(&self, rc: &mut RecursionGuard) -> Cow<str> {
        Cow::from(format!("enumerate({})", self.inner.safe_to_repr_str(rc)))
    }

    pub(super) fn to_bool(&self) -> bool {
        self.inner.to_bool()
    }

    pub(super) fn len(&self) -> ErrorResult<usize> {
        self.inner.len()
    }
}


#[cfg(test)]
mod tests {
    use crate::vm::{IntoValue, RuntimeError};
    use crate::vm::value::range::Range;

    #[test]
    fn test_range_from_with_zero_step() {
        assert_eq!(Range::from(1, 5, 0), RuntimeError::ValueErrorStepCannotBeZero.err());
    }

    #[test]
    fn test_range_from_with_empty() {
        assert_eq!(Range::from(5, 4, 2), Range::new(0, 0, 0).to_value().ok());
        assert_eq!(Range::from(10, 20, -1), Range::new(0, 0, 0).to_value().ok());
    }
}