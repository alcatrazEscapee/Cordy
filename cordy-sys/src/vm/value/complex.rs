use std::borrow::Cow;
use std::cmp::Ordering;

use crate::util::impl_partial_ord;
use crate::vm::value::{ConstValue, ptr, Value};
use crate::vm::{IntoValue, Type, ValuePtr};
use crate::vm::value::ptr::Prefix;


/// The value type of a Cordy `complex` number, currently represented by `num_complex::Complex<i64>`
pub type ComplexValue = num_complex::Complex<i64>;


/// # Complex
///
/// This is Cordy's complex integer type. For now, it is represented by a 64-bit complex pair of (real, imaginary) components.
/// Since normal integer values are represented with a `int`, we require that this type **only** holds values with a nonzero imaginary component
///
/// It is a shared, and immutable type.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Complex(ComplexValue);


impl Value for Complex {}
impl ConstValue for Complex {}


impl IntoValue for ComplexValue {
    fn to_value(self) -> ValuePtr {
        match self.im == 0 {
            true => ptr::from_i64(self.re),
            false => ptr::from_shared(Prefix::new(Type::Complex, Complex::new(self)))
        }
    }
}

impl IntoValue for Complex {
    fn to_value(self) -> ValuePtr {
        self.0.to_value()
    }
}


impl ValuePtr {
    /// Returns this value as the underlying numeric type of the complex number.
    ///
    /// This does not do any conversions from `int` or `bool` types to complex, and will only accept `Type::Complex`
    /// For converting variants, or if returning an owned type is required, use `to_complex()`
    pub fn as_complex(&self) -> &ComplexValue {
        debug_assert!(self.ty() == Type::Complex);
        &self.as_prefix::<Complex>().borrow_const().0
    }

    /// If the current type is int-like, then automatically converts it to a complex number.
    pub fn to_complex(self) -> ComplexValue {
        debug_assert!(self.ty() == Type::Bool || self.ty() == Type::Int || self.ty() == Type::Complex);
        match self.ty() {
            Type::Bool => ComplexValue::new(self.is_true() as i64, 0),
            Type::Int => ComplexValue::new(self.as_precise_int(), 0),
            Type::Complex => self.as_complex().clone(),
            _ => unreachable!(),
        }
    }

    /// Returns `true` if this value is **convertable** to a complex number. Note that this only allows use of
    /// `to_complex()` - if calling `as_complex()` is desired, the `ty()` must be checked against `Type::Complex` directly.
    pub fn is_complex(&self) -> bool {
        self.is_int() || self.ty() == Type::Complex
    }
}

impl_partial_ord!(Complex);
impl Ord for Complex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.re.cmp(&other.0.re).then(self.0.im.cmp(&other.0.im))
    }
}

impl Complex {
    const fn new(value: ComplexValue) -> Complex {
        Complex(value)
    }

    pub(super) fn to_repr_str(value: &ComplexValue) -> Cow<'static, str> {
        let str = if value.re == 0 {
            format!("{}i", value.im)
        } else {
            // This complicated-ness handles things like 1 - 1i and 1 + 1i
            let mut re = format!("{} ", value.re);
            let mut im = format!("{:+}i", value.im);

            im.insert(1, ' '); // Legal, as the first character should be `+` or `-`
            re.push_str(im.as_str());
            re
        };
        Cow::from(str)
    }
}