use std::borrow::Cow;

use crate::vm::value::{ConstValue, impl_into, ptr, Value};
use crate::vm::{IntoValue, Type, ValuePtr};
use crate::vm::value::ptr::Prefix;


pub type ComplexType = num_complex::Complex<i64>;
pub type RationalType = rug::Rational;



/// # Complex
///
/// This is Cordy's complex integer type. For now, it is represented by a 64-bit complex pair of (real, imaginary) components.
/// Since normal integer values are represented with a `int`, we require that this type **only** holds values with a nonzero imaginary component
///
/// It is a shared, and immutable type.
///
/// Due to conversion rules we don't use `impl_const!` and `impl_into!` and instead implement them manually due to key differences:
///
/// - All methods acting on `complex` return the underlying `ComplexType`, rather than a `&Complex`
/// - Both `as_complex` and `to_complex` exist, with converting semantics
/// - `is_complex` is a compatibility check, not an exact type check
///
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Complex(ComplexType);

impl Complex {
    const fn new(value: ComplexType) -> Complex {
        Complex(value)
    }

    pub(super) fn to_repr_str(value: &ComplexType) -> Cow<'static, str> {
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

impl Value for Complex {}
impl ConstValue for Complex {}

impl_into!(Complex, self, self.0.to_value());
impl_into!(ComplexType, self, match self.im == 0 {
    true => ptr::from_i64(self.re),
    false => ptr::from_shared(Prefix::new(Type::Complex, Complex::new(self)))
});

impl ValuePtr {
    /// Returns this value as the underlying numeric type of the complex number.
    ///
    /// This does not do any conversions from `int` or `bool` types to complex, and will only accept `Type::Complex`
    /// For converting variants, or if returning an owned type is required, use `to_complex()`
    pub fn as_complex(&self) -> &ComplexType {
        debug_assert!(self.ty() == Type::Complex);
        &self.as_prefix::<Complex>().borrow_const().0
    }

    /// If the current type is int-like, then automatically converts it to a complex number.
    ///
    /// **N.B.** The caller is responsible for first checking that this value is able to convert to a `complex`
    pub fn to_complex(self) -> ComplexType {
        debug_assert!(self.ty() == Type::Bool || self.ty() == Type::Int || self.ty() == Type::Complex);
        match self.ty() {
            Type::Bool => ComplexType::new(self.is_true() as i64, 0),
            Type::Int => ComplexType::new(self.as_precise_int(), 0),
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


/// # Rational
///
/// `rational` in Cordy is an arbitrary precision, integer rational value. Unlike `complex`, which shares values with `int` and will
/// convert down to `int`, `rational` will _pollute_ any expressions done in the rationals, and the resultant value must be converted,
/// either via `int`, `numer`, or `denom`
///
/// todo: finalize names for `numer` and `denom`
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Rational(RationalType);

impl Rational {
    const fn new(rational: RationalType) -> Rational {
        Rational(rational)
    }

    pub(super) fn to_repr_str(value: &RationalType) -> Cow<str> {
        Cow::from(format!("{} / {}", value.numer(), value.denom()))
    }
}

impl Value for Rational {}
impl ConstValue for Rational {}

impl_into!(RationalType, self, Rational::new(self).to_value());
impl_into!(Rational, self, ptr::from_shared(Prefix::new(Type::Rational, self)));

impl ValuePtr {

    /// Returns this value as the underlying numeric type of the rational number.
    ///
    /// This does not do any conversions from `int` or `bool` types to rational, and will only accept `Type::Rational`
    /// For converting variants, or if returning an owned type is required, use `to_rational()`
    pub fn as_rational(&self) -> &RationalType {
        debug_assert!(self.ty() == Type::Rational);
        &self.as_prefix::<Rational>().borrow_const().0
    }

    /// If the current type is real int-like, then automatically converts it to a rational number.
    ///
    /// **N.B.** The caller is responsible for first checking that this value is able to convert to a `rational`
    pub fn to_rational(self) -> RationalType {
        debug_assert!(self.ty() == Type::Bool || self.ty() == Type::Int || self.ty() == Type::Rational);
        match self.ty() {
            Type::Bool | Type::Int => RationalType::from((self.as_int(), 1)),
            Type::Rational => self.as_rational().clone(),
            _ => unreachable!()
        }
    }

    /// Returns `true` if this value is **convertable** to a rational number. Note that this only allows use of
    /// `to_rational()` - if calling `to_rational()` is desired, the `ty()` must be checked against `Type::Rational` directly.
    pub fn is_rational(&self) -> bool {
        self.is_int() || self.ty() == Type::Rational
    }
}

