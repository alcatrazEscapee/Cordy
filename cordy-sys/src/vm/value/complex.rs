use std::cmp::Ordering;
use crate::util::impl_partial_ord;
use crate::vm::value::OwnedValue;
use crate::vm::value::str::{IntoRefStr, RefStr};
use crate::vm::{Prefix, Type, ValuePtr};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ComplexImpl {
    pub inner: num_complex::Complex<i64>,
}

impl OwnedValue for ComplexImpl {}


impl ValuePtr {
    pub fn as_precise_complex(self) -> Box<Prefix<ComplexImpl>> {
        debug_assert!(self.ty() == Type::Complex);
        self.as_box()
    }

    pub fn as_precise_complex_ref(&self) -> &ComplexImpl {
        debug_assert!(self.ty() == Type::Complex);
        self.as_ref()
    }

    pub fn is_precise_complex(&self) -> bool {
        self.ty() == Type::Complex
    }

    pub fn is_complex(&self) -> bool {
        self.is_int() || self.is_precise_complex()
    }
}

impl_partial_ord!(ComplexImpl);
impl Ord for ComplexImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.re.cmp(&other.inner.re)
            .then(self.inner.im.cmp(&other.inner.im))
    }
}

impl ComplexImpl {

    pub fn to_repr_str(&self) -> RefStr {
        let c = self.inner;
        let str = if c.re == 0 {
            format!("{}i", c.im)
        } else {
            // This complicated-ness handles things like 1 - 1i and 1 + 1i
            let mut re = format!("{} ", c.re);
            let mut im = format!("{:+}i", c.im);

            im.insert(1, ' '); // Legal, as the first character should be `+` or `-`
            re.push_str(im.as_str());
            re
        };
        str.to_ref_str()
    }
}