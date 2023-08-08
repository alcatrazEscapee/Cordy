use std::ptr::NonNull;

// If you use `main()`, declare it as `pub` to see it in the output:
// pub fn main() { ... }


pub fn is_int_1(ptr: ValuePtr) -> bool {
    ptr.is_int()
}

pub fn is_int_2(ptr: ValuePtr) -> bool {
    ptr.ty() == Type::Int
}

pub fn is_int_3(ptr: ValuePtr) -> bool {
    ptr.is_inline_ty(Type::Int)
}


pub fn is_str_1(ptr: ValuePtr) -> bool {
    ptr.ty() == Type::Str
}

pub fn is_str_2(ptr: ValuePtr) -> bool {
    ptr.is_owned_ty(Type::Str)
}


#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Type {
    Nil,
    Bool,
    Int,
    Native,
    Complex,
    Str,
    List,
    Set,
    Dict,
    Heap,
    Vector,
    Struct,
    StructType,
    Range,
    Enumerate,
    Slice,
    Iter,
    Memoized,
    GetField,
    Function,
    PartialFunction,
    PartialNativeFunction,
    Closure,
    Error,
    None, // Useful when we would otherwise hold an `Option<ValuePtr>` - this compresses the `None` state
    Never,
}

pub struct Prefix<T> {
    ty: Type,
    pub value: T,
}

pub union ValuePtr {
    _ptr: NonNull<Prefix<()>>,
    tag: usize,
    int: i16,
}

const TAG_INT: usize       = 0b______000;
const TAG_NIL: usize       = 0b__000_001;
const TAG_BOOL: usize      = 0b__001_001;
const TAG_FALSE: usize     = 0b0_001_001;
const TAG_TRUE: usize      = 0b1_001_001;
const TAG_NATIVE: usize    = 0b__010_001;
const TAG_NONE: usize      = 0b__011_001;
const TAG_FIELD: usize     = 0b__100_001;
const TAG_ERR: usize       = 0b______011;
const TAG_PTR: usize       = 0b______101;
const TAG_SHARED: usize    = 0b______111;

const MASK_INT: usize      = 0b________1;
const MASK_NIL: usize      = 0b__111_111;
const MASK_BOOL: usize     = 0b__111_111;
const MASK_NATIVE: usize   = 0b__111_111;
const MASK_FIELD: usize    = 0b__111_111;
const MASK_NONE: usize     = 0b__111_111;
const MASK_ERR: usize      = 0b______111;
const MASK_PTR: usize      = 0b______111;
const MASK_SHARED: usize   = 0b______111;

const TY_MASK: usize   = 0b111;
const PTR_MASK: usize = 0xffff_ffff_ffff_fff8;

const MAX_INT: i64 = 0x3fff_ffff_ffff_ffffu64 as i64;
const MIN_INT: i64 = 0xc000_0000_0000_0000u64 as i64;

impl ValuePtr {
    fn ty(&self) -> Type {
        unsafe {
            match self.tag & MASK_PTR {
                TAG_NIL => match self.tag & MASK_BOOL {
                    TAG_NIL => Type::Nil,
                    TAG_BOOL => Type::Bool,
                    TAG_NATIVE => Type::Native,
                    TAG_NONE => Type::None,
                    _ => Type::Never,
                },
                TAG_ERR => Type::Error,
                TAG_PTR | TAG_SHARED => (*self.as_ptr()).ty, // Check the prefix for the type
                _ => Type::Int, // Includes all remaining bit patterns with a `0` LSB
            }
        }
    }

    unsafe fn as_ptr(&self) -> *mut Prefix<()> {
        unsafe {
            (self.tag & PTR_MASK) as *mut Prefix<()>
        }
    }

    fn is_inline_ty(&self, ty: Type) -> bool {
        ty == unsafe {
            match self.tag & MASK_PTR {
                TAG_NIL => match self.tag & MASK_BOOL {
                    TAG_NIL => Type::Nil,
                    TAG_BOOL => Type::Bool,
                    TAG_NATIVE => Type::Native,
                    TAG_NONE => Type::None,
                    _ => Type::Never,
                },
                TAG_ERR => Type::Error,
                TAG_PTR | TAG_SHARED => Type::None,
                _ => Type::Int, // Includes all remaining bit patterns with a `0` LSB
            }
        }
    }

    fn is_owned_ty(&self, ty: Type) -> bool {
        unsafe {
            self.is_ptr() && (*self.as_ptr()).ty == ty
        }
    }

    fn is_nil(&self) -> bool { (unsafe { self.tag } & MASK_NIL) == TAG_NIL }
    fn is_bool(&self) -> bool { (unsafe { self.tag } & MASK_BOOL) == TAG_BOOL }
    fn is_true(&self) -> bool { (unsafe { self.tag }) == TAG_TRUE }
    fn is_int(&self) -> bool { (unsafe { self.tag } & MASK_INT) == TAG_INT }
    fn is_native(&self) -> bool { (unsafe { self.tag } & MASK_NATIVE) == TAG_NATIVE }
    fn is_none(&self) -> bool { (unsafe { self.tag } & MASK_NONE) == TAG_NONE }
    fn is_err(&self) -> bool { (unsafe { self.tag } & MASK_ERR) == TAG_ERR }

    fn is_ptr(&self) -> bool { (unsafe { self.tag } & MASK_PTR) == TAG_PTR }
    fn is_shared(&self) -> bool { (unsafe { self.tag } & MASK_SHARED) == TAG_SHARED }
}

