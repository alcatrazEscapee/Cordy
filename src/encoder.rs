use std::collections::HashMap;
use std::hash::Hash;
use std::io::{Cursor, Read, Write};
use std::rc::Rc;

use crate::compiler::{CompileResult, Fields};
use crate::vm::{FunctionImpl, Opcode, StructTypeImpl};

use Opcode::{*};


type Maybe<T> = Result<T, ()>;

const SEGMENT_BITS: u8 = 0x7F;
const SEGMENT_BITS_MASK: u64 = !(SEGMENT_BITS as u64);
const CONTINUE_BIT: u8 = 0x80;


/// Encodes a compile result into a series of bytes
pub fn encode(compile: &CompileResult) -> Vec<u8> {
    let mut encoder: Encoder = Encoder(Vec::new());
    compile.encode(&mut encoder);
    encoder.0
}

/// Attempts to decode a compile result from a series of bytes
pub fn decode(bytes: Vec<u8>) -> Maybe<CompileResult> {
    let mut decoder: Decoder = Decoder(Cursor::new(bytes));
    decoder.decode()
}

// ===== Encoder and Decoder Structs ===== //

struct Encoder(Vec<u8>);
struct Decoder(Cursor<Vec<u8>>);

impl Encoder {
    fn encode<E: Encode + ?Sized>(&mut self, e: &E) -> &mut Self {
        e.encode(self);
        self
    }

    fn encode_u8(&mut self, value: u8) -> &mut Self {
        self.0.push(value);
        self
    }

    fn encode_uv(&mut self, mut value: u64) -> &mut Self {
        loop {
            if (value & SEGMENT_BITS_MASK) == 0 {
                return self.encode_u8(value as u8)
            }

            self.encode_u8((value as u8 & SEGMENT_BITS) | CONTINUE_BIT);
            value >>= 7;
        }
    }
}

impl Decoder {
    fn decode_u8(&mut self) -> Maybe<u8> {
        let mut buf = [0; 1];
        self.0.read_exact(&mut buf).map_err(|_| ())?;
        Ok(buf[0])
    }

    fn decode_uv(&mut self) -> Maybe<u64> {
        let mut value: u64 = 0;
        let mut position = 0;
        loop {
            let byte = self.decode_u8()?;
            value |= ((byte & SEGMENT_BITS) as u64) << position;
            if (byte & CONTINUE_BIT) == 0 { return Ok(value); }
            position += 7;
        }
    }

    fn decode_usize(&mut self) -> Maybe<usize> { self.decode_uv().map(|u| u as usize) }
}

/// A trait that is responsible for decoding an arbitrary type from a `Decoder`.
trait Decode<T> {
    fn decode(&mut self) -> Maybe<T>;
}

/// A trait that is responsible for encoding an arbitrary type into a sequence of bytes
trait Encode {
    fn encode(&self, encoder: &mut Encoder);
}

// ===== Encode + Decode Trait Implementations ===== //

impl Encode for u8 { fn encode(&self, encoder: &mut Encoder) { encoder.encode_u8(*self); } }
impl Encode for u32 { fn encode(&self, encoder: &mut Encoder) { encoder.encode_uv(*self as u64); } }
impl Encode for i32 { fn encode(&self, encoder: &mut Encoder) { encoder.encode_uv(*self as u64); } }
impl Encode for i64 { fn encode(&self, encoder: &mut Encoder) { encoder.encode_uv(*self as u64); } }
impl Encode for usize { fn encode(&self, encoder: &mut Encoder) { encoder.encode_uv(*self as u64); } }
impl Encode for String { fn encode(&self, encoder: &mut Encoder) { self.len().encode(encoder); encoder.0.write(&self.as_bytes()).unwrap(); } }

impl<T : Encode> Encode for Rc<T> { fn encode(&self, encoder: &mut Encoder) { (**self).encode(encoder) } }
impl<T : Encode> Encode for Vec<T> { fn encode(&self, encoder: &mut Encoder) { self.len().encode(encoder); for e in self { e.encode(encoder); } } }
impl<K : Encode, V : Encode> Encode for HashMap<K, V> { fn encode(&self, encoder: &mut Encoder) { self.len().encode(encoder); for (k, v) in self { k.encode(encoder); v.encode(encoder) } } }
impl<T1: Encode, T2: Encode> Encode for (T1, T2) { fn encode(&self, encoder: &mut Encoder) { self.0.encode(encoder); self.1.encode(encoder); } }

impl Encode for FunctionImpl { fn encode(&self, encoder: &mut Encoder) { encoder.encode(&self.head).encode(&self.tail).encode(&self.name).encode(&self.args); } }
impl Encode for StructTypeImpl { fn encode(&self, encoder: &mut Encoder) { encoder.encode(&self.name).encode(&self.field_names).encode(&self.type_index); } }
impl Encode for Fields { fn encode(&self, encoder: &mut Encoder) { encoder.encode(&self.fields).encode(&self.lookup); } }
impl Encode for CompileResult { fn encode(self: &Self, encoder: &mut Encoder) { encoder.encode(&self.code).encode(&self.strings).encode(&self.constants).encode(&self.functions).encode(&self.structs).encode(&self.fields); } }


impl Decode<u8> for Decoder { fn decode(&mut self) -> Maybe<u8> { self.decode_u8() } }
impl Decode<u32> for Decoder { fn decode(&mut self) -> Maybe<u32> { self.decode_uv().map(|u| u as u32) } }
impl Decode<i32> for Decoder { fn decode(&mut self) -> Maybe<i32> { self.decode_uv().map(|u| u as i32) } }
impl Decode<i64> for Decoder { fn decode(&mut self) -> Maybe<i64> { self.decode_uv().map(|u| u as i64) } }
impl Decode<usize> for Decoder { fn decode(&mut self) -> Maybe<usize> { self.decode_usize() } }
impl Decode<String> for Decoder { fn decode(&mut self) -> Maybe<String> { String::from_utf8(self.decode()?).map_err(|_| ()) } }

impl<T> Decode<Rc<T>> for Decoder where Decoder : Decode<T> { fn decode(&mut self) -> Maybe<Rc<T>> { Ok(Rc::new(self.decode()?)) } }
impl<T1, T2> Decode<(T1, T2)> for Decoder where Decoder : Decode<T1> + Decode<T2> { fn decode(&mut self) -> Maybe<(T1, T2)> { Ok((self.decode()?, self.decode()?)) } }

impl<T> Decode<Vec<T>> for Decoder where Decoder : Decode<T> {
    fn decode(&mut self) -> Maybe<Vec<T>> {
        let size: usize = self.decode_usize()?;
        let mut ret: Vec<T> = Vec::with_capacity(size);
        for _ in 0..size { ret.push(self.decode()?); }
        Ok(ret)
    }
}

impl<K, V> Decode<HashMap<K, V>> for Decoder where Decoder : Decode<K> + Decode<V>, K : Eq + Hash {
    fn decode(&mut self) -> Maybe<HashMap<K, V>> {
        let size: usize = self.decode_usize()?;
        let mut ret: HashMap<K, V> = HashMap::with_capacity(size);
        for _ in 0..size { ret.insert(self.decode()?, self.decode()?); }
        Ok(ret)
    }
}

impl Decode<FunctionImpl> for Decoder { fn decode(&mut self) -> Maybe<FunctionImpl> { Ok(FunctionImpl::new(self.decode()?, self.decode()?, self.decode()?, self.decode()?)) } }
impl Decode<StructTypeImpl> for Decoder { fn decode(&mut self) -> Maybe<StructTypeImpl> { Ok(StructTypeImpl::new(self.decode()?, self.decode()?, self.decode()?)) } }
impl Decode<Fields> for Decoder { fn decode(&mut self) -> Maybe<Fields> { Ok(Fields { fields: self.decode()?, lookup: self.decode()? }) } }
impl Decode<CompileResult> for Decoder { fn decode(&mut self) -> Maybe<CompileResult> { Ok(CompileResult {
    code: self.decode()?,
    errors: Vec::new(),
    strings: self.decode()?,
    constants: self.decode()?,
    functions: self.decode()?,
    structs: self.decode()?,
    locations: Vec::new(),
    locals: Vec::new(),
    globals: Vec::new(),
    fields: self.decode()?
}) } }


// ===== Encode + Decode implementations for `Opcode` ===== //

/// A stable representation of individual opcodes, which can be converted to and from a `u8`.
///
/// We use this intermediary between `Opcode` <---> `Op` <---> `u8` for rwo reasons:
///
/// - Conversion to a `u8` can be easily inferred via the natural order of these opcodes, without having to explicitly declare any opcode values.
/// - It is much more programmer-safe, by matching each `Opcode` <---> `Op` via it's name, than having to make sure each opcode encodes and decodes with it's byte correctly.
#[repr(u8)]
enum Op {
    JumpIfFalse, JumpIfFalsePop, JumpIfTrue, JumpIfTruePop, Jump, Return, Pop, PopN, Dup, Swap, PushLocal, StoreLocal, PushGlobal, StoreGlobal, PushUpValue, StoreUpValue, StoreArray, IncGlobalCount, Closure, CloseLocal, CloseUpValue, LiftUpValue, InitIterable, TestIterable, Nil, True, False, Int, Str, Function, NativeFunction, List, Vector, Set, Dict, Struct, CheckLengthGreaterThan, CheckLengthEqualTo, OpFuncEval, OpIndex, OpIndexPeek, OpSlice, OpSliceWithStep, GetField, GetFieldPeek, GetFieldFunction, SetField, Unary, Binary, Slice, SliceWithStep, Exit, Yield, AssertFailed
}


#[allow(unused_variables)]
impl Encode for Opcode {
    fn encode(&self, encoder: &mut Encoder) {
        macro_rules! encode {
            ($op: expr) => { encoder.encode_u8($op as u8) };
            ($op:expr, $i:expr) => { encoder.encode_u8($op as u8).encode($i) };
        }
        match self {
            Noop => panic!(),
            JumpIfFalse(i) => encode!(Op::JumpIfFalse, i),
            JumpIfFalsePop(i) => encode!(Op::JumpIfFalsePop, i),
            JumpIfTrue(i) => encode!(Op::JumpIfTrue, i),
            JumpIfTruePop(i) => encode!(Op::JumpIfTruePop, i),
            Jump(i) => encode!(Op::Jump, i),
            Return => encode!(Op::Return),
            Pop => encode!(Op::Pop),
            PopN(i) => encode!(Op::PopN, i),
            Dup => encode!(Op::Dup),
            Swap => encode!(Op::Swap),
            PushLocal(i) => encode!(Op::PushLocal, i),
            StoreLocal(i) => encode!(Op::StoreLocal, i),
            PushGlobal(i) => encode!(Op::PushGlobal, i),
            StoreGlobal(i) => encode!(Op::StoreGlobal, i),
            PushUpValue(i) => encode!(Op::PushUpValue, i),
            StoreUpValue(i) => encode!(Op::StoreUpValue, i),
            StoreArray => encode!(Op::StoreArray),
            IncGlobalCount => encode!(Op::IncGlobalCount),
            Closure => encode!(Op::Closure),
            CloseLocal(i) => encode!(Op::CloseLocal, i),
            CloseUpValue(i) => encode!(Op::CloseUpValue, i),
            LiftUpValue(i) => encode!(Op::LiftUpValue, i),
            InitIterable => encode!(Op::InitIterable),
            TestIterable => encode!(Op::TestIterable),
            Nil => encode!(Op::Nil),
            True => encode!(Op::True),
            False => encode!(Op::False),
            Int(i) => encode!(Op::Int, i),
            Str(i) => encode!(Op::Str, i),
            Function(i) => encode!(Op::Function, i),
            NativeFunction(i) => encode!(Op::NativeFunction, &(*i as u8)),
            List(i) => encode!(Op::List, i),
            Vector(i) => encode!(Op::Vector, i),
            Set(i) => encode!(Op::Set, i),
            Dict(i) => encode!(Op::Dict, i),
            Struct(i) => encode!(Op::Struct, i),
            CheckLengthGreaterThan(i) => encode!(Op::CheckLengthGreaterThan, i),
            CheckLengthEqualTo(i) => encode!(Op::CheckLengthEqualTo, i),
            OpFuncEval(i) => encode!(Op::OpFuncEval, i),
            OpIndex => encode!(Op::OpIndex),
            OpIndexPeek => encode!(Op::OpIndexPeek),
            OpSlice => encode!(Op::OpSlice),
            OpSliceWithStep => encode!(Op::OpSliceWithStep),
            GetField(i) => encode!(Op::GetField, i),
            GetFieldPeek(i) => encode!(Op::GetFieldPeek, i),
            GetFieldFunction(i) => encode!(Op::GetFieldFunction, i),
            SetField(i) => encode!(Op::SetField, i),
            Unary(i) => encode!(Op::Unary, &(*i as u8)),
            Binary(i) => encode!(Op::Binary, &(*i as u8)),
            Slice => encode!(Op::Slice),
            SliceWithStep => encode!(Op::SliceWithStep),
            Exit => encode!(Op::Exit),
            Yield => encode!(Op::Yield),
            AssertFailed => encode!(Op::AssertFailed),
        };
    }
}

impl Decode<Opcode> for Decoder {
    fn decode(&mut self) -> Maybe<Opcode> {
        macro_rules! from_byte {
            () => { unsafe { std::mem::transmute(self.decode_u8()?) } };
        }

        Ok(match from_byte!() {
            Op::JumpIfFalse => JumpIfFalse(self.decode()?),
            Op::JumpIfFalsePop => JumpIfFalsePop(self.decode()?),
            Op::JumpIfTrue => JumpIfTrue(self.decode()?),
            Op::JumpIfTruePop => JumpIfTruePop(self.decode()?),
            Op::Jump => Jump(self.decode()?),
            Op::Return => Return,
            Op::Pop => Pop,
            Op::PopN => PopN(self.decode()?),
            Op::Dup => Dup,
            Op::Swap => Swap,
            Op::PushLocal => PushLocal(self.decode()?),
            Op::StoreLocal => StoreLocal(self.decode()?),
            Op::PushGlobal => PushGlobal(self.decode()?),
            Op::StoreGlobal => StoreGlobal(self.decode()?),
            Op::PushUpValue => PushUpValue(self.decode()?),
            Op::StoreUpValue => StoreUpValue(self.decode()?),
            Op::StoreArray => StoreArray,
            Op::IncGlobalCount => IncGlobalCount,
            Op::Closure => Closure,
            Op::CloseLocal => CloseLocal(self.decode()?),
            Op::CloseUpValue => CloseUpValue(self.decode()?),
            Op::LiftUpValue => LiftUpValue(self.decode()?),
            Op::InitIterable => InitIterable,
            Op::TestIterable => TestIterable,
            Op::Nil => Nil,
            Op::True => True,
            Op::False => False,
            Op::Int => Int(self.decode()?),
            Op::Str => Str(self.decode()?),
            Op::Function => Function(self.decode()?),
            Op::NativeFunction => NativeFunction(from_byte!()),
            Op::List => List(self.decode()?),
            Op::Vector => Vector(self.decode()?),
            Op::Set => Set(self.decode()?),
            Op::Dict => Dict(self.decode()?),
            Op::Struct => Struct(self.decode()?),
            Op::CheckLengthGreaterThan => CheckLengthGreaterThan(self.decode()?),
            Op::CheckLengthEqualTo => CheckLengthEqualTo(self.decode()?),
            Op::OpFuncEval => OpFuncEval(self.decode()?),
            Op::OpIndex => OpIndex,
            Op::OpIndexPeek => OpIndexPeek,
            Op::OpSlice => OpSlice,
            Op::OpSliceWithStep => OpSliceWithStep,
            Op::GetField => GetField(self.decode()?),
            Op::GetFieldPeek => GetFieldPeek(self.decode()?),
            Op::GetFieldFunction => GetFieldFunction(self.decode()?),
            Op::SetField => SetField(self.decode()?),
            Op::Unary => Unary(from_byte!()),
            Op::Binary => Binary(from_byte!()),
            Op::Slice => Slice,
            Op::SliceWithStep => SliceWithStep,
            Op::Exit => Exit,
            Op::Yield => Yield,
            Op::AssertFailed => AssertFailed,
        })
    }
}


#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::io::Cursor;

    use crate::encoder::{Decode, Decoder, Encode, Encoder, Maybe};
    use crate::vm::Opcode;

    #[test] fn test_opcode() { run(Opcode::JumpIfFalse(5), vec![0, 5]); }
    #[test] fn test_opcodes() { run(vec![Opcode::Int(0), Opcode::Pop, Opcode::Exit], vec![3, 27, 0, 6, 51]); }

    fn run<E>(e: E, bytes: Vec<u8>) where
        E : Encode + Sized + Eq + Debug,
        Decoder : Decode<E>
    {
        let mut encoder: Encoder = Encoder(Vec::new());
        encoder.encode(&e);

        assert_eq!(encoder.0, bytes);

        let mut decoder: Decoder = Decoder(Cursor::new(encoder.0));
        let ret: Maybe<E> = decoder.decode();

        assert!(ret.is_ok());
        assert_eq!(e, ret.unwrap());
    }
}
