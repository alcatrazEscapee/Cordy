use std::collections::HashMap;
use std::hash::Hash;
use std::io::{Cursor, Read, Write};
use std::rc::Rc;

use crate::compiler::{CompileResult, Fields};
use crate::stdlib;
use crate::vm::{FunctionImpl, Opcode, StructTypeImpl};

use Opcode::{*};


type Maybe<T> = Result<T, ()>;

const SEGMENT_BITS: u8 = 0x7F;
const SEGMENT_BITS_MASK: u64 = !(SEGMENT_BITS as u64);
const CONTINUE_BIT: u8 = 0x80;

const MAX_NATIVE_FUNCTION_OPCODE: u8 = 180;


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
// Opcodes are serialized using a custom binary encoding
// - Noop is ignored, as it should never be present in compiled code
// - Each opcode has a single byte which identifies the type of the opcode. `Unary`, `Binary`, and `NativeFunction` are flattened and appended to the end of the range:
// - `Unary` maps to indexes [49, 50], `Binary` to indexes [51, 71]`, and `NativeFunction` to indexes [72, 256]
// - Opcodes with arguments include the argument following the identifier byte (so this is a variable encoding)


#[allow(unused_variables)]
impl Encode for Opcode {
    fn encode(&self, encoder: &mut Encoder) {
        macro_rules! encode {
            ($op: expr) => { encoder.encode(&($op as u8)) };
            ($op:expr, $i:expr) => { encoder.encode(&($op as u8)).encode($i) };
        }
        match self {
            Noop => panic!(),
            JumpIfFalse(i) => encode!(0, i),
            JumpIfFalsePop(i) => encode!(1, i),
            JumpIfTrue(i) => encode!(2, i),
            JumpIfTruePop(i) => encode!(3, i),
            Jump(i) => encode!(4, i),
            Return => encode!(5),
            Pop => encode!(6),
            PopN(i) => encode!(7, i),
            Dup => encode!(8),
            Swap => encode!(9),
            PushLocal(i) => encode!(10, i),
            StoreLocal(i) => encode!(11, i),
            PushGlobal(i) => encode!(12, i),
            StoreGlobal(i) => encode!(13, i),
            PushUpValue(i) => encode!(14, i),
            StoreUpValue(i) => encode!(15, i),
            StoreArray => encode!(16),
            IncGlobalCount => encode!(17),
            Closure => encode!(18),
            CloseLocal(i) => encode!(19, i),
            CloseUpValue(i) => encode!(20, i),
            LiftUpValue(i) => encode!(21, i),
            InitIterable => encode!(22),
            TestIterable => encode!(23),
            Nil => encode!(24),
            True => encode!(25),
            False => encode!(26),
            Int(i) => encode!(27, i),
            Str(i) => encode!(28, i),
            Function(i) => encode!(29, i),
            NativeFunction(i) => encode!(74 + (*i as u8)), // [72, 256) - currently uses up to MAX_NATIVE_FUNCTION_OPCODE
            List(i) => encode!(30, i),
            Vector(i) => encode!(31, i),
            Set(i) => encode!(32, i),
            Dict(i) => encode!(33, i),
            Struct(i) => encode!(34, i),
            CheckLengthGreaterThan(i) => encode!(35, i),
            CheckLengthEqualTo(i) => encode!(36, i),
            OpFuncEval(i) => encode!(37, i),
            OpIndex => encode!(38),
            OpIndexPeek => encode!(39),
            OpSlice => encode!(40),
            OpSliceWithStep => encode!(41),
            GetField(i) => encode!(42, i),
            GetFieldPeek(i) => encode!(43, i),
            GetFieldFunction(i) => encode!(44, i),
            SetField(i) => encode!(45, i),
            Unary(i) => encode!(49 + (*i as u8)), // [49, 50]
            Binary(i) => encode!(51 + (*i as u8)), // [51, 71]
            Slice => encode!(72),
            SliceWithStep => encode!(73),
            Exit => encode!(46),
            Yield => encode!(47),
            AssertFailed => encode!(48),
        };
    }
}

impl Decode<Opcode> for Decoder {
    fn decode(&mut self) -> Maybe<Opcode> {
        let op = self.decode_u8()?;
        Ok(match op {
            0 => JumpIfFalse(self.decode()?),
            1 => JumpIfFalsePop(self.decode()?),
            2 => JumpIfTrue(self.decode()?),
            3 => JumpIfTruePop(self.decode()?),
            4 => Jump(self.decode()?),
            5 => Return,
            6 => Pop,
            7 => PopN(self.decode()?),
            8 => Dup,
            9 => Swap,
            10 => PushLocal(self.decode()?),
            11 => StoreLocal(self.decode()?),
            12 => PushGlobal(self.decode()?),
            13 => StoreGlobal(self.decode()?),
            14 => PushUpValue(self.decode()?),
            15 => StoreUpValue(self.decode()?),
            16 => StoreArray,
            17 => IncGlobalCount,
            18 => Closure,
            19 => CloseLocal(self.decode()?),
            20 => CloseUpValue(self.decode()?),
            21 => LiftUpValue(self.decode()?),
            22 => InitIterable,
            23 => TestIterable,
            24 => Nil,
            25 => True,
            26 => False,
            27 => Int(self.decode()?),
            28 => Str(self.decode()?),
            29 => Function(self.decode()?),
            74..=MAX_NATIVE_FUNCTION_OPCODE => NativeFunction(stdlib::NativeFunction::get(op - 74)),
            30 => List(self.decode()?),
            31 => Vector(self.decode()?),
            32 => Set(self.decode()?),
            33 => Dict(self.decode()?),
            34 => Struct(self.decode()?),
            35 => CheckLengthGreaterThan(self.decode()?),
            36 => CheckLengthEqualTo(self.decode()?),
            37 => OpFuncEval(self.decode()?),
            38 => OpIndex,
            39 => OpIndexPeek,
            40 => OpSlice,
            41 => OpSliceWithStep,
            42 => GetField(self.decode()?),
            43 => GetFieldPeek(self.decode()?),
            44 => GetFieldFunction(self.decode()?),
            45 => SetField(self.decode()?),
            49..=50 => Unary(unsafe { std::mem::transmute(op - 49) }),
            51..=71 => Binary(unsafe { std::mem::transmute(op - 51) }),
            72 => Slice,
            73 => SliceWithStep,
            46 => Exit,
            47 => Yield,
            48 => AssertFailed,
            _ => return Err(()),
        })
    }
}


#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::io::Cursor;

    use crate::encoder::{Decode, Decoder, Encode, Encoder, MAX_NATIVE_FUNCTION_OPCODE, Maybe};
    use crate::stdlib::NativeFunction;
    use crate::vm::Opcode;
    use crate::vm::operator::{BinaryOp, UnaryOp};

    #[test] fn test_opcode_jump_if_false() { run(Opcode::JumpIfFalse(5), vec![0, 5]); }
    #[test] fn test_opcode_exit() { run(Opcode::Exit, vec![46]); }
    #[test] fn test_opcode_unary_lo() { run(Opcode::Unary(UnaryOp::Minus), vec![49]); }
    #[test] fn test_opcode_unary_hi() { run(Opcode::Unary(UnaryOp::Not), vec![50]); }
    #[test] fn test_opcode_binary_lo() { run(Opcode::Binary(BinaryOp::Mul), vec![51]); }
    #[test] fn test_opcode_binary_hi() { run(Opcode::Binary(BinaryOp::Min), vec![71]); }
    #[test] fn test_opcode_native_function_lo() { run(Opcode::NativeFunction(NativeFunction::Read), vec![74]); }
    #[test] fn test_opcode_native_function_hi() { run(Opcode::NativeFunction(NativeFunction::CountZeros), vec![MAX_NATIVE_FUNCTION_OPCODE]); }

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
