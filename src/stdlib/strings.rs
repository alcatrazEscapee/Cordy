use std::iter::Peekable;
use std::str::Chars;
use crate::vm::error::RuntimeError;
use crate::vm::value::{Value, ValueIter};

use Value::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


macro_rules! fn_str_to_str {
    ($func:ident, $s:ident, $expr:expr) => {
        pub fn $func(v: Value) -> ValueResult {
            match &v {
                Str($s) => Ok($expr),
                _ => TypeErrorFunc1(concat!(stringify!($ident), "(str): str"), v).err()
            }
        }
    };
}


fn_str_to_str!(to_lower, s, Str(Box::new(s.to_lowercase())));
fn_str_to_str!(to_upper, s, Str(Box::new(s.to_uppercase())));
fn_str_to_str!(trim, s, Str(Box::new(String::from(s.trim()))));

pub fn replace(v0: Value, v1: Value, v2: Value) -> ValueResult {
    match (&v0, &v1, &v2) {
        (Str(a0), Str(a1), Str(a2)) => Ok(Str(Box::new(a2.replace(a0.as_ref(), a1.as_str())))),
        _ => TypeErrorFunc3("replace(str, str, str): str", v0, v1, v2).err()
    }
}

pub fn split(v0: Value, v1: Value) -> ValueResult {
    match (&v0, &v1) {
        (Str(a0), Str(a1)) => Ok(Value::iter_list(a1.split(a0.as_ref()).map(|v| Str(Box::new(String::from(v)))))),
        _ => TypeErrorFunc2("split(str, str): [str]", v0, v1).err()
    }
}

pub fn to_char(a1: Value) -> ValueResult {
    match &a1 {
        Int(i) if i > &0 => match char::from_u32(*i as u32) {
            Some(c) => Ok(Value::str(c)),
            None => ValueErrorInvalidCharacterOrdinal(*i).err()
        },
        Int(i) => ValueErrorInvalidCharacterOrdinal(*i).err(),
        _ => TypeErrorArgMustBeInt(a1).err()
    }
}

pub fn to_ord(a1: Value) -> ValueResult {
    match &a1 {
        Str(s) if s.len() == 1 => Ok(Int(s.chars().next().unwrap() as u32 as i64)),
        _ => TypeErrorArgMustBeChar(a1).err()
    }
}

pub fn to_hex(a1: Value) -> ValueResult {
    Ok(Str(Box::new(format!("{:x}", a1.as_int()?))))
}

pub fn to_bin(a1: Value) -> ValueResult {
    Ok(Str(Box::new(format!("{:b}", a1.as_int()?))))
}

pub fn format_string(literal: &String, args: Value) -> ValueResult {
    StringFormatter::format(literal, args)
}

struct StringFormatter<'a> {
    chars: Peekable<Chars<'a>>,
    args: ValueIter<'a>,

    output: String,
}

impl<'a> StringFormatter<'a> {

    fn format(literal: &String, args: Value) -> ValueResult {
        let args = args.as_iter()?;
        let len = literal.len();

        let formatter = StringFormatter {
            chars: literal.chars().peekable(),
            args: args.into_iter(),
            output: String::with_capacity(len)
        };

        formatter.run()
    }

    fn run(mut self: Self) -> ValueResult {
        loop {
            match self.next() {
                Some('%') => {
                    match self.peek() {
                        Some('%') => {
                            self.next();
                            self.push('%');
                            continue
                        },
                        _ => {},
                    }

                    let is_zero_padded: bool = match self.peek() {
                        Some('0') => {
                            self.next();
                            true
                        },
                        _ => false
                    };

                    let mut buffer: String = String::new();
                    loop {
                        match self.peek() {
                            Some(c @ ('1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9')) => {
                                buffer.push(*c);
                                self.next();
                            },
                            Some('0') => {
                                self.next();
                                if buffer.is_empty() {
                                    return ValueErrorInvalidFormatCharacter(Some('0')).err()
                                }
                                buffer.push('0');
                            },
                            _ => break
                        }
                    }

                    let padding: usize = if buffer.is_empty() { 0 } else { buffer.parse::<usize>().unwrap() };

                    let text = match (self.peek(), is_zero_padded) {
                        (Some('d'), false) => format!("{:width$}", self.arg()?.as_int()?, width = padding),
                        (Some('d'), true) => format!("{:0width$}", self.arg()?.as_int()?, width = padding),
                        (Some('x'), false) => format!("{:width$x}", self.arg()?.as_int()?, width = padding),
                        (Some('x'), true) => format!("{:0width$x}", self.arg()?.as_int()?, width = padding),
                        (Some('b'), false) => format!("{:width$b}", self.arg()?.as_int()?, width = padding),
                        (Some('b'), true) => format!("{:0width$b}", self.arg()?.as_int()?, width = padding),
                        (Some('s'), true) => format!("{:width$}", self.arg()?.to_str(), width = padding),
                        (Some('s'), false) => format!("{:0width$}", self.arg()?.to_str(), width = padding),
                        (c, _) => return ValueErrorInvalidFormatCharacter(c.cloned()).err(),
                    };

                    self.next();
                    self.output.push_str(text.as_str());
                },
                Some(c) => self.push(c),
                None => break
            }
        }
        match self.args.next() {
            Some(e) => ValueErrorNotAllArgumentsUsedInStringFormatting(e.clone()).err(),
            None => Ok(Str(Box::new(self.output))),
        }
    }

    fn next(self: &mut Self) -> Option<char> { self.chars.next() }
    fn peek(self: &mut Self) -> Option<&char> { self.chars.peek() }
    fn push(self: &mut Self, c: char) { self.output.push(c); }
    fn arg(self: &mut Self) -> Result<&Value, Box<RuntimeError>> {
        match self.args.next() {
            Some(v) => Ok(v),
            None => ValueErrorMissingRequiredArgumentInStringFormatting.err(),
        }
    }
}


