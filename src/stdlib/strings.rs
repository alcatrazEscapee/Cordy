use std::iter::Peekable;
use std::str::Chars;

use crate::vm::{IntoIterableValue, IntoValue, Iterable, Value, RuntimeError};

use Value::{*};
use RuntimeError::{*};

type ValueResult = Result<Value, Box<RuntimeError>>;


pub fn to_lower(a0: Value) -> ValueResult {
    Ok(a0.as_str()?.to_lowercase().to_value())
}

pub fn to_upper(a0: Value) -> ValueResult {
    Ok(a0.as_str()?.to_uppercase().to_value())
}

pub fn trim(a0: Value) -> ValueResult {
    Ok(a0.as_str()?.trim().to_value())
}

pub fn replace(a0: Value, a1: Value, a2: Value) -> ValueResult {
    Ok(a2.as_str()?.replace(a0.as_str()?, a1.as_str()?.as_str()).to_value())
}

pub fn split(a0: Value, a1: Value) -> ValueResult {
    Ok(a1.as_str()?.split(a0.as_str()?).map(|u| u.to_value()).to_list())
}

pub fn to_char(a1: Value) -> ValueResult {
    match &a1 {
        Int(i) if i > &0 => match char::from_u32(*i as u32) {
            Some(c) => Ok(c.to_value()),
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
    Ok(format!("{:x}", a1.as_int()?).to_value())
}

pub fn to_bin(a1: Value) -> ValueResult {
    Ok(format!("{:b}", a1.as_int()?).to_value())
}

pub fn format_string(literal: &String, args: Value) -> ValueResult {
    StringFormatter::format(literal, args)
}

struct StringFormatter<'a> {
    chars: Peekable<Chars<'a>>,
    args: Iterable,

    output: String,
}

impl<'a> StringFormatter<'a> {

    fn format(literal: &String, args: Value) -> ValueResult {
        let args = args.as_iter_or_unit();
        let len = literal.len();

        let formatter = StringFormatter {
            chars: literal.chars().peekable(),
            args,
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
        match (&mut self.args).next() {
            Some(e) => ValueErrorNotAllArgumentsUsedInStringFormatting(e.clone()).err(),
            None => Ok(self.output.to_value()),
        }
    }

    fn next(self: &mut Self) -> Option<char> { self.chars.next() }
    fn peek(self: &mut Self) -> Option<&char> { self.chars.peek() }
    fn push(self: &mut Self, c: char) { self.output.push(c); }
    fn arg(self: &mut Self) -> Result<Value, Box<RuntimeError>> {
        match (&mut self.args).next() {
            Some(v) => Ok(v),
            None => ValueErrorMissingRequiredArgumentInStringFormatting.err(),
        }
    }
}


