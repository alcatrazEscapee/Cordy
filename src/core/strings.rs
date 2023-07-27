use std::iter::{FusedIterator, Peekable};
use std::str::Chars;
use fancy_regex::{Captures, Matches, Regex};
use itertools::Itertools;

use crate::core::InvokeArg1;
use crate::vm::{IntoIterableValue, IntoValue, Iterable, Value, RuntimeError, VirtualInterface, ValueResult};
use crate::util;

use Value::{*};
use RuntimeError::{*};


pub fn to_lower(value: Value) -> ValueResult {
    Ok(value.as_str()?.to_lowercase().to_value())
}

pub fn to_upper(value: Value) -> ValueResult {
    Ok(value.as_str()?.to_uppercase().to_value())
}

pub fn trim(value: Value) -> ValueResult {
    Ok(value.as_str()?.trim().to_value())
}

pub fn replace<VM: VirtualInterface>(vm: &mut VM, pattern: Value, replacer: Value, target: Value) -> ValueResult {
    let regex: Regex = compile_regex(pattern)?;
    let text = target.as_str()?;
    if replacer.is_function() {
        let replacer: InvokeArg1 = InvokeArg1::from(replacer)?;
        let mut err = None;
        let replaced: Value = regex.replace_all(text, |captures: &Captures| {
            let arg: Value = as_result(captures);
            util::yield_result(&mut err, || replacer.invoke(arg, vm)?.as_str().cloned(), String::new())
        }).to_value();
        util::join_result(replaced, err)
    } else {
        Ok(regex.replace_all(text, replacer.as_str()?).to_value())
    }
}

pub fn search(pattern: Value, target: Value) -> ValueResult {
    let regex: Regex = compile_regex(pattern)?;
    let text: &String = target.as_str()?;

    let mut start: usize = 0;
    Ok(std::iter::from_fn(move || {
        match regex.captures_from_pos(text, start).unwrap() {
            Some(captures) => {
                start = captures.get(0).unwrap().end();
                Some(as_result(&captures))
            },
            None => None
        }
    }).to_list())
}

pub fn split(pattern: Value, target: Value) -> ValueResult {
    if pattern.as_str()? == "" {
        // Special case for empty string
        return Ok(target.as_str()?.chars().map(|u| u.to_value()).to_list());
    }
    let regex: Regex = compile_regex(pattern)?;
    let text: &String = target.as_str()?;
    Ok(fancy_split(&regex, text.as_str())
        .map(|u| u.to_value())
        .to_list())
}

fn as_result(captures: &Captures) -> Value {
    captures.iter()
        .map(|group| group.unwrap().as_str().to_value())
        .to_vector()
}

fn compile_regex(a1: Value) -> Result<Regex, Box<RuntimeError>> {
    let raw = escape_regex(a1.as_str()?);
    match Regex::new(&raw) {
        Ok(regex) => Ok(regex),
        Err(e) => ValueErrorCannotCompileRegex(raw, e.to_string()).err()
    }
}

/// Replaces escaped characters `\t`, `\n`, `\r` with their original un-escaped sequences.
fn escape_regex(raw: &String) -> String {
    let mut result = String::with_capacity(raw.len());
    for c in raw.chars() {
        match c {
            '\t' => result.push_str("\\t"),
            '\r' => result.push_str("\\r"),
            '\n' => result.push_str("\\n"),
            _ => result.push(c)
        };
    }
    result
}

/// For some reason the `fancy_regex` crate does not support `.split()`
/// However, it does support `find_iter()`, so we just create the same extension to allow `Split` to work.
/// This implementation is completely borrowed from the `re_unicode.rs` module in the `regex` crate, and adapted to use `Regex` from the `fancy-regex` crate.
///
/// `'r` is the lifetime of the compiled regular expression and `'t` is the lifetime of the string being split.
fn fancy_split<'r, 't>(regex: &'r Regex, text: &'t str) -> FancySplit<'r, 't> {
    FancySplit { finder: regex.find_iter(text), last: 0 }
}

#[derive(Debug)]
struct FancySplit<'r, 't> {
    finder: Matches<'r, 't>,
    last: usize,
}

impl<'r, 't> Iterator for FancySplit<'r, 't> {
    type Item = &'t str;

    fn next(&mut self) -> Option<&'t str> {
        let text = self.finder.text();
        match self.finder.next() {
            None => {
                if self.last > text.len() {
                    None
                } else {
                    let s = &text[self.last..];
                    self.last = text.len() + 1; // Next call will return None
                    Some(s)
                }
            }
            Some(Ok(m)) => {
                let matched = &text[self.last..m.start()];
                self.last = m.end();
                Some(matched)
            },
            _ => None
        }
    }
}

impl<'r, 't> FusedIterator for FancySplit<'r, 't> {}


pub fn join(joiner: Value, it: Value) -> ValueResult {
    Ok(it.as_iter()?
        .map(|u| u.to_str())
        .join(joiner.as_str()?)
        .to_value())
}


pub fn to_char(value: Value) -> ValueResult {
    match &value {
        Int(i) if i > &0 => match char::from_u32(*i as u32) {
            Some(c) => Ok(c.to_value()),
            None => ValueErrorInvalidCharacterOrdinal(*i).err()
        },
        Int(i) => ValueErrorInvalidCharacterOrdinal(*i).err(),
        _ => TypeErrorArgMustBeInt(value).err()
    }
}

pub fn to_ord(value: Value) -> ValueResult {
    match &value {
        Str(s) if s.len() == 1 => Ok(Int(s.chars().next().unwrap() as u32 as i64)),
        _ => TypeErrorArgMustBeChar(value).err()
    }
}

pub fn to_hex(value: Value) -> ValueResult {
    Ok(format!("{:x}", value.as_int()?).to_value())
}

pub fn to_bin(value: Value) -> ValueResult {
    Ok(format!("{:b}", value.as_int()?).to_value())
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


