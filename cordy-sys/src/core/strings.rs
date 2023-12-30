use std::iter::{FusedIterator, Peekable};
use std::str::Chars;
use fancy_regex::{Captures, Matches, Regex};

use crate::core::InvokeArg1;
use crate::util;
use crate::vm::{ErrorPtr, ErrorResult, IntoIterableValue, IntoValue, Iterable, RuntimeError, Type, ValuePtr, ValueResult, VirtualInterface};

use RuntimeError::{*};


pub fn to_lower(value: ValuePtr) -> ValueResult {
    value.check_str()?
        .as_str_slice()
        .to_lowercase()
        .to_value()
        .ok()
}

pub fn to_upper(value: ValuePtr) -> ValueResult {
    value.check_str()?
        .as_str_slice()
        .to_uppercase()
        .to_value()
        .ok()
}

pub fn trim(value: ValuePtr) -> ValueResult {
    value.check_str()?
        .as_str_slice()
        .trim()
        .to_value()
        .ok()
}

pub fn replace<VM: VirtualInterface>(vm: &mut VM, pattern: ValuePtr, replacer: ValuePtr, target: ValuePtr) -> ValueResult {
    let regex: Regex = compile_regex(pattern)?;
    let target = target.check_str()?;
    let text = target.as_str_slice();
    if replacer.is_evaluable() {
        let replacer: InvokeArg1 = InvokeArg1::from(replacer)?;
        let mut err = None;
        let replaced: ValuePtr = regex.replace_all(text, |captures: &Captures| {
            let arg: ValuePtr = as_result(captures);
            util::catch::<String>(&mut err, || Ok(replacer.invoke(arg, vm)?
                .check_str()?
                .as_str_owned()),
                String::new())
        }).to_value();
        util::join::<ValuePtr, ErrorPtr, ValueResult>(replaced, err)
    } else {
        regex.replace_all(text, replacer.check_str()?.as_str_slice())
            .to_value()
            .ok()
    }
}

pub fn search(pattern: ValuePtr, target: ValuePtr) -> ValueResult {
    let regex: Regex = compile_regex(pattern)?;
    let target = target.check_str()?;
    let text: &str = target.as_str_slice();

    let mut start: usize = 0;
    std::iter::from_fn(move || {
        match regex.captures_from_pos(text, start).unwrap() {
            Some(captures) => {
                let end = captures.get(0).unwrap().end();
                if start == end {
                    // If we match a zero-length group, this regex will run infinitely - so the only sensible way to handle this here is to abort
                    // Regex captures are greedy by default, so this will trigger only,
                    // - If we can *only* match an empty string, which will return nothing
                    // - If we've already matched the whole string, and now are matching *against* an empty string, which will return the existing match
                    return None
                }
                start = end;
                Some(as_result(&captures))
            },
            None => None
        }
    }).to_list().ok()
}

pub fn split(pattern: ValuePtr, target: ValuePtr) -> ValueResult {
    let pattern = pattern.check_str()?;
    let target = target.check_str()?;

    if pattern.as_str_slice().is_empty() { // Special case for empty string
        return target.as_str_slice()
            .chars()
            .map(|u| u.to_value())
            .to_list()
            .ok();
    }

    let regex: Regex = compile_regex(pattern)?;

    fancy_split(&regex, target.as_str_slice())
        .map(|u| u.to_value())
        .to_list()
        .ok()
}

fn as_result(captures: &Captures) -> ValuePtr {
    if captures.len() == 1 {
        captures.get(0)
            .map_or(ValuePtr::nil(), |u| u.as_str().to_value())
    } else {
        captures.iter()
            .map(|group| group.map_or(ValuePtr::nil(), |u| u.as_str().to_value()))
            .to_vector()
    }
}

fn compile_regex(a1: ValuePtr) -> ErrorResult<Regex> {
    let raw = escape_regex(a1.check_str()?.as_str_slice());
    match Regex::new(&raw) {
        Ok(regex) => Ok(regex),
        Err(e) => ValueErrorCannotCompileRegex(raw, e.to_string()).err()
    }
}

/// Replaces escaped characters `\t`, `\n`, `\r` with their original un-escaped sequences.
fn escape_regex(raw: &str) -> String {
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


pub fn join(joiner: ValuePtr, it: ValuePtr) -> ValueResult {
    map_join(it.to_iter()?, joiner.check_str()?.as_str_slice()).ok()
}


fn map_join<'a, I : Iterator<Item=ValuePtr>>(mut iter: I, sep: &str) -> ValuePtr {
    // Avoids issues with `.map().join()` that create temporaries in the `map()` and then destroy them
    match iter.next() {
        None => ValuePtr::str(),
        Some(first) => {
            let (lower, _) = iter.size_hint();
            let mut result = String::with_capacity(lower * sep.len());
            result.push_str(&first.to_str());
            while let Some(next) = iter.next() {
                result.push_str(sep);
                result.push_str(&next.to_str());
            }
            result.to_value()
        }
    }
}


pub fn chr(value: ValuePtr) -> ValueResult {
    let i = value.check_int()?.as_int();
    if i <= 0 {
        return ValueErrorInvalidCharacterOrdinal(i).err()
    }
    match char::from_u32(i as u32) {
        Some(c) => c.to_value().ok(),
        None => ValueErrorInvalidCharacterOrdinal(i).err()
    }
}

pub fn ord(value: ValuePtr) -> ValueResult {
    let value = value.check_str()?;
    let s = value.as_str_slice();
    match s.len() {
        1 => (s.chars().next().unwrap() as u32 as i64)
            .to_value()
            .ok(),
        _ => TypeErrorArgMustBeChar(value).err(),
    }
}

pub fn hex(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Int => format!("{:x}", value.as_precise_int()).to_value().ok(),
        Type::ShortStr | Type::LongStr => match i64::from_str_radix(value.as_str_slice(), 16).ok() {
            Some(int) => int.to_value().ok(),
            _ => TypeErrorCannotConvertToInt(value).err()
        },
        _ => TypeErrorArgMustBeIntOrStr(value).err()
    }
}

pub fn bin(value: ValuePtr) -> ValueResult {
    match value.ty() {
        Type::Int => format!("{:b}", value.as_precise_int()).to_value().ok(),
        Type::ShortStr | Type::LongStr => match i64::from_str_radix(value.as_str_slice(), 2).ok() {
            Some(int) => int.to_value().ok(),
            _ => TypeErrorCannotConvertToInt(value).err()
        }
        _ => TypeErrorArgMustBeIntOrStr(value).err()
    }
}

pub fn format_string(literal: &str, args: ValuePtr) -> ValueResult {
    StringFormatter::format(literal, args)
}

struct StringFormatter<'a> {
    chars: Peekable<Chars<'a>>,
    args: Iterable,

    output: String,
}

impl<'a> StringFormatter<'a> {

    fn format(literal: &str, args: ValuePtr) -> ValueResult {
        let args = args.as_iter_or_unit();
        let len = literal.len();

        let formatter = StringFormatter {
            chars: literal.chars().peekable(),
            args,
            output: String::with_capacity(len)
        };

        formatter.run()
    }

    fn run(mut self) -> ValueResult {
        loop {
            match self.next() {
                Some('%') => {
                    if let Some('%') = self.peek() {
                        self.next();
                        self.push('%');
                        continue
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
                        (Some('d'), false) => format!("{:width$}", self.arg()?.check_int()?.as_int(), width = padding),
                        (Some('d'), true) => format!("{:0width$}", self.arg()?.check_int()?.as_int(), width = padding),
                        (Some('x'), false) => format!("{:width$x}", self.arg()?.check_int()?.as_int(), width = padding),
                        (Some('x'), true) => format!("{:0width$x}", self.arg()?.check_int()?.as_int(), width = padding),
                        (Some('b'), false) => format!("{:width$b}", self.arg()?.check_int()?.as_int(), width = padding),
                        (Some('b'), true) => format!("{:0width$b}", self.arg()?.check_int()?.as_int(), width = padding),
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
            None => self.output.to_value().ok(),
        }
    }

    fn next(&mut self) -> Option<char> { self.chars.next() }
    fn peek(&mut self) -> Option<&char> { self.chars.peek() }
    fn push(&mut self, c: char) { self.output.push(c); }
    fn arg(&mut self) -> ValueResult {
        match self.args.next() {
            Some(v) => v.ok(),
            None => ValueErrorMissingRequiredArgumentInStringFormatting.err(),
        }
    }
}


