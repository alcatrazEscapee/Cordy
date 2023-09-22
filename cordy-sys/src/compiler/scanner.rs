use std::fmt::Debug;
use std::iter::Peekable;
use std::num::ParseIntError;
use std::str::Chars;

use crate::core::NativeFunction;
use crate::reporting::{AsErrorWithContext, Location};
use crate::SourceView;

use self::ScanErrorType::{*};
use self::ScanToken::{*};


pub fn scan(view: &SourceView) -> ScanResult {
    let text = view.text();
    let mut scanner: Scanner = Scanner {
        chars: text.chars().peekable(),
        tokens: Vec::new(),
        errors: Vec::new(),
        cursor: 0,
        index: view.index(),
    };
    scanner.scan();
    ScanResult {
        tokens: scanner.tokens,
        errors: scanner.errors
    }
}


#[derive(Debug, Clone)]
pub struct ScanResult {
    pub tokens: Vec<(Location, ScanToken)>,
    pub errors: Vec<ScanError>
}


#[derive(Debug, Clone)]
pub struct ScanError {
    pub error: ScanErrorType,
    pub loc: Location,
}

impl ScanError {
    pub fn is_eof(&self) -> bool {
        matches!(self.error, UnterminatedStringLiteral | UnterminatedBlockComment)
    }
}

impl AsErrorWithContext for ScanError {
    fn location(&self) -> Location {
        self.loc
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ScanErrorType {
    InvalidNumericPrefix(char),
    InvalidNumericValue(ParseIntError),
    InvalidCharacter(char),
    UnterminatedStringLiteral,
    UnterminatedBlockComment,
}

#[repr(u8)]
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub enum ScanTokenType {
    Keyword,
    Constant,
    Native,
    Type,
    Number,
    String,
    Syntax,
    Blank, // Used for newline tokens
    Comment, // Special type, used for comments, which don't have any emitted tokens, but still may need to highlight.
}


#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ScanToken {
    // Special
    Identifier(String),
    StringLiteral(String),
    IntLiteral(i64),
    ComplexLiteral(i64),

    // Keywords
    KeywordLet,
    KeywordFn,
    KeywordReturn,
    KeywordIf,
    KeywordElif,
    KeywordElse,
    KeywordThen,
    KeywordLoop,
    KeywordWhile,
    KeywordFor,
    KeywordIn,
    KeywordIs,
    KeywordNot,
    KeywordBreak,
    KeywordContinue,
    KeywordDo,
    KeywordTrue,
    KeywordFalse,
    KeywordNil,
    KeywordStruct,
    KeywordExit,
    KeywordAssert,

    // Syntax
    Equals,
    PlusEquals,
    MinusEquals,
    MulEquals,
    DivEquals,
    AndEquals,
    OrEquals,
    XorEquals,
    LeftShiftEquals,
    RightShiftEquals,
    ModEquals,
    PowEquals,
    DotEquals,

    Plus,
    Minus,
    Mul,
    Div,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Mod,
    Pow,
    LeftShift,
    RightShift,

    Not,

    LogicalAnd,
    LogicalOr,

    NotEquals,
    DoubleEquals,
    LessThan,
    LessThanEquals,
    GreaterThan,
    GreaterThanEquals,

    OpenParen, // ( )
    CloseParen,
    OpenSquareBracket, // [ ]
    CloseSquareBracket,
    OpenBrace, // { }
    CloseBrace,

    Comma,
    Dot,
    Colon,
    Arrow,
    Underscore,
    Semicolon,
    At,
    Ellipsis,
    QuestionMark,

    NewLine,
}

impl ScanToken {
    pub(super) fn ty(self) -> ScanTokenType {
        match self {
            StringLiteral(_) => ScanTokenType::String,
            IntLiteral(_) | ComplexLiteral(_) => ScanTokenType::Number,
            KeywordTrue | KeywordFalse | KeywordNil | LogicalAnd | LogicalOr => ScanTokenType::Constant,
            KeywordLet | KeywordFn | KeywordReturn | KeywordIf | KeywordElif | KeywordElse | KeywordThen | KeywordLoop | KeywordWhile | KeywordFor | KeywordIn | KeywordIs | KeywordNot | KeywordBreak | KeywordContinue | KeywordDo | KeywordStruct | KeywordExit | KeywordAssert => ScanTokenType::Keyword,
            Identifier(it)  => match NativeFunction::find(it.as_str()) {
                Some(NativeFunction::Int | NativeFunction::Str | NativeFunction::Function | NativeFunction::List | NativeFunction::Heap | NativeFunction::Dict | NativeFunction::Set | NativeFunction::Vector | NativeFunction::Any | NativeFunction::Bool | NativeFunction::Iterable | NativeFunction::Complex) => ScanTokenType::Type,
                Some(_) => ScanTokenType::Native,
                _ => ScanTokenType::Syntax,
            }
            NewLine => ScanTokenType::Blank,
            _ => ScanTokenType::Syntax
        }
    }
}


struct Scanner<'a> {
    chars: Peekable<Chars<'a>>,
    tokens: Vec<(Location, ScanToken)>,
    errors: Vec<ScanError>,
    cursor: usize,
    index: u32,
}


impl<'a> Scanner<'a> {
    
    fn scan(&mut self) {
        loop {
           match self.advance() {
               Some(c) => {
                   match c {
                       ' ' | '\t' | '\r' | '\n' => {},

                       'a'..='z' | 'A'..='Z' => {
                           let mut buffer: String = String::new();
                           buffer.push(c);
                           loop {
                               match self.peek() {
                                   Some('a'..='z' | 'A'..='Z' | '0'..='9' | '_') => self.push_advance(&mut buffer),
                                   _ => break
                               }
                           }
                           self.screen_identifier(buffer);
                       },
                       '0' => {
                            match self.peek() {
                                Some('x') => {
                                    self.advance();
                                    let mut buffer: String = String::new();
                                    let mut extra_len = 0;
                                    loop {
                                        match self.peek() {
                                            Some('0'..='9' | 'A'..='F' | 'a'..='f') => self.push_advance(&mut buffer),
                                            Some('_') => {
                                                self.skip();
                                                extra_len += 1;
                                            },
                                            _ => break
                                        };
                                    }
                                    self.screen_int(buffer, 16, extra_len);
                                },
                                Some('b') => {
                                    self.advance();
                                    let mut buffer: String = String::new();
                                    let mut extra_len = 0;
                                    loop {
                                        match self.peek() {
                                            Some('1' | '0') => self.push_advance(&mut buffer),
                                            Some('_') => {
                                                self.skip();
                                                extra_len += 1;
                                            },
                                            _ => break
                                        }
                                    }
                                    self.screen_int(buffer, 2, extra_len);
                                },
                                Some('i' | 'j') => { // Complex literal `0i` which is equal to `0`
                                    self.advance();
                                    self.push(2, IntLiteral(0));
                                }
                                Some(e @ ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) => self.push_err(1, 2, InvalidNumericPrefix(e)),
                                Some(_) => {
                                    // Don't consume, as this isn't part of the number, just a '0' literal followed by some other syntax
                                    self.push(1, IntLiteral(0));
                                },
                                None => {
                                    // The last element in the input is `0`, so still emit `Int(0)`
                                    self.advance();
                                    self.push(1, IntLiteral(0));
                                },
                            }
                       }
                       '1'..='9' => {
                           let mut buffer: String = String::new();
                           let mut extra_len = 0;
                           buffer.push(c);
                           loop {
                               match self.peek() {
                                   Some('0'..='9') => self.push_advance(&mut buffer),
                                   Some('_') => {
                                       self.skip();
                                       extra_len += 1;
                                   },
                                   _ => break
                               }
                           }
                           self.screen_int(buffer, 10, extra_len);
                       },

                       open @ ('\'' | '"') => {
                           let mut buffer: Vec<char> = Vec::new();
                           let mut escaped: bool = false;
                           let start: usize = self.cursor;
                           loop {
                               match self.advance() {
                                   // Escaped quote always emits the single character
                                   // Un-escaped will emit if it's not the same as the open
                                   Some(quote @ ('\'' | '"')) => {
                                       if escaped {
                                           buffer.push(quote);
                                           escaped = false;
                                       } else if open != quote {
                                            buffer.push(quote);
                                       } else {
                                           break
                                       }
                                   },
                                   Some('\\') => { // Escaped backslash will emit a backslash, un-escaped will begin an escape sequence (skipping the backslash)
                                       if escaped {
                                           buffer.push('\\');
                                           escaped = false;
                                       } else {
                                           escaped = true;
                                       }
                                   },
                                   Some('\r') => {}, // A natural `\r` never gets included in a string, when present in source
                                   Some('n') if escaped => { // `\n` escape sequence -> emit a single `\n`
                                       buffer.push('\n');
                                       escaped = false;
                                   },
                                   Some('r') if escaped => { // `\r` escape sequence -> emit a single `\r`
                                       buffer.push('\r');
                                       escaped = false;
                                   }
                                   Some('t') if escaped => {
                                       buffer.push('\t'); // `\t` escape sequence -> emit a single `\t`
                                       escaped = false;
                                   },
                                   Some(c0) => { // Any other character, emits itself. If escaped, the backslash is also included as part of the string
                                       if escaped {
                                           buffer.push('\\');
                                       }
                                       buffer.push(c0);
                                       escaped = false;
                                   }
                                   None => {
                                       // Manually report this error at the source point, not at the destination point of the string
                                       // It makes it much easier to read.
                                       self.push_err_at(start, 1, UnterminatedStringLiteral);
                                       break
                                   }
                               }
                           }
                           self.push(self.cursor - start + 1, StringLiteral(buffer.iter().collect()))
                       },

                       '!' => match self.peek() {
                           Some('=') => self.push_skip(2, NotEquals),
                           _ => self.push(1, Not)
                       },

                       '=' => match self.peek() {
                           Some('=') => self.push_skip(2, DoubleEquals),
                           _ => self.push(1, Equals)
                       },
                       '>' => match self.peek() {
                           Some('>') => match self.advance_peek() {
                               Some('=') => self.push_skip(3, RightShiftEquals),
                               _ => self.push(2, RightShift)
                           }
                           Some('=') => self.push_skip(2, GreaterThanEquals),
                           _ => self.push(1, GreaterThan)
                       },
                       '<' => match self.peek() {
                           Some('<') => match self.advance_peek() {
                               Some('=') => self.push_skip(3, LeftShiftEquals),
                               _ => self.push(2, LeftShift)
                           },
                           Some('=') => self.push_skip(2, LessThanEquals),
                           _ => self.push(1, LessThan)
                       },

                       '+' => match self.peek() {
                           Some('=') => self.push_skip(2, PlusEquals),
                           _ => self.push(1, Plus)
                       },
                       '-' => match self.peek() {
                           Some('=') => self.push_skip(2, MinusEquals),
                           Some('>') => self.push_skip(2, Arrow),
                           _ => self.push(1, Minus)
                       },
                       '*' => match self.peek() {
                           Some('=') => self.push_skip(2, MulEquals),
                           Some('*') => match self.advance_peek() {
                               Some('=') => self.push_skip(3, PowEquals),
                               _ => self.push(2, Pow)
                           },
                           _ => self.push(1, Mul)
                       },
                       '/' => match self.peek() {
                           Some('/') => {
                               // Single-line comment
                               loop {
                                   match self.advance() {
                                       Some('\n') => break,
                                       Some(_) => {},
                                       None => break
                                   }
                               }
                           }
                           Some('*') => {
                               let start: usize = self.cursor;
                               loop {
                                   match self.advance() {
                                       Some('*') => {
                                           match self.advance() {
                                               Some('/') => break,
                                               Some(_) => {},
                                               None => {
                                                   self.push_err_at(start, 2, UnterminatedBlockComment);
                                                   break
                                               }
                                           }
                                       },
                                       Some(_) => {},
                                       None => {
                                           self.push_err_at(start, 2, UnterminatedBlockComment);
                                           break
                                       }
                                   }
                               }
                           }
                           Some('=') => self.push_skip(2, DivEquals),
                           _ => self.push(1, Div)
                       },
                       '|' => match self.peek() {
                           Some('=') => self.push_skip(2, OrEquals),
                           _ => self.push(1, BitwiseOr)
                       },
                       '&' => match self.peek() {
                           Some('=') => self.push_skip(2, AndEquals),
                           _ => self.push(1, BitwiseAnd)
                       },
                       '^' => match self.peek() {
                           Some('=') => self.push_skip(2, XorEquals),
                           _ => self.push(1, BitwiseXor)
                       },
                       '%' => match self.peek() {
                           Some('=') => self.push_skip(2, ModEquals),
                           _ => self.push(1, Mod)
                       },
                       '.' => match self.peek() {
                           Some('=') => self.push_skip(2, DotEquals),
                           Some('.') => match self.advance_peek() {
                               Some('.') => self.push_skip(3, Ellipsis),
                               _ => {
                                   self.cursor -= 1;
                                   self.push(1, Dot);
                                   self.cursor += 1;
                                   self.push(1, Dot);
                               }
                           }
                           _ => self.push(1, Dot)
                       },

                       '(' => self.push(1, OpenParen),
                       ')' => self.push(1, CloseParen),
                       '[' => self.push(1, OpenSquareBracket),
                       ']' => self.push(1 , CloseSquareBracket),
                       '{' => self.push(1, OpenBrace),
                       '}' => self.push(1, CloseBrace),

                       ',' => self.push(1, Comma),
                       ':' => self.push(1, Colon),
                       '_' => self.push(1, Underscore),
                       ';' => self.push(1, Semicolon),
                       '@' => self.push(1, At),
                       '?' => self.push(1, QuestionMark),

                       e => self.push_err(0, 1, InvalidCharacter(e))
                   }
               }
               None => break // eof
           }
        }
    }

    fn screen_identifier(&mut self, buffer: String) {
        let len: usize = buffer.len();
        let token: ScanToken = match buffer.as_str() {
            "let" => KeywordLet,
            "fn" => KeywordFn,
            "return" => KeywordReturn,
            "if" => KeywordIf,
            "elif" => KeywordElif,
            "else" => KeywordElse,
            "then" => KeywordThen,
            "loop" => KeywordLoop,
            "while" => KeywordWhile,
            "for" => KeywordFor,
            "in" => KeywordIn,
            "is" => KeywordIs,
            "not" => KeywordNot,
            "break" => KeywordBreak,
            "continue" => KeywordContinue,
            "do" => KeywordDo,
            "true" => KeywordTrue,
            "false" => KeywordFalse,
            "nil" => KeywordNil,
            "struct" => KeywordStruct,
            "exit" => KeywordExit,
            "assert" => KeywordAssert,
            "and" => LogicalAnd,
            "or" => LogicalOr,
             _ => Identifier(buffer)
        };
        self.push(len, token);
    }

    fn screen_int(&mut self, buffer: String, radix: u32, extra_len: usize) {
        let mut len: usize = buffer.len() + extra_len;
        let is_complex: bool = match self.peek() {
            Some('i' | 'j') => {
                self.advance();
                len += 1;
                true
            },
            _ => false,
        };
        if radix != 10 {
            len += 2; // To account for the numeric prefix
        }

        match i64::from_str_radix(buffer.as_str(), radix) {
            Ok(value) => self.push(len, if is_complex { ComplexLiteral(value) } else { IntLiteral(value) }),
            Err(e) => self.push_err(0, len, InvalidNumericValue(e))
        }
    }


    fn push(&mut self, width: usize, token: ScanToken) {
        self.tokens.push((Location::new(self.cursor - width, width as u32, self.index), token));
    }

    fn push_skip(&mut self, width: usize, token: ScanToken) {
        self.skip();
        self.push(width, token);
    }

    fn push_err(&mut self, offset: usize, width: usize, error: ScanErrorType) {
        self.errors.push(ScanError {
            error,
            loc: Location::new(self.cursor - width + offset, width as u32, self.index)
        });
    }

    /// Reports an error at a given source point, not until the end. Used for 'unterminated' errors, as it makes them easier to read.
    fn push_err_at(&mut self, start: usize, width: u32, error: ScanErrorType) {
        // Manually report this error at the source point, not at the destination point of the string
        // It makes it much easier to read.
        self.errors.push(ScanError {
            error,
            loc: Location::new(start, (self.cursor - start) as u32 + width, self.index)
        });
    }


    /// Consumes the next character (unconditionally) and adds it to the buffer
    /// **Note**: This function must only be invoked after `Some()` has been matched to a `peek()` variant.
    fn push_advance(&mut self, buffer: &mut String) {
        buffer.push(self.advance().unwrap());
    }

    /// Consumes the next character without returning it.
    /// Also see `advance()`
    fn skip(&mut self) {
        self.advance();
    }

    /// Consumes the next character, and peeks one character ahead
    /// Chains together `advance()` and `peek()`
    fn advance_peek(&mut self) -> Option<char> {
        self.advance();
        self.peek()
    }

    /// Consumes the next character and returns it
    /// Also see `advance()`
    fn advance(&mut self) -> Option<char> {
        let c: Option<char> = self.chars.next();
        if c.is_some() {
            self.cursor += 1;
        }
        if let Some('\n') = c {
            self.push(1, NewLine);
        }
        c
    }

    /// Inspects the next character and returns it, without consuming it
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }
}


#[cfg(test)]
mod tests {
    use crate::compiler::scanner;
    use crate::compiler::scanner::{ScanResult, ScanToken};
    use crate::reporting::SourceView;
    use crate::test_util;

    use ScanToken::{*};


    #[test] fn test_empty() { run_str("", vec![]); }
    #[test] fn test_keywords() { run_str("let fn return if elif else then loop while for in is not break continue do true false nil struct exit assert", vec![KeywordLet, KeywordFn, KeywordReturn, KeywordIf, KeywordElif, KeywordElse, KeywordThen, KeywordLoop, KeywordWhile, KeywordFor, KeywordIn, KeywordIs, KeywordNot, KeywordBreak, KeywordContinue, KeywordDo, KeywordTrue, KeywordFalse, KeywordNil, KeywordStruct, KeywordExit, KeywordAssert]); }
    #[test] fn test_identifiers() { run_str("foobar big_bad_wolf ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz", vec![Identifier(String::from("foobar")), Identifier(String::from("big_bad_wolf")), Identifier(String::from("ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"))]); }
    #[test] fn test_str_literals() { run_str("'abc' 'a \n 3' '\\''", vec![StringLiteral(String::from("abc")), NewLine, StringLiteral(String::from("a \n 3")), StringLiteral(String::from("'"))]); }
    #[test] fn test_double_quote_str_literals() { run_str("\"abc\" '\"' \"'\"", vec![StringLiteral(String::from("abc")), StringLiteral(String::from("\"")), StringLiteral(String::from("'"))]); }
    #[test] fn test_str_escaping() { run_str("'\\.' '\\\\.' '\\n' '\\\\n'", vec![StringLiteral(String::from("\\.")), StringLiteral(String::from("\\.")), StringLiteral(String::from("\n")), StringLiteral(String::from("\\n"))]); }
    #[test] fn test_ints() { run_str("1234 654 10_00_00 0 1", vec![IntLiteral(1234), IntLiteral(654), IntLiteral(100000), IntLiteral(0), IntLiteral(1)]); }
    #[test] fn test_binary_ints() { run_str("0b11011011 0b0 0b1 0b1_01", vec![IntLiteral(0b11011011), IntLiteral(0b0), IntLiteral(0b1), IntLiteral(0b101)]); }
    #[test] fn test_hex_ints() { run_str("0x12345678 0xabcdef90 0xABCDEF 0xF_f", vec![IntLiteral(0x12345678), IntLiteral(0xabcdef90), IntLiteral(0xABCDEF), IntLiteral(0xFF)])}
    #[test] fn test_complex_ints() { run_str("0i 0j 1i 1j 0b101i 0xfi 123i", vec![IntLiteral(0), IntLiteral(0), ComplexLiteral(1), ComplexLiteral(1), ComplexLiteral(5), ComplexLiteral(0xf), ComplexLiteral(123)]); }
    #[test] fn test_unary_operators() { run_str("- !", vec![Minus, Not]); }
    #[test] fn test_comparison_operators() { run_str("> < >= > = <= < =", vec![GreaterThan, LessThan, GreaterThanEquals, GreaterThan, Equals, LessThanEquals, LessThan, Equals]); }
    #[test] fn test_equality_operators() { run_str("!= ! = == =", vec![NotEquals, Not, Equals, DoubleEquals, Equals]); }
    #[test] fn test_binary_logical_operators() { run_str("and & or |", vec![LogicalAnd, BitwiseAnd, LogicalOr, BitwiseOr]); }
    #[test] fn test_arithmetic_operators() { run_str("+ - += -= * = *= / = /=", vec![Plus, Minus, PlusEquals, MinusEquals, Mul, Equals, MulEquals, Div, Equals, DivEquals]); }
    #[test] fn test_other_arithmetic_operators() { run_str("% %= ** *= **= * *=", vec![Mod, ModEquals, Pow, MulEquals, PowEquals, Mul, MulEquals]); }
    #[test] fn test_bitwise_operators() { run_str("| ^ & &= |= ^=", vec![BitwiseOr, BitwiseXor, BitwiseAnd, AndEquals, OrEquals, XorEquals]); }
    #[test] fn test_groupings() { run_str("( [ { } ] )", vec![OpenParen, OpenSquareBracket, OpenBrace, CloseBrace, CloseSquareBracket, CloseParen]); }
    #[test] fn test_syntax() { run_str(". .. ... .= , -> - > : @", vec![Dot, Dot, Dot, Ellipsis, DotEquals, Comma, Arrow, Minus, GreaterThan, Colon, At]); }


    #[test] fn test_hello_world() { run("hello_world"); }
    #[test] fn test_invalid_character() { run("invalid_character"); }
    #[test] fn test_invalid_numeric_prefix() { run("invalid_numeric_prefix"); }
    #[test] fn test_invalid_numeric_value() { run("invalid_numeric_value"); }
    #[test] fn test_string_with_newlines() { run("string_with_newlines"); }
    #[test] fn test_unterminated_block_comment() { run("unterminated_block_comment"); }
    #[test] fn test_unterminated_string_literal() { run("unterminated_string_literal"); }


    fn run_str(text: &str, expected: Vec<ScanToken>) {
        let view: SourceView = SourceView::new(String::new(), String::from(text));
        let result: ScanResult = scanner::scan(&view);
        let actual: Vec<ScanToken> = result.tokens
            .into_iter()
            .map(|c| c.1)
            .collect();

        assert!(result.errors.is_empty());
        assert_eq!(expected, actual);
    }

    fn run(path: &'static str) {
        let resource = test_util::get_resource("scanner", path);
        let view = resource.view();
        let result: ScanResult = scanner::scan(&view);

        let mut actual: Vec<String> = Vec::new();
        if !result.tokens.is_empty() {
            actual.push(String::from("=== Scan Tokens ===\n"));
            for token in result.tokens {
                actual.push(format!("{:?}", token.1));
            }
        }
        if !result.errors.is_empty() {
            actual.push(String::from("\n=== Scan Errors ===\n"));
            for error in &result.errors {
                actual.push(view.format(error))
            }
        }

        resource.assert_eq(actual);
    }
}