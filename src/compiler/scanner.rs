use std::fmt::Debug;
use std::iter::Peekable;
use std::num::ParseIntError;
use std::str::Chars;

use self::ScanErrorType::{*};
use self::ScanTokenType::{*};


pub fn scan(text: &String) -> ScanResult {
    let mut scanner: Scanner = Scanner {
        chars: text.chars().peekable(),
        index: 0,

        tokens: Vec::new(),
        errors: Vec::new(),
    };
    scanner.scan();
    ScanResult {
        tokens: scanner.tokens,
        errors: scanner.errors
    }
}


pub struct ScanResult {
    tokens: Vec<ScanToken>,
    errors: Vec<ScanError>
}

struct Scanner<'a> {
    chars: Peekable<Chars<'a>>,
    index: usize,

    tokens: Vec<ScanToken>,
    errors: Vec<ScanError>,
}

#[derive(Debug)]
pub struct ScanError {
    error: ScanErrorType,
    index: usize,
}

#[derive(Debug, Clone)]
pub enum ScanErrorType {
    InvalidNumericPrefix(char),
    InvalidNumericValue(ParseIntError),
    InvalidCharacter(char),
    UnterminatedStringLiteral,
}

impl ScanError {
    fn format_error(self: &Self) -> String {
        match &self.error {
            InvalidNumericPrefix(c) => format!("Invalid numeric prefix: '0{}'", c),
            InvalidNumericValue(e) => format!("Invalid numeric value: {}", e),
            InvalidCharacter(c) => format!("Invalid character: '{}'", c),
            UnterminatedStringLiteral => String::from("Unterminated string literal (missing a closing single quote)")
        }
    }
}

#[derive(Debug)]
pub struct ScanToken {
    token: ScanTokenType,
    index: usize,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Debug, Clone)]
pub enum ScanTokenType {
    // Special
    Identifier(String),
    StringLiteral(String),
    Int(i64),

    // Keywords
    KeywordLet,
    KeywordConst,
    KeywordIf,
    KeywordElif,
    KeywordElse,
    KeywordLoop,
    KeywordFor,
    KeywordIn,
    KeywordIs,

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

    LogicalNot,
    BitwiseNot,

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
}


impl<'a> Scanner<'a> {
    
    fn scan(self: &mut Self) {
        loop {
           match self.advance() {
               Some(c) => {
                   match c {
                       ' ' | '\t' | '\r' | '\n' => {},

                       'a'..='z' | 'A'..='Z' => {
                           let mut buffer: Vec<char> = Vec::new();
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
                                    let mut buffer: Vec<char> = Vec::new();
                                    loop {
                                        match self.peek() {
                                            Some('0'..='9' | 'A'..='F' | 'a'..='f') => self.push_advance(&mut buffer),
                                            Some('_') => self.skip(),
                                            _ => break
                                        };
                                    }
                                    self.screen_int(buffer, 16);
                                },
                                Some('b') => {
                                    self.advance();
                                    let mut buffer: Vec<char> = Vec::new();
                                    loop {
                                        match self.peek() {
                                            Some('1' | '0') => self.push_advance(&mut buffer),
                                            Some('_') => self.skip(),
                                            _ => break
                                        }
                                    }
                                    self.screen_int(buffer, 2);
                                },
                                Some(e) => self.push_err(InvalidNumericPrefix(e)),
                                _ => {}
                            }
                       }
                       '1'..='9' => {
                           let mut buffer: Vec<char> = Vec::new();
                           buffer.push(c);
                           loop {
                               match self.peek() {
                                   Some('0'..='9') => self.push_advance(&mut buffer),
                                   Some('_') => self.skip(),
                                   _ => break
                               }
                           }
                           self.screen_int(buffer, 10);
                       },

                       '\'' => {
                           let mut buffer: Vec<char> = Vec::new();
                           let mut escaped: bool = false;
                           loop {
                               match self.advance() {
                                   Some('\'') if !escaped => break,
                                   Some('\\') if !escaped => {
                                       escaped = true;
                                   }
                                   Some(c0) => buffer.push(c0),
                                   None => {
                                       self.push_err(UnterminatedStringLiteral);
                                       break
                                   }
                               }
                           }
                           self.push(StringLiteral(buffer.iter().collect()))
                       },

                       '!' => match self.peek() {
                           Some('=') => self.push_skip(NotEquals),
                           _ => self.push(LogicalNot)
                       },
                       '~' => self.push(BitwiseNot),

                       '=' => match self.peek() {
                           Some('=') => self.push_skip(DoubleEquals),
                           _ => self.push(Equals)
                       },
                       '>' => match self.peek() {
                           Some('>') => match self.advance_peek() {
                               Some('=') => self.push_skip(RightShiftEquals),
                               _ => self.push(RightShift)
                           }
                           Some('=') => self.push_skip(GreaterThanEquals),
                           _ => self.push(GreaterThan)
                       },
                       '<' => match self.peek() {
                           Some('<') => match self.advance_peek() {
                               Some('=') => self.push_skip(LeftShiftEquals),
                               _ => self.push(LeftShift)
                           },
                           Some('=') => self.push_skip(LessThanEquals),
                           _ => self.push(LessThan)
                       },

                       '+' => match self.peek() {
                           Some('=') => self.push_skip(PlusEquals),
                           _ => self.push(Plus)
                       },
                       '-' => match self.peek() {
                           Some('=') => self.push_skip(MinusEquals),
                           Some('>') => self.push_skip(Arrow),
                           _ => self.push(Minus)
                       },
                       '*' => match self.peek() {
                           Some('=') => self.push_skip(MulEquals),
                           Some('*') => match self.advance_peek() {
                               Some('=') => self.push_skip(PowEquals),
                               _ => self.push(Pow)
                           },
                           _ => self.push(Mul)
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
                               loop {
                                   match self.advance() {
                                       Some('*') => {
                                           match self.advance() {
                                               Some('/') => break,
                                               Some(_) => {},
                                               None => break
                                           }
                                       },
                                       Some(_) => {},
                                       None => break
                                   }
                               }
                           }
                           Some('=') => self.push_skip(DivEquals),
                           _ => self.push(Div)
                       },
                       '|' => match self.peek() {
                           Some('=') => self.push_skip(OrEquals),
                           Some('|') => self.push_skip(LogicalOr),
                           _ => self.push(BitwiseOr)
                       },
                       '&' => match self.peek() {
                           Some('=') => self.push_skip(AndEquals),
                           Some('&') => self.push_skip(LogicalAnd),
                           _ => self.push(BitwiseAnd)
                       },
                       '^' => match self.peek() {
                           Some('=') => self.push_skip(XorEquals),
                           _ => self.push(BitwiseXor)
                       },
                       '%' => match self.peek() {
                           Some('=') => self.push_skip(ModEquals),
                           _ => self.push(Mod)
                       }


                       '(' => self.push(OpenParen),
                       ')' => self.push(CloseParen),
                       '[' => self.push(OpenSquareBracket),
                       ']' => self.push(CloseSquareBracket),
                       '{' => self.push(OpenBrace),
                       '}' => self.push(CloseBrace),

                       ',' => self.push(Comma),
                       '.' => self.push(Dot),
                       ':' => self.push(Colon),

                       e => self.push_err(InvalidCharacter(e))
                   }
               }
               None => break // eof
           }
        }
    }

    fn screen_identifier(self: &mut Self, buffer: Vec<char>) {
        let string: String = buffer.iter().collect();
        let token: ScanTokenType = match string.as_str() {
            "let" => KeywordLet,
            "const" => KeywordConst,
            "if" => KeywordIf,
            "elif" => KeywordElif,
            "else" => KeywordElse,
            "loop" => KeywordLoop,
            "for" => KeywordFor,
            "in" => KeywordIn,
            "is" => KeywordIs,
            _ => Identifier(string)
        };
        self.push(token);
    }

    fn screen_int(self: &mut Self, buffer: Vec<char>, radix: u32) {
        let string: String = buffer.iter().collect();
        match i64::from_str_radix(string.as_str(), radix) {
            Ok(value) => self.push(Int(value)),
            Err(e) => self.push_err(InvalidNumericValue(e))
        }
    }


    fn push(self: &mut Self, token: ScanTokenType) {
        self.tokens.push(ScanToken {
            token,
            index: self.index
        });
    }

    fn push_skip(self: &mut Self, token: ScanTokenType) {
        self.push(token);
        self.skip();
    }


    fn push_err(self: &mut Self, error: ScanErrorType) {
        self.errors.push(ScanError {
            error,
            index: self.index,
        })
    }


    /// Consumes the next character (unconditionally) and adds it to the buffer
    /// **Note**: This function must only be invoked after `Some()` has been matched to a `peek()` variant.
    fn push_advance(self: &mut Self, buffer: &mut Vec<char>) {
        buffer.push(self.advance().unwrap());
    }

    /// Consumes the next character without returning it.
    /// Also see `advance()`
    fn skip(self: &mut Self) {
        self.advance();
    }

    /// Consumes the next character, and peeks one character ahead
    /// Chains together `advance()` and `peek()`
    fn advance_peek(self: &mut Self) -> Option<char> {
        self.advance();
        self.peek()
    }

    /// Consumes the next character and returns it
    /// Also see `advance()`
    fn advance(self: &mut Self) -> Option<char> {
        let c: Option<char> = self.chars.next();
        if c.is_some() { self.index += 1; }
        c
    }

    /// Inspects the next character and returns it, without consuming it
    fn peek(self: &mut Self) -> Option<char> {
        self.chars.peek().map(|c| *c)
    }
}


#[cfg(test)]
mod tests {
    use std::{env, fs};
    use crate::compiler::{error_reporting, scanner};
    use crate::compiler::scanner::{ScanResult, ScanTokenType};
    use crate::compiler::scanner::ScanTokenType::{*};

    #[test] fn test_str_empty() { run_str("", vec![]); }

    #[test] fn test_str_keywords() { run_str("let const if elif else loop for in is", vec![KeywordLet, KeywordConst, KeywordIf, KeywordElif, KeywordElse, KeywordLoop, KeywordFor, KeywordIn, KeywordIs]); }
    #[test] fn test_str_identifiers() { run_str("foobar big_bad_wolf ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz", vec![Identifier(String::from("foobar")), Identifier(String::from("big_bad_wolf")), Identifier(String::from("ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"))]); }

    #[test] fn test_str_ints() { run_str("1234 654 10_00_00", vec![Int(1234), Int(654), Int(100000)]); }
    #[test] fn test_str_binary_ints() { run_str("0b11011011 0b0 0b1 0b1_01", vec![Int(0b11011011), Int(0b0), Int(0b1), Int(0b101)]); }
    #[test] fn test_str_hex_ints() { run_str("0x12345678 0xabcdef90 0xABCDEF 0xF_f", vec![Int(0x12345678), Int(0xabcdef90), Int(0xABCDEF), Int(0xFF)])}

    #[test] fn test_str_unary_operators() { run_str("! ~", vec![LogicalNot, BitwiseNot]); }
    #[test] fn test_str_comparison_operators() { run_str("> < >= > = <= < =", vec![GreaterThan, LessThan, GreaterThanEquals, GreaterThan, Equals, LessThanEquals, LessThan, Equals]); }
    #[test] fn test_str_equality_operators() { run_str("!= ! = == =", vec![NotEquals, LogicalNot, Equals, DoubleEquals, Equals]); }
    #[test] fn test_str_binary_logical_operators() { run_str("&& & || |", vec![LogicalAnd, BitwiseAnd, LogicalOr, BitwiseOr]); }
    #[test] fn test_str_arithmetic_operators() { run_str("+ - += -= * = *= / = /=", vec![Plus, Minus, PlusEquals, MinusEquals, Mul, Equals, MulEquals, Div, Equals, DivEquals]); }
    #[test] fn test_str_other_arithmetic_operators() { run_str("% %= ** *= **= * *=", vec![Mod, ModEquals, Pow, MulEquals, PowEquals, Mul, MulEquals]); }
    #[test] fn test_str_bitwise_operators() { run_str("| ^ ~ & &= |= ^=", vec![BitwiseOr, BitwiseXor, BitwiseNot, BitwiseAnd, AndEquals, OrEquals, XorEquals]); }
    #[test] fn test_str_groupings() { run_str("( [ { } ] )", vec![OpenParen, OpenSquareBracket, OpenBrace, CloseBrace, CloseSquareBracket, CloseParen]); }
    #[test] fn test_str_syntax() { run_str(". , -> - > :", vec![Dot, Comma, Arrow, Minus, GreaterThan, Colon]); }

    fn run_str(text: &str, tokens: Vec<ScanTokenType>) {
        let result: ScanResult = scanner::scan(&String::from(text));
        assert!(result.errors.is_empty());
        assert_eq!(result.tokens
            .into_iter()
            .map(|t| t.token)
            .collect::<Vec<ScanTokenType>>(), tokens);
    }


    #[test] fn test_empty() { run("empty"); }
    #[test] fn test_hello_world() { run("hello_world"); }
    #[test] fn test_invalid_character() { run("invalid_character"); }
    #[test] fn test_invalid_numeric_value() { run("invalid_numeric_value"); }
    #[test] fn test_noop() { run("noop"); }
    #[test] fn test_unterminated_string_literal() { run("unterminated_string_literal"); }


    fn run(path: &'static str) {
        let mut root: String = env::var("CARGO_MANIFEST_DIR").unwrap();
        root.push_str("\\resources\\");
        root.push_str(path);
        root.push_str(".aocl");

        let text: String = fs::read_to_string(&root).unwrap();
        let result: ScanResult = scanner::scan(&text);

        let mut actual_path: String = root.clone();
        actual_path.push_str(".out");

        let mut lines: Vec<String> = Vec::new();
        if !result.tokens.is_empty() {
            lines.push(String::from("=== Scan Tokens ===\n"));
            for token in result.tokens {
                lines.push(format!("{:?}", token));
            }
        }
        if !result.errors.is_empty() {
            lines.push(String::from("\n=== Scan Errors ===\n"));
            for error in &result.errors {
                lines.push(format!("{:?}", error));
            }
            lines.push(String::from("\n=== Formatted Scan Errors ===\n"));
            let mut source: String = String::from(path);
            source.push_str(".aocl");
            for error in &result.errors {
                lines.push(error.format_error());
                lines.push(error_reporting::find_error_in_source(&text, error.index, Some(&source)));
            }
        }

        let actual: String = lines.join("\n");

        fs::write(actual_path, &actual).unwrap();

        let mut expected_path: String = root.clone();
        expected_path.push_str(".scan");

        let expected: String = fs::read_to_string(expected_path).unwrap();

        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = actual.lines().collect();
        assert_eq!(expected_lines, actual_lines);
    }
}