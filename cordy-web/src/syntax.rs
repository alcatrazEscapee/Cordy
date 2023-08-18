use std::iter::Peekable;
use std::str::Chars;

use cordy_sys::{compiler, ScanTokenType, SourceView};


/// Scans a given input string `input`, and applies syntax highlighting to it. Intended to be used with the JQuery 'terminal' formatter.
///
/// This has a few notable requirements:
/// - The input string must be in raw text - so all escaped `[`, `\`, and `]` are replaced with literal characters.
/// - The output string will have formatting codes appended, _and_ will have notable non-formatting `[`, `\`, and `]` escaped.
///
/// ### Parameters:
/// - `input` is the current line of text to be formatted
/// - `prefix` is the prefix of the current line already present in the scanner (i.e. via a continuation)
pub fn scan(input: String, prefix: &String) -> String {
    let mut full_input: String = prefix.clone();
    full_input.push_str(input.as_str());

    let view = SourceView::new(String::new(), full_input);
    let scan = compiler::scan(&view);
    let mut output: EscapedString = EscapedString::new(String::with_capacity(view.text().len()));
    let mut chars = input.chars().peekable();

    // The index of the next character of `chars` to be consumed
    // We start at the length of the `prefix`, so we don't output anything that occurred before the prefix
    let mut index = prefix.len();

    for (loc, ty) in scan {
        if index > loc.end() {
            continue; // In the beginning, skip any tokens that end before the start of the current text
        }

        if matches!(ty, ScanTokenType::Blank) {
            continue; // Also skip any tokens labeled as 'blank' (like newline tokens), since we don't need to highlight them.
        }

        // Consume any leading whitespace and/or comments, leading up to the token
        consume_whitespace_and_comment(&mut chars, &mut output, &mut index, |_, index| *index < loc.start());

        output.push_syntax(Some(ty));

        while index <= loc.end() { // Consume the token itself
            output.push(chars.next().unwrap());
            index += 1;
        }

        output.pop_syntax();
    }

    // Consume any trailing whitespace or comment characters, after all tokens have been parsed
    consume_whitespace_and_comment(&mut chars, &mut output, &mut index, |c, _| c.peek().is_some());

    output.inner
}


fn consume_whitespace_and_comment<F>(chars: &mut Peekable<Chars>, output: &mut EscapedString, index: &mut usize, mut end: F)
    where F : FnMut(&mut Peekable<Chars>, &mut usize) -> bool {
    while let Some(' ' | '\t' | '\r' | '\n') = chars.peek() {
        output.push(chars.next().unwrap());
        *index += 1;
    }

    if end(chars, index) {
        output.push_syntax(Some(ScanTokenType::Comment));
        while end(chars, index) {
            output.push(chars.next().unwrap());
            *index += 1;
        }
        output.pop_syntax();
    }
}


/// Handles escaping control (formatting) characters in the output
struct EscapedString {
    inner: String,
    /// Since formatting in the JQuery terminal cannot encompass newlines, we have to break and re-emit formatting characters every '\n' that gets emitted
    last: Option<ScanTokenType>,
}

impl EscapedString {
    fn new(inner: String) -> EscapedString {
        EscapedString { inner, last: None }
    }

    fn push(&mut self, c: char) {
        match c {
            '\n' => {
                // Formatting needs to close, then re-enter
                let ty = self.pop_syntax();
                self.inner.push('\n');
                self.push_syntax(ty);
            }
            '[' => self.inner.push_str("&#91;"),
            '\\' => self.inner.push_str("&#92;"),
            ']' => self.inner.push_str("&#93;"),
            _ => self.inner.push(c),
        }
    }

    fn push_syntax(&mut self, ty: Option<ScanTokenType>) {
        self.last = ty;
        if let Some(ty) = ty {
            self.inner.push_str(to_color_prefix(ty)); // Don't escape formatting characters
        }
    }

    fn pop_syntax(&mut self) -> Option<ScanTokenType> {
        let ty = self.last.take();
        if let Some(ty) = ty {
            self.inner.push_str(to_color_suffix(ty)); // Don't escape formatting characters
        }
        ty
    }
}


fn to_color_prefix(ty: ScanTokenType) -> &'static str {
    match ty {
        ScanTokenType::Keyword => "[[b;#b5f;]",
        ScanTokenType::Constant => "[[b;#27f;]",
        ScanTokenType::Native => "[[;#b80;]",
        ScanTokenType::Type => "[[;#2aa;]",
        ScanTokenType::Number => "[[;#385;]",
        ScanTokenType::String => "[[;#b10;]",
        ScanTokenType::Comment => "[[;#aaa;]",
        _ => "",
    }
}

fn to_color_suffix(ty: ScanTokenType) -> &'static str {
    match ty {
        ScanTokenType::Syntax |
        ScanTokenType::Blank => "",
        _ => "]"
    }
}


#[cfg(test)]
mod tests {
    use crate::syntax;

    #[test] fn test_empty() { run("", "", "") }
    #[test] fn test_string() { run("'hello'", "", "[[;#b10;]'hello']") }
    #[test] fn test_multiline_string_with_prefix() { run("world'", "'hello", "[[;#b10;]world']") }
    #[test] fn test_multiline_string_with_newline() { run("'hello\nworld'", "", "[[;#b10;]'hello]\n[[;#b10;]world']") }
    #[test] fn test_no_color_for_whitespace() { run("foo bar", "", "foo bar") }
    #[test] fn test_no_color_for_newline() { run("foo\nbar", "", "foo\nbar") }
    #[test] fn test_comment_at_end_of_line() { run("foobar // comment", "", "foobar [[;#aaa;]// comment]") }
    #[test] fn test_multiline_comment_with_prefix() { run("world */ bar", "foo /* hello", "[[;#aaa;]world */ ]bar") }
    #[test] fn test_multiline_comment_with_newline() { run("foo /* hello\nworld */ bar", "", "foo [[;#aaa;]/* hello]\n[[;#aaa;]world */ ]bar") }
    #[test] fn test_escaped_square_brackets() { run("[ 123 ]", "", "&#91; [[;#385;]123] &#93;") }
    #[test] fn test_escaped_backslash() { run("'\\''", "", "[[;#b10;]'&#92;'']") }

    fn run(input: &'static str, prefix: &'static str, expected: &'static str) {
        assert_eq!(syntax::scan(String::from(input), &String::from(prefix)), String::from(expected));
    }
}