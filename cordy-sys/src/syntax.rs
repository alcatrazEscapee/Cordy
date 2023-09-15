use std::iter::Peekable;
use std::str::Chars;

use crate::{compiler, ScanTokenType, SourceView};


/// A trait which define syntax highlighting operations on a given piece of code.
///
/// This supports three primary operations:
/// - `begin()`, which begins a certain section by type (such as a keyword, number, string, etc.)
/// - `end()`, which ends the previous section
/// - `push()`, which pushes a sequence of text, using the current formatting.
///
/// Note that `begin()` and `end()` will only ever be called in sequence, and there are no nested syntax elements. That is, the sequence of
/// `begin`, `push`, and `end` calls will be equivalent to the following grammar: `(push * begin push * end push *) *`
pub trait Formatter {
    fn begin(&mut self, ty: ScanTokenType);
    fn end(&mut self, ty: ScanTokenType);
    fn push(&mut self, c: char);
}

/// A formatter-wrapper which blocks over new lines. So instead of `<syntax>foo\nbar<end>`, we would see `<syntax>foo<end>\n<syntax>bar<end>`.
pub struct BlockingFormatter<F : Formatter> {
    pub fmt: F,
    ty: Option<ScanTokenType>,
}


/// Scans a given input string `input`, and applies Cordy syntax highlighting to it, through the provided formatter.
///
/// ### Parameters:
/// - `input` is the current line of text to be formatted.
///   It must be legal Cordy with no escape characters present.
/// - `prefix` is the prefix of the current line already present in the scanner (i.e. via a continuation)
/// - `fmt` is the formatter that will be used to format the output (of `input`, not including `prefix`)
pub fn scan<Fmt : Formatter>(input: String, prefix: &String, fmt: &mut Fmt) {
    let mut full_input: String = prefix.clone();
    full_input.push_str(input.as_str());

    let view = SourceView::new(String::new(), full_input);
    let scan = compiler::scan(&view);
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
        consume_whitespace_and_comment(&mut chars, fmt, &mut index, |_, index| *index < loc.start());

        fmt.begin(ty);

        while index <= loc.end() { // Consume the token itself
            fmt.push(chars.next().unwrap());
            index += 1;
        }

        fmt.end(ty);
    }

    // Consume any trailing whitespace or comment characters, after all tokens have been parsed
    consume_whitespace_and_comment(&mut chars, fmt, &mut index, |c, _| c.peek().is_some())
}


fn consume_whitespace_and_comment<
    F : FnMut(&mut Peekable<Chars>, &mut usize) -> bool,
    Fmt : Formatter
>(chars: &mut Peekable<Chars>, output: &mut Fmt, index: &mut usize, mut end: F) {
    while let Some(' ' | '\t' | '\r' | '\n') = chars.peek() {
        output.push(chars.next().unwrap());
        *index += 1;
    }

    if end(chars, index) {
        output.begin(ScanTokenType::Comment);
        while end(chars, index) {
            output.push(chars.next().unwrap());
            *index += 1;
        }
        output.end(ScanTokenType::Comment);
    }
}


impl<F : Formatter> BlockingFormatter<F> {
    pub fn new(fmt: F) -> BlockingFormatter<F> {
        BlockingFormatter { fmt, ty: None }
    }

    fn begin(&mut self, ty: Option<ScanTokenType>) {
        self.ty = ty;
        if let Some(ty) = ty {
            self.fmt.begin(ty);
        }
    }

    fn end(&mut self) -> Option<ScanTokenType> {
        let ty = self.ty.take();
        if let Some(ty) = ty {
            self.fmt.end(ty);
        }
        ty
    }
}

impl<F : Formatter> Formatter for BlockingFormatter<F> {
    fn begin(&mut self, ty: ScanTokenType) { self.begin(Some(ty)); }
    fn end(&mut self, _: ScanTokenType) { self.end(); }

    fn push(&mut self, c: char) {
        match c {
            '\n' => {
                // Formatting needs to close, then re-enter
                let ty = self.end();
                self.fmt.push('\n');
                self.begin(ty);
            }
            _ => self.fmt.push(c),
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::{ScanTokenType, syntax, test_util};
    use crate::syntax::{BlockingFormatter, Formatter};

    struct TestFormatter(String);

    impl Formatter for TestFormatter {
        fn begin(&mut self, ty: ScanTokenType) { self.0.push_str(format!("<{:?}>", ty).to_lowercase().as_str()) }
        fn end(&mut self, _: ScanTokenType) { self.0.push_str("<end>"); }
        fn push(&mut self, c: char) { self.0.push(c); }
    }

    #[test] fn test_empty() { run("", "", "", None) }
    #[test] fn test_string() { run("'hello'", "", "<string>'hello'<end>", None) }
    #[test] fn test_multiline_string_with_prefix() { run("world'", "'hello", "<string>world'<end>", None) }
    #[test] fn test_multiline_string_with_newline() { run("'hello\nworld'", "", "<string>'hello\nworld'<end>", Some("<string>'hello<end>\n<string>world'<end>")) }
    #[test] fn test_syntax_over_whitespace() { run("foo bar", "", "<syntax>foo<end> <syntax>bar<end>", None) }
    #[test] fn test_syntax_over_newline() { run("foo\nbar", "", "<syntax>foo<end>\n<syntax>bar<end>", None) }
    #[test] fn test_comment_at_end_of_line() { run("foobar // comment", "", "<syntax>foobar<end> <comment>// comment<end>", None) }
    #[test] fn test_multiline_comment_with_prefix() { run("world */ bar", "foo /* hello", "<comment>world */ <end><syntax>bar<end>", None) }
    #[test] fn test_multiline_comment_with_newline() { run(
        "foo /* hello\nworld */ bar", "",
        "<syntax>foo<end> <comment>/* hello\nworld */ <end><syntax>bar<end>",
        Some("<syntax>foo<end> <comment>/* hello<end>\n<comment>world */ <end><syntax>bar<end>")
    ) }
    #[test] fn test_numbers() { run("0 123 10000000000", "", "<number>0<end> <number>123<end> <number>10000000000<end>", None) }
    #[test] fn test_numbers_with_separator() { run("1_000 1_2_3_4 1____2", "", "<number>1_000<end> <number>1_2_3_4<end> <number>1____2<end>", None) }

    fn run(input: &'static str, prefix: &'static str, expected: &'static str, expected_blocking: Option<&'static str>) {
        let mut fmt = TestFormatter(String::new());
        syntax::scan(String::from(input), &String::from(prefix), &mut fmt);

        test_util::assert_eq(fmt.0, String::from(expected));

        let mut fmt = BlockingFormatter::new(TestFormatter(String::new()));
        syntax::scan(String::from(input), &String::from(prefix), &mut fmt);

        test_util::assert_eq(fmt.fmt.0, String::from(expected_blocking.unwrap_or(expected)));
    }
}