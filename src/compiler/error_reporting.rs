

/// Locates the presence of an error based on a index.
/// Returns a formatted `String` with the error, source line and file, and exact column hint.
pub fn find_error_in_source(text: &String, index: usize, src: Option<&String>) -> String {
    let mut value: String = String::new();
    let mut current: usize = 0;
    for (lineno, line) in text.split('\n').enumerate() {
        if current + 1 + line.len() > index {
            value.push_str("  at line ");
            value.push_str(lineno.to_string().as_str());
            if let Some(s) = src {
                value.push_str(" (");
                value.push_str(s.as_str());
                value.push(')');
            }
            value.push('\n');
            value.push_str(line);
            value.push('\n');
            if index > current + 1 {
                value.push_str(" ".repeat(index - current - 1).as_str());
            }
            value.push_str("^--- here\n");
            break
        }
        current += 1 + line.len();
    }
    value
}