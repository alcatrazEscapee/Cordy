

pub fn strip_line_ending(buffer: &mut String) {
    if buffer.ends_with('\n') {
        buffer.pop();
    }
    if buffer.ends_with('\r') {
        buffer.pop();
    }
}