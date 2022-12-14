
mod scanner;
mod parser;

mod error_reporter;


#[cfg(test)]
mod test_common {
    use std::{env, fs};

    pub fn get_test_resource_path(source: &str, path: &str) -> String {
        let mut root: String = env::var("CARGO_MANIFEST_DIR").unwrap();
        root.push_str("\\resources\\");
        root.push_str(source);
        root.push('\\');
        root.push_str(path);
        root.push_str(".aocl");
        root
    }

    pub fn get_test_resource_src(root: &String) -> String {
        fs::read_to_string(root).unwrap()
    }

    pub fn compare_test_resource_content(root: &String, lines: Vec<String>) {
        let actual: String = lines.join("\n");

        let mut actual_path: String = root.clone();
        actual_path.push_str(".out");

        fs::write(actual_path, &actual).unwrap();

        let mut expected_path: String = root.clone();
        expected_path.push_str(".trace");

        let expected: String = fs::read_to_string(expected_path).unwrap();
        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = actual.lines().collect();

        assert_eq!(expected_lines, actual_lines);
    }
}