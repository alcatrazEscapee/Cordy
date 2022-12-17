

macro_rules! trace_parser {
    ($($e:expr),+) => {
        {
            #[cfg(trace_parser = "on")]
            {
                print!("[parser] ");
                println!($($e),+)
            }
        }
    };
}

macro_rules! trace_interpreter {
    ($($e:expr),+) => {
        {
            #[cfg(trace_interpreter = "on")]
            {
                print!("[vm] ");
                println!($($e),+);
            }
        }
    };
}

macro_rules! trace_interpreter_stack {
    ($($e:expr),+) => {
        {
            #[cfg(trace_interpreter_stack = "on")]
            {
                print!("[stack] ");
                println!($($e),+);
            }
        }
    };
}

#[cfg(test)]
pub mod test {
    use std::{env, fs};

    pub fn get_test_resource_path(source: &str, path: &str) -> String {
        let mut root: String = env::var("CARGO_MANIFEST_DIR").unwrap();
        root.push_str("\\resources\\");
        root.push_str(source);
        root.push('\\');
        root.push_str(path);
        root.push_str(".cor");
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

        assert_eq!(actual, expected.replace("\r", ""));
    }
}

pub(crate) use trace_parser;
pub(crate) use trace_interpreter;
pub(crate) use trace_interpreter_stack;