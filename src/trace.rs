

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

pub mod test {
    use std::{env, fs};
    use std::path::PathBuf;

    pub fn get_test_resource_path(source: &'static str, path: &str) -> PathBuf {
        [env::var("CARGO_MANIFEST_DIR").unwrap(), String::from("resources"), String::from(source), format!("{}.cor", path)].iter().collect::<PathBuf>()
    }

    pub fn get_test_resource_src(root: &PathBuf) -> String {
        fs::read_to_string(root).expect(format!("Reading: {:?}", root).as_str())
    }

    pub fn compare_test_resource_content(root: &PathBuf, lines: Vec<String>) {
        let actual: String = lines.join("\n");
        let expected: String = fs::read_to_string(root.with_extension("cor.trace"))
            .expect(format!("Reading: {:?}", root).as_str());

        fs::write(root.with_extension("cor.out"), &actual).unwrap();

        assert_eq!(actual, expected.replace("\r", ""));
    }
}

pub(crate) use trace_parser;
pub(crate) use trace_interpreter;
pub(crate) use trace_interpreter_stack;