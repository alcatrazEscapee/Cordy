

macro_rules! trace_parser {
    ($($e:expr),+) => {
        {
            #[cfg(trace_parser = "on")]
            {
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
                println!($($e),+)
            }
        }
    };
}

macro_rules! trace_interpreter_stack {
    ($($e:expr),+) => {
        {
            #[cfg(trace_interpreter_stack = "on")]
            {
                println!($($e),+)
            }
        }
    };
}

pub(crate) use trace_parser;
pub(crate) use trace_interpreter;
pub(crate) use trace_interpreter_stack;