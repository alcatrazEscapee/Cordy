macro_rules! trace_parser {
    ($($e:expr),+) => {
        {
            #[cfg(feature = "trace_parser")]
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
            #[cfg(feature = "trace_interpreter")]
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
            #[cfg(feature = "trace_interpreter_stack")]
            {
                print!("[stack] ");
                println!($($e),+);
            }
        }
    };
}

pub(crate) use {trace_parser, trace_interpreter, trace_interpreter_stack};