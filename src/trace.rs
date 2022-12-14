

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

pub(crate) use trace_parser;