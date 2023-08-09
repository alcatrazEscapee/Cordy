#![feature(variant_count)]
#![feature(try_trait_v2)]

pub use crate::reporting::{AsError, Location, SourceView};
pub use crate::compiler::ScanTokenType;

pub mod compiler;
pub mod repl;
pub mod util;
pub mod vm;

mod reporting;
mod trace;
mod core;

#[cfg(test)]
mod test_util;

pub const SYS_VERSION: &str = version();

const fn version() -> &'static str {
    match option_env!("CARGO_PKG_VERSION") {
        Some(v) => v,
        None => "unknown"
    }
}