#![feature(variant_count)]
#![feature(try_trait_v2)]

pub use crate::reporting::{AsError, SourceView};

pub mod compiler;
pub mod repl;
pub mod vm;

mod reporting;
mod trace;
mod core;
mod util;

#[cfg(test)]
mod test_util;