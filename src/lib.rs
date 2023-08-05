#![feature(variant_count)]
#![feature(try_trait_v2)]

use mimalloc::MiMalloc;

pub use crate::reporting::{AsError, SourceView};

pub mod compiler;
pub mod repl;
pub mod vm;

mod core;
mod reporting;
mod trace;
mod util;

#[cfg(test)]
mod test_util;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
