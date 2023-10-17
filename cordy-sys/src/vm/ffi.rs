use crate::compiler::FunctionLibrary;
use crate::util::Noop;
use crate::vm::{ValuePtr, ValueResult};


pub trait FunctionInterface {

    /// Handles a foreign function call from the given ID.
    fn handle(&mut self, functions: &FunctionLibrary, handle_id: u32, args: Vec<ValuePtr>) -> ValueResult;
}

impl FunctionInterface for Noop {
    fn handle(&mut self, _ : &FunctionLibrary, _: u32, _: Vec<ValuePtr>) -> ValueResult {
        panic!("Natives are not supported for Noop");
    }
}