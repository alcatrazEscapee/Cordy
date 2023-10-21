use crate::compiler::optimizer::Optimize;
use crate::compiler::parser::{Block, Code};
use crate::vm::Opcode;

use Opcode::{*};


impl<'a> Optimize for &'a mut Code {

    /// Optimizes the code control flow graph using a number of techniques. It currently implements:
    ///
    /// - `Pop` merging into `PopN`
    /// - `StoreGlobal, Pop` merge into `StoreGlobalPop`
    ///
    fn optimize(self) -> Self {
        for block in &mut self.blocks {
            let cursor = &mut BlockCursor::new(block);

            // todo: more blocks optimizations
            // Ideas:
            // - Try to simplify / merge blocks and jumps
            // - Constant / Pure push + Pop
            // - Jump Inlining
            // - Remove code after Exit/Return branches
            // - Inline non-virtual (aka method references that only have one implementation) methods
            //     1. We need to prove the module/struct is const, so we're tracking assignment
            //     2. Look for loads of a given known const field (so PushGlobal()) followed by GetField() of a known non-virtual field (aka method)
            // For cases like this:
            // module Foo { fn bar() {} }
            // Foo->bar // this is normally a `PushGlobal, GetField` but could be a `Constant`
            //
            // Open Issues:
            // - Removing any `Store` / `Push` opcodes breaks the continuity of local references visible in the disassembly.
            merge_store_pop_load(cursor);
            merge_store_global_pop(cursor);
            merge_pops(cursor);
        }

        self
    }
}

// ===== Optimization Passes ===== //


// Store, Pop, Load
fn merge_store_pop_load(cursor: &mut BlockCursor) {
    cursor.reset();
    while let Some(op) = cursor.next() {
        match op {
            StoreGlobal(store, false) if Some(Pop) == cursor.at(1) && Some(PushGlobal(store)) == cursor.at(2) => cursor.remove(1, 2),
            StoreLocal(store, false) if Some(Pop) == cursor.at(1) && Some(PushLocal(store)) == cursor.at(2) => cursor.remove(1, 2),
            StoreUpValue(store) if Some(Pop) == cursor.at(1) && Some(PushUpValue(store)) == cursor.at(2) => cursor.remove(1, 2),
            _ => {}
        }
    }
}

/// StoreGlobal, Pop -> StoreGlobalPop
fn merge_store_global_pop(cursor: &mut BlockCursor) {
    cursor.reset();
    while let Some(op) = cursor.next() {
        match op {
            StoreGlobal(index, false) if Some(Pop) == cursor.peek() => {
                cursor.set(0, StoreGlobal(index, true));
                cursor.remove(1, 1);
            },
            StoreLocal(index, false) if Some(Pop) == cursor.at(1) => {
                cursor.set(0, StoreLocal(index, true));
                cursor.remove(1, 1);
            },
            _ => {},
        }
    }
}

/// Pop, ... Pop -> PopN
fn merge_pops(cursor: &mut BlockCursor) {
    cursor.reset();
    while let Some(op) = cursor.next() {
        if op == Pop {
            let mut count = 1;
            while Some(Pop) == cursor.peek() {
                cursor.remove(1, 1);
                count += 1;
            }
            if count > 1 {
                cursor.set(0, PopN(count))
            }
        }
    }
}



/// An optimizer-visible API for basic block optimizations.
struct BlockCursor<'a> {
    block: &'a mut Block,
    ptr: isize,
}

impl<'a> BlockCursor<'a> {
    fn new(block: &'a mut Block) -> BlockCursor<'a> {
        BlockCursor { block, ptr: -1 }
    }

    fn reset(&mut self) { self.ptr = -1; }

    fn next(&mut self) -> Option<Opcode> { self.move_by(1); self.at(0) }
    //fn back(&mut self) -> Option<Opcode> { self.move_by(-1); self.at(0) }

    fn peek(&mut self) -> Option<Opcode> { self.at(1) }
    //fn prev(&mut self) -> Option<Opcode> { self.at(-1) }

    fn set(&mut self, offset: isize, op: Opcode) { self.block.set((self.ptr + offset) as usize, op) }
    fn remove(&mut self, offset: isize, repeat: usize) {
        for _ in 0..repeat {
            self.block.remove((self.ptr + offset) as usize)
        }
    }

    fn move_by(&mut self, n: isize) { self.ptr += n; }
    fn at(&mut self, n: isize) -> Option<Opcode> { self.block.get((self.ptr + n) as usize) }
}
