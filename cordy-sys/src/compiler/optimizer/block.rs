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

            // todo: more optimizations!
            // todo: merge + transform blocks and jumps
            // todo: remove pop in store/pop/load
            // todo: jump following
            // todo: remove code after Exit/Return
            merge_store_global_pop(cursor);
            merge_pops(cursor);
        }

        self
    }
}

// ===== Optimization Passes ===== //


/// StoreGlobal, Pop -> StoreGlobalPop
fn merge_store_global_pop(cursor: &mut BlockCursor) {
    cursor.reset();
    while let Some(op) = cursor.next() {
        if let StoreGlobal(index, false) = op {
            if Some(Pop) == cursor.peek() {
                cursor.set(0, StoreGlobal(index, true));
                cursor.remove(1);
            }
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
                cursor.remove(1);
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
    fn back(&mut self) -> Option<Opcode> { self.move_by(-1); self.at(0) }

    fn peek(&mut self) -> Option<Opcode> { self.at(1) }
    fn prev(&mut self) -> Option<Opcode> { self.at(-1) }

    fn set(&mut self, n: isize, op: Opcode) { self.block.set((self.ptr + n) as usize, op) }
    fn remove(&mut self, n: isize) { self.block.remove((self.ptr + n) as usize) }

    fn move_by(&mut self, n: isize) { self.ptr += n; }
    fn at(&mut self, n: isize) -> Option<Opcode> { self.block.get((self.ptr + n) as usize) }
}
