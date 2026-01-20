mod ops;
pub use ops::*;

use luminal::op::EgglogOp;
use luminal::prelude::*;

/// Trait for CPU kernel operations with SIMD optimization.
///
/// This trait defines the interface for CPU-based tensor operations
/// that can be executed with SIMD optimizations for performance.
pub trait CpuKernelOp: EgglogOp {
    /// Returns the output size of this operation
    fn output_size(&self) -> Expression;

    /// Execute the operation on CPU with SIMD optimizations
    ///
    /// # Arguments
    /// * `inputs` - Slice of input buffer references
    /// * `output` - Mutable output buffer
    /// * `dyn_map` - Dynamic dimension mappings
    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>);
}

luminal::impl_into_ops!(CpuKernelOp);
