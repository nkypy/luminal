mod ops;
pub use ops::*;

use cubecl::{Runtime as CubeCLRuntime, prelude::ComputeClient, server::Handle};
use luminal::op::EgglogOp;
use luminal::prelude::{Expression, FxHashMap};

pub trait CubeKernelOp: EgglogOp {
    type R: CubeCLRuntime;

    fn output_size(&self) -> Expression;

    fn execute(
        &self,
        client: &ComputeClient<Self::R>,
        inputs: &[Handle],
        output: Handle,
        dyn_map: &FxHashMap<char, usize>,
    );
}

luminal::impl_into_ops!(CubeKernelOp);
