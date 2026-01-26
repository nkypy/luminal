mod ops;
pub use ops::*;

use cubecl::{prelude::ComputeClient, server::Handle};
use luminal::op::EgglogOp;
use luminal::prelude::{Expression, FxHashMap};

pub trait CubeKernelOp: EgglogOp {
    fn output_size(&self) -> Expression;

    fn execute(
        &self,
        client: &ComputeClient<super::runtime::CubeCLRuntime>,
        inputs: &[Handle],
        output: Handle,
        dyn_map: &FxHashMap<char, usize>,
    );
}

luminal::impl_into_ops!(CubeKernelOp);
