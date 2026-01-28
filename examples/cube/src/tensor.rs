use cubecl::{
    prelude::{CubeElement, Float, Runtime},
    server::Handle,
};
use petgraph::graph::NodeIndex;

use crate::{
    graph::Graph,
    op::{CustomOp, EgglogOp, HLIROp, LLIROp},
};

#[derive(Clone, Debug)]
pub struct GraphTensor<R, F, H, L, E, C>
where
    R: Runtime,
    F: Float + CubeElement,
    H: HLIROp,
    L: LLIROp,
    E: EgglogOp,
    C: CustomOp,
{
    pub id: NodeIndex,
    pub graph_ref: *mut Graph<R, F, H, L, E, C>,
    pub data: Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}
