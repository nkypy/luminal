use cubecl::prelude::{CubeElement, Float, Runtime};
use petgraph::graph::NodeIndex;

use crate::{
    graph::Graph,
    op::{CustomOp, EgglogOp},
};

#[derive(Clone, Debug)]
pub struct GraphTensor<R, F, E, C>
where
    R: Runtime,
    F: Float + CubeElement,
    E: EgglogOp,
    C: CustomOp,
{
    pub id: NodeIndex,
    pub graph_ref: *mut Graph<R, F, E, C>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}
