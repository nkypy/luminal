use cubecl::{
    prelude::{CubeElement, Float, Runtime},
    server::Handle,
};
use petgraph::graph::NodeIndex;

use crate::{
    graph::Graph,
    op::{CustomOp, EgglogOp, Output},
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
    pub data: Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<R, F, E, C> GraphTensor<R, F, E, C>
where
    R: Runtime,
    F: Float + CubeElement,
    E: EgglogOp,
    C: CustomOp,
{
    // Mark this tensor as an output
    // pub fn output(&self) -> Self {
    //     let graph = unsafe { &mut *self.graph_ref };
    //     graph.graph.add_node(
    //         Output {
    //             node: self.id.index(),
    //         }
    //         .into(),
    //     );
    //     let t = self.clone();
    //     *t
    // }
}
