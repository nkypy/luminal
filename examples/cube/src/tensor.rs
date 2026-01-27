use cubecl::prelude::{CubeElement, Float, Runtime};
use petgraph::graph::NodeIndex;

use crate::graph::Graph;

#[derive(Clone)]
pub struct GraphTensor<R: Runtime, F: Float + CubeElement> {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph<R, F>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}
