use std::marker::PhantomData;

use cubecl::{
    client::ComputeClient,
    prelude::{CubeElement, Float, Runtime},
};
use petgraph::{graph::NodeIndex, stable_graph::StableGraph};
use rustc_hash::FxHashMap;

use crate::{
    op::{CustomOp, EgglogOp, HLIROp, LLIROp},
    tensor::GraphTensor,
};

#[derive(Clone)]
pub struct Graph<R, F, H, L, E, C>
where
    R: Runtime,
    F: Float + CubeElement,
    H: HLIROp,
    L: LLIROp,
    E: EgglogOp,
    C: CustomOp,
{
    pub client: ComputeClient<R>,
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: StableGraph<H, Vec<usize>>,
    pub llir_graph: StableGraph<L, ()>,
    // /// E-Graph search space
    // egraph: Option<SerializedEGraph>,
    /// Available ops
    pub ops: Vec<E>,
    /// Custom ops
    pub custom_ops: Vec<C>,
    _r: PhantomData<R>,
    _f: PhantomData<F>,
}

impl<R, F, H, L, E, C> Graph<R, F, H, L, E, C>
where
    R: Runtime,
    F: Float + CubeElement,
    H: HLIROp,
    L: LLIROp,
    E: EgglogOp,
    C: CustomOp,
{
    /// Create a new graph
    pub fn new(device: &R::Device) -> Self {
        Self {
            client: R::client(device),
            dyn_map: FxHashMap::default(),
            graph: StableGraph::new(),
            llir_graph: StableGraph::new(),
            ops: Vec::new(),
            custom_ops: Vec::new(),
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn tensor(
        &mut self,
        op: impl Into<H>,
        data: Vec<F>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> GraphTensor<R, F, H, L, E, C> {
        let id = self.graph.add_node(op.into());
        let data = self.client.create_from_slice(F::as_bytes(&data));
        GraphTensor {
            id,
            graph_ref: self,
            data,
            shape,
            strides,
        }
    }

    pub fn add_op(&mut self, op: impl Into<H>, _id: NodeIndex, _shape: Vec<usize>) {
        self.graph.add_node(op.into());
    }
}
