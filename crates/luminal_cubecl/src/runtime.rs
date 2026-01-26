use crate::kernel::CubeKernelOp;
use cubecl::{prelude::*, server::Handle};
use itertools::Itertools;
use luminal::{
    graph::LLIRGraph,
    hlir::{Input, Output},
    op::Runtime,
    prelude::{
        FxHashMap, NodeIndex, ToId,
        petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef},
    },
};
use std::time::Duration;

pub type CubeCLRuntime = cubecl::wgpu::WgpuRuntime;

pub struct CubeRuntime {
    client: ComputeClient<CubeCLRuntime>,
    /// Buffers for HLIR input tensors (set by user)
    pub hlir_buffers: FxHashMap<NodeIndex, Handle>,
    /// Buffers for LLIR intermediate/output tensors
    pub buffers: FxHashMap<NodeIndex, Handle>,
    /// The current LLIR graph
    llir_graph: LLIRGraph,
}

impl CubeRuntime {
    pub fn new() -> Self {
        let client = cubecl::Runtime::client(&Default::default());
        Self {
            client,
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            llir_graph: StableGraph::default(),
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: &[f32]) {
        let handle = self.client.create_from_slice(f32::as_bytes(data));
        self.hlir_buffers.insert(id.to_id(), handle);
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let id = id.to_id();
        let output_id = self
            .llir_graph
            .node_indices()
            .find(|n| {
                if let Some(Output { node }) = self.llir_graph[*n].to_op::<Output>() {
                    *node == id.index()
                } else {
                    false
                }
            })
            .expect("Cannot find output tensor!");

        let data_id = self
            .llir_graph
            .neighbors_directed(output_id, Direction::Incoming)
            .next()
            .unwrap();

        let handle = self
            .buffers
            .get(&data_id)
            .or_else(|| {
                // If data_id is an Input node, get from hlir_buffers
                if let Some(Input { node, .. }) = self.llir_graph[data_id].to_op::<Input>() {
                    self.hlir_buffers.get(&NodeIndex::new(*node))
                } else {
                    None
                }
            })
            .expect("Cannot find tensor in runtime!");

        let bytes = self.client.read_one(handle.clone());
        let f32s: &[f32] = bytemuck::cast_slice(&bytes);
        f32s.to_vec()
    }
}

impl Runtime for CubeRuntime {
    type Ops = crate::kernel::CubeOps;
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(_: Self::CompileArg) -> Self {
        Self::new()
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.buffers.clear();
        self.llir_graph = llir_graph.clone();
    }

    #[tracing::instrument(skip_all)]
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
    ) -> (Self::ProfileMetric, String) {
        self.load_llir(llir_graph);
        self.allocate_intermediate_buffers(dyn_map);

        let start = std::time::Instant::now();
        self.execute(dyn_map);
        let elapsed = start.elapsed();

        (elapsed, format!("{:.2?}", elapsed))
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        let llir_to_hlir: FxHashMap<NodeIndex, NodeIndex> = self
            .llir_graph
            .node_indices()
            .filter_map(|n| {
                if let Some(Input { node, .. }) = self.llir_graph[n].to_op::<Input>() {
                    Some((n, NodeIndex::new(*node)))
                } else {
                    None
                }
            })
            .collect();

        let topo_order = toposort(&self.llir_graph, None).expect("Graph has cycles!");

        for node in topo_order {
            if self.llir_graph[node].to_op::<Input>().is_some()
                || self.llir_graph[node].to_op::<Output>().is_some()
            {
                continue;
            }

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn CubeKernelOp>() {
                let input_nodes: Vec<NodeIndex> = self
                    .llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect();

                let input_buffers: Vec<Handle> = input_nodes
                    .iter()
                    .map(|&n| {
                        if let Some(hlir_node) = llir_to_hlir.get(&n) {
                            self.hlir_buffers
                                .get(hlir_node)
                                .expect("Input buffer not set!")
                                .clone()
                        } else {
                            self.buffers
                                .get(&n)
                                .expect("Intermediate buffer not found!")
                                .clone()
                        }
                    })
                    .collect();

                let output_buffer = self
                    .buffers
                    .get(&node)
                    .expect("Output buffer not allocated!")
                    .clone();

                kernel_op.execute(&self.client, &input_buffers, output_buffer, dyn_map);
            }
        }
    }
}

impl CubeRuntime {
    pub fn allocate_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        for node in self.llir_graph.node_indices() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn CubeKernelOp>() {
                let size = kernel_op.output_size().exec(dyn_map).unwrap();
                let buffer = self.client.empty(size * core::mem::size_of::<f32>());
                self.buffers.insert(node, buffer);
            }
        }
    }
}
