use crate::kernel::CpuKernelOp;
use half::f16;
use itertools::Itertools;
use luminal::{
    graph::LLIRGraph,
    hlir::{Input, Output},
    op::Runtime,
    prelude::{
        petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef},
        *,
    },
};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::{fs::File, time::Duration};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub struct CpuRuntime {
    /// Buffers for HLIR input tensors (set by user)
    pub hlir_buffers: FxHashMap<NodeIndex, Vec<f32>>,
    /// Buffers for LLIR intermediate/output tensors
    pub buffers: FxHashMap<NodeIndex, Vec<f32>>,
    /// The current LLIR graph
    llir_graph: LLIRGraph,
}

impl CpuRuntime {
    /// Load tensors from a SafeTensors file
    ///
    /// This will load all tensors that match Input nodes in the graph
    /// by their label and store them in the HLIR buffers.
    #[tracing::instrument(skip_all)]
    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                println!("NEON detected");
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                println!("AVX2 detected");
            }
        }
        let f = File::open(file_path).expect("Failed to open safetensors file");
        let mmap = unsafe { MmapOptions::new().map(&f).expect("Failed to mmap file") };
        let st = SafeTensors::deserialize(&mmap).expect("Failed to deserialize safetensors");

        for node in cx.graph.node_indices() {
            if let Some(Input { label, .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>() {
                if let Ok(tensor) = st.tensor(label) {
                    // Load into hlir_buffers using the HLIR node index
                    match tensor.dtype() {
                        safetensors::Dtype::F32 => {
                            let bytes = tensor.data();
                            let f32s: &[f32] = bytemuck::cast_slice(bytes);
                            self.hlir_buffers.insert(node, f32s.to_vec());
                        }
                        safetensors::Dtype::F16 => {
                            let bytes = tensor.data();
                            let f32s: Vec<f32> = bytes
                                .chunks_exact(2)
                                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                                .collect();
                            self.hlir_buffers.insert(node, f32s);
                        }
                        dtype => {
                            tracing::warn!(
                                "Skipping tensor '{}' with unsupported dtype: {:?}",
                                label,
                                dtype
                            );
                        }
                    }
                }
            }
        }
    }

    /// Set input data for a tensor
    pub fn set_data(&mut self, id: impl ToId, data: &[f32]) {
        self.hlir_buffers.insert(id.to_id(), data.to_vec());
    }

    /// Get output data as f32 vector
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

        self.buffers
            .get(&data_id)
            .or_else(|| {
                // If data_id is an Input node, get from hlir_buffers
                if let Some(Input { node, .. }) = self.llir_graph[data_id].to_op::<Input>() {
                    self.hlir_buffers.get(&NodeIndex::new(*node))
                } else {
                    None
                }
            })
            .expect("Cannot find tensor in runtime!")
            .clone()
    }
}

impl Runtime for CpuRuntime {
    type Ops = crate::kernel::CpuOps;
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(_: Self::CompileArg) -> Self {
        Self {
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            llir_graph: StableGraph::default(),
        }
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

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn CpuKernelOp>() {
                let input_nodes: Vec<NodeIndex> = self
                    .llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect();

                // Collect input buffer data first (before mutable borrow)
                let input_data: Vec<Vec<f32>> = input_nodes
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

                // Convert to slice references
                let input_slices: Vec<&[f32]> = input_data.iter().map(|v| v.as_slice()).collect();

                // Now get mutable output buffer
                let output_buffer = self
                    .buffers
                    .get_mut(&node)
                    .expect("Output buffer not allocated!");

                kernel_op.execute(&input_slices, output_buffer, dyn_map);
            }
        }
    }
}

impl CpuRuntime {
    /// Allocate intermediate buffers for all operations
    pub fn allocate_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        for node in self.llir_graph.node_indices() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn CpuKernelOp>() {
                let size = kernel_op.output_size().exec(dyn_map).unwrap();
                let buffer = vec![0.0f32; size];
                self.buffers.insert(node, buffer);
            }
        }
    }
}
