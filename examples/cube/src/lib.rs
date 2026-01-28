mod graph;
mod tensor;

pub mod op;

#[cfg(feature = "cpu")]
pub use cubecl::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
pub use cubecl::cuda::CudaRuntime;
#[cfg(feature = "wgpu")]
pub use cubecl::wgpu::WgpuRuntime;

pub mod prelude {
    pub use crate::graph::Graph;
    pub use crate::tensor::GraphTensor;
}
