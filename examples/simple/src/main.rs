use luminal::prelude::*;

#[cfg(all(not(feature = "cuda"), not(feature = "metal"), not(feature = "cpu")))]
use luminal::hlir::NativeRuntime;
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "metal")))]
use luminal_cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use luminal_cuda::{cudarc::driver::CudaContext, CudaRuntime};
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use luminal_metal::MetalRuntime;

fn main() {
    // Create compute graph
    let mut cx = Graph::new();
    let a = cx.tensor((3, 1));
    let b = cx.tensor((1, 4));

    let c = a.matmul(b).output();

    #[cfg(feature = "cuda")]
    cx.build_search_space::<CudaRuntime>();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    cx.build_search_space::<MetalRuntime>();
    #[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "metal")))]
    cx.build_search_space::<CpuRuntime>();
    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), not(feature = "cpu")))]
    cx.build_search_space::<NativeRuntime>();

    // Compile
    #[cfg(feature = "cuda")]
    let mut rt = {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        CudaRuntime::initialize(stream)
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let mut rt = MetalRuntime::initialize(());
    #[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "metal")))]
    let mut rt = CpuRuntime::initialize(());
    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), not(feature = "cpu")))]
    let mut rt = NativeRuntime::initialize(());
    // Set input tensors
    rt.set_data(a, vec![1.0, 2.0, 3.0].as_slice());
    rt.set_data(b, vec![1.0, 2.0, 3.0, 3.0].as_slice());

    rt = cx.search(rt, 1);

    // Run
    #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    // Get output tensor
    println!("Result: {:?}", rt.get_f32(c));
}
