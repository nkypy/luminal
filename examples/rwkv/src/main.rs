use luminal::prelude::*;

#[cfg(feature = "cuda")]
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};

mod model;

use model::rwkv7::Model;

fn main() {
    // let tokenizer = rwkv_tokenizer::WorldTokenizer::new(None).unwrap();

    // Create compute graph
    let mut cx = Graph::default();

    let input = cx.named_tensor("input", (1, 1)).as_dtype(DType::Int);
    let model = Model::init(&mut cx);
    println!("Init model...");
    let logits = model.forward(input).output();

    // Build search space
    println!("Building E-Graph...");
    #[cfg(feature = "cuda")]
    cx.build_search_space::<CudaRuntime>();
    #[cfg(not(feature = "cuda"))]
    cx.build_search_space::<NativeRuntime>();

    // Load model weights from safetensors file
    #[cfg(feature = "cuda")]
    let mut rt = {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        CudaRuntime::initialize(stream)
    };
    #[cfg(not(feature = "cuda"))]
    let mut rt = NativeRuntime::default();

    println!("Compiling...");
    rt = cx.search(rt, 1);

    println!("Loading weights...");
    rt.load_safetensors(&cx, "setup/model_combined.safetensors");

    rt.set_data(input, vec![1]);

    for i in rt.buffers.keys() {
        println!("{:?}", i);
    }

    println!("Executing...");
    rt.execute(&cx.dyn_map);
    // Get output tensor
    println!("Result: {:?}", rt.get_f32(logits));
}
