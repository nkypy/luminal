use luminal::prelude::*;

#[cfg(feature = "cuda")]
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};

mod model;

use model::rwkv7::Model;

fn main() {
    // Set up tracing to perfetto
    let _trace_session = luminal_tracing::subscriber()
        .perfetto("trace.pftrace")
        .env_filter(format!(
            "{}=trace,luminal=trace,luminal_cuda=trace",
            env!("CARGO_PKG_NAME")
        ))
        .init();

    let tokenizer = rwkv_tokenizer::WorldTokenizer::new(None).unwrap();

    // Create compute graph
    let mut cx = Graph::default();

    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
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
    println!("Loading weights...");
    #[cfg(feature = "cuda")]
    let mut rt = {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        CudaRuntime::initialize(stream)
    };
    #[cfg(not(feature = "cuda"))]
    let mut rt = NativeRuntime::default();
    rt.load_safetensors(
        &cx,
        "setup/model_combined.safetensors",
    );

    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 0);
    rt.set_data(input, vec![1]);
    rt = cx.search(rt, 1);

    println!("Executing...");
    rt.execute(&cx.dyn_map);
    // Get output tensor
    println!("Result: {:?}", rt.get_f32(logits));
}
