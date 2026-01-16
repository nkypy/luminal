use luminal::prelude::*;

// cfg not, any, all
#[cfg(feature = "cuda")]
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};

mod model;

use crate::model::rwkv7::Model;
use crate::model::{Config, State};

fn main() {
    // let tokenizer = rwkv_tokenizer::WorldTokenizer::new(None).unwrap();

    // Create compute graph
    let mut cx = Graph::default();

    let input = cx.named_tensor("input", (1, 1)).as_dtype(DType::Int);

    println!("Initializing model...");
    let cfg = Config {
        hidden_size: 768,
        num_hidden_layers: 12,
        head_size: 64,
    };
    let model = Model::init(&mut cx, &cfg);
    let mut state = State::init(&mut cx, &cfg);
    let logits = model.forward(input, &mut state).output();

    // Build search space
    println!("Building E-Graph...");
    #[cfg(feature = "cuda")]
    cx.build_search_space::<CudaRuntime>();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    cx.build_search_space::<CudaRuntime>();
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    cx.build_search_space::<NativeRuntime>();

    // Load model weights from safetensors file
    #[cfg(feature = "cuda")]
    let mut rt = {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        CudaRuntime::initialize(stream)
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let mut rt = {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        CudaRuntime::initialize(stream)
    };
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    let mut rt = NativeRuntime::default();

    println!("Compiling...");
    rt = cx.search(rt, 5);

    println!("Loading weights...");
    rt.load_safetensors(&cx, "setup/model.safetensors");

    rt.set_data(input, vec![1]);

    // for i in 0..state.per_layer.len() {
    //     rt.set_data(
    //         state.per_layer[i].extract_key_value,
    //         vec![0.0; cfg.hidden_size],
    //     );
    //     rt.set_data(
    //         state.per_layer[i].linear_attention,
    //         vec![0.0; cfg.hidden_size * cfg.head_size * cfg.head_size],
    //     );
    //     rt.set_data(state.per_layer[i].feed_forward, vec![0.0; cfg.hidden_size]);
    // }

    println!("Executing...");
    rt.execute(&cx.dyn_map);
    // Get output tensor
    println!("Result: {:?}", &rt.get_f32(logits)[..50]);
}
