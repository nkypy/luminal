use luminal::prelude::*;

// cfg not, any, all
#[cfg(feature = "cuda")]
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};

#[cfg(all(feature = "metal", not(feature = "cuda")))]
use luminal_metal::runtime::MetalRuntime;

#[cfg(all(feature = "cpu", not(feature = "metal"), not(feature = "cuda")))]
use luminal_cpu::runtime::CpuRuntime;

#[cfg(all(not(feature = "cpu"), not(feature = "metal"), not(feature = "cuda")))]
use luminal::hlir::NativeRuntime;

mod model;

use crate::model::rwkv7::Model;
use crate::model::{Config, State};

fn main() {
    // let tokenizer = rwkv_tokenizer::WorldTokenizer::new(None).unwrap();

    // Create compute graph
    let mut cx = Graph::default();

    // Runtime
    #[cfg(feature = "cuda")]
    let mut rt = {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        CudaRuntime::initialize(stream)
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let mut rt = MetalRuntime::initialize(());
    #[cfg(all(feature = "cpu", not(feature = "metal"), not(feature = "cuda")))]
    let mut rt = CpuRuntime::initialize(());
    #[cfg(all(not(feature = "cpu"), not(feature = "metal"), not(feature = "cuda")))]
    let mut rt = NativeRuntime::initialize(());

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

    let att = model.blocks[0].attention.forward(input, &mut state);
    let ffn = model.blocks[0].feed_forward.forward(input, &mut state);

    // Build search space
    println!("Building E-Graph...");
    #[cfg(feature = "cuda")]
    cx.build_search_space::<CudaRuntime>();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    cx.build_search_space::<MetalRuntime>();
    #[cfg(all(feature = "cpu", not(feature = "metal"), not(feature = "cuda")))]
    cx.build_search_space::<CpuRuntime>();
    #[cfg(all(not(feature = "cpu"), not(feature = "metal"), not(feature = "cuda")))]
    cx.build_search_space::<NativeRuntime>();

    println!("Compiling...");
    rt = cx.search(rt, 5);

    println!("Loading weights...");
    rt.load_safetensors(&cx, "setup/model.safetensors");
    // println!("Loading state...");
    // rt.load_state(&cx, cfg.hidden_size, None);

    #[cfg(any(feature = "metal", feature = "cpu"))]
    rt.set_data(input, vec![1.0].as_ref());
    #[cfg(not(any(feature = "metal", feature = "cpu")))]
    rt.set_data(input, vec![1]);

    println!("Executing...");
    rt.execute(&cx.dyn_map);
    // Get output tensor
    println!("Result: {:?}\n==========", &rt.get_f32(logits)[..20]);
    println!("att Result: {:?}\n==========", &rt.get_f32(att)[..20]);
    println!("ffn Result: {:?}", &rt.get_f32(ffn)[..20]);
}
