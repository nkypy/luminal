use luminal::prelude::*;

mod model;

use model::rwkv7::Model;

fn main() {
    // Create compute graph
    let mut cx = Graph::default();

    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let model = Model::init(&mut cx);
    println!("Model init...");
    let logits = model.forward(input).output();

    // Compile
    cx.build_search_space::<NativeRuntime>();

    let mut rt = NativeRuntime::default();

    rt.load_safetensors(
        &cx,
        "/Users/tajan/codes/RWKV7_Pytorh/rwkv7-g1a-0.1b-20250728-ctx4096.safetensors",
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
