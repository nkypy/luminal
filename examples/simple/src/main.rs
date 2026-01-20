use luminal::prelude::*;

use luminal_cpu::CpuRuntime;
use luminal_metal::MetalRuntime;

fn main() {
    // Create compute graph
    let mut cx = Graph::new();
    let a = cx.tensor((3, 1));
    let b = cx.tensor((1, 4));

    let c = a.matmul(b).output();

    // Compile
    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    // Set input tensors
    rt.set_data(a, &[1.0, 2.0, 3.0]);
    rt.set_data(b, &[1.0, 2.0, 3.0, 3.0]);

    rt = cx.search(rt, 1);

    // Run
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    // Get output tensor
    println!("Result: {:?}", rt.get_f32(c));
}
