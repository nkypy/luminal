use crate::runtime::CubeRuntime;
use luminal::prelude::*;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel_err = diff / e.abs().max(1.0);
        assert!(
            rel_err < tolerance,
            "Mismatch at index {}: got {}, expected {}, rel_err={}",
            i,
            a,
            e,
            rel_err
        );
    }
}

#[test]
fn cube_simple_add() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a + b).output();

    cx.build_search_space::<CubeRuntime<cubecl::wgpu::WgpuRuntime>>();
    let mut rt = CubeRuntime::<cubecl::wgpu::WgpuRuntime>::new();
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn cube_simple_mul() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a * b).output();

    cx.build_search_space::<CubeRuntime<cubecl::wgpu::WgpuRuntime>>();
    let mut rt = CubeRuntime::<cubecl::wgpu::WgpuRuntime>::new();
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    // 1*5, 2*6, 3*7, 4*8
    assert_eq!(out, vec![5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn cube_simple_exp() {
    let mut cx = Graph::default();
    let input = cx.tensor(3);
    let output = input.exp().output();

    cx.build_search_space::<CubeRuntime<cubecl::wgpu::WgpuRuntime>>();
    let mut rt = CubeRuntime::<cubecl::wgpu::WgpuRuntime>::new();
    rt.set_data(input, &[0.0, 1.0, -1.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, std::f32::consts::E, (-1.0f32).exp()], 0.001);
}
