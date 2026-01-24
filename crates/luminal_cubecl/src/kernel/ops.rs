use super::CubeKernelOp;
use cubecl::{
    Runtime as CubeCLRuntime, cube,
    prelude::{ABSOLUTE_POS, Array, ComputeClient, CubeCount, CubeDim},
    server::Handle,
};
use luminal::{egglog_utils::SerializedEGraph, op::OpParam::*, op::*, prelude::*};

pub type CubeOps = (CubeExp, CubeLog, CubeAdd, CubeMul, CubeSumReduce);

#[cube]
fn strided_index(idx: u32, shape: &Array<u32>, strides: &Array<u32>, ndim: u32) -> u32 {
    let mut offset = 0;
    let mut remaining = idx;

    for i in 0..ndim {
        // Reverse iteration: ndim-1 down to 0
        let dim_idx = ndim - 1 - i;
        let dim_size = shape[dim_idx];
        let stride = strides[dim_idx];
        let coord = remaining % dim_size;
        offset += coord * stride;
        remaining /= dim_size;
    }
    offset
}

#[cube(launch)]
fn unary_kernel_exp(
    input: &Array<f32>,
    output: &mut Array<f32>,
    shape: &Array<u32>,
    inp_strides: &Array<u32>,
    out_strides: &Array<u32>,
    ndim: u32,
) {
    if ABSOLUTE_POS < output.len() {
        let idx = ABSOLUTE_POS;
        let in_idx = strided_index(idx, shape, inp_strides, ndim);
        let out_idx = strided_index(idx, shape, out_strides, ndim);
        output[out_idx] = cubecl::prelude::exp(input[in_idx]);
    }
}

#[cube(launch)]
fn binary_kernel_add(
    lhs: &Array<f32>,
    rhs: &Array<f32>,
    output: &mut Array<f32>,
    shape: &Array<u32>,
    lhs_strides: &Array<u32>,
    rhs_strides: &Array<u32>,
    out_strides: &Array<u32>,
    ndim: u32,
) {
    if ABSOLUTE_POS < output.len() {
        let idx = ABSOLUTE_POS;
        let lhs_idx = strided_index(idx, shape, lhs_strides, ndim);
        let rhs_idx = strided_index(idx, shape, rhs_strides, ndim);
        let out_idx = strided_index(idx, shape, out_strides, ndim);
        output[out_idx] = lhs[lhs_idx] + rhs[rhs_idx];
    }
}

#[cube(launch)]
fn binary_kernel_mul(
    lhs: &Array<f32>,
    rhs: &Array<f32>,
    output: &mut Array<f32>,
    shape: &Array<u32>,
    lhs_strides: &Array<u32>,
    rhs_strides: &Array<u32>,
    out_strides: &Array<u32>,
    ndim: u32,
) {
    if ABSOLUTE_POS < output.len() {
        let idx = ABSOLUTE_POS;
        let lhs_idx = strided_index(idx, shape, lhs_strides, ndim);
        let rhs_idx = strided_index(idx, shape, rhs_strides, ndim);
        let out_idx = strided_index(idx, shape, out_strides, ndim);
        output[out_idx] = lhs[lhs_idx] * rhs[rhs_idx];
    }
}

// Helpers
fn exprs_to_u32_vec(exprs: &[Expression], dyn_map: &FxHashMap<char, usize>) -> Vec<u32> {
    exprs
        .iter()
        .map(|e| e.exec(dyn_map).unwrap() as u32)
        .collect()
}

macro_rules! impl_unary {
    ($name:ident, $op_name:expr, $kernel:ident, $rewrite_op:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            input_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn term(&self) -> (String, Vec<OpParam>) {
                ($op_name.to_string(), vec![EList, Input, EList, EList])
            }
            fn rewrites(&self) -> Vec<String> {
                vec![format!(
                    r#"(rule
                        ((= ?e ({} ?shape ?x ?x_stride ?out_stride))
                         (= ?dt (dtype ?x)))
                        ((let ?me ({} ?shape ?x ?x_stride ?out_stride))
                         (union ?e ?me)
                         (set (dtype ?me) ?dt))
                    )"#,
                    $rewrite_op, $op_name
                )]
            }
            fn cleanup(&self) -> bool {
                false
            }
            fn extract<'a>(
                &'a self,
                egraph: &'a SerializedEGraph,
                children: &[&'a ENodeId],
                list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
            ) -> (LLIROp, Vec<&'a ENodeId>) {
                use luminal::graph::extract_expr_list;
                (
                    LLIROp::new::<dyn CubeKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                            .unwrap(),
                        input_strides: extract_expr_list(
                            egraph,
                            children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            children[3],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    vec![children[1]],
                )
            }
        }

        impl CubeKernelOp for $name {
            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .cloned()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn execute<R: CubeCLRuntime>(
                &self,
                client: &ComputeClient<R>,
                inputs: &[Handle],
                output: Handle,
                dyn_map: &FxHashMap<char, usize>,
            ) {
                let shape_u32 = exprs_to_u32_vec(&self.shape, dyn_map);
                let inp_strides_u32 = exprs_to_u32_vec(&self.input_strides, dyn_map);
                let out_strides_u32 = exprs_to_u32_vec(&self.output_strides, dyn_map);
                let ndim = shape_u32.len() as u32;
                let num_elements = shape_u32.iter().map(|&x| x as usize).product::<usize>();

                let shape_handle = client.create(unsafe {
                    std::slice::from_raw_parts(shape_u32.as_ptr() as *const u8, shape_u32.len() * 4)
                });
                let inp_strides_handle = client.create(unsafe {
                    std::slice::from_raw_parts(
                        inp_strides_u32.as_ptr() as *const u8,
                        inp_strides_u32.len() * 4,
                    )
                });
                let out_strides_handle = client.create(unsafe {
                    std::slice::from_raw_parts(
                        out_strides_u32.as_ptr() as *const u8,
                        out_strides_u32.len() * 4,
                    )
                });

                $kernel::launch::<R>(
                    client,
                    CubeCount::Static(1, 1, 1), // TODO: Calculate proper grid
                    CubeDim::default(),
                    unsafe { Arg::from_raw(inputs[0].clone()) },
                    unsafe { Arg::from_raw(output) },
                    unsafe { Arg::from_raw(shape_handle) },
                    unsafe { Arg::from_raw(inp_strides_handle) },
                    unsafe { Arg::from_raw(out_strides_handle) },
                    Scalar::new(ndim),
                );
            }
        }
    };
}

// Implement CubeExp
impl_unary!(CubeExp, "CubeExp", unary_kernel_exp, "Exp");

// Placeholder for Log (implement properly later or map to Exp for now if kernel missing)
// I need a log kernel
#[cube(launch)]
fn unary_kernel_log(
    input: &Array<f32>,
    output: &mut Array<f32>,
    shape: &Array<u32>,
    inp_strides: &Array<u32>,
    out_strides: &Array<u32>,
    ndim: u32,
) {
    if ABSOLUTE_POS < output.len() {
        let idx = ABSOLUTE_POS;
        let in_idx = strided_index(idx, shape, inp_strides, ndim);
        let out_idx = strided_index(idx, shape, out_strides, ndim);
        output[out_idx] = cubecl::prelude::log(input[in_idx]);
    }
}
impl_unary!(CubeLog, "CubeLog", unary_kernel_log, "Log");

macro_rules! impl_binary {
    ($name:ident, $op_name:expr, $kernel:ident, $rewrite_op:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            a_strides: Vec<Expression>,
            b_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn term(&self) -> (String, Vec<OpParam>) {
                (
                    $op_name.to_string(),
                    vec![EList, Input, EList, Input, EList, EList],
                )
            }
            fn rewrites(&self) -> Vec<String> {
                vec![format!(
                    r#"(rule
                        ((= ?e ({} ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
                         (= ?dt (dtype ?a)))
                        ((let ?me ({} ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
                         (union ?e ?me)
                         (set (dtype ?me) ?dt))
                    )"#,
                    $rewrite_op, $op_name
                )]
            }
            fn cleanup(&self) -> bool {
                false
            }
            fn extract<'a>(
                &'a self,
                egraph: &'a SerializedEGraph,
                children: &[&'a ENodeId],
                list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
            ) -> (LLIROp, Vec<&'a ENodeId>) {
                use luminal::graph::extract_expr_list;
                (
                    LLIROp::new::<dyn CubeKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                            .unwrap(),
                        a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                            .unwrap(),
                        b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                            .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            children[5],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    vec![children[1], children[3]],
                )
            }
        }

        impl CubeKernelOp for $name {
            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .cloned()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn execute<R: CubeCLRuntime>(
                &self,
                client: &ComputeClient<R>,
                inputs: &[Handle],
                output: Handle,
                dyn_map: &FxHashMap<char, usize>,
            ) {
                let shape_u32 = exprs_to_u32_vec(&self.shape, dyn_map);
                let lhs_strides_u32 = exprs_to_u32_vec(&self.a_strides, dyn_map);
                let rhs_strides_u32 = exprs_to_u32_vec(&self.b_strides, dyn_map);
                let out_strides_u32 = exprs_to_u32_vec(&self.output_strides, dyn_map);
                let ndim = shape_u32.len() as u32;

                let shape_handle = client.create(unsafe {
                    std::slice::from_raw_parts(shape_u32.as_ptr() as *const u8, shape_u32.len() * 4)
                });
                let lhs_strides_handle = client.create(unsafe {
                    std::slice::from_raw_parts(
                        lhs_strides_u32.as_ptr() as *const u8,
                        lhs_strides_u32.len() * 4,
                    )
                });
                let rhs_strides_handle = client.create(unsafe {
                    std::slice::from_raw_parts(
                        rhs_strides_u32.as_ptr() as *const u8,
                        rhs_strides_u32.len() * 4,
                    )
                });
                let out_strides_handle = client.create(unsafe {
                    std::slice::from_raw_parts(
                        out_strides_u32.as_ptr() as *const u8,
                        out_strides_u32.len() * 4,
                    )
                });

                $kernel::launch::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::default(),
                    unsafe { Arg::from_raw(inputs[0].clone()) },
                    unsafe { Arg::from_raw(inputs[1].clone()) },
                    unsafe { Arg::from_raw(output) },
                    unsafe { Arg::from_raw(shape_handle) },
                    unsafe { Arg::from_raw(lhs_strides_handle) },
                    unsafe { Arg::from_raw(rhs_strides_handle) },
                    unsafe { Arg::from_raw(out_strides_handle) },
                    Scalar::new(ndim),
                );
            }
        }
    };
}

impl_binary!(CubeAdd, "CubeAdd", binary_kernel_add, "Add");
impl_binary!(CubeMul, "CubeMul", binary_kernel_mul, "Mul");

// Reduce Ops - Stub for now or simple implementation
#[derive(Debug, Default, Clone)]
pub struct CubeSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

impl EgglogOp for CubeSumReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CubeSum".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Sum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (= ?dt (dtype ?inp)))
            ((let ?me (CubeSum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
            .to_string(),
        ]
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr;
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn CubeKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![children[2]],
        )
    }
}

impl CubeKernelOp for CubeSumReduce {
    type R = CubeCLRuntime;

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }
    fn execute(
        &self,
        client: &ComputeClient<Self::R>,
        inputs: &[Handle],
        output: Handle,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        // TODO: Implement reduction kernel
        // For now panic or use a dummy implementation (won't work for real graphs)
        panic!("CubeSumReduce not implemented yet");
    }
}
