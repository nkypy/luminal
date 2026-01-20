use super::CpuKernelOp;
use luminal::{egglog_utils::SerializedEGraph, op::OpParam::*, op::*, prelude::*};

pub type CpuOps = (
    // Unary ops
    CpuExp2,
    CpuLog2,
    CpuSin,
    CpuSqrt,
    CpuRecip,
    // Binary ops
    CpuAdd,
    CpuMul,
    CpuMod,
    CpuLessThan,
    // Reduce ops
    CpuSumReduce,
    CpuMaxReduce,
    // Data ops
    CpuConstant,
    CpuIota,
    CpuGather,
);

/// Apply strided indexing to calculate the actual index in the buffer
#[inline]
fn strided_index(
    idx: usize,
    shape: &[Expression],
    strides: &[Expression],
    dyn_map: &FxHashMap<char, usize>,
) -> usize {
    let ndim = shape.len();
    let mut result = 0;
    let mut remaining = idx;

    for i in (0..ndim).rev() {
        let dim_size = shape[i].exec(dyn_map).unwrap();
        let stride = strides[i].exec(dyn_map).unwrap();
        let coord = remaining % dim_size;
        result += coord * stride;
        remaining /= dim_size;
    }
    result
}

macro_rules! cpu_unary_op {
    ($name:ident, $op_name:expr, $cpu_fn:expr) => {
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
                    $op_name.replace("Cpu", ""),
                    $op_name
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
                    LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
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

        impl CpuKernelOp for $name {
            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .cloned()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn execute(
                &self,
                inputs: &[&[f32]],
                output: &mut [f32],
                dyn_map: &FxHashMap<char, usize>,
            ) {
                let n_elements = self.output_size().exec(dyn_map).unwrap();
                let input = inputs[0];

                // Process in chunks of 4 for SIMD-friendly access patterns
                let chunks = n_elements / 4;

                for chunk_idx in 0..chunks {
                    let base = chunk_idx * 4;
                    for offset in 0..4 {
                        let idx = base + offset;
                        let in_idx = strided_index(idx, &self.shape, &self.input_strides, dyn_map);
                        let out_idx =
                            strided_index(idx, &self.shape, &self.output_strides, dyn_map);
                        output[out_idx] = $cpu_fn(input[in_idx]);
                    }
                }

                // Handle remainder
                for idx in (chunks * 4)..n_elements {
                    let in_idx = strided_index(idx, &self.shape, &self.input_strides, dyn_map);
                    let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
                    output[out_idx] = $cpu_fn(input[in_idx]);
                }
            }
        }
    };
}

cpu_unary_op!(CpuExp2, "CpuExp2", |x: f32| 2.0f32.powf(x));
cpu_unary_op!(CpuLog2, "CpuLog2", |x: f32| x.log2());
cpu_unary_op!(CpuSin, "CpuSin", |x: f32| x.sin());
cpu_unary_op!(CpuSqrt, "CpuSqrt", |x: f32| x.sqrt());
cpu_unary_op!(CpuRecip, "CpuRecip", |x: f32| 1.0 / x);

#[derive(Debug, Default, Clone)]
pub struct CpuAdd {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for CpuAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CpuAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Add ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (CpuAdd ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
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
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl CpuKernelOp for CpuAdd {
    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_elements = self.output_size().exec(dyn_map).unwrap();
        let (a, b) = (inputs[0], inputs[1]);

        // Process in chunks of 4 for SIMD-friendly pattern
        let chunks = n_elements / 4;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;
            for offset in 0..4 {
                let idx = base + offset;
                let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
                let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
                let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
                output[out_idx] = a[a_idx] + b[b_idx];
            }
        }

        // Handle remainder
        for idx in (chunks * 4)..n_elements {
            let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
            let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
            let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
            output[out_idx] = a[a_idx] + b[b_idx];
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuMul {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for CpuMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CpuMul".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Mul ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (CpuMul ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
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
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl CpuKernelOp for CpuMul {
    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_elements = self.output_size().exec(dyn_map).unwrap();
        let (a, b) = (inputs[0], inputs[1]);

        let chunks = n_elements / 4;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;
            for offset in 0..4 {
                let idx = base + offset;
                let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
                let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
                let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
                output[out_idx] = a[a_idx] * b[b_idx];
            }
        }

        for idx in (chunks * 4)..n_elements {
            let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
            let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
            let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
            output[out_idx] = a[a_idx] * b[b_idx];
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuMod {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for CpuMod {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CpuMod".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Mod ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (CpuMod ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
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
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl CpuKernelOp for CpuMod {
    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_elements = self.output_size().exec(dyn_map).unwrap();
        let (a, b) = (inputs[0], inputs[1]);

        let chunks = n_elements / 4;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;
            for offset in 0..4 {
                let idx = base + offset;
                let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
                let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
                let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
                output[out_idx] = a[a_idx] % b[b_idx];
            }
        }

        for idx in (chunks * 4)..n_elements {
            let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
            let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
            let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
            output[out_idx] = a[a_idx] % b[b_idx];
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuLessThan {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for CpuLessThan {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CpuLessThan".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (LessThan ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (CpuLessThan ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
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
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl CpuKernelOp for CpuLessThan {
    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_elements = self.output_size().exec(dyn_map).unwrap();
        let (a, b) = (inputs[0], inputs[1]);

        let chunks = n_elements / 4;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;
            for offset in 0..4 {
                let idx = base + offset;
                let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
                let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
                let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
                output[out_idx] = if a[a_idx] < b[b_idx] { 1.0 } else { 0.0 };
            }
        }

        for idx in (chunks * 4)..n_elements {
            let a_idx = strided_index(idx, &self.shape, &self.a_strides, dyn_map);
            let b_idx = strided_index(idx, &self.shape, &self.b_strides, dyn_map);
            let out_idx = strided_index(idx, &self.shape, &self.output_strides, dyn_map);
            output[out_idx] = if a[a_idx] < b[b_idx] { 1.0 } else { 0.0 };
        }
    }
}

// Reduce Operations

#[derive(Debug, Default, Clone)]
pub struct CpuSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

impl EgglogOp for CpuSumReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CpuSum".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Sum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (= ?dt (dtype ?inp)))
            ((let ?me (CpuSum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
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
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
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

impl CpuKernelOp for CpuSumReduce {
    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_outputs = self.output_size().exec(dyn_map).unwrap();
        let input = inputs[0];
        let iters = self.iters.exec(dyn_map).unwrap();
        let iter_stride_val = self.iter_stride.exec(dyn_map).unwrap();

        for out_idx in 0..n_outputs {
            let in_start = strided_index(out_idx, &self.out_shape, &self.in_stride, dyn_map);
            let out_offset = strided_index(out_idx, &self.out_shape, &self.out_stride, dyn_map);

            // Sum reduction with loop unrolling for SIMD
            let mut sum = 0.0f32;
            let chunks = iters / 4;

            for chunk in 0..chunks {
                let base = chunk * 4;
                sum += input[in_start + base * iter_stride_val];
                sum += input[in_start + (base + 1) * iter_stride_val];
                sum += input[in_start + (base + 2) * iter_stride_val];
                sum += input[in_start + (base + 3) * iter_stride_val];
            }

            for i in (chunks * 4)..iters {
                sum += input[in_start + i * iter_stride_val];
            }

            output[out_offset] = sum;
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

impl EgglogOp for CpuMaxReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CpuMax".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Max ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (= ?dt (dtype ?inp)))
            ((let ?me (CpuMax ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
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
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
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

impl CpuKernelOp for CpuMaxReduce {
    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_outputs = self.output_size().exec(dyn_map).unwrap();
        let input = inputs[0];
        let iters = self.iters.exec(dyn_map).unwrap();
        let iter_stride_val = self.iter_stride.exec(dyn_map).unwrap();

        for out_idx in 0..n_outputs {
            let in_start = strided_index(out_idx, &self.out_shape, &self.in_stride, dyn_map);
            let out_offset = strided_index(out_idx, &self.out_shape, &self.out_stride, dyn_map);

            let mut max_val = f32::NEG_INFINITY;
            let chunks = iters / 4;

            for chunk in 0..chunks {
                let base = chunk * 4;
                max_val = max_val.max(input[in_start + base * iter_stride_val]);
                max_val = max_val.max(input[in_start + (base + 1) * iter_stride_val]);
                max_val = max_val.max(input[in_start + (base + 2) * iter_stride_val]);
                max_val = max_val.max(input[in_start + (base + 3) * iter_stride_val]);
            }

            for i in (chunks * 4)..iters {
                max_val = max_val.max(input[in_start + i * iter_stride_val]);
            }

            output[out_offset] = max_val;
        }
    }
}

// Data Operations

#[derive(Debug, Default, Clone)]
pub struct CpuConstant {
    value: f32,
}

impl EgglogOp for CpuConstant {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("CpuConstant".to_string(), vec![Float])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Constant ?f)))
            ((let ?me (CpuConstant ?f))
             (union ?e ?me)
             (set (dtype ?me) (F32)))
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
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                value: egraph.enodes[children[0]]
                    .0
                    .replace("\"", "")
                    .parse::<f32>()
                    .unwrap(),
            })),
            vec![],
        )
    }
}

impl CpuKernelOp for CpuConstant {
    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn execute(&self, _inputs: &[&[f32]], output: &mut [f32], _dyn_map: &FxHashMap<char, usize>) {
        output[0] = self.value;
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for CpuIota {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("CpuIota".to_string(), vec![Expr, Expr])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            r#"(rule
            ((= ?e (Iota ?expr ?range)))
            ((let ?me (CpuIota ?expr ?range))
             (union ?e ?me)
             (set (dtype ?me) (Int)))
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
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                expr: extract_expr(egraph, children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl CpuKernelOp for CpuIota {
    fn output_size(&self) -> Expression {
        self.range.clone()
    }

    fn execute(&self, _inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_elements = self.range.exec(dyn_map).unwrap();

        // For Iota, evaluate the expression for each index
        // Replace the symbolic 'z' variable with the actual index
        for idx in 0..n_elements {
            let mut dyn_map_extended = dyn_map.clone();
            dyn_map_extended.insert('z', idx);
            let value = self.expr.exec(&dyn_map_extended).unwrap();
            output[idx] = value as f32;
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
}

impl EgglogOp for CpuGather {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CpuGather".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?a (Gather ?indexes ?out_shape ?index_strides ?data ?data_shape ?data_strides))
             (= ?dty (dtype ?data)))
            ((let ?out_strides (RowMajor ?out_shape))
             (let ?me (CpuGather ?out_shape ?indexes ?index_strides ?data ?data_strides ?out_strides))
             (union ?a ?me)
             (set (dtype ?me) ?dty))
        )"#
        .to_string()]
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
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                index_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                data_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl CpuKernelOp for CpuGather {
    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn execute(&self, inputs: &[&[f32]], output: &mut [f32], dyn_map: &FxHashMap<char, usize>) {
        let n_elements = self.output_size().exec(dyn_map).unwrap();
        let indexes = inputs[0];
        let data = inputs[1];

        for idx in 0..n_elements {
            let index_idx = strided_index(idx, &self.out_shape, &self.index_stride, dyn_map);
            let out_idx = strided_index(idx, &self.out_shape, &self.out_stride, dyn_map);

            // Get the index value (stored as f32 but represents an integer)
            let gathered_index = indexes[index_idx] as usize;

            // Calculate the data index using the gathered index
            let data_idx =
                strided_index(gathered_index, &self.out_shape, &self.data_stride, dyn_map);
            output[out_idx] = data[data_idx];
        }
    }
}
