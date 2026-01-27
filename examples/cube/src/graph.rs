use std::marker::PhantomData;

use cubecl::prelude::{CubeElement, Float, Runtime};
use rustc_hash::FxHashMap;

#[derive(Debug, Default)]
pub struct Graph<R: Runtime, F: Float + CubeElement> {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    // pub graph: HLIRGraph,
    // /// E-Graph search space
    // egraph: Option<SerializedEGraph>,
    // /// Available ops
    // pub ops: Option<Vec<Arc<Box<dyn EgglogOp>>>>,
    // /// Custom ops
    // pub custom_ops: Vec<Box<dyn CustomOp>>,
    _r: PhantomData<R>,
    _f: PhantomData<F>,
}
