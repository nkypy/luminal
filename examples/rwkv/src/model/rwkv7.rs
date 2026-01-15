use luminal::prelude::*;
use luminal_nn::{GroupNorm, LayerNorm, Linear};

const HIDDEN: usize = 768;
const VOCAB_SIZE: usize = 65536;

pub struct Model {
    // embeddings: Embedding,
    embeddings: GraphTensor,
    // blocks: Vec<Block>,
    ln_out: LayerNorm,
    // head: Linear,
}

impl Model {
    pub fn init(cx: &mut Graph) -> Self {
        let embeddings = cx.named_tensor("rwkv.embeddings.weight", (VOCAB_SIZE, HIDDEN));
        // let blocks = vec![];
        let ln_out = LayerNorm::new(
            HIDDEN,
            Some("rwkv.ln_out.weight"),
            Some("rwkv.ln_out.bias"),
            false,
            1e-5,
            cx,
        );
        // let head = Linear::new(VOCAB_SIZE, HIDDEN, false, cx);
        Self {
            embeddings,
            // blocks,
            ln_out,
            // head,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = self.embeddings.gather(x);
        self.ln_out.forward(x)
    }
}

// struct Block {
//     pre_ln: Option<LayerNorm>,
//     ln1: LayerNorm,
//     ln2: LayerNorm,
//     attention: SelfAttention,
//     feed_forward: FeedForward,
// }

// struct SelfAttention {
//     x_r: GraphTensor,
//     x_w: GraphTensor,
//     x_k: GraphTensor,
//     x_v: GraphTensor,
//     x_a: GraphTensor,
//     x_g: GraphTensor,
//     r_k: GraphTensor,
//     w0: GraphTensor,
//     w1: GraphTensor,
//     w2: GraphTensor,
//     a0: GraphTensor,
//     a1: GraphTensor,
//     a2: GraphTensor,
//     g1: GraphTensor,
//     g2: GraphTensor,
//     v0: Option<GraphTensor>,
//     v1: Option<GraphTensor>,
//     v2: Option<GraphTensor>,
//     k_k: GraphTensor,
//     k_a: GraphTensor,
//     receptance: Linear,
//     key: Linear,
//     value: Linear,
//     output: Linear,
//     ln_x: GroupNorm,
//     layer_id: usize,
// }

// struct FeedForward {
//     x_k: GraphTensor,
//     key: Linear,
//     value: Linear,
//     layer_id: usize,
// }
