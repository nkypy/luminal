use luminal::prelude::*;
use luminal_nn::{GroupNorm, LayerNorm, Linear};

const HIDDEN: usize = 768;
const VOCAB_SIZE: usize = 65536;
const HEAD: usize = 64;
const NUM_LAYERS: usize = 12;
const EPSILON: f32 = 1e-5;

pub struct Model {
    // embeddings: Embedding,
    embeddings: GraphTensor,
    blocks: Vec<Block>,
    ln_out: LayerNorm,
    head: Linear,
}

impl Model {
    pub fn init(cx: &mut Graph) -> Self {
        let embeddings = cx.named_tensor("rwkv.embeddings.weight", (VOCAB_SIZE, HIDDEN));
        let blocks = (0..NUM_LAYERS).map(|i| Block::init(i, cx)).collect();
        let ln_out = LayerNorm::new(
            HIDDEN,
            Some("rwkv.ln_out.weight"),
            Some("rwkv.ln_out.bias"),
            false,
            EPSILON,
            cx,
        );
        let head = Linear::new(VOCAB_SIZE, HIDDEN, "head.weight", None, cx);
        Self {
            embeddings,
            blocks,
            ln_out,
            head,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let mut x = self.embeddings.gather(x);
        for block in &self.blocks {
            x = block.forward(x);
        }
        x = self.ln_out.forward(x);
        self.head.forward(x)
    }
}

pub struct Block {
    pre_ln: Option<LayerNorm>,
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: SelfAttention,
    feed_forward: FeedForward,
}

impl Block {
    pub fn init(layer_id: usize, cx: &mut Graph) -> Self {
        let prefix = format!("rwkv.blocks.{layer_id}");

        let pre_ln = if layer_id == 0 {
            Some(LayerNorm::new(
                HIDDEN,
                Some(&format!("{prefix}.pre_ln.weight")),
                Some(&format!("{prefix}.pre_ln.bias")),
                false,
                EPSILON,
                cx,
            ))
        } else {
            None
        };
        let ln1 = LayerNorm::new(
            HIDDEN,
            Some(&format!("{prefix}.ln1.weight")),
            Some(&format!("{prefix}.ln1.bias")),
            false,
            EPSILON,
            cx,
        );
        let ln2 = LayerNorm::new(
            HIDDEN,
            Some(&format!("{prefix}.ln2.weight")),
            Some(&format!("{prefix}.ln2.bias")),
            false,
            EPSILON,
            cx,
        );
        let attention = SelfAttention::init(layer_id, cx);
        let feed_forward = FeedForward::init(layer_id, cx);
        Self {
            pre_ln,
            ln1,
            ln2,
            attention,
            feed_forward,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = self.pre_ln.as_ref().map(|ln| ln.forward(x)).unwrap_or(x);
        let x = self.ln1.forward(x);
        // let x = self.attention.forward(x);
        let x = self.ln2.forward(x);
        let x = self.feed_forward.forward(x);
        x
    }
}

pub struct SelfAttention {
    x_r: GraphTensor,
    x_w: GraphTensor,
    x_k: GraphTensor,
    x_v: GraphTensor,
    x_a: GraphTensor,
    x_g: GraphTensor,
    r_k: GraphTensor,
    w0: GraphTensor,
    w1: GraphTensor,
    w2: GraphTensor,
    a0: GraphTensor,
    a1: GraphTensor,
    a2: GraphTensor,
    g1: GraphTensor,
    g2: GraphTensor,
    v0: Option<GraphTensor>,
    v1: Option<GraphTensor>,
    v2: Option<GraphTensor>,
    k_k: GraphTensor,
    k_a: GraphTensor,
    receptance: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    ln_x: GroupNorm,
}

impl SelfAttention {
    pub fn init(layer_id: usize, cx: &mut Graph) -> Self {
        let prefix = format!("rwkv.blocks.{layer_id}.attention");

        let x_r = cx.named_tensor(&format!("{prefix}.x_r"), (1, 1, HIDDEN));
        let x_w = cx.named_tensor(&format!("{prefix}.x_w"), (1, 1, HIDDEN));
        let x_k = cx.named_tensor(&format!("{prefix}.x_k"), (1, 1, HIDDEN));
        let x_v = cx.named_tensor(&format!("{prefix}.x_v"), (1, 1, HIDDEN));
        let x_a = cx.named_tensor(&format!("{prefix}.x_a"), (1, 1, HIDDEN));
        let x_g = cx.named_tensor(&format!("{prefix}.x_g"), (1, 1, HIDDEN));
        let r_k = cx.named_tensor(&format!("{prefix}.r_k"), (HIDDEN / HEAD, HEAD));
        let w0 = cx.named_tensor(&format!("{prefix}.w0"), (1, 1, HIDDEN));
        let w1 = cx.named_tensor(&format!("{prefix}.w1"), (HIDDEN, HEAD));
        let w2 = cx.named_tensor(&format!("{prefix}.w2"), (HEAD, HIDDEN));
        let a0 = cx.named_tensor(&format!("{prefix}.a0"), (1, 1, HIDDEN));
        let a1 = cx.named_tensor(&format!("{prefix}.a1"), (HIDDEN, HEAD));
        let a2 = cx.named_tensor(&format!("{prefix}.a2"), (HEAD, HIDDEN));
        let g1 = cx.named_tensor(&format!("{prefix}.g1"), (HIDDEN, HEAD * 2));
        let g2 = cx.named_tensor(&format!("{prefix}.g2"), (HEAD * 2, HIDDEN));
        let v0 = if layer_id == 0 {
            None
        } else {
            Some(cx.named_tensor(&format!("{prefix}.v0"), (1, 1, HIDDEN)))
        };
        let v1 = if layer_id == 0 {
            None
        } else {
            Some(cx.named_tensor(&format!("{prefix}.v1"), (HIDDEN, HEAD / 2)))
        };
        let v2 = if layer_id == 0 {
            None
        } else {
            Some(cx.named_tensor(&format!("{prefix}.v2"), (HEAD / 2, HIDDEN)))
        };
        let k_k = cx.named_tensor(&format!("{prefix}.k_k"), (1, 1, HIDDEN));
        let k_a = cx.named_tensor(&format!("{prefix}.k_a"), (1, 1, HIDDEN));
        let receptance = Linear::new(
            HIDDEN,
            HIDDEN,
            &format!("{prefix}.receptance.weight"),
            None,
            cx,
        );
        let key = Linear::new(HIDDEN, HIDDEN, &format!("{prefix}.key.weight"), None, cx);
        let value = Linear::new(HIDDEN, HIDDEN, &format!("{prefix}.value.weight"), None, cx);
        let output = Linear::new(HIDDEN, HIDDEN, &format!("{prefix}.output.weight"), None, cx);
        let ln_x = GroupNorm::new(
            HIDDEN,
            HEAD,
            Some(&format!("{prefix}.ln_x.weight")),
            Some(&format!("{prefix}.ln_x.bias")),
            EPSILON as f64,
            cx,
        );
        Self {
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            r_k,
            w0,
            w1,
            w2,
            a0,
            a1,
            a2,
            g1,
            g2,
            v0,
            v1,
            v2,
            k_k,
            k_a,
            receptance,
            key,
            value,
            output,
            ln_x,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = self.x_r + x;
        let x = self.x_w + x;
        let x = self.x_k + x;
        let x = self.x_v + x;
        let x = self.x_a + x;
        let x = self.x_g + x;
        let x = self.r_k + x;
        let x = self.w0 + x;
        let x = self.w1 + x;
        let x = self.w2 + x;
        let x = self.a0 + x;
        let x = self.a1 + x;
        let x = self.a2 + x;
        let x = self.g1 + x;
        let x = self.g2 + x;
        let x = self.v0.as_ref().map(|v| *v + x).unwrap_or(x);
        let x = self.v1.as_ref().map(|v| *v + x).unwrap_or(x);
        let x = self.v2.as_ref().map(|v| *v + x).unwrap_or(x);
        let x = self.k_k + x;
        let x = self.k_a + x;
        // let x = self.receptance.forward(x);
        // let x = self.key.forward(x);
        // let x = self.value.forward(x);
        // let x = self.output.forward(x);
        x
    }
}

pub struct FeedForward {
    x_k: GraphTensor,
    key: Linear,
    value: Linear,
}

impl FeedForward {
    pub fn init(layer_id: usize, cx: &mut Graph) -> Self {
        let prefix = format!("rwkv.blocks.{layer_id}.feed_forward");

        let x_k = cx.named_tensor(&format!("{prefix}.x_k"), (1, 1, HIDDEN));
        let key = Linear::new(
            HIDDEN * 4,
            HIDDEN,
            &format!("{prefix}.key.weight"),
            None,
            cx,
        );
        let value = Linear::new(
            HIDDEN,
            HIDDEN * 4,
            &format!("{prefix}.value.weight"),
            None,
            cx,
        );
        Self { x_k, key, value }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = self.x_k + x;
        // let x = self.key.forward(x);
        // let x = self.value.forward(x);
        x
    }
}
