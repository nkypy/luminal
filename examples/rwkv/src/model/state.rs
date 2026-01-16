use luminal::prelude::*;

#[derive(Debug)]
pub struct StatePerLayer {
    pub extract_key_value: GraphTensor,
    pub linear_attention: GraphTensor,
    pub feed_forward: GraphTensor,
}

#[derive(Debug)]
pub struct State {
    pub per_layer: Vec<StatePerLayer>,
    pub pos: usize,
}

impl State {
    pub fn init(cx: &mut Graph, num_layers: usize, hidden_size: usize, head_size: usize) -> Self {
        let mut per_layer = Vec::with_capacity(num_layers);
        let num_attention_heads = hidden_size / head_size;
        for _idx in 0..num_layers {
            let extract_key_value = cx.tensor((1, 1, hidden_size));
            let linear_attention = cx.tensor((num_attention_heads, head_size, head_size));
            let feed_forward = cx.tensor((1, 1, hidden_size));
            per_layer.push(StatePerLayer {
                extract_key_value,
                linear_attention,
                feed_forward,
            });
        }
        Self { per_layer, pos: 0 }
    }
}
