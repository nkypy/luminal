use luminal::prelude::*;

pub mod rwkv7;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub head_size: usize,
}

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
    pub fn init(cx: &mut Graph, cfg: &Config) -> Self {
        let mut per_layer = Vec::with_capacity(cfg.num_hidden_layers);
        let num_attention_heads = cfg.hidden_size / cfg.head_size;
        for _idx in 0..cfg.num_hidden_layers {
            let extract_key_value = cx.tensor((1, cfg.hidden_size));
            let linear_attention = cx.tensor((num_attention_heads, cfg.head_size, cfg.head_size));
            let feed_forward = cx.tensor((1, cfg.hidden_size));
            per_layer.push(StatePerLayer {
                extract_key_value,
                linear_attention,
                feed_forward,
            });
        }
        Self { per_layer, pos: 0 }
    }
}
