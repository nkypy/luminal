// Unary - Log2, Exp2, Sin, Sqrt, Recip
// Binary - Add, Mul, Mod, LessThan
// Other - SumReduce, MaxReduce, Contiguous

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait EgglogOp: std::fmt::Debug {
    fn name(&self) -> &str;
}

#[enum_dispatch(EgglogOp)]
#[derive(Debug)]
pub enum HLIROp {
    Input(Input),
    Output(Output),
}

#[derive(Debug)]
pub struct Input {
    pub node: usize,
    pub label: String,
}

impl EgglogOp for Input {
    fn name(&self) -> &str {
        &self.label
    }
}

#[derive(Debug)]
pub struct Output {
    pub node: usize,
}

impl EgglogOp for Output {
    fn name(&self) -> &str {
        "output"
    }
}
