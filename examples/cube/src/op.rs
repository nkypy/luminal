// Unary - Log2, Exp2, Sin, Sqrt, Recip
// Binary - Add, Mul, Mod, LessThan
// Other - SumReduce, MaxReduce, Contiguous

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait EgglogOp: std::fmt::Debug {
    fn name(&self) -> &str;
}

#[enum_dispatch]
pub trait CustomOp: std::fmt::Debug {
    fn to_llir_op(&self) -> &str {
        "custom_op"
    }
}

#[enum_dispatch(EgglogOp, CustomOp)]
#[derive(Clone, Debug)]
pub enum HLIROp {
    Input(Input),
    Output(Output),
}

#[enum_dispatch(EgglogOp, CustomOp)]
#[derive(Clone, Debug)]
pub enum LLIROp {
    Input(Input),
    Output(Output),
}

#[derive(Clone, Debug)]
pub struct Input {
    pub node: usize,
    pub label: String,
}

impl EgglogOp for Input {
    fn name(&self) -> &str {
        &self.label
    }
}

impl CustomOp for Input {}

#[derive(Clone, Debug)]
pub struct Output {
    pub node: usize,
}

impl EgglogOp for Output {
    fn name(&self) -> &str {
        "output"
    }
}

impl CustomOp for Output {}
