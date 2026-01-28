// Unary - Log2, Exp2, Sin, Sqrt, Recip
// Binary - Add, Mul, Mod, LessThan
// Other - SumReduce, MaxReduce, Contiguous

use enum_dispatch::enum_dispatch;
use petgraph::graph::NodeIndex;

pub enum OpParam {
    EList,
    Expr,
    Input,
    Int,
    Float,
    Str,
    Dty,
    IList,
}

impl std::fmt::Debug for OpParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpParam::EList => write!(f, "EList"),
            OpParam::Expr => write!(f, "Expression"),
            OpParam::Input => write!(f, "IR"),
            OpParam::Int => write!(f, "i64"),
            OpParam::Str => write!(f, "String"),
            OpParam::Dty => write!(f, "DType"),
            OpParam::Float => write!(f, "f64"),
            OpParam::IList => write!(f, "IList"),
        }
    }
}

#[enum_dispatch]
pub trait EgglogOp: std::fmt::Debug {
    fn term(&self) -> (String, Vec<OpParam>);
    fn rewrites(&self) -> Vec<String> {
        vec![]
    }
    fn early_rewrites(&self) -> Vec<String> {
        vec![]
    }
    fn cleanup(&self) -> bool {
        false
    }
    // #[allow(unused_variables)]
    // fn extract<'a>(
    //     &'a self,
    //     egraph: &'a SerializedEGraph,
    //     children: &[&'a ENodeId],
    //     list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
    //     expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    // ) -> (LLIROp, Vec<&'a ENodeId>) {
    //     panic!("Extraction not implemented for {self:?}!");
    // }
}

#[enum_dispatch]
pub trait CustomOp: std::fmt::Debug {
    fn to_llir_op(&self) -> &str {
        "custom_op"
    }
}

#[enum_dispatch]
pub trait HLIROp: std::fmt::Debug {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, Vec<usize>)]) -> String;
}

#[enum_dispatch]
pub trait LLIROp: std::fmt::Debug {}

#[enum_dispatch(EgglogOp, CustomOp, HLIROp, LLIROp)]
#[derive(Clone, Debug)]
pub enum HLIROps {
    Input(Input),
    Output(Output),
}

#[enum_dispatch(EgglogOp, CustomOp, HLIROp, LLIROp)]
#[derive(Clone, Debug)]
pub enum LLIROps {
    Input(Input),
    Output(Output),
}

#[derive(Clone, Debug)]
pub struct Input {
    pub node: usize,
    pub label: String,
}

impl EgglogOp for Input {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Input".to_string(),
            vec![OpParam::Int, OpParam::Str, OpParam::Dty],
        )
    }
}

impl CustomOp for Input {}

impl HLIROp for Input {
    fn to_egglog(&self, _inputs: &[(NodeIndex, String, Vec<usize>)]) -> String {
        "".to_string()
    }
}

impl LLIROp for Input {}

#[derive(Clone, Debug)]
pub struct Output {
    pub node: usize,
}

impl EgglogOp for Output {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Output".to_string(), vec![OpParam::Input, OpParam::Int])
    }
}

impl CustomOp for Output {}

impl HLIROp for Output {
    fn to_egglog(&self, _inputs: &[(NodeIndex, String, Vec<usize>)]) -> String {
        "".to_string()
    }
}

impl LLIROp for Output {}
