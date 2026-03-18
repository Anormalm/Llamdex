from src.multimodal.eval.plan1_eval import EvalPlan1Args
from src.multimodal.trainers.plan1_trainer import TrainPlan1Args


def test_train_args_defaults_match_adapter_style_surface():
    a = TrainPlan1Args()
    assert a.use_adapters is True
    assert a.inject_location == "layer_input"
    assert a.fusion_policy == "post_attn_router_parallel"
    assert a.tune_layernorm is False


def test_eval_args_defaults_match_adapter_style_surface():
    a = EvalPlan1Args()
    assert a.use_adapters is True
    assert a.inject_location == "layer_input"
    assert a.fusion_policy == "post_attn_router_parallel"
    assert a.tune_layernorm is False
