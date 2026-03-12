from src.multimodal.eval.plan1_eval import EvalPlan1Args
from src.multimodal.trainers.plan1_trainer import TrainPlan1Args


def test_train_args_backward_defaults_match_old_behavior():
    a = TrainPlan1Args()
    assert a.use_adapters is True
    assert a.inject_location == "post_attn"
    assert a.tune_layernorm is False


def test_eval_args_backward_defaults_match_old_behavior():
    a = EvalPlan1Args()
    assert a.use_adapters is True
    assert a.inject_location == "post_attn"
    assert a.tune_layernorm is False
