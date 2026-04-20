import torch

from src.multimodal.injection.fusion_policies import (
    FusionContext,
    PostAttnRouterLayersPolicy,
    PostAttnRouterParallelPolicy,
    PreFFNRouterParallelPolicy,
)


def test_pre_ffn_router_parallel_biases_binary_allowed_logits_from_context():
    policy = PreFFNRouterParallelPolicy(hidden_size=4)
    policy.set_context(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    logits = torch.zeros(1, 8)

    fused = policy.forward_logits(
        model=None,
        outputs=None,
        logits=logits,
        z_ctx=None,
        ctx=FusionContext(task_name="grounded_generation", allowed_token_ids=[2, 5]),
    )

    assert fused[0, 5] > fused[0, 2]


def test_pre_ffn_router_parallel_matches_post_attn_heuristic_for_population_bins():
    z_ctx = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    logits = torch.zeros(1, 16)
    ctx = FusionContext(task_name="population", allowed_token_ids=list(range(11)))

    post = PostAttnRouterParallelPolicy(hidden_size=4).forward_logits(
        model=None,
        outputs=type("Out", (), {"hidden_states": None})(),
        logits=logits.clone(),
        z_ctx=z_ctx,
        ctx=ctx,
    )

    pre = PreFFNRouterParallelPolicy(hidden_size=4)
    pre.set_context(z_ctx)
    fused = pre.forward_logits(
        model=None,
        outputs=None,
        logits=logits.clone(),
        z_ctx=None,
        ctx=ctx,
    )

    assert torch.equal(fused, post)


def test_post_attn_router_layers_registers_all_layer_attention_hooks():
    class _Attn(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    class _Layer:
        def __init__(self):
            self.self_attn = _Attn()

    model = type("Model", (), {"model": type("Inner", (), {"layers": [_Layer(), _Layer()]})()})()
    policy = PostAttnRouterLayersPolicy(hidden_size=4)
    with torch.no_grad():
        policy.expert_out.bias.fill_(1.0)

    policy.register_to_model(model)
    policy.set_context(torch.ones(2, 4))
    out = model.model.layers[0].self_attn(torch.zeros(2, 3, 4))

    assert policy._layer_indices == [0, 1]
    assert out.shape == (2, 3, 4)
    assert out.abs().sum() > 0

    policy.clear_context()
    policy.clear_hooks()


def test_post_attn_router_layers_supports_qwen_linear_attention_layers():
    class _LinearAttn(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    class _Layer:
        def __init__(self):
            self.linear_attn = _LinearAttn()

    model = type("Model", (), {"model": type("Inner", (), {"layers": [_Layer()]})()})()
    policy = PostAttnRouterLayersPolicy(hidden_size=4)
    with torch.no_grad():
        policy.expert_out.bias.fill_(1.0)

    policy.register_to_model(model)
    policy.set_context(torch.ones(1, 4))
    out = model.model.layers[0].linear_attn(torch.zeros(1, 2, 4))

    assert policy._layer_indices == [0]
    assert out.abs().sum() > 0

    policy.clear_context()
    policy.clear_hooks()
