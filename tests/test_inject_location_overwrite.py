import torch
import torch.nn as nn

from src.model.DomainMistralModel import DomainMistralForCausalLM


class _ConstExpert(nn.Module):
    def __init__(self, num_tokens: int, hidden_size: int, value: float = 0.5):
        super().__init__()
        self.tokens = nn.Parameter(torch.full((1, num_tokens, hidden_size), value), requires_grad=False)

    def forward_with_features(self, x):
        b = x.shape[0]
        return self.tokens.expand(b, -1, -1)


def test_overwrite_happens_at_requested_inject_location():
    model = DomainMistralForCausalLM.from_pretrained_mistral(
        "hf-internal-testing/tiny-random-MistralForCausalLM",
        cache_dir="runs/hf_cache_tiny",
        torch_dtype=torch.float32,
    )
    layer = model.model.layers[0]
    hidden = model.config.hidden_size
    num_tokens = 2
    layer.add_expert_(_ConstExpert(num_tokens=num_tokens, hidden_size=hidden), map_to_expert_emb=None)

    x = torch.randn(1, 8, hidden)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)
    expert_in = (torch.zeros(1, 3),)
    raw = torch.full((1, num_tokens, hidden), 0.5)
    expected = layer.expert_output_layernorm(raw)

    for loc in ("layer_input", "post_attn", "pre_ffn", "post_ffn"):
        _ = layer(
            x,
            expert_inputs=expert_in,
            attention_mask=None,
            position_ids=pos,
            output_attentions=False,
            use_cache=False,
            inject_layer_id=0,
            inject_location=loc,
        )
        assert layer._debug_last_injection_location == loc
        assert layer._debug_last_overwrite_slice is not None
        assert torch.allclose(layer._debug_last_overwrite_slice, expected, atol=1e-6)
