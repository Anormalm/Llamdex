import torch
import torch.nn as nn

from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.multimodal.evidence.base import EvidenceBuilder
from src.multimodal.injection.evidence_domain_expert import SemanticEvidenceDomainExpert
from src.multimodal.injection.projector import EvidenceProjector


class _DummyBuilder(EvidenceBuilder):
    def __init__(self, evidence_dim: int):
        super().__init__(evidence_dim=evidence_dim)
        self.proj = nn.Linear(3, evidence_dim)

    def forward(self, x):
        return self.proj(x.float())


def _build_model(use_adapters: bool, tune_layernorm: bool):
    model = DomainMistralForCausalLM.from_pretrained_mistral(
        "hf-internal-testing/tiny-random-MistralForCausalLM",
        cache_dir="runs/hf_cache_tiny",
        torch_dtype=torch.float32,
    )
    model.num_tokens = 2
    for p in model.parameters():
        p.requires_grad = False
    model.configure_adapters_(use_adapters=use_adapters, adapter_bottleneck=16, adapter_dropout=0.0, adapter_activation="gelu")
    model.set_layernorm_tuning_(requires_grad=tune_layernorm and use_adapters)
    return model


def test_adapters_and_connectors_are_trainable_scope():
    model = _build_model(use_adapters=True, tune_layernorm=False)
    hidden = model.config.hidden_size
    expert = SemanticEvidenceDomainExpert(
        evidence_builder=_DummyBuilder(64),
        projector=EvidenceProjector(evidence_dim=64, hidden_size=hidden, num_tokens=2),
    )
    model.model.layers[0].add_expert_(expert, map_to_expert_emb=None)

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names, "Expected non-empty trainable params"
    for n in names:
        ok = (
            "experts.0.evidence_builder" in n
            or "experts.0.projector" in n
            or "attn_adapter" in n
            or "ffn_adapter" in n
        )
        assert ok, f"Unexpected trainable parameter: {n}"
        assert "self_attn" not in n and "mlp" not in n


def test_tune_layernorm_flag_unfreezes_only_layernorms():
    model = _build_model(use_adapters=True, tune_layernorm=True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("input_layernorm" in n or "post_attention_layernorm" in n for n in names)
    assert not any("self_attn" in n or ".mlp." in n for n in names)

