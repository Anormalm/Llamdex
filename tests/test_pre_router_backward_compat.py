import torch

from src.multimodal.evidence.base import EvidenceBuilder
from src.multimodal.injection.evidence_domain_expert import SemanticEvidenceDomainExpert
from src.multimodal.injection.projector import EvidenceProjector


class _LinearBuilder(EvidenceBuilder):
    def __init__(self, evidence_dim: int):
        super().__init__(evidence_dim=evidence_dim)
        self.proj = torch.nn.Linear(6, evidence_dim, bias=True)

    def forward(self, inputs):
        return self.proj(inputs)


def test_disabling_pre_router_preserves_forward_behavior():
    torch.manual_seed(7)
    builder = _LinearBuilder(evidence_dim=8)
    projector = EvidenceProjector(evidence_dim=8, hidden_size=16, num_tokens=2, alpha=1.0)
    expert_a = SemanticEvidenceDomainExpert(builder, projector, pre_router=None)

    builder_b = _LinearBuilder(evidence_dim=8)
    projector_b = EvidenceProjector(evidence_dim=8, hidden_size=16, num_tokens=2, alpha=1.0)
    builder_b.load_state_dict(builder.state_dict())
    projector_b.load_state_dict(projector.state_dict())
    expert_b = SemanticEvidenceDomainExpert(builder_b, projector_b, pre_router=None)

    x = torch.randn(4, 6)
    out_a = expert_a.forward_with_features(x)
    out_b = expert_b.forward_with_features(x)
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)
