import torch
import torch.nn as nn

from src.multimodal.evidence.base import EvidenceBuilder
from src.multimodal.injection import EvidencePreRouter, EvidenceProjector, SemanticEvidenceDomainExpert
from src.multimodal.trainers.plan1_trainer import _collect_trainable_named_parameters


class _FrozenBuilder(EvidenceBuilder):
    def __init__(self, evidence_dim: int):
        super().__init__(evidence_dim=evidence_dim)
        self.frozen = nn.Linear(5, evidence_dim, bias=True)
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, inputs):
        return self.frozen(inputs)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False)
        self.adapter = nn.Linear(4, 4, bias=False)
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True


def test_pre_router_trainable_parameter_report_targets_expected_modules():
    model = _Model()
    builder = _FrozenBuilder(evidence_dim=6)
    projector = EvidenceProjector(evidence_dim=6, hidden_size=8, num_tokens=2, alpha=1.0)
    pre_router = EvidencePreRouter(evidence_dim=6, mode="global_feature", hidden_dim=12, task_conditioning=True)
    expert = SemanticEvidenceDomainExpert(builder, projector, pre_router=pre_router)

    rows = _collect_trainable_named_parameters(model, expert)
    names = {name for name, _ in rows}
    assert any(name.startswith("model.adapter") for name in names)
    assert any(name.startswith("semantic_expert.projector") for name in names)
    assert any(name.startswith("semantic_expert.pre_router") for name in names)
    assert all(not name.startswith("model.base") for name in names)
    assert all(not name.startswith("semantic_expert.evidence_builder") for name in names)
