import torch
import torch.nn as nn

from src.multimodal.evidence.vision_builder import VisionEvidenceBuilder
from src.multimodal.experts.vision_experts import build_vision_expert
from src.multimodal.injection.evidence_domain_expert import SemanticEvidenceDomainExpert
from src.multimodal.injection.projector import EvidenceProjector


def test_plan1_vision_forward_and_train_step():
    torch.manual_seed(0)
    expert = build_vision_expert(
        expert_kind="classifier",
        checkpoint_path=None,
        num_classes=10,
        init_weights="random",
    )
    builder = VisionEvidenceBuilder(
        evidence_dim=32,
        vision_expert=expert,
        expert_output_mode="logits",
        num_classes=10,
    )
    projector = EvidenceProjector(evidence_dim=32, hidden_size=64, num_tokens=4, alpha=1.0)
    semantic_expert = SemanticEvidenceDomainExpert(builder, projector)

    images = torch.randn(2, 3, 32, 32)
    injected = semantic_expert.forward_with_features(images)
    assert injected.shape == (2, 4, 64)

    head = nn.Linear(64, 10)
    optimizer = torch.optim.AdamW(list(projector.parameters()) + list(builder.parameters()) + list(head.parameters()), lr=1e-3)
    labels = torch.tensor([1, 2], dtype=torch.long)

    logits = head(injected[:, -1, :])
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert loss.item() > 0

