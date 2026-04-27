import torch

from src.multimodal.injection.pre_router import EvidencePreRouter


def test_task_conditioning_changes_gate_values_for_same_evidence():
    torch.manual_seed(123)
    router = EvidencePreRouter(
        evidence_dim=10,
        mode="global_feature",
        hidden_dim=16,
        task_conditioning=True,
        task_embedding_dim=6,
        max_task_ids=16,
    )
    z = torch.randn(1, 10)
    _, g_a, c_a = router(z, task_id=torch.tensor([1]))
    _, g_b, c_b = router(z, task_id=torch.tensor([2]))
    assert not torch.allclose(g_a, g_b)
    assert not torch.allclose(c_a, c_b)
