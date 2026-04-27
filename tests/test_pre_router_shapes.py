import torch

from src.multimodal.injection.pre_router import EvidencePreRouter


def test_pre_router_shapes_global_feature():
    router = EvidencePreRouter(evidence_dim=12, mode="global_feature", hidden_dim=16, task_conditioning=True, task_embedding_dim=4)
    z = torch.randn(3, 12)
    task_id = torch.tensor([0, 1, 2], dtype=torch.long)
    z_prime, g0, c0 = router(z, task_id=task_id)
    assert z_prime.shape == (3, 12)
    assert g0.shape == (3, 1)
    assert c0.shape == (3, 12)


def test_pre_router_shapes_global_only():
    router = EvidencePreRouter(evidence_dim=10, mode="global", hidden_dim=8, task_conditioning=False)
    z = torch.randn(2, 10)
    z_prime, g0, c0 = router(z)
    assert z_prime.shape == (2, 10)
    assert g0.shape == (2, 1)
    assert c0.shape == (2, 10)
