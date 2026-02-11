from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.multimodal.evidence.base import EvidenceBuilder


class PopulationStatsEvidenceBuilder(EvidenceBuilder):
    """
    Build z from aggregated class probability statistics over N images.
    stats = [mean_prob(C), log(1+N)] then projected to z.
    """

    def __init__(self, evidence_dim: int, num_classes: int, sigma: float = 0.0):
        super().__init__(evidence_dim=evidence_dim)
        self.num_classes = num_classes
        self.sigma = sigma
        self.stats_to_z = nn.Linear(num_classes + 1, evidence_dim, bias=True)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean_prob = inputs["mean_prob"].float()
        n_images = inputs["n_images"].float().unsqueeze(-1)
        stats = torch.cat([mean_prob, torch.log1p(n_images)], dim=-1)
        if self.sigma > 0:
            stats = stats + torch.randn_like(stats) * self.sigma
        return self.stats_to_z(stats)

