from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceProjector(nn.Module):
    """
    z(B,D) -> injected tokens(B,T,H)
    Linear(D,4D) + GELU + Linear(4D,T*H) -> reshape -> LayerNorm(H) -> alpha scaling
    """

    def __init__(self, evidence_dim: int, hidden_size: int, num_tokens: int, alpha: float = 1.0):
        super().__init__()
        self.evidence_dim = evidence_dim
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(evidence_dim, evidence_dim * 4, bias=True),
            nn.GELU(),
            nn.Linear(evidence_dim * 4, num_tokens * hidden_size, bias=True),
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(dtype=self.net[0].weight.dtype)
        x = self.net(z)
        x = x.view(z.size(0), self.num_tokens, self.hidden_size)
        x = self.ln(x)
        return self.alpha * x
