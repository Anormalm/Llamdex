from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.multimodal.evidence.base import EvidenceBuilder


class EncoderEvidenceBuilder(EvidenceBuilder):
    """
    Generic bridge: ExpertEncoder output -> evidence vector z in R^D.
    Supports pluggable expert encoders while preserving downstream interface.
    """

    def __init__(self, expert_encoder: nn.Module, evidence_dim: int, output_dim: int):
        super().__init__(evidence_dim=evidence_dim)
        self.expert_encoder = expert_encoder
        self.output_dim = int(output_dim)
        self.to_z = nn.Identity() if self.output_dim == evidence_dim else nn.Linear(self.output_dim, evidence_dim, bias=True)
        for p in self.expert_encoder.parameters():
            p.requires_grad = False

    def forward(self, inputs: Any) -> torch.Tensor:
        x = self.expert_encoder.encode(inputs)
        if x.ndim != 2:
            raise ValueError(f"Expert encoder must return rank-2 tensor (B, M), got shape={tuple(x.shape)}")
        dtype = self.to_z.weight.dtype if isinstance(self.to_z, nn.Linear) else x.dtype
        x = x.to(dtype=dtype)
        z = self.to_z(x)
        if z.shape[-1] != self.evidence_dim:
            raise ValueError(f"Evidence dim mismatch: expected {self.evidence_dim}, got {z.shape[-1]}")
        return z
