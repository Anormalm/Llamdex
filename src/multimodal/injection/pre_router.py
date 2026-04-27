from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


KNOWN_TASK_IDS = {
    "single_label": 0,
    "single_label_code": 1,
    "single_yesno": 2,
    "single_yesno_set2": 3,
    "population": 4,
    "grounded_generation": 5,
    "finegrained": 6,
    "strict_yesno": 7,
    "vqa": 8,
}


def task_name_to_id(task_name: str, max_task_ids: int = 32) -> int:
    name = str(task_name or "").strip().lower()
    if name in KNOWN_TASK_IDS:
        return int(KNOWN_TASK_IDS[name]) % max(1, int(max_task_ids))
    if name.startswith("single_image"):
        return int(KNOWN_TASK_IDS["single_label"]) % max(1, int(max_task_ids))
    # Stable deterministic fallback.
    return sum(ord(c) for c in name) % max(1, int(max_task_ids))


class EvidencePreRouter(nn.Module):
    """
    Lightweight admission router that gates evidence vectors before projection.

    Modes:
    - global:          z' = g0 * z
    - feature:         z' = c0 ⊙ z
    - global_feature:  z' = g0 * (c0 ⊙ z)
    """

    def __init__(
        self,
        evidence_dim: int,
        mode: str = "global_feature",
        hidden_dim: int = 128,
        *,
        task_conditioning: bool = False,
        task_embedding_dim: int = 16,
        max_task_ids: int = 32,
    ):
        super().__init__()
        m = str(mode or "global_feature").strip().lower()
        if m not in {"global", "feature", "global_feature"}:
            raise ValueError(f"Unsupported pre-router mode: {mode}")
        self.evidence_dim = int(evidence_dim)
        self.mode = m
        self.task_conditioning = bool(task_conditioning)
        self.max_task_ids = max(1, int(max_task_ids))
        self.task_embedding_dim = int(task_embedding_dim) if self.task_conditioning else 0

        in_dim = self.evidence_dim + self.task_embedding_dim
        h = max(8, int(hidden_dim))
        self.task_embedding = (
            nn.Embedding(self.max_task_ids, self.task_embedding_dim) if self.task_conditioning else None
        )
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h, bias=True),
            nn.SiLU(),
        )
        self.global_head = nn.Linear(h, 1, bias=True)
        self.feature_head = nn.Linear(h, self.evidence_dim, bias=True)

    def _resolve_task_emb(
        self,
        batch_size: int,
        *,
        task_id: Optional[torch.Tensor],
        task_emb: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.task_conditioning:
            return None
        if task_emb is not None:
            te = task_emb.to(device=device, dtype=dtype)
            if te.dim() == 1:
                te = te.unsqueeze(0).expand(batch_size, -1)
            if te.size(0) != batch_size:
                raise ValueError(f"task_emb batch mismatch: got {te.size(0)}, expected {batch_size}")
            return te
        if task_id is None:
            tid = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            tid = task_id.to(device=device, dtype=torch.long).view(-1)
            if tid.numel() == 1 and batch_size > 1:
                tid = tid.expand(batch_size)
            if tid.numel() != batch_size:
                raise ValueError(f"task_id batch mismatch: got {tid.numel()}, expected {batch_size}")
        tid = tid.remainder(self.max_task_ids)
        return self.task_embedding(tid).to(dtype=dtype)

    def forward(
        self,
        z: torch.Tensor,
        *,
        task_id: Optional[torch.Tensor] = None,
        task_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError(f"Expected z shape [B, d_e], got {tuple(z.shape)}")
        batch_size = int(z.size(0))
        inp = [z]
        t = self._resolve_task_emb(
            batch_size,
            task_id=task_id,
            task_emb=task_emb,
            device=z.device,
            dtype=z.dtype,
        )
        if t is not None:
            inp.append(t)
        x = torch.cat(inp, dim=-1)
        h = self.trunk(x)

        if self.mode in {"global", "global_feature"}:
            g0 = torch.sigmoid(self.global_head(h))
        else:
            g0 = torch.ones(batch_size, 1, device=z.device, dtype=z.dtype)

        if self.mode in {"feature", "global_feature"}:
            c0 = torch.sigmoid(self.feature_head(h))
        else:
            c0 = torch.ones_like(z)

        z_prime = g0 * (c0 * z)
        return z_prime, g0, c0
