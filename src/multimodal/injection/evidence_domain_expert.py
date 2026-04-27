from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.multimodal.evidence.base import EvidenceBuilder
from src.multimodal.injection.pre_router import EvidencePreRouter
from src.multimodal.injection.projector import EvidenceProjector


class SemanticEvidenceDomainExpert(nn.Module):
    """
    DomainExpert-compatible wrapper that:
    raw evidence input -> z -> projected tokens.
    """

    def __init__(
        self,
        evidence_builder: EvidenceBuilder,
        projector: EvidenceProjector,
        pre_router: Optional[EvidencePreRouter] = None,
    ):
        super().__init__()
        self.evidence_builder = evidence_builder
        self.projector = projector
        self.pre_router = pre_router
        self.intermediate_result = None
        self.raw_intermediate_result = None
        self.output = None
        self.pre_router_gate = None
        self.pre_router_feature_gate = None
        self._cache_key = None
        self._cache_tokens = None
        self._task_context = {}

    def _build_cache_key(self, x: Any, *, task_id=None, task_emb=None) -> str:
        if torch.is_tensor(x):
            base = f"tensor:{id(x)}:{tuple(x.shape)}"
        elif isinstance(x, dict):
            parts = []
            for k in sorted(x.keys()):
                v = x[k]
                if torch.is_tensor(v):
                    parts.append(f"{k}:{id(v)}:{tuple(v.shape)}")
                else:
                    parts.append(f"{k}:{id(v)}")
            base = "|".join(parts)
        else:
            base = f"obj:{id(x)}"
        if torch.is_tensor(task_id):
            base += f"|task_id:{id(task_id)}:{tuple(task_id.shape)}"
        if torch.is_tensor(task_emb):
            base += f"|task_emb:{id(task_emb)}:{tuple(task_emb.shape)}"
        return base

    def set_task_context(
        self,
        *,
        task_id: Optional[torch.Tensor] = None,
        task_emb: Optional[torch.Tensor] = None,
    ) -> None:
        self._task_context = {
            "task_id": task_id,
            "task_emb": task_emb,
        }

    def clear_task_context(self) -> None:
        self._task_context = {}

    def _resolve_task_context(self, task_id=None, task_emb=None):
        if task_id is None:
            task_id = self._task_context.get("task_id")
        if task_emb is None:
            task_emb = self._task_context.get("task_emb")
        return task_id, task_emb

    def route_evidence(
        self,
        z: torch.Tensor,
        *,
        task_id: Optional[torch.Tensor] = None,
        task_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        task_id, task_emb = self._resolve_task_context(task_id=task_id, task_emb=task_emb)
        self.raw_intermediate_result = z.detach()
        if self.pre_router is None:
            g0 = torch.ones(z.size(0), 1, device=z.device, dtype=z.dtype)
            c0 = torch.ones_like(z)
            z_prime = z
        else:
            z_prime, g0, c0 = self.pre_router(z, task_id=task_id, task_emb=task_emb)
        self.intermediate_result = z_prime.detach()
        self.output = z_prime.detach()
        self.pre_router_gate = g0.detach()
        self.pre_router_feature_gate = c0.detach()
        return z_prime

    def project_from_evidence(
        self,
        z: torch.Tensor,
        *,
        task_id: Optional[torch.Tensor] = None,
        task_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_prime = self.route_evidence(z, task_id=task_id, task_emb=task_emb)
        return self.projector(z_prime)

    def forward_with_features(
        self,
        x: Any,
        *,
        task_id: Optional[torch.Tensor] = None,
        task_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        task_id, task_emb = self._resolve_task_context(task_id=task_id, task_emb=task_emb)
        use_cache = not torch.is_grad_enabled()
        key = self._build_cache_key(x, task_id=task_id, task_emb=task_emb) if use_cache else None
        if use_cache and self._cache_key == key and self._cache_tokens is not None:
            return self._cache_tokens

        z = self.evidence_builder(x)
        injected = self.project_from_evidence(z, task_id=task_id, task_emb=task_emb)
        if use_cache:
            self._cache_key = key
            self._cache_tokens = injected
        else:
            self._cache_key = None
            self._cache_tokens = None
        return injected

    def pre_router_gate_stats(self):
        if self.pre_router_gate is None:
            return {"mean": 1.0, "std": 0.0}
        g = self.pre_router_gate.float().view(-1)
        if g.numel() == 0:
            return {"mean": 1.0, "std": 0.0}
        return {"mean": float(g.mean().item()), "std": float(g.std(unbiased=False).item())}

    def get_expert_input(self, scale: bool = True):
        return self.intermediate_result

    def get_expert_output(self):
        return self.output

    def setup_encoder(self, requires_grad: bool):
        for p in self.evidence_builder.parameters():
            p.requires_grad = requires_grad

    def setup_decoder(self, requires_grad: bool):
        for p in self.projector.parameters():
            p.requires_grad = requires_grad

    def clone(self):
        return self
