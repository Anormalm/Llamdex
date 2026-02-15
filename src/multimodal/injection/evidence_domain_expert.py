from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.multimodal.evidence.base import EvidenceBuilder
from src.multimodal.injection.projector import EvidenceProjector


class SemanticEvidenceDomainExpert(nn.Module):
    """
    DomainExpert-compatible wrapper that:
    raw evidence input -> z -> projected tokens.
    """

    def __init__(self, evidence_builder: EvidenceBuilder, projector: EvidenceProjector):
        super().__init__()
        self.evidence_builder = evidence_builder
        self.projector = projector
        self.intermediate_result = None
        self.output = None
        self._cache_key = None
        self._cache_tokens = None

    def _build_cache_key(self, x: Any) -> str:
        if torch.is_tensor(x):
            return f"tensor:{id(x)}:{tuple(x.shape)}"
        if isinstance(x, dict):
            parts = []
            for k in sorted(x.keys()):
                v = x[k]
                if torch.is_tensor(v):
                    parts.append(f"{k}:{id(v)}:{tuple(v.shape)}")
                else:
                    parts.append(f"{k}:{id(v)}")
            return "|".join(parts)
        return f"obj:{id(x)}"

    def forward_with_features(self, x: Any) -> torch.Tensor:
        use_cache = not torch.is_grad_enabled()
        key = self._build_cache_key(x) if use_cache else None
        if use_cache and self._cache_key == key and self._cache_tokens is not None:
            return self._cache_tokens

        z = self.evidence_builder(x)
        self.intermediate_result = z.detach()
        self.output = z.detach()
        injected = self.projector(z)
        if use_cache:
            self._cache_key = key
            self._cache_tokens = injected
        else:
            self._cache_key = None
            self._cache_tokens = None
        return injected

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
