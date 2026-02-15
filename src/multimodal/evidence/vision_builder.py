from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.multimodal.evidence.base import EvidenceBuilder


class VisionEvidenceBuilder(EvidenceBuilder):
    """
    Build z from frozen vision expert output.
    - logits -> trainable logits_to_z
    - embedding -> optional small adapter (off by default)
    """

    def __init__(
        self,
        evidence_dim: int,
        vision_expert: nn.Module,
        expert_output_mode: str,
        num_classes: int = 10,
        embedding_dim: int = 512,
        use_embedding_adapter: bool = False,
    ):
        super().__init__(evidence_dim=evidence_dim)
        self.vision_expert = vision_expert
        self.expert_output_mode = expert_output_mode
        self.use_embedding_adapter = use_embedding_adapter
        self._last_probs = None

        if expert_output_mode == "logits":
            self.logits_to_z = nn.Linear(num_classes, evidence_dim, bias=True)
            self.embedding_adapter = None
        elif expert_output_mode == "embedding":
            self.logits_to_z = None
            if use_embedding_adapter:
                self.embedding_adapter = nn.Sequential(
                    nn.Linear(embedding_dim, evidence_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(evidence_dim, evidence_dim, bias=True),
                )
            else:
                self.embedding_adapter = nn.Identity()
        else:
            raise ValueError(f"Unsupported expert_output_mode: {expert_output_mode}")

        for p in self.vision_expert.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        out = self.vision_expert(images)
        if getattr(out, "logits", None) is not None:
            return out.logits
        if getattr(out, "embedding", None) is not None and hasattr(self.vision_expert, "model"):
            # best-effort fallback for embedding-only experts with classifier head available
            model = self.vision_expert.model
            if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
                return model.fc(out.embedding)
        raise RuntimeError("Vision expert did not provide logits.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        try:
            p = next(self.vision_expert.parameters())
            inputs = inputs.to(device=p.device, dtype=p.dtype)
        except StopIteration:
            pass

        with torch.no_grad():
            out = self.vision_expert(inputs)
            if out.logits is not None:
                self._last_probs = torch.softmax(out.logits.float(), dim=-1)
            elif out.embedding is not None:
                self._last_probs = None

        if self.expert_output_mode == "logits":
            logits = out.logits.to(dtype=self.logits_to_z.weight.dtype)
            return self.logits_to_z(logits)

        emb = out.embedding.to(dtype=next(self.embedding_adapter.parameters(), out.embedding).dtype if isinstance(self.embedding_adapter, nn.Module) else out.embedding.dtype)
        z = self.embedding_adapter(emb)
        if z.shape[-1] != self.evidence_dim:
            raise ValueError(f"Embedding evidence dim mismatch: expected {self.evidence_dim}, got {z.shape[-1]}")
        return z
