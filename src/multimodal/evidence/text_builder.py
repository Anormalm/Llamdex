from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel
import warnings

from src.multimodal.evidence.base import EvidenceBuilder


class TextEvidenceBuilder(EvidenceBuilder):
    """
    Frozen encoder-only text model + trainable projection to z.
    """

    def __init__(
        self,
        evidence_dim: int,
        text_encoder_model_id: str = "distilroberta-base",
        cache_dir: str = "model/llm",
    ):
        super().__init__(evidence_dim=evidence_dim)
        try:
            self.encoder = AutoModel.from_pretrained(text_encoder_model_id, cache_dir=cache_dir)
        except Exception as e:
            fallback_id = "hf-internal-testing/tiny-random-roberta"
            warnings.warn(
                f"Failed to load text encoder '{text_encoder_model_id}' from cache/network ({e}). "
                f"Falling back to '{fallback_id}'.",
                UserWarning,
            )
            self.encoder = AutoModel.from_pretrained(fallback_id, cache_dir=cache_dir)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.text_to_z = nn.Linear(self.encoder.config.hidden_size, evidence_dim, bias=True)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        pooled = outputs.last_hidden_state[:, 0, :].to(dtype=self.text_to_z.weight.dtype)
        return self.text_to_z(pooled)
