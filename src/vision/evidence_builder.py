"""
Evidence builder abstractions for Llamdex-IMG Plan-1 upgrade.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .vision_expert import VisionExpert


class EvidenceBuilder(nn.Module, ABC):
    """Base class for producing a generic evidence vector z in R^D."""

    def __init__(self, evidence_dim: int):
        super().__init__()
        self.evidence_dim = evidence_dim

    @abstractmethod
    def build(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch) -> torch.Tensor:
        return self.build(batch)


class VisionEvidenceBuilder(EvidenceBuilder):
    """Build evidence from a frozen vision expert."""

    def __init__(
        self,
        vision_expert: VisionExpert,
        evidence_dim: int,
    ):
        super().__init__(evidence_dim=evidence_dim)
        self.vision_expert = vision_expert
        expert_output_dim = vision_expert.get_output_dim()
        self.to_evidence = nn.Identity() if expert_output_dim == evidence_dim else nn.Linear(expert_output_dim, evidence_dim)

        for p in self.vision_expert.parameters():
            p.requires_grad = False

        self.last_output: Optional[torch.Tensor] = None
        self.last_logits: Optional[torch.Tensor] = None
        self.last_probs: Optional[torch.Tensor] = None

    def build(self, batch) -> torch.Tensor:
        if isinstance(batch, dict):
            images = batch["images"]
        else:
            images = batch

        with torch.no_grad():
            out = self.vision_expert(images)

        self.last_output = out
        if out.dim() == 2 and out.size(-1) == self.vision_expert.num_classes:
            self.last_logits = out
            self.last_probs = torch.softmax(out.float(), dim=-1)
        else:
            self.last_logits = None
            self.last_probs = None

        return self.to_evidence(out)


class TextEvidenceBuilder(EvidenceBuilder):
    """Build evidence from frozen distilroberta encoder outputs."""

    def __init__(
        self,
        evidence_dim: int,
        text_encoder_model: str = "distilroberta-base",
        cache_dir: str = "model/llm",
    ):
        super().__init__(evidence_dim=evidence_dim)
        self.text_encoder_model = text_encoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_model, cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(text_encoder_model, cache_dir=cache_dir)

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.to_evidence = nn.Linear(self.encoder.config.hidden_size, evidence_dim)

    def build(self, batch) -> torch.Tensor:
        if isinstance(batch, dict) and "input_ids" in batch and "attention_mask" in batch:
            encoded = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
        else:
            if isinstance(batch, dict):
                texts = batch["texts"]
            else:
                texts = batch
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        device = self.to_evidence.weight.device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)
        pooled = outputs.last_hidden_state[:, 0, :].float()
        return self.to_evidence(pooled)


class StatsAggregator(nn.Module):
    """
    Aggregate per-sample class logits/probabilities into summary stats.
    Output: [mean_probabilities, optional_count_feature].
    """

    def __init__(self, num_classes: int, include_count_feature: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.include_count_feature = include_count_feature

    def forward(self, per_sample: torch.Tensor, input_is_probs: bool = False) -> torch.Tensor:
        if per_sample.dim() != 3 or per_sample.size(-1) != self.num_classes:
            raise ValueError(
                f"Expected per_sample shape (batch, group, {self.num_classes}), got {tuple(per_sample.shape)}"
            )

        if input_is_probs:
            probs = per_sample.float()
        else:
            probs = torch.softmax(per_sample.float(), dim=-1)

        mean_prob = probs.mean(dim=1)
        if not self.include_count_feature:
            return mean_prob

        counts = torch.full(
            (mean_prob.size(0), 1),
            per_sample.size(1),
            dtype=mean_prob.dtype,
            device=mean_prob.device,
        )
        return torch.cat([mean_prob, counts], dim=-1)


class PopulationEvidenceBuilder(EvidenceBuilder):
    """Build evidence from population-level aggregated vision statistics."""

    def __init__(
        self,
        vision_expert: VisionExpert,
        evidence_dim: int,
        num_classes: int,
        include_count_feature: bool = True,
    ):
        super().__init__(evidence_dim=evidence_dim)
        self.vision_expert = vision_expert
        self.aggregator = StatsAggregator(num_classes=num_classes, include_count_feature=include_count_feature)
        stats_dim = num_classes + (1 if include_count_feature else 0)
        self.stats_to_evidence = nn.Linear(stats_dim, evidence_dim)

        for p in self.vision_expert.parameters():
            p.requires_grad = False

        self.last_summary: Optional[torch.Tensor] = None
        self.last_mean_probs: Optional[torch.Tensor] = None

    def build(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch["images"]
        if images.dim() != 5:
            raise ValueError(f"Expected population images shape (batch, group, c, h, w), got {tuple(images.shape)}")

        bsz, group_size, c, h, w = images.shape
        flat = images.view(bsz * group_size, c, h, w)
        with torch.no_grad():
            out = self.vision_expert(flat)

        if out.dim() != 2:
            raise ValueError(f"Population path expects 2D expert outputs, got {tuple(out.shape)}")
        if out.size(-1) != self.aggregator.num_classes:
            classifier = getattr(self.vision_expert, "classifier", None)
            if classifier is None:
                raise ValueError(
                    f"Population aggregation requires logits with {self.aggregator.num_classes} classes, got {out.size(-1)}"
                )
            with torch.no_grad():
                out = classifier(out)
        logits = out.view(bsz, group_size, -1)

        summary = self.aggregator(logits, input_is_probs=False)
        self.last_summary = summary
        self.last_mean_probs = summary[:, :self.aggregator.num_classes]
        return self.stats_to_evidence(summary)
