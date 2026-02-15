from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torchvision.models as tvm
import warnings


def _resnet18(weights: str = "imagenet") -> nn.Module:
    if weights == "imagenet":
        return tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    return tvm.resnet18(weights=None)


@dataclass
class VisionExpertOutput:
    logits: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None


class VisionClassifierExpert(nn.Module):
    """
    Frozen vision classifier expert returning logits.
    Supports loading client checkpoints trained with torchvision ResNet18.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str],
        num_classes: int,
        init_weights: Literal["imagenet", "random"] = "imagenet",
    ):
        super().__init__()
        self.model = _resnet18(weights=init_weights)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
        else:
            warnings.warn(
                "VisionClassifierExpert loaded without checkpoint_path. "
                "The classifier head is randomly initialized; accuracy will be near chance. "
                "Provide a CIFAR-finetuned checkpoint for meaningful results."
            )

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> VisionExpertOutput:
        return VisionExpertOutput(logits=self.model(images), embedding=None)


class VisionEmbeddingExpert(nn.Module):
    """
    Frozen vision embedding expert returning pooled embeddings.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        init_weights: Literal["imagenet", "random"] = "imagenet",
    ):
        super().__init__()
        model = _resnet18(weights=init_weights)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        self.embedding_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.model = model

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> VisionExpertOutput:
        return VisionExpertOutput(logits=None, embedding=self.model(images))


def build_vision_expert(
    expert_kind: Literal["classifier", "embedding"],
    checkpoint_path: Optional[str],
    num_classes: int,
    init_weights: Literal["imagenet", "random"] = "imagenet",
) -> nn.Module:
    if expert_kind == "classifier":
        return VisionClassifierExpert(checkpoint_path=checkpoint_path, num_classes=num_classes, init_weights=init_weights)
    if expert_kind == "embedding":
        return VisionEmbeddingExpert(checkpoint_path=checkpoint_path, init_weights=init_weights)
    raise ValueError(f"Unsupported expert_kind: {expert_kind}")
