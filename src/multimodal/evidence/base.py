from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class EvidenceBuilder(nn.Module, ABC):
    """
    Abstract interface for producing semantic evidence vectors z in R^D.
    """

    def __init__(self, evidence_dim: int):
        super().__init__()
        self.evidence_dim = evidence_dim

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

