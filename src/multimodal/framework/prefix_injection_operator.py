from __future__ import annotations

import torch
import torch.nn as nn


class PrefixInjectionOperator(nn.Module):
    """
    Overwrite the last T hidden-state slots with projected evidence tokens.
    Interface is fixed: z -> projector -> (B,T,H) -> overwrite hidden[:, -T:, :].
    """

    def forward(self, hidden_states: torch.Tensor, evidence_tokens: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or evidence_tokens.ndim != 3:
            raise ValueError("Expected hidden_states and evidence_tokens with shape (B,L,H) and (B,T,H).")
        b1, l, h1 = hidden_states.shape
        b2, t, h2 = evidence_tokens.shape
        if b1 != b2 or h1 != h2:
            raise ValueError(f"Batch/hidden mismatch: hidden={hidden_states.shape}, evidence={evidence_tokens.shape}")
        if t <= 0 or t > l:
            raise ValueError(f"Invalid evidence token count T={t} for sequence length L={l}.")
        out = hidden_states.clone()
        out[:, -t:, :] = evidence_tokens
        return out
