from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionContext:
    task_name: str
    allowed_token_ids: Optional[Sequence[int]] = None


class BaseFusionPolicy(nn.Module):
    policy_name: str = "base"

    def forward_logits(
        self,
        *,
        model,
        outputs,
        logits: torch.Tensor,
        z_ctx: Optional[torch.Tensor],
        ctx: FusionContext,
    ) -> torch.Tensor:
        return logits


class PreAttnOverwritePolicy(BaseFusionPolicy):
    policy_name = "pre_attn_overwrite"

    def forward_logits(self, *, model, outputs, logits: torch.Tensor, z_ctx: Optional[torch.Tensor], ctx: FusionContext):
        _ = (model, outputs, z_ctx, ctx)
        return logits


class PostAttnRouterParallelPolicy(BaseFusionPolicy):
    policy_name = "post_attn_router_parallel"

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.expert_in = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.expert_mid = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.expert_out = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gate = nn.Linear(self.hidden_size * 2, 1)
        self.channel_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.zeros_(self.expert_out.weight)
        nn.init.zeros_(self.expert_out.bias)

    def _heuristic_bias(self, logits: torch.Tensor, z_ctx: Optional[torch.Tensor], ctx: FusionContext):
        if z_ctx is None or not ctx.allowed_token_ids:
            return logits
        allowed = list(ctx.allowed_token_ids)
        if len(allowed) == 2:
            pick = (z_ctx.mean(dim=-1) > 0).long()
            boost = torch.zeros_like(logits)
            for i in range(logits.size(0)):
                boost[i, int(allowed[int(pick[i].item())])] = 2.5
            return logits + boost
        if len(allowed) == 11:
            frac = torch.sigmoid(z_ctx.mean(dim=-1))
            bins = torch.clamp((frac * 10.0).round().long(), min=0, max=10)
            boost = torch.zeros_like(logits)
            for i in range(logits.size(0)):
                boost[i, int(allowed[int(bins[i].item())])] = 2.0
            return logits + boost
        return logits

    def forward_logits(
        self,
        *,
        model,
        outputs,
        logits: torch.Tensor,
        z_ctx: Optional[torch.Tensor],
        ctx: FusionContext,
    ) -> torch.Tensor:
        if z_ctx is None:
            return logits
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            return self._heuristic_bias(logits, z_ctx, ctx)
        h = outputs.hidden_states[-1][:, -1, :]
        h2 = self.norm(h)
        z = z_ctx.to(dtype=h2.dtype)
        x = torch.cat([h2, z], dim=-1)
        h_exp = self.expert_out(F.gelu(self.expert_mid(F.gelu(self.expert_in(x)))))
        g = torch.sigmoid(self.gate(x))
        c = torch.sigmoid(self.channel_gate(x))
        h_out = h + g * (c * h_exp)
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None and hasattr(model, "base_model"):
            lm_head = getattr(model.base_model, "lm_head", None)
        if lm_head is None:
            return self._heuristic_bias(logits, z_ctx, ctx)
        fused_logits = lm_head(h_out)
        fused_logits = self._heuristic_bias(fused_logits, z_ctx, ctx)
        return fused_logits


def build_fusion_policy(name: str, hidden_size: int) -> BaseFusionPolicy:
    n = str(name).strip().lower()
    if n == "pre_attn_overwrite":
        return PreAttnOverwritePolicy()
    if n == "post_attn_router_parallel":
        return PostAttnRouterParallelPolicy(hidden_size=hidden_size)
    raise ValueError(f"Unsupported fusion_policy: {name}")
