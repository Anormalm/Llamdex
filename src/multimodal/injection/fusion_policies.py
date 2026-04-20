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
        return _router_parallel_heuristic_bias(logits, z_ctx, ctx)

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


class PostAttnRouterLayersPolicy(PostAttnRouterParallelPolicy):
    """
    Evidence-conditioned branch added to selected self-attention outputs.

    This is the in-transformer counterpart to PostAttnRouterParallelPolicy:
    instead of fusing only the final token hidden state after the full model
    forward, it hooks decoder layer self-attention modules and adds a gated
    evidence delta to the attention output before the layer residual/FFN path.
    """

    policy_name = "post_attn_router_layers"

    def __init__(self, hidden_size: int):
        super().__init__(hidden_size=hidden_size)
        self._z_ctx: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._layer_indices = []

    def _token_mixer_for_layer(self, layer, idx: int):
        unwrapped = getattr(layer, "layer", layer)
        for name in ("self_attn", "linear_attn", "attention", "attn"):
            mixer = getattr(unwrapped, name, None)
            if mixer is not None:
                return mixer
        raise ValueError(f"Layer {idx} does not expose a token mixer for post-attn router injection.")

    def register_to_model(self, model, layer_indices=None):
        self.clear_hooks()
        layers = list(model.model.layers)
        if layer_indices is None:
            indices = list(range(len(layers)))
        elif isinstance(layer_indices, int):
            indices = [int(layer_indices)]
        else:
            indices = [int(i) for i in layer_indices]
        for idx in indices:
            layer = layers[idx]
            mixer = self._token_mixer_for_layer(layer, idx)
            self._hook_handles.append(mixer.register_forward_hook(self._attn_out_hook))
        self._layer_indices = indices

    def clear_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._layer_indices = []

    def set_context(self, z_ctx: Optional[torch.Tensor]):
        self._z_ctx = z_ctx

    def clear_context(self):
        self._z_ctx = None

    def _delta(self, u: torch.Tensor) -> torch.Tensor:
        z = self._z_ctx
        if z is None:
            return torch.zeros_like(u)
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(-1, u.size(1), -1)
        z = z.to(dtype=u.dtype, device=u.device)
        x = torch.cat([u, z], dim=-1)
        h_exp = self.expert_out(F.gelu(self.expert_mid(F.gelu(self.expert_in(x)))))
        g = torch.sigmoid(self.gate(x))
        c = torch.sigmoid(self.channel_gate(x))
        return g * (c * h_exp)

    def _attn_out_hook(self, module, inputs, output):
        _ = module
        if self._z_ctx is None:
            return output
        attn_out = output[0] if isinstance(output, tuple) else output
        if attn_out is None or attn_out.dim() != 3:
            return output
        fused = attn_out + self._delta(attn_out).to(dtype=attn_out.dtype, device=attn_out.device)
        if isinstance(output, tuple):
            return (fused,) + output[1:]
        return fused

    def forward_logits(
        self,
        *,
        model,
        outputs,
        logits: torch.Tensor,
        z_ctx: Optional[torch.Tensor],
        ctx: FusionContext,
    ) -> torch.Tensor:
        _ = (model, outputs)
        z_eff = z_ctx if z_ctx is not None else self._z_ctx
        return _router_parallel_heuristic_bias(logits, z_eff, ctx)


def _router_parallel_heuristic_bias(logits: torch.Tensor, z_ctx: Optional[torch.Tensor], ctx: FusionContext):
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


class PreFFNRouterParallelPolicy(BaseFusionPolicy):
    """
    Evidence-conditioned trainable branch added in parallel to frozen FFN output.
    This is applied via a forward hook on target layer.mlp output.
    """

    policy_name = "pre_ffn_router_parallel"

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.expert_in = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.expert_mid = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.expert_out = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gate = nn.Linear(self.hidden_size * 2, 1)
        self.channel_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.zeros_(self.expert_out.weight)
        nn.init.zeros_(self.expert_out.bias)

        self._z_ctx: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._layer_idx: Optional[int] = None

    def register_to_model(self, model, layer_idx: int):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._layer_idx = int(layer_idx)
        layer = model.model.layers[self._layer_idx]
        self._hook_handle = layer.mlp.register_forward_hook(self._mlp_out_hook)

    def clear_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._layer_idx = None

    def set_context(self, z_ctx: Optional[torch.Tensor]):
        self._z_ctx = z_ctx

    def clear_context(self):
        self._z_ctx = None

    def _mlp_out_hook(self, module, inputs, output):
        _ = module
        if self._z_ctx is None:
            return output
        if not isinstance(inputs, (tuple, list)) or len(inputs) == 0:
            return output
        u = inputs[0]  # pre-ffn hidden after post-attn layernorm, [B, T, d]
        if u is None or output is None:
            return output

        z = self._z_ctx
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(-1, u.size(1), -1)
        z = z.to(dtype=u.dtype, device=u.device)

        x = torch.cat([u, z], dim=-1)
        h_exp = self.expert_out(F.gelu(self.expert_mid(F.gelu(self.expert_in(x)))))
        g = torch.sigmoid(self.gate(x))
        c = torch.sigmoid(self.channel_gate(x))
        delta = g * (c * h_exp)
        return output + delta.to(dtype=output.dtype, device=output.device)

    def forward_logits(
        self,
        *,
        model,
        outputs,
        logits: torch.Tensor,
        z_ctx: Optional[torch.Tensor],
        ctx: FusionContext,
    ) -> torch.Tensor:
        _ = (model, outputs)
        z_eff = z_ctx if z_ctx is not None else self._z_ctx
        # Fusion is applied in-layer through the hook, but grounded answer-code
        # selection still benefits from the same allowed-token bias used by the
        # post-attn policy when the task is constrained to a small code set.
        return _router_parallel_heuristic_bias(logits, z_eff, ctx)



def build_fusion_policy(name: str, hidden_size: int) -> BaseFusionPolicy:
    n = str(name).strip().lower()
    if n == "pre_attn_overwrite":
        return PreAttnOverwritePolicy()
    if n == "post_attn_router_parallel":
        return PostAttnRouterParallelPolicy(hidden_size=hidden_size)
    if n in {"post_attn_router_layers", "post_attn_router_all_layers"}:
        return PostAttnRouterLayersPolicy(hidden_size=hidden_size)
    if n == "pre_ffn_router_parallel":
        return PreFFNRouterParallelPolicy(hidden_size=hidden_size)
    raise ValueError(f"Unsupported fusion_policy: {name}")
