from __future__ import annotations

from typing import Callable, Optional, Union
import os
import warnings

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationConfig, PretrainedConfig


class DomainQwenDecoderLayer(nn.Module):
    """
    Legacy Llamdex-style layer wrapper.

    Before the wrapped Qwen decoder layer runs, frozen expert outputs are projected
    to token embeddings and overwrite the rightmost reserved hidden-state slots.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.experts = nn.ModuleList()
        self.maps_to_expert_emb = nn.ModuleList()
        self.num_experts = 0
        self._active_expert_inputs = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            layer = self._modules.get("layer")
            if layer is not None and hasattr(layer, name):
                return getattr(layer, name)
            raise exc

    def add_expert_(self, expert: nn.Module, map_to_expert_emb: Optional[Callable] = None):
        self.experts.append(expert)
        if map_to_expert_emb is None:
            warnings.warn("No map_to_expert_emb function provided. Direct expert_inputs are expected.")
            map_to_expert_emb = nn.Identity()
        self.maps_to_expert_emb.append(map_to_expert_emb if isinstance(map_to_expert_emb, nn.Module) else _CallableModule(map_to_expert_emb))
        self.num_experts += 1

    def set_active_expert_inputs_(self, expert_inputs):
        self._active_expert_inputs = expert_inputs

    def clear_active_expert_inputs_(self):
        self._active_expert_inputs = None

    def _expert_input_for(self, idx: int, hidden_states: torch.Tensor):
        if self._active_expert_inputs is not None:
            seq = list(self._active_expert_inputs) if isinstance(self._active_expert_inputs, (tuple, list)) else [self._active_expert_inputs]
            return seq[min(idx, len(seq) - 1)]
        return self.maps_to_expert_emb[idx](hidden_states)

    def _run_expert(self, idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        expert_input = self._expert_input_for(idx, hidden_states)
        expert = self.experts[idx]
        if isinstance(expert_input, tuple) and len(expert_input) == 2 and hasattr(expert, "forward"):
            return expert(expert_input[0], expert_input[1])
        if hasattr(expert, "forward_with_features"):
            return expert.forward_with_features(expert_input)
        return expert(expert_input)

    def _inject(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_experts <= 0:
            return hidden_states
        expert_tokens = [self._run_expert(i, hidden_states) for i in range(self.num_experts)]
        hidden_states_right = torch.cat(expert_tokens, dim=1).to(dtype=hidden_states.dtype, device=hidden_states.device)
        if hidden_states_right.size(1) <= 0:
            return hidden_states
        if hidden_states_right.size(1) > hidden_states.size(1):
            raise ValueError(
                f"Expert token count {hidden_states_right.size(1)} exceeds sequence length {hidden_states.size(1)}."
            )
        norm = getattr(self.layer, "input_layernorm", None)
        if norm is not None:
            hidden_states_right = norm(hidden_states_right)
        hidden_states_left = hidden_states[:, : hidden_states.size(1) - hidden_states_right.size(1), :]
        return torch.cat((hidden_states_left, hidden_states_right), dim=1)

    def forward(self, hidden_states: Optional[torch.Tensor] = None, *args, **kwargs):
        if hidden_states is None:
            if not args:
                return self.layer(*args, **kwargs)
            hidden_states, args = args[0], args[1:]
        hidden_states = self._inject(hidden_states)
        return self.layer(hidden_states, *args, **kwargs)


class _CallableModule(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class DomainQwenForCausalLM(nn.Module):
    """
    Lightweight compatibility model used by Plan-1++ scripts.

    It wraps a Hugging Face causal LM, keeps backbone frozen by caller policy,
    and exposes the adapter/injection APIs expected by legacy Llamdex code.
    """

    def __init__(self, base_model: nn.Module, tokenizer=None):
        super().__init__()
        self.base_model = base_model
        self.model = getattr(base_model, "model", base_model)
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None) or GenerationConfig.from_model_config(
            self.config
        )
        self.tokenizer = tokenizer
        self.num_tokens = 0
        self.inject_layer_id = None
        self.inject_location = "post_attn"
        self.use_adapters = False
        self._external_experts = nn.ModuleList()
        self.is_quantized_4bit = bool(getattr(base_model, "is_quantized_4bit", False))

    @classmethod
    def from_pretrained_qwen(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        tokenizer=None,
        **kwargs,
    ):
        # Keep compatibility with legacy callers that still pass bitsandbytes knobs.
        # Modern HF Qwen constructors may reject these kwargs directly.
        kwargs.pop("load_in_4bit", None)
        kwargs.pop("bnb_4bit_compute_dtype", None)
        kwargs.pop("bnb_4bit_quant_type", None)
        kwargs.pop("bnb_4bit_use_double_quant", None)
        kwargs.pop("bnb_4bit_cpu_offload", None)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("weights_only", False)
        base = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(base_model=base, tokenizer=tokenizer)

    def configure_injection_(self, layer_id: Optional[int], inject_location: str = "post_attn"):
        self.inject_layer_id = layer_id
        self.inject_location = inject_location

    def _normalize_layer_ids(self, layer_id):
        layers = getattr(self.model, "layers", None)
        if layers is None:
            raise AttributeError("Wrapped Qwen model does not expose model.layers for legacy expert injection.")
        if layer_id is None:
            return list(range(len(layers)))
        if isinstance(layer_id, (list, tuple)):
            raw_ids = list(layer_id)
        else:
            raw_ids = [layer_id]
        out = []
        for idx in raw_ids:
            idx = int(idx)
            if idx < 0:
                idx = len(layers) + idx
            if not 0 <= idx < len(layers):
                raise ValueError(f"Invalid layer ID: {idx}")
            out.append(idx)
        return out

    def add_expert_(self, expert: nn.Module, layer_id=None, copy: bool = True, map_to_expert_emb: Optional[Callable] = None):
        """
        Attach an expert using original Llamdex reserved-slot overwrite semantics.

        The expert is expected to expose forward_with_features(raw_features) and
        return (batch, num_tokens, hidden_size) token embeddings.
        """
        for idx in self._normalize_layer_ids(layer_id):
            layer = self.model.layers[idx]
            if not isinstance(layer, DomainQwenDecoderLayer):
                layer = DomainQwenDecoderLayer(layer)
                self.model.layers[idx] = layer
            layer_expert = expert.clone() if copy and hasattr(expert, "clone") else expert
            layer.add_expert_(layer_expert, map_to_expert_emb=map_to_expert_emb)

    def _legacy_expert_layers(self):
        layers = getattr(self.model, "layers", [])
        return [layer for layer in layers if isinstance(layer, DomainQwenDecoderLayer) and layer.num_experts > 0]

    def _set_layer_expert_inputs_(self, expert_inputs):
        for layer in self._legacy_expert_layers():
            layer.set_active_expert_inputs_(expert_inputs)

    def _clear_layer_expert_inputs_(self):
        for layer in self._legacy_expert_layers():
            layer.clear_active_expert_inputs_()

    def setup_encoder(self, layer_id: int, expert_id: int, requires_grad: bool):
        self.model.layers[layer_id].experts[expert_id].setup_encoder(requires_grad)

    def setup_decoder(self, layer_id: int, expert_id: int, requires_grad: bool):
        self.model.layers[layer_id].experts[expert_id].setup_decoder(requires_grad)

    def configure_adapters_(
        self,
        use_adapters: bool = False,
        adapter_bottleneck: Optional[int] = None,
        adapter_dropout: float = 0.0,
        adapter_activation: str = "gelu",
    ):
        self.use_adapters = bool(use_adapters)
        _ = (adapter_bottleneck, adapter_dropout, adapter_activation)

    def set_layernorm_tuning_(self, requires_grad: bool = False):
        _ = requires_grad

    def adapter_state_dict(self):
        return {}

    def load_adapter_state_dict(self, state, strict: bool = False):
        _ = (state, strict)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_inputs=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        expert_weight: Optional[float] = None,
        **kwargs,
    ):
        _ = (position_ids, cache_position, expert_weight, kwargs)
        use_layer_overwrite = expert_inputs is not None and len(self._legacy_expert_layers()) > 0
        use_external = (
            expert_inputs is not None
            and not use_layer_overwrite
            and hasattr(self, "_external_experts")
            and isinstance(self._external_experts, nn.ModuleList)
            and len(self._external_experts) > 0
        )
        if use_layer_overwrite:
            self._set_layer_expert_inputs_(expert_inputs)
        try:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True if use_external else output_hidden_states,
                return_dict=True if use_external else return_dict,
            )
        finally:
            if use_layer_overwrite:
                self._clear_layer_expert_inputs_()
        if not use_external:
            return outputs

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            return outputs

        if isinstance(expert_inputs, (tuple, list)):
            expert_inputs_seq = list(expert_inputs)
        else:
            expert_inputs_seq = [expert_inputs]

        h_last = outputs.hidden_states[-1][:, -1, :]
        ctx = torch.zeros_like(h_last)
        for i, expert in enumerate(self._external_experts):
            inp = expert_inputs_seq[min(i, len(expert_inputs_seq) - 1)]
            toks = expert.forward_with_features(inp)
            ctx = ctx + toks.mean(dim=1).to(dtype=h_last.dtype, device=h_last.device)

        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            return outputs
        logits = outputs.logits
        fused_last = lm_head(h_last + ctx)
        logits = logits.clone()
        logits[:, -1, :] = fused_last
        outputs.logits = logits
        return outputs
