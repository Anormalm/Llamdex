from __future__ import annotations

from typing import Optional, Union
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationConfig, PretrainedConfig


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
        use_external = (
            expert_inputs is not None
            and hasattr(self, "_external_experts")
            and isinstance(self._external_experts, nn.ModuleList)
            and len(self._external_experts) > 0
        )
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
