import os
import warnings
from typing import Callable, Dict

from pandas.core.roperator import rand_
from transformers import PretrainedConfig
try:
    from transformers import MistralConfig
except Exception:
    MistralConfig = None
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralForCausalLM, MistralModel
try:
    from transformers.models.mistral.modeling_mistral import MISTRAL_INPUTS_DOCSTRING
except Exception:
    MISTRAL_INPUTS_DOCSTRING = ""
try:
    from transformers.models.mistral.modeling_mistral import _prepare_4d_causal_attention_mask
except Exception:
    _prepare_4d_causal_attention_mask = None
try:
    from transformers.models.mistral.modeling_mistral import create_causal_mask, create_sliding_window_causal_mask
except Exception:
    create_causal_mask = None
    create_sliding_window_causal_mask = None
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import joblib
import numpy as np
from torch.nn import CrossEntropyLoss

from .DomainExpert import DomainExpert
from .util import SwiGLU, SimpleMLP, XGBoostModule

logger = logging.get_logger(__name__)


def _build_offline_tiny_mistral_config():
    if MistralConfig is None:
        raise RuntimeError("transformers.MistralConfig is unavailable; cannot build offline tiny mistral fallback.")
    cfg = MistralConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        sliding_window=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    setattr(cfg, "_attn_implementation", "eager")
    return cfg


class AdapterLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        up_init_std: float = 1e-4,
    ):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported adapter activation: {activation}")
        nn.init.normal_(self.up.weight, mean=0.0, std=up_init_std)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.act(self.down(x))))
        return x + delta


class DomainMistralDecoderLayer(nn.Module):
    def __init__(self, layer: MistralDecoderLayer, maps_to_expert_emb=None, layer_idx: int = -1):
        super().__init__()
        self.hidden_size = layer.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = layer.self_attn

        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.experts = nn.ModuleList([])
        self.num_experts = 0
        self.maps_to_expert_emb = maps_to_expert_emb if maps_to_expert_emb is not None else []

        self.expert_output_layernorm = nn.LayerNorm(self.hidden_size)
        self.use_adapters = False
        self.adapter_bottleneck = None
        self.attn_adapter = None
        self.ffn_adapter = None
        self._debug_last_injection_location = None
        self._debug_last_overwrite_slice = None

    def enable_adapters_(
        self,
        adapter_bottleneck: int,
        adapter_dropout: float = 0.0,
        adapter_activation: str = "gelu",
    ):
        self.use_adapters = True
        self.adapter_bottleneck = int(adapter_bottleneck)
        self.attn_adapter = AdapterLayer(
            hidden_size=self.hidden_size,
            bottleneck=self.adapter_bottleneck,
            dropout=adapter_dropout,
            activation=adapter_activation,
        )
        self.ffn_adapter = AdapterLayer(
            hidden_size=self.hidden_size,
            bottleneck=self.adapter_bottleneck,
            dropout=adapter_dropout,
            activation=adapter_activation,
        )

    def disable_adapters_(self):
        self.use_adapters = False
        self.attn_adapter = None
        self.ffn_adapter = None

    def _apply_injection(
        self,
        hidden_states: torch.Tensor,
        injected_tokens: Optional[torch.Tensor],
        location: str,
    ) -> torch.Tensor:
        if injected_tokens is None:
            return hidden_states
        t = injected_tokens.size(1)
        if t <= 0:
            return hidden_states
        hidden_states = hidden_states.clone()
        hidden_states[:, -t:, :] = injected_tokens
        self._debug_last_injection_location = location
        self._debug_last_overwrite_slice = hidden_states[:, -t:, :].detach().cpu()
        return hidden_states

    def add_expert_(self, expert: DomainExpert, map_to_expert_emb: Callable = None):
        self.experts.append(expert)
        if map_to_expert_emb is None:
            warnings.warn("No map_to_expert_emb function provided. Using identity function.")
            map_to_expert_emb = (lambda x: (nn.Identity()(x), None))
        self.maps_to_expert_emb.append(map_to_expert_emb)
        self.num_experts += 1

    # Modified from original code to support the use of domain expert
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_inputs: Tuple[torch.FloatTensor, ...] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        expert_weight: Optional[float] = None,
        inject_layer_id: Optional[int] = None,
        inject_location: str = "post_attn",
        **kwargs,

    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            expert_inputs (`Tuple[torch.FloatTensor]`): input to the domain expert
            position_ids (`torch.LongTensor`, *optional*): position ids of input tokens of shape `(batch, seq_len)`
            expert_inputs (`Tuple[torch.FloatTensor]`): input to the domain expert
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            expert_weight (`float`, *optional*):
                Weight of the expert output
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        ############### Modified from original code to support the use of domain expert #########################
        injected_tokens = None
        if self.num_experts > 0:
            expert_out_tuple = ()
            for i in range(self.num_experts):
                if expert_inputs is None:
                    expert_input, expert_mask = self.maps_to_expert_emb[i](hidden_states)
                    expert_out = self.experts[i](expert_input, expert_mask)
                else:
                    # direct forward ground truth expert input to the expert
                    expert_input = expert_inputs[i]
                    expert_out = self.experts[i].forward_with_features(expert_input)

                if expert_weight is not None:
                    expert_out = expert_out * expert_weight

                expert_out_tuple += (expert_out,)

            # append the expert output to the hidden states, split can concat to avoid inplace operation
            hidden_states_right = torch.cat(expert_out_tuple, dim=1)
            hidden_states_left = hidden_states[:, :-hidden_states_right.size(1), :]

            # LayerNorm for hidden state right
            hidden_states_right = self.expert_output_layernorm(hidden_states_right)

            injected_tokens = hidden_states_right

        should_inject_here = (inject_layer_id is None and self.num_experts > 0) or (
            inject_layer_id is not None and self.layer_idx == int(inject_layer_id) and self.num_experts > 0
        )
        if should_inject_here and inject_location == "layer_input":
            hidden_states = self._apply_injection(hidden_states, injected_tokens, location="layer_input")

        ##########################################################################################################

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_out, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=None,
        )
        if self.use_adapters and self.attn_adapter is not None:
            attn_out = self.attn_adapter(attn_out)
        hidden_states = residual + attn_out
        if should_inject_here and inject_location == "post_attn":
            hidden_states = self._apply_injection(hidden_states, injected_tokens, location="post_attn")

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if should_inject_here and inject_location == "pre_ffn":
            hidden_states = self._apply_injection(hidden_states, injected_tokens, location="pre_ffn")
        ffn_out = self.mlp(hidden_states)
        if self.use_adapters and self.ffn_adapter is not None:
            ffn_out = self.ffn_adapter(ffn_out)

        hidden_states = residual + ffn_out
        if should_inject_here and inject_location == "post_ffn":
            hidden_states = self._apply_injection(hidden_states, injected_tokens, location="post_ffn")

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DomainMistralModel(MistralModel):
    def __init__(self, model: MistralModel):
        for attr_name, attr_value in model.__dict__.items():
            setattr(self, attr_name, attr_value)

        for i in range(len(self.layers)):
            layer = DomainMistralDecoderLayer(self.layers[i], layer_idx=i)
            self.layers[i] = layer

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_inputs: Tuple[torch.FloatTensor, ...] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        expert_weight: Optional[float] = None,
        inject_layer_id: Optional[int] = None,
        inject_location: str = "post_attn",
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        past_key_values_length = 0
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            return_legacy_cache = True
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if use_cache and past_key_values is not None:
            if hasattr(past_key_values, "get_usable_length"):
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            elif hasattr(past_key_values, "get_seq_length"):
                past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if getattr(self, "_use_flash_attention_2", False):
            causal_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _prepare_4d_causal_attention_mask is not None:
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        elif create_causal_mask is not None:
            if cache_position is None:
                cache_position = torch.arange(
                    past_key_values_length,
                    past_key_values_length + seq_length,
                    device=inputs_embeds.device,
                )
            mask_fn = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
            causal_mask = mask_fn(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        else:
            causal_mask = attention_mask

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    expert_inputs,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    expert_weight,
                    inject_layer_id,
                    inject_location,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    expert_inputs,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    expert_weight=expert_weight,
                    inject_layer_id=inject_layer_id,
                    inject_location=inject_location,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DomainMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config, tokenizer=None, llamdex_padding:str = 'gaussian'):
        super().__init__(config)
        self.config = config
        self.num_tokens = 0
        self.tokenizer = tokenizer  # tokenizer for expert embedding translation
        self.llamdex_padding = llamdex_padding
        self.inject_layer_id = None
        self.inject_location = "post_attn"
        self.use_adapters = False

    def configure_injection_(self, layer_id: Optional[int], inject_location: str = "post_attn"):
        valid = {"layer_input", "post_attn", "pre_ffn", "post_ffn"}
        if inject_location not in valid:
            raise ValueError(f"Unsupported inject_location: {inject_location}")
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
        if adapter_bottleneck is None:
            adapter_bottleneck = max(1, self.config.hidden_size // 16)
        for layer in self.model.layers:
            if not isinstance(layer, DomainMistralDecoderLayer):
                continue
            if self.use_adapters:
                layer.enable_adapters_(
                    adapter_bottleneck=adapter_bottleneck,
                    adapter_dropout=adapter_dropout,
                    adapter_activation=adapter_activation,
                )
            else:
                layer.disable_adapters_()

    def set_layernorm_tuning_(self, requires_grad: bool = False):
        for layer in self.model.layers:
            if not isinstance(layer, DomainMistralDecoderLayer):
                continue
            for p in layer.input_layernorm.parameters():
                p.requires_grad = requires_grad
            for p in layer.post_attention_layernorm.parameters():
                p.requires_grad = requires_grad

    def adapter_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for i, layer in enumerate(self.model.layers):
            if not isinstance(layer, DomainMistralDecoderLayer):
                continue
            if layer.attn_adapter is not None and layer.ffn_adapter is not None:
                out[str(i)] = {
                    "attn_adapter": layer.attn_adapter.state_dict(),
                    "ffn_adapter": layer.ffn_adapter.state_dict(),
                }
        return out

    def load_adapter_state_dict(self, state: Dict[str, Dict[str, Dict[str, torch.Tensor]]], strict: bool = False):
        for k, layer_state in state.items():
            i = int(k)
            if i < 0 or i >= len(self.model.layers):
                continue
            layer = self.model.layers[i]
            if not isinstance(layer, DomainMistralDecoderLayer):
                continue
            if layer.attn_adapter is None or layer.ffn_adapter is None:
                continue
            if "attn_adapter" in layer_state:
                layer.attn_adapter.load_state_dict(layer_state["attn_adapter"], strict=strict)
            if "ffn_adapter" in layer_state:
                layer.ffn_adapter.load_state_dict(layer_state["ffn_adapter"], strict=strict)

    def add_expert_(self, expert: DomainExpert, layer_id: int | list[int] = None, copy=True, disable_emb_translate=False):
        """
        Add domain experts to the model at layer_id.
        Args:
            expert: DomainExpert object (passed by value or reference depending on the copy parameter)
            layer_id: ID of the layer to add the domain expert.
                        If None, the expert is added to all layers.
                        If a list of integers, the expert is added to the layers with the corresponding IDs.
            copy: If True, the expert is copied before adding it to the model.
            disable_emb_translate: If True, the embedding translation is disabled.

        Returns: None
        """
        if layer_id is None:
            layer_ids = list(range(len(self.model.layers)))
        elif not isinstance(layer_id, list):
            layer_ids = [layer_id]
        else:
            layer_ids = layer_id

        for i in layer_ids:
            if not 0 <= i < len(self.model.layers):
                raise ValueError(f"Invalid layer ID: {i}")

            if copy:
                expert = expert.clone()

            layer = self.model.layers[i]
            if disable_emb_translate:
                layer.add_expert_(expert, map_to_expert_emb=None)
            else:
                layer.add_expert_(expert, map_to_expert_emb=lambda x: self.emb_translate(x, expert.tokenizer,
                                                                                         expert.expert_input_size))
            self.model.layers[i] = layer

    def replace_expert_(self, expert_dir, expert_type="mlp"):
        """
        Replace the domain expert in the model.
        Args:
            expert_dir: Path to the domain expert
        Returns: None
        """
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            if isinstance(layer, DomainMistralDecoderLayer):
                for j in range(layer.num_experts):
                    layer.experts[j].expert_type = expert_type
                    if expert_type == "mlp":
                        layer.experts[j].domain_expert = torch.load(expert_dir, map_location='cpu')
                    elif expert_type == "gbdt":
                        # layer.experts[j].domain_expert = layer.experts[j]._load_gbdt_model
                        layer.experts[j].domain_expert = XGBoostModule(expert_dir)
                    else:
                        raise ValueError(f"Invalid expert type: {expert_type}")

    @classmethod
    def from_pretrained_mistral(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        tokenizer=None,
        llamdex_padding: str = 'gaussian',
        **kwargs,
    ):
        """
        Load a model from a pretrained mistral model without domain experts.
        Args:
            pretrained_model_name_or_path: path to the pretrained model or model ID of mistral model
            *model_args: Additional model specific arguments.
            config: path to the model configuration file.
            cache_dir: path to the cache directory.
            force_download: force download the model.
            local_files_only: load the model from local files only.
            tokenizer: tokenizer for expert embedding translation
            **kwargs:

        Returns: DomainMistralForCausalLM model with pretrained weights from MistralForCausalLM model
        """
        try:
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                llamdex_padding=llamdex_padding,
                **kwargs,
            )
        except OSError as exc:
            name = str(pretrained_model_name_or_path or "")
            if "tiny-random-MistralForCausalLM" not in name:
                raise
            warnings.warn(
                "Falling back to an offline tiny Mistral config because the requested tiny test model is not cached locally.",
                RuntimeWarning,
            )
            cfg = config if isinstance(config, PretrainedConfig) else _build_offline_tiny_mistral_config()
            dtype = kwargs.get("torch_dtype", None)
            model = cls(cfg)
            if dtype is not None:
                model = model.to(dtype=dtype)

        # wrap model with DomainMistralModel
        model.model = DomainMistralModel(model.model)

        # initialize tokenizer
        model.tokenizer = tokenizer

        # initialize each layer
        for i in range(len(model.model.layers)):
            model.model.layers[i] = DomainMistralDecoderLayer(model.model.layers[i], layer_idx=i)

        return model

    # override the from_pretrained method to initialize the model with domain experts
    @classmethod
    def from_pretrained(cls, dataset_name: str, base_model_root="model/", model_id="mistralai/Mistral-7B-Instruct-v0.3",
                        encoder_dir="roberta-large", expert_path=None, model_state_dict_path=None):
        raise NotImplementedError("This method is not implemented yet.")
        expert_root_dir = os.path.join(base_model_root, "experts")
        llm_root_dir = os.path.join(base_model_root, "llm")

        if expert_path is None:
            expert_path = os.path.join(expert_root_dir, f"{dataset_name}_mlp.pth")
        if model_state_dict_path is None:
            model_state_dict_path = os.path.join(llm_root_dir, f"domain_mistral_{dataset_name}/model_final.pth")

        # Load the model
        base_model = cls.from_pretrained_mistral(model_id, cache_dir=base_model_root, torch_dtype=torch.bfloat16)

        expert = DomainExpert

    def get_expert_input(self, layer_id: int, expert_id: int):
        """
        Get the input to the domain expert.
        Args:
            layer_id: ID of the layer
            expert_id: ID of the expert
        Returns: Input to the domain expert
        """
        return self.model.layers[layer_id].experts[expert_id].get_expert_input()

    def get_expert_output(self, layer_id: int, expert_id: int):
        """
        Get the output of the domain expert.
        Args:
            layer_id: ID of the layer
            expert_id: ID of the expert
        Returns: Output of the domain expert
        """
        return self.model.layers[layer_id].experts[expert_id].get_expert_output()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_inputs: Tuple[torch.FloatTensor, ...] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        expert_weight: Optional[float] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""


        mask_pad = torch.ones((input_ids.shape[0], self.num_tokens), dtype=torch.long).to(input_ids.device)
        attention_mask = torch.cat((attention_mask, mask_pad), dim=1).long()

        # pad the input_embeds with Gaussian values
        inputs_embeds = self.model.embed_tokens(input_ids)
        if self.llamdex_padding == 'gaussian':
            pad = torch.normal(0, 1, (input_ids.shape[0], self.num_tokens, inputs_embeds.shape[-1])).to(input_ids.device).to(torch.bfloat16)
        elif self.llamdex_padding == 'zero':
            pad = torch.zeros((input_ids.shape[0], self.num_tokens, inputs_embeds.shape[-1])).to(input_ids.device).to(torch.bfloat16)
        inputs_embeds = torch.cat((inputs_embeds, pad), dim=1)
        input_ids = None    # Use inputs_embeds instead of input_ids

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            expert_inputs=expert_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            expert_weight=expert_weight,
            inject_layer_id=self.inject_layer_id,
            inject_location=self.inject_location,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def setup_encoder(self, layer_id: int, expert_id: int, requires_grad: bool):
        """
        Setup the encoder of the domain expert.
        Args:
            layer_id: ID of the layer
            expert_id: ID of the expert
            requires_grad: If True, the encoder parameters are trainable
        Returns: None
        """
        self.model.layers[layer_id].experts[expert_id].setup_encoder(requires_grad)

    def setup_decoder(self, layer_id: int, expert_id: int, requires_grad: bool):
        """
        Setup the decoder of the domain expert.
        Args:
            layer_id: ID of the layer
            expert_id: ID of the expert
            requires_grad: If True, the decoder parameters are trainable
        Returns: None
        """
        self.model.layers[layer_id].experts[expert_id].setup_decoder(requires_grad)

    @torch.no_grad()
    def emb_translate(self, x, expert_tokenizer, expert_input_size):
        """
        Translate the embeddings of the domain expert.
        Args:
            x: Embeddings of the domain expert
            expert_tokenizer: [AutoTokenizer] tokenizer for the domain
            expert_input_size: [int] input size of the domain expert
        Returns: input_ids, attention_mask for DomainExpertEncoder
        """
        logits = self.lm_head(self.model.norm(x.to(torch.bfloat16)))
        token_ids = logits.argmax(dim=-1)

        text_batch = []
        for ids in token_ids:
            filtered_ids = [id.item() for id in ids if 0 <= id < self.tokenizer.vocab_size]
            try:
                text = self.tokenizer.decode(filtered_ids)
            except Exception as e:
                print(f"Error in decoding: {e}")
                text = self.tokenizer.unk_token * len(filtered_ids)
            text_batch.append(text)

        padded_text_batch = [text + " " + expert_tokenizer.pad_token * expert_input_size for text in text_batch]

        encoded_inputs = expert_tokenizer(padded_text_batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_inputs['input_ids'].to(x.device)
        attention_mask = encoded_inputs['attention_mask'].to(x.device)

        return input_ids, attention_mask
