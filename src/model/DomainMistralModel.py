import os
import warnings
from typing import Callable

from pandas.core.roperator import rand_
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralModel,
    MISTRAL_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
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


class DomainMistralDecoderLayer(nn.Module):
    def __init__(self, layer: MistralDecoderLayer, maps_to_expert_emb=None):
        super().__init__()
        self.hidden_size = layer.hidden_size

        self.self_attn = layer.self_attn

        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.experts = nn.ModuleList([])
        self.num_experts = 0
        self.maps_to_expert_emb = maps_to_expert_emb if maps_to_expert_emb is not None else []

        self.expert_output_layernorm = nn.LayerNorm(self.hidden_size)

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

            hidden_states = torch.cat((hidden_states_left, hidden_states_right), dim=1)

        ##########################################################################################################

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=None,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

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
            layer = DomainMistralDecoderLayer(self.layers[i])
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            return_legacy_cache = True
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

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
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config,
                                        cache_dir=cache_dir, force_download=force_download,
                                        local_files_only=local_files_only, llamdex_padding=llamdex_padding, **kwargs)

        # wrap model with DomainMistralModel
        model.model = DomainMistralModel(model.model)

        # initialize tokenizer
        model.tokenizer = tokenizer

        # initialize each layer
        for i in range(len(model.model.layers)):
            model.model.layers[i] = DomainMistralDecoderLayer(model.model.layers[i])

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
