from typing import Optional
import torch
from peft import PeftType
from peft.utils import TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING
from transformers import DynamicCache


"""
A workaround to make PEFT 0.13 work on transformers 4.44.2 according to https://github.com/huggingface/peft/issues/869
"""
def custom_get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
    """
    peft_config = self.active_peft_config
    prompt_encoder = self.prompt_encoder[self.active_adapter]
    prompt_tokens = (
        self.prompt_tokens[self.active_adapter]
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(prompt_encoder.embedding.weight.device)
    )
    if peft_config.peft_type == PeftType.PREFIX_TUNING:
        prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
        if peft_config.inference_mode:
            past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
        else:
            past_key_values = prompt_encoder(prompt_tokens)
        if self.base_model_torch_dtype is not None:
            past_key_values = past_key_values.to(self.base_model_torch_dtype)
        past_key_values = past_key_values.view(
            batch_size,
            peft_config.num_virtual_tokens,
            peft_config.num_layers * 2,
            peft_config.num_attention_heads,
            peft_config.token_dim // peft_config.num_attention_heads,
        )
        if peft_config.num_transformer_submodules == 2:
            past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
            peft_config.num_transformer_submodules * 2
        )
        if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
            post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
            past_key_values = post_process_fn(past_key_values)

        ############################## Workaround for PEFT 0.13 ##############################
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        ######################################################################################

        return past_key_values
    else:
        if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompts = prompt_encoder(prompt_tokens, task_ids)
        else:
            if peft_config.inference_mode:
                prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = prompt_encoder(prompt_tokens)
        return prompts