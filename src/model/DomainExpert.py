import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModel, AutoTokenizer
import xgboost as xgb
import os
import numpy as np
import joblib

from .util import SwiGLU, SimpleMLP, XGBoostModule



class ExpertEncoder(nn.Module):
    def __init__(self, input_size, output_size, model_id="roberta-large", cache_dir="model/llm",
                 disable_emb_translate=False):
        super(ExpertEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, add_pooling_layer=False)
        # self.tokenizer = tokenizer
        # self.encoder_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, clean_up_tokenization_spaces=False)

        self.input_size = input_size
        self.output_size = output_size
        self.disable_emb_translate = disable_emb_translate
        if self.disable_emb_translate:
            self.model_embed_size = 4096    # hard-coded for mistral
        else:
            self.model_embed_size = self.model.config.hidden_size

        self.output_linear_list = nn.ModuleList([nn.Linear(self.model_embed_size, 1) for _ in range(output_size)])

    def forward(self, x, attention_mask=None):
        if len(x.shape) == 3:
            # the input is embeddings instead of token ids
            last_hidden_states = x
        else:
            outputs = self.model(input_ids=x, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        output_tokens = last_hidden_states[:, -self.output_size:, :]  # [batch_size, output_size, hidden_size]

        outputs = []
        for i in range(self.output_size):
            output = self.output_linear_list[i](output_tokens[:, i, :])
            outputs.append(output)

        output = torch.cat(outputs, dim=1)  # [batch_size, output_size]

        return output


class ExpertDecoder(nn.Module):
    def __init__(self, expert_output_size, ffn_hidden_size, embed_size, num_tokens, dropout=0.0):
        super().__init__()
        # self.proj = nn.Linear(expert_output_size, embed_size, bias=True)
        if ffn_hidden_size != -1:
            self.proj = SwiGLU(expert_output_size, ffn_hidden_size, embed_size * num_tokens, dropout=dropout)
        else:
            self.proj = nn.Linear(expert_output_size, embed_size * num_tokens, bias=True)
        self.embed_size = embed_size
        self.num_tokens = num_tokens

    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.size(0), -1, self.embed_size)

        return x



class DomainExpert(nn.Module):
    """
    Input: x (batch_size, num_tokens, embed_size) or (batch_size, expert_input_size)
    Output: x (batch_size, num_tokens, embed_size)
    """

    def __init__(self, embed_size, ffn_hidden_size, expert_input_size, expert_output_size, expert_dir,
                 num_heads=8, dropout=0.0, use_norm=True, max_length=512,
                 dataset_json=None, dataset_columns=None, expert_input_size_scaled=None,
                 mapping_hidden_size=64, num_tokens=10, cache_dir="model/llm", encoder_model_id="roberta-large",
                 expert_input_scaler=None, disable_emb_translate=False, expert_type="mlp", is_wine=False):
        super().__init__()
        self.embed_size = embed_size
        self.ffn_hidden_size = ffn_hidden_size
        self.expert_input_size = expert_input_size
        self.expert_output_size = expert_output_size
        self.expert_dir = expert_dir
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_norm = use_norm
        self.max_length = max_length
        self.dataset_json = dataset_json
        self.dataset_columns = dataset_columns
        self.expert_input_size_scaled = expert_input_size_scaled if expert_input_size_scaled is not None else expert_input_size
        self.mapping_hidden_size = mapping_hidden_size
        self.num_tokens = num_tokens
        self.cache_dir = cache_dir
        self.encoder_model_id = encoder_model_id
        self.expert_input_scaler = expert_input_scaler
        self.disable_emb_translate = disable_emb_translate
        self.expert_type = expert_type
        self.is_wine = is_wine

        self.expert_encoder = ExpertEncoder(embed_size, expert_input_size, cache_dir=cache_dir,
                                            model_id=encoder_model_id, disable_emb_translate=disable_emb_translate)

        if self.expert_type == "mlp":
            self.domain_expert = torch.load(expert_dir, map_location='cpu')

        elif self.expert_type == "gbdt":
            # self.domain_expert = self._load_gbdt_model()
            self.domain_expert = XGBoostModule(expert_dir)

        else:
            raise ValueError("Invalid expert_type.")

        for param in self.domain_expert.parameters():
            param.requires_grad = False

        self.expert_decoder = ExpertDecoder(expert_output_size, ffn_hidden_size, embed_size, num_tokens, dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_id, cache_dir=cache_dir,
                                                       clean_up_tokenization_spaces=False)
        self.intermediate_result = None
        self.output = None

    # def _load_gbdt_model(self, expert_dir=None):
    #     """Loads XGBoost model, moves to GPU if available."""
    #     if expert_dir is None:
    #         expert_dir = self.expert_dir
    #
    #     if os.path.exists(expert_dir):
    #         model = xgb.XGBClassifier()
    #         model.load_model(expert_dir)
    #         if torch.cuda.is_available():
    #             model.set_params(device="cuda:0")
    #         return model
    #     else:
    #         print(f"Warning: GBDT model not found at {expert_dir}. Returning None.")
    #         return None

    def clone(self):
        return DomainExpert(embed_size=self.embed_size, ffn_hidden_size=self.ffn_hidden_size,
                            expert_input_size=self.expert_input_size, expert_output_size=self.expert_output_size,
                            expert_dir=self.expert_dir,
                            num_heads=self.num_heads, dropout=self.dropout, use_norm=self.use_norm,
                            max_length=self.max_length, dataset_json=self.dataset_json,
                            dataset_columns=self.dataset_columns,
                            expert_input_size_scaled=self.expert_input_size_scaled,
                            mapping_hidden_size=self.mapping_hidden_size, num_tokens=self.num_tokens,
                            cache_dir=self.cache_dir, encoder_model_id=self.encoder_model_id,
                            expert_input_scaler=self.expert_input_scaler,
                            disable_emb_translate=self.disable_emb_translate,
                            expert_type=self.expert_type, is_wine=self.is_wine)

    def forward(self, x, attention_mask=None):
        x = self.expert_encoder(x, attention_mask=attention_mask)
        if self.expert_input_scaler is not None:
            x = self.expert_input_scaler(x).to(torch.bfloat16)
        self.intermediate_result = x.clone()

        x = self.domain_expert(x)

        if self.expert_output_size == 1:
            if x.shape[-1] > 2:
                raise ValueError(f"GBDT output size mismatch! Expected 2, got {x.shape[-1]}")
            elif x.shape[-1] == 2:
                x = x[:, 1].unsqueeze(1)  # only keep the positive class probability

        if x.shape[-1] != self.expert_output_size:
            if self.is_wine:
                # Wine quality has 11 classes, but the GBDT model may output 7 probabilities since other labels are not present in the actual data.
                # To avoid this, we pad the output with zeros, 3 zeros in the front and 1 zero in the back.
                x = torch.cat([torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype), x, torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)
                assert x.shape[-1] == self.expert_output_size
            else:
                raise ValueError(f"GBDT output size mismatch! Expected {self.expert_output_size}, got {x.shape[-1]}")

        self.output = x.clone()
        x = self.expert_decoder(x)
        return x

    def forward_with_features(self, x):
        self.intermediate_result = x.clone()

        x = self.domain_expert(x)

        if self.expert_output_size == 1:
            if x.shape[-1] > 2:
                raise ValueError(f"GBDT output size mismatch! Expected 2, got {x.shape[-1]}")
            elif x.shape[-1] == 2:
                x = x[:, 1].unsqueeze(1)  # only keep the positive class probability

        if x.shape[-1] != self.expert_output_size:
            if self.is_wine:
                # Wine quality has 11 classes, but the GBDT model may output 7 probabilities since other labels are not present in the actual data.
                # To avoid this, we pad the output with zeros, 3 zeros in the front and 1 zero in the back.
                x = torch.cat([torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype), x,
                               torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)
                assert x.shape[-1] == self.expert_output_size
            else:
                raise ValueError(
                    f"GBDT output size mismatch! Expected {self.expert_output_size}, got {x.shape[-1]}")

        self.output = x.clone()
        x = self.expert_decoder(x)
        return x

    def get_expert_input(self, scale=True):
        return self.intermediate_result

    def get_expert_output(self):
        return self.output

    def setup_encoder(self, requires_grad):
        for param in self.expert_encoder.parameters():
            param.requires_grad = requires_grad

    def setup_decoder(self, requires_grad):
        for param in self.expert_decoder.parameters():
            param.requires_grad = requires_grad

    
