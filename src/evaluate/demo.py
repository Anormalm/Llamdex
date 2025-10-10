import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import re

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_collate_fn, get_dataset
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.analysis.TokenMonitor import DomainMistralTokenMonitor
from src.model.DomainExpert import DomainExpert
from src.utils.constants import MISTRAL_YES_TOKEN, MISTRAL_NO_TOKEN


def evaluate_domain_mistral(mistral_models_path, model_name, dataset_type, expert_dir, encoder_dir,
                            encoder_tokenizer_name, ffn_hidden_size, num_tokens, num_heads, dropout, use_norm,
                            expert_input_size, expert_output_size, max_new_tokens, batch_size, n_instances,
                            system_prompt,
                            model_state_dict_dir, output_file, mapping_hidden_size,
                            expert_input_size_scaled, dataset_columns, tune_lora, use_baseline, layer_to_add,
                            direct_input):
    if expert_dir is None:
        expert_dir = f"model/experts/{dataset_type}_mlp.pth"

    if encoder_dir is None:
        # encoder_dir = f"model/llm/encoder_{dataset_type}"
        encoder_dir = "roberta-large"

    if encoder_tokenizer_name is None:
        encoder_tokenizer_name = "roberta-large"

    if model_state_dict_dir is None:
        model_state_dict_dir = f"model/llm/domain_mistral_{dataset_type}/model_final.pt"

    if output_file is None:
        os.makedirs(f"data/{dataset_type}/res", exist_ok=True)
        output_file = f"data/{dataset_type}/res/domain_mistral_test.csv"

    data_file = f"data/{dataset_type}/text/{dataset_type}_test_text.csv"
    dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    if use_baseline:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    else:
        model = DomainMistralForCausalLM.from_pretrained_mistral(model_name,
                                                                 cache_dir=mistral_models_path,
                                                                 torch_dtype=torch.bfloat16)

    chat_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    chat_model_tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                         cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    chat_model.generation_config.pad_token_id = chat_model_tokenizer.pad_token_id
    if chat_model_tokenizer.pad_token is None:
        chat_model_tokenizer.pad_token = chat_model_tokenizer.unk_token

    embed_size = model.config.hidden_size

    yes_token_list = MISTRAL_YES_TOKEN
    no_token_list = MISTRAL_NO_TOKEN

    if not use_baseline:
        model.num_tokens = num_tokens

        print("Adding the custom expert to the model...")

        for p in model.parameters():
            p.requires_grad = False

        if tune_lora:
            config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )

            model = get_peft_model(model, config)
            base_model = model.base_model.model
        else:
            base_model = model

        if layer_to_add is None:
            layer_to_add = list(range(len(base_model.model.layers)))

        embed_pad = base_model.model.embed_tokens(torch.tensor([tokenizer.pad_token_id]))[0]
        embed_yes = base_model.model.embed_tokens(torch.tensor(yes_token_list))[0]
        embed_no = base_model.model.embed_tokens(torch.tensor(no_token_list))[0]

        lm_head = model.get_output_embeddings()
        norm = model.model.norm

        expert = DomainExpert(embed_size, ffn_hidden_size, expert_input_size, expert_output_size, expert_dir, embed_yes,
                              embed_no, embed_pad, tokenizer, lm_head, norm, num_heads,
                              dropout, use_norm, max_new_tokens, dataset_json, None,
                              None, mapping_hidden_size, num_tokens, mistral_models_path, encoder_dir)

        base_model.add_expert_(expert, [(layer if layer >= 0 else len(base_model.model.layers) + layer) for layer in
                                        layer_to_add])

        print("Loading the model state dict...")

        model.load_state_dict(torch.load(model_state_dict_dir, map_location='cpu'))

        if tune_lora:
            model.base_model.model.replace_expert_(expert_dir)
        else:
            model.replace_expert_(expert_dir)

    model = model.to(torch.bfloat16)
    model = model.to(torch.device('cuda'))

    chat_model = chat_model.to(torch.bfloat16)
    chat_model = chat_model.to(torch.device('cuda'))

    model.eval()
    chat_model.eval()

    with torch.no_grad():
        msg_text = '''
            You are a data analysis assistant with access to a Titanic survival prediction model. Your task:

            Step 1 (Current): Generate multiple queries for survival rate predictions. Each query should include:
            - Age
            - Fare
            - Parents/Children Aboard
            - Ticket Class
            - Sex
            - Siblings/Spouses Aboard
            
            Enclose each query in <<>>. Example:
            <<This passenger was a female traveling on the Titanic. She had two parents or children aboard the ship, as well as two siblings or a spouse. Her ticket was purchased in the Third class, and she paid a fare of $34.375. The age of this passenger was 21 years old. Did she survive the tragic sinking of the Titanic?>>
            
            Step 2 (Next round): Analyze results and answer the question.

            Rules:
            1. Enclose each query in <<>>.
            2. Wrap each query separately. Like this: 
                <<1. ...>> 
                <<2. ...>>
                ...
            3. Ensure diverse and sufficient sample size for analysis.
            4. Do not assume or infer any results in the first step.
            
            Question: Which gender has a higher survival rate on Titanic?
        '''

        messages = [{"role": "user", "content": msg_text}]
        tokens = chat_model_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        attn_mask = (tokens != chat_model_tokenizer.pad_token_id).long()

        generated_ids = chat_model.generate(tokens, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=chat_model_tokenizer.pad_token_id, eos_token_id=chat_model_tokenizer.eos_token_id)

        result = chat_model_tokenizer.decode(generated_ids[0][tokens.size(1):-1].tolist())
        messages.append({"role": "assistant", "content": result})

        def extract_bracketed_strings(text):
            pattern = r'<<(.*?)>>'
            matches = re.findall(pattern, text, re.DOTALL)
            return matches

        bracketed_strings = extract_bracketed_strings(result)
        survival_rates = []

        for string in bracketed_strings:
            if system_prompt is None:
                model_message = [{"role": "user", "content": string}]
            else:
                model_message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": string}]

            model_tokens = tokenizer.apply_chat_template(model_message, return_tensors="pt").to("cuda")
            model_attn_mask = (model_tokens != tokenizer.pad_token_id).long()

            outputs = model(model_tokens, attention_mask=model_attn_mask, past_key_values=None, use_cache=False)

            logits = outputs.logits
            yes_logits = logits[:, -1, yes_token_list].mean(dim=-1)
            no_logits = logits[:, -1, no_token_list].mean(dim=-1)
            probs = torch.stack([no_logits, yes_logits], dim=-1).softmax(dim=-1)
            survival_rates.append(probs[:, 1].item())

        messages.append({"role": "user", "content": "The survival rates are: " + ", ".join([str(round(rate, 4)) for rate in survival_rates]) + ". Now proceed and give your conclusion."})

        tokens = chat_model_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        attn_mask = (tokens != chat_model_tokenizer.pad_token_id).long()

        generated_ids = chat_model.generate(tokens, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=chat_model_tokenizer.pad_token_id, eos_token_id=chat_model_tokenizer.eos_token_id)

        result = chat_model_tokenizer.decode(generated_ids[0][tokens.size(1):-1].tolist())
        messages.append({"role": "assistant", "content": result})

        print(result)

        for item in messages:
            print(f"{item['role']}: {item['content']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Domain Mistral model.')

    parser.add_argument('--mistral_models_path', type=str, default="model/llm", help='Path to the model cache')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help='Model name')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to the domain expert, None for default")
    parser.add_argument("--encoder_dir", type=str, default=None, help="Path to the encoder model, None for default")
    parser.add_argument("--encoder_tokenizer", type=str, default=None,
                        help="Path to the encoder tokenizer, None for default")
    parser.add_argument('--ffn_hidden_size', type=int, default=512,
                        help='Hidden size of the feed-forward neural network in the Domain Expert')
    parser.add_argument('--num_tokens', type=int, default=10, help='Number of tokens that expert appends to the input')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of heads in the multi-head self-attention layer in the MappingBlock')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in the MappingBlock')
    parser.add_argument('--use_norm', type=bool, default=True,
                        help='Whether to use layer normalization after attention in the MappingBlock')
    parser.add_argument('--expert_input_size', type=int, default=9, help='Input size of the Domain Expert')
    parser.add_argument('--expert_output_size', type=int, default=1, help='Output size of the Domain Expert')
    parser.add_argument('--max_new_tokens', type=int, default=2000, help='Maximum number of tokens to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_instances', type=int, default=None,
                        help='Number of instances to use for testing, None to use the whole dataset')
    parser.add_argument('--system_prompt', type=str, default="Respond the user's question in only one word: Yes or No.",
                        help='System prompt for the model')
    parser.add_argument('--model_state_dict_dir', type=str, default=None, help='Path to the model state dict')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions')
    parser.add_argument('--mapping_hidden_size', type=int, default=32, help='Hidden size for the mapping')
    parser.add_argument('--expert_input_size_scaled', type=int, default=None,
                        help='Scaled input size of the Domain Expert')
    parser.add_argument('--dataset_columns', type=str, default=None, help='Columns of the dataset')
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--baseline', action='store_true', help='Whether to use the baseline model')
    parser.add_argument("--layer", type=int, nargs='*', default=None,
                        help="Layers to add the expert, negative indexing supported")
    parser.add_argument("--direct_input", action='store_true',
                        help="Whether to directly use the ground truth features as input to the expert")

    args = parser.parse_args()

    evaluate_domain_mistral(args.mistral_models_path, args.model_name, args.dataset, args.expert_dir, args.encoder_dir,
                            args.encoder_tokenizer, args.ffn_hidden_size, args.num_tokens, args.num_heads,
                            args.dropout, args.use_norm, args.expert_input_size, args.expert_output_size,
                            args.max_new_tokens,
                            args.batch_size, args.n_instances, args.system_prompt, args.model_state_dict_dir,
                            args.output_file, args.mapping_hidden_size, args.expert_input_size_scaled,
                            args.dataset_columns, args.tune_lora, args.baseline, args.layer, args.direct_input)

