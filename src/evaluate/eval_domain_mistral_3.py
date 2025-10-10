import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor, PreTrainedModel
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from typing import List, Optional, Tuple, Union
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from peft import LoraConfig, get_peft_model
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import deepspeed
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.adult.InstructTextDataset import AdultTextDataset, get_collate_fn
from src.model.DomainMistralModel import DomainMistralForCausalLM
from src.analysis.TokenMonitor import DomainMistralTokenMonitor
from src.model.DomainExpert import DomainExpert
from src.utils.constants import MISTRAL_YES_TOKEN, MISTRAL_NO_TOKEN

def evaluate_domain_mistral(mistral_models_path, model_name, expert_dir, ffn_hidden_size, num_heads, dropout, use_norm,
                            expert_input_size, expert_output_size, max_new_tokens, batch_size, n_instances, system_prompt,
                            model_state_dict_dir, data_file, output_file, dataset_json, mapping_hidden_size,
                            expert_input_size_scaled, dataset_columns, tune_lora, layer_to_add):

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                        cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    model = DomainMistralForCausalLM.from_pretrained_mistral(model_name,
                                                             cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    embed_size = model.config.hidden_size

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
        layer_to_add = list(range(len(model.base_model.model.model.layers)))

    # Get the token ids for "Yes" and "No"
    yes_token_list = MISTRAL_YES_TOKEN
    no_token_list = MISTRAL_NO_TOKEN

    embed_yes = base_model.model.embed_tokens(torch.tensor(yes_token_list))[0]
    embed_no = base_model.model.embed_tokens(torch.tensor(no_token_list))[0]

    expert = DomainExpert(embed_size, ffn_hidden_size, expert_input_size, expert_output_size, expert_dir, embed_yes, embed_no,
                          num_heads, dropout, use_norm, max_new_tokens, dataset_json, dataset_columns,
                          expert_input_size_scaled, mapping_hidden_size)

    base_model.add_expert_(expert, [(layer if layer >= 0 else len(base_model.model.layers) + layer) for layer in layer_to_add])
    
    model.load_state_dict(torch.load(model_state_dict_dir, map_location='cpu'))

    if tune_lora:
        model.base_model.model.replace_expert_(expert_dir)
    else:
        model.replace_expert_(expert_dir)
    
    model = model.to(torch.bfloat16)
    model = model.to(torch.device('cuda'))

    monitor = DomainMistralTokenMonitor(model, tokenizer)

    print("Loading the dataset...")
    if n_instances is not None:
        dataset = AdultTextDataset(pd.read_csv(data_file).head(n_instances), tokenizer, 
                                         system_prompt=system_prompt, dataset_json=dataset_json, add_features=True)
    else:
        # Use the whole dataset for training
        dataset = AdultTextDataset(pd.read_csv(data_file), tokenizer, 
                                         system_prompt=system_prompt, dataset_json=dataset_json, add_features=True)
    
    custom_dataloader = DataLoader(dataset, batch_size=batch_size,
                                   collate_fn=get_collate_fn(tokenizer.pad_token_id, add_features=True))
    
    print("Testing the model...")

    torch.cuda.empty_cache()

    model.eval()

    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(custom_dataloader, desc="Evaluating"):
            tokens = batch['tokens'].to(torch.device('cuda'))
            attn_mask = batch['attention_mask'].to(torch.device('cuda'))
            labels = batch['labels'].to(torch.device('cuda'))
            features = batch['features'].to(torch.device('cuda'))

            outputs = model(tokens, expert_input=(features,), attention_mask=attn_mask, past_key_values=None, use_cache=False)
            logits = outputs.logits

            # # debug
            # generated_ids = model.generate(tokens, max_new_tokens=10, do_sample=False, temperature=0.1,
            #                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            # generated_tokens = tokenizer.batch_decode(generated_ids[:, tokens.size(1): ], skip_special_tokens=False)
            # print(f"Generated tokens: {generated_tokens}")
            # print(f"Generated token ids: {generated_ids[:, tokens.size(1): ]}")
            # # end debug
            '''

            auxiliary_loss = 0

            for layer in layer_to_add:
                if tune_lora:
                    expert_input = model.base_model.model.get_expert_input(layer, 0).to(torch.bfloat16)
                else:
                    expert_input = model.get_expert_input(layer, 0).to(torch.bfloat16)
                """
                expert_output = model.get_expert_output(layer, 0).to(torch.bfloat16)
                values, indices = model.lm_head(model.model.norm(expert_output[0, -1, :])).topk(20)
                decoded = [tokenizer.decode(idx) for idx in indices]
                print(decoded)
                """

                features = features.to(torch.bfloat16).to(expert_input.device)
                auxiliary_loss += nn.MSELoss()(expert_input, features).to(torch.bfloat16)
            '''

            # print(auxiliary_loss.item())
            # print(features[0], expert_input[0])

            yes_logits = logits[:, -1, yes_token_list].mean(dim=-1)
            no_logits = logits[:, -1, no_token_list].mean(dim=-1)
            probs = torch.stack([no_logits, yes_logits], dim=-1).softmax(dim=-1)
            preds = probs.argmax(dim=-1)

            all_probs.extend(probs.cpu().numpy()[:, 1])  # Use probability for the positive class ("Yes")
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(all_preds, all_labels)
    auc_roc = roc_auc_score(all_labels, all_probs)
    print(f"AUC-ROC Score on the test set: {auc_roc:.4f}")

    # Save predictions and ground truth to a CSV file
    results_df = pd.DataFrame({
        'ground_truth': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    })
    results_df.to_csv(output_file, index=False)
    print(f"Predictions and ground truth saved to {output_file}")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Evaluate the Domain Mistral model on the Adult dataset.')

    parser.add_argument('--mistral_models_path', type=str, default="model/llm", help='Path to the model cache')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help='Model name')
    parser.add_argument('--expert_dir', type=str, default="model/experts/adult_mlp.pth", help='Path to the domain expert')
    parser.add_argument('--ffn_hidden_size', type=int, default=14336, help='Hidden size of the feed-forward neural network in the Domain Expert')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the multi-head self-attention layer in the MappingBlock')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the MappingBlock')
    parser.add_argument('--use_norm', type=bool, default=True, help='Whether to use layer normalization after attention in the MappingBlock')
    parser.add_argument('--expert_input_size', type=int, default=105, help='Input size of the Domain Expert')
    parser.add_argument('--expert_output_size', type=int, default=1, help='Output size of the Domain Expert')
    parser.add_argument('--max_new_tokens', type=int, default=300, help='Maximum number of tokens to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_instances', type=int, default=None, help='Number of instances to use for testing, None to use the whole dataset')
    parser.add_argument('--system_prompt', type=str, default=None, help='System prompt for the model')
    parser.add_argument('--model_state_dict_dir', type=str, default="model/llm/domain_single_token/model_final.pt", help='Path to the model state dict')
    parser.add_argument('--data_file', type=str, default="data/adult/text/adult_test_text.csv", help='Path to the test data file')
    parser.add_argument('--output_file', type=str, default="data/adult/res/domain_mistral_test.csv", help='Path to save predictions')
    parser.add_argument('--dataset_json', type=str, default="src/dataset/adult/adult.json", help='Path to the dataset JSON file')
    parser.add_argument('--mapping_hidden_size', type=int, default=32, help='Hidden size for the mapping')
    parser.add_argument('--expert_input_size_scaled', type=int, default=None, help='Scaled input size of the Domain Expert')
    parser.add_argument('--dataset_columns', type=str, default=None, help='Columns of the dataset')
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument("--layer", type=int, nargs='*', default=None, help="Layers to add the expert, negative indexing supported")

    args = parser.parse_args()

    evaluate_domain_mistral(args.mistral_models_path, args.model_name, args.expert_dir, args.ffn_hidden_size, args.num_heads,
                            args.dropout, args.use_norm, args.expert_input_size, args.expert_output_size, args.max_new_tokens,
                            args.batch_size, args.n_instances, args.system_prompt, args.model_state_dict_dir, args.data_file,
                            args.output_file, args.dataset_json, args.mapping_hidden_size, args.expert_input_size_scaled,
                            args.dataset_columns, args.tune_lora, args.layer)

    