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

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_collate_fn_for_evaluation, get_dataset_for_evaluation
from src.model.DomainMistralModel import DomainMistralForCausalLM, EncoderModel
from src.analysis.TokenMonitor import DomainMistralTokenMonitor
from src.model.DomainExpert import DomainExpert
from src.utils.constants import MISTRAL_YES_TOKEN, MISTRAL_NO_TOKEN


def evaluate_domain_mistral_text(mistral_models_path, model_name, dataset_type, expert_dir, expert_description, encoder_dir,
                                 encoder_tokenizer_name, max_new_tokens, batch_size, n_instances, system_prompt, output_file,
                                 dataset_columns, baseline, gt_features):

    if expert_dir is None:
        expert_dir = f"model/experts/{dataset_type}_mlp.pth"

    if expert_description is None:
        expert_description = f"{dataset_type}-model prediction of the answer to the user's question"

    if encoder_dir is None:
        encoder_dir = f"model/llm/encoder_{dataset_type}"

    if encoder_tokenizer_name is None:
        encoder_tokenizer_name = "gpt2-large"

    if output_file is None:
        os.makedirs(f"data/{dataset_type}/res", exist_ok=True)
        output_file = f"data/{dataset_type}/res/domain_mistral_test.csv"

    data_file = f"data/{dataset_type}/text/{dataset_type}_test_text.csv"
    dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    encoder_model = AutoModelForCausalLM.from_pretrained(encoder_dir, cache_dir=mistral_models_path,
                                                         torch_dtype=torch.bfloat16)

    encoder_model = encoder_model.to(torch.bfloat16)
    encoder_model = encoder_model.to(torch.device('cuda'))

    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_name, cache_dir=mistral_models_path,
                                                      torch_dtype=torch.bfloat16)
    encoder_model.generation_config.pad_token_id = encoder_tokenizer.pad_token_id
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.unk_token
    encoder = EncoderModel(encoder_model, encoder_tokenizer, dataset_json, max_new_tokens=max_new_tokens)

    encoder.to(torch.bfloat16)
    encoder.to(torch.device('cuda'))

    yes_token_list = MISTRAL_YES_TOKEN
    no_token_list = MISTRAL_NO_TOKEN

    model = model.to(torch.bfloat16)
    model = model.to(torch.device('cuda'))

    # monitor = DomainMistralTokenMonitor(model, tokenizer)

    print("Loading the dataset...")

    dataset = get_dataset_for_evaluation(dataset_type, data_file, dataset_json=dataset_json, n_instances=n_instances)

    if not baseline:
        expert = torch.load(expert_dir, map_location='cuda')
        expert = expert.to(torch.bfloat16)
        custom_dataloader = DataLoader(dataset, batch_size=batch_size,
                                        collate_fn=get_collate_fn_for_evaluation(tokenizer, [(encoder, expert, expert_description)], system_prompt=system_prompt, gt_features=gt_features))

    else:
        custom_dataloader = DataLoader(dataset, batch_size=batch_size,
                                        collate_fn=get_collate_fn_for_evaluation(tokenizer, [], system_prompt=system_prompt, gt_features=gt_features))

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

            outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)

            logits = outputs.logits

            yes_logits = logits[:, -1, yes_token_list].mean(dim=-1)
            no_logits = logits[:, -1, no_token_list].mean(dim=-1)
            probs = torch.stack([no_logits, yes_logits], dim=-1).softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            # print(probs, preds)

            all_probs.extend(probs.cpu().numpy()[:, 1])  # Use probability for the positive class ("Yes")
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(all_labels, all_probs)
    # get the AUC-ROC score
    auc_roc = roc_auc_score(all_labels, all_probs)
    print(f"AUC-ROC Score on the test set: {auc_roc:.4f}")
    # get the accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    # get the F1 score
    f1 = f1_score(all_labels, all_preds)
    print(f"F1 Score on the test set: {f1:.4f}")
    print(len(all_labels), len(all_probs))

    # Save predictions and ground truth to a CSV file
    results_df = pd.DataFrame({
        'ground_truth': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    })
    # Also save the metrics
    results_df['auc_roc'] = auc_roc
    results_df['accuracy'] = accuracy
    results_df['f1'] = f1

    results_df.to_csv(output_file, index=False)
    print(f"Predictions and ground truth saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Domain Mistral model.')

    parser.add_argument('--mistral_models_path', type=str, default="model/llm", help='Path to the model cache')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help='Model name')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to the domain expert, None for default")
    parser.add_argument("--expert_description", type=str, default=None, help="Description of the domain expert")
    parser.add_argument("--encoder_dir", type=str, default=None, help="Path to the encoder model, None for default")
    parser.add_argument("--encoder_tokenizer", type=str, default=None,
                        help="Path to the encoder tokenizer, None for default")
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of tokens to generate for encoder model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_instances', type=int, default=None,
                        help='Number of instances to use for testing, None to use the whole dataset')
    parser.add_argument('--system_prompt', type=str, default="Respond the user's question in only one word: Yes or No.",
                        help='System prompt for the model')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions')
    parser.add_argument('--dataset_columns', type=str, default=None, help='Columns of the dataset')
    parser.add_argument('--baseline', action='store_true', help='Whether to use the baseline model')
    parser.add_argument('--gt_features', action='store_true', help='Whether to use ground truth features instead of the encoder model')

    args = parser.parse_args()

    evaluate_domain_mistral_text(args.mistral_models_path, args.model_name, args.dataset, args.expert_dir, args.expert_description,
                                 args.encoder_dir, args.encoder_tokenizer, args.max_new_tokens, args.batch_size, args.n_instances,
                                 args.system_prompt, args.output_file, args.dataset_columns, args.baseline, args.gt_features)

