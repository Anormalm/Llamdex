import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from tqdm.auto import tqdm
import numpy as np
from src.preprocess.DataScaler import TableScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

import argparse
import re
import json
import csv
import textwrap
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_dataset_for_evaluation


@torch.no_grad()
def evaluate_baseline(mistral_models_path, model_name, dataset_type, max_new_tokens, batch_size, n_instances, system_prompt, output_file,
                      expert_dir, seed, print_responses):

    if expert_dir is None:
        expert_dir = f"model/experts/{dataset_type}_mlp.pth"

    if output_file is None:
        os.makedirs(f"data/{dataset_type}/res", exist_ok=True)
        output_file = f"data/{dataset_type}/res/baseline_seed{seed}.csv"

    data_file = f"data/{dataset_type}/text/{dataset_type}_test_text.csv"
    dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left",
                                              cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=mistral_models_path, torch_dtype=torch.bfloat16)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    dataset = get_dataset_for_evaluation(dataset_type, data_file, dataset_json=dataset_json, n_instances=n_instances)

    print("Evaluating the model...")

    model = model.to(torch.device('cuda')).to(torch.bfloat16)
    model.eval()
    expert = torch.load(expert_dir, map_location='cuda')
    expert = expert.to(torch.bfloat16)
    device = torch.device('cuda')

    if isinstance(dataset_json, str):
        with open(dataset_json, 'r') as f:
            dataset_json = json.load(f)

    feature_columns = list(sorted(dataset_json['X'].keys()))

    instruction = f"You are a data analysis assistant who strictly adheres to instructions. You have access to a model. It receives input in the following json format:\n"
    instruction += json.dumps(dataset_json, indent=4)
    instruction += f"\nNow you should generate a query to answer the user's question. Use the following format to generate queries:\n"
    instruction += " ".join([f"{feature}: [value]" for feature in feature_columns]) + "\n"

    print("Instruction to model: ", instruction)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_expert_responses = []
    all_labels = []
    all_preds = []
    all_extracted_features = []
    all_probabilities = []

    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        texts = batch['text']
        labels = batch['labels']
        features = batch['features']

        prompts = [instruction + text for text in texts]
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        tokens = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True).to(device)
        attn_mask = (tokens != tokenizer.pad_token_id).long()

        generated_ids = model.generate(
            tokens,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        responses = tokenizer.batch_decode(generated_ids[:, tokens.size(1):], skip_special_tokens=True)

        if print_responses:
            for response in responses:
                print("Response: ", response)

        scaler = dataset.scaler

        def analyze_feature(feature_name, value):
            feature_info = dataset_json['X'].get(feature_name)
            if feature_info is None:
                return np.nan

            if feature_info['type'] == 'category':
                categories = feature_info['categories']
                if categories and isinstance(categories[0], (int, float)):
                    try:
                        converted_value = int(value) if isinstance(categories[0], int) else float(value)
                        return converted_value if converted_value in categories else np.nan
                    except ValueError:
                        return np.nan
                else:
                    return value if value in categories else np.nan

            elif feature_info['type'] in ['int', 'float']:
                try:
                    value = float(value)
                    min_val, max_val = feature_info['range']
                    if min_val <= value <= max_val:
                        return int(value) if feature_info['type'] == 'int' else value
                    else:
                        return np.nan
                except ValueError:
                    return np.nan
            else:
                return np.nan

        def get_extracted_features(response, scaler):
            features = {}
            for feature_name in feature_columns:
                pattern = rf'{re.escape(feature_name)}:\s*(\S+)'
                match = re.search(pattern, response)
                if match:
                    value = match.group(1)
                    features[feature_name] = analyze_feature(feature_name, value)
                else:
                    feature_info = dataset_json['X'].get(feature_name)
                    features[feature_name] = np.nan if feature_info['type'] == 'category' else 0

            feature_array = np.array([features[f] for f in feature_columns], dtype=object)
            raw_feature_array = feature_array.copy()
            # replace nan with 0 for numerical features
            for feature_name in feature_columns:
                if dataset_json['X'][feature_name]['type'] in ['int', 'float'] and np.isnan(features[feature_name]):
                    feature_array[feature_columns.index(feature_name)] = 0

            return raw_feature_array, torch.tensor(scaler.transform(feature_array.reshape(1, -1)))

        for text, label, feature, response in zip(texts, labels, features, responses):
            label_tensor = label.to(device)

            raw_features, extracted_features = get_extracted_features(response, scaler)
            extracted_features = extracted_features.to(device).to(torch.bfloat16)
            expert_response = expert(extracted_features).float().detach().cpu()

            if expert_response.shape[-1] == 1:
                prob = expert_response.item()
                pred = (expert_response > 0.5).int().item()
                all_probabilities.append(prob)
            else:
                pred = expert_response.argmax().item()
                all_probabilities.append(expert_response.numpy())

            all_expert_responses.append(expert_response.numpy())
            all_labels.append(label_tensor.cpu().numpy().item())
            all_preds.append(pred)
            all_extracted_features.append(raw_features)

            correct_predictions += (pred == label_tensor.item())
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on the test set: {accuracy:.4f}")

    if all_expert_responses[0].shape[-1] == 1:
        auc_roc = roc_auc_score(all_labels, all_probabilities)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"AUC-ROC Score on the test set: {auc_roc:.4f}")
        print(f"Weighted F1 Score on the test set: {f1:.4f}")

    csv_data = []
    for i in range(len(all_labels)):
        # print(list(all_extracted_features[i]), all_expert_responses[i].flatten().tolist(), [all_preds[i]], [all_labels[i]])
        row = list(all_extracted_features[i]) + all_expert_responses[i].flatten().tolist() + [all_preds[i]] + [all_labels[i]]
        csv_data.append(row)

    expert_response_cols = [f'expert_response_{i}' for i in range(all_expert_responses[0].size)]
    header = feature_columns + expert_response_cols + ['prediction', 'label']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Domain Mistral model.')

    parser.add_argument('--mistral_models_path', type=str, default="model/llm", help='Path to the model cache')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help='Model name')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate for encoder model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_instances', type=int, default=None,
                        help='Number of instances to use for testing, None to use the whole dataset')
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='System prompt for the model')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions')
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to the domain expert, None for default")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--print_responses", action='store_true', help="Print responses of the LLM")

    args = parser.parse_args()

    evaluate_baseline(args.mistral_models_path, args.model_name, args.dataset, args.max_new_tokens, args.batch_size,
                      args.n_instances, args.system_prompt, args.output_file, args.expert_dir,
                      args.seed, args.print_responses)

