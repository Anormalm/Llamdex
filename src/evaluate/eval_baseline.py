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
import re
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_collate_fn_for_evaluation, get_dataset_for_evaluation
from src.model.DomainMistralModel import DomainMistralForCausalLM, EncoderModel
from src.analysis.TokenMonitor import DomainMistralTokenMonitor
from src.model.DomainExpert import DomainExpert
from src.utils.constants import MISTRAL_YES_TOKEN, MISTRAL_NO_TOKEN
from src.preprocess.DataScaler import TableScaler



def evaluate_baseline(mistral_models_path, model_name, dataset_type, max_new_tokens, batch_size, n_instances, system_prompt, output_file,
                      dataset_columns, expert_dir, instruction):

    if expert_dir is None:
        expert_dir = f"model/experts/{dataset_type}_mlp.pth"

    if output_file is None:
        os.makedirs(f"data/{dataset_type}/res", exist_ok=True)
        output_file = f"data/{dataset_type}/res/baseline.csv"

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

    dataset = get_dataset_for_evaluation(dataset_type, data_file, dataset_json=dataset_json, n_instances=n_instances)

    print("Evaluating the model...")

    yes_token_list = MISTRAL_YES_TOKEN
    no_token_list = MISTRAL_NO_TOKEN

    model = model.to(torch.device('cuda')).to(torch.bfloat16)
    model.eval()
    expert = torch.load(expert_dir, map_location='cuda')
    expert = expert.to(torch.bfloat16)

    if isinstance(dataset_json, str):
        with open(dataset_json, 'r') as f:
            dataset_json = json.load(f)

    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating"):
            labels = torch.tensor([item["labels"]], dtype=torch.long).to(torch.device('cuda'))
            prompt = instruction + item["text"]
            messages = [{"role": "user", "content": prompt}]
            tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            attn_mask = (tokens != tokenizer.pad_token_id).long()

            generated_ids = model.generate(tokens, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

            result = tokenizer.decode(generated_ids[0][tokens.size(1):].tolist())
            messages.append({"role": "assistant", "content": result})

            def extract_bracketed_strings(text):
                pattern = r'<<(.*?)>>'
                matches = re.findall(pattern, text, re.DOTALL)
                return matches

            bracketed_strings = extract_bracketed_strings(result)
            expert_responses = []
            for string in bracketed_strings:
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
                                return 0
                        except ValueError:
                            return 0
                    else:
                        return np.nan

                def extracted_features(response, scaler):
                    features = {}
                    for feature_name in dataset_json['X']:
                        pattern = rf'{re.escape(feature_name)}:\s*(\S+)'
                        match = re.search(pattern, response)
                        if match:
                            value = match.group(1)
                            features[feature_name] = analyze_feature(feature_name, value)
                        else:
                            feature_info = dataset_json['X'].get(feature_name)
                            features[feature_name] = np.nan if feature_info['type'] == 'category' else 0

                    feature_names = list(dataset_json['X'].keys())
                    feature_array = np.array([features[f] for f in feature_names], dtype=object)
                    feature_array = feature_array.reshape(1, -1)

                    return torch.tensor(scaler.transform(feature_array))

                scaler = TableScaler(dataset_json)
                features = extracted_features(string, scaler)
                features = features.to(torch.device('cuda')).to(torch.bfloat16)
                expert_response = expert(features).detach().cpu().item()
                expert_responses.append(expert_response)

            messages.append({"role": "user", "content": f"The predicted survival rate of your queries are: {expert_responses}, next, please answer in only one word: 'Yes' or 'No'."})
            print(messages)
            tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            attn_mask = (tokens != tokenizer.pad_token_id).long()

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
    parser.add_argument('--max_new_tokens', type=int, default=1000, help='Maximum number of tokens to generate for encoder model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_instances', type=int, default=None,
                        help='Number of instances to use for testing, None to use the whole dataset')
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='System prompt for the model')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions')
    parser.add_argument('--dataset_columns', type=str, default=None, help='Columns of the dataset')
    parser.add_argument("--expert_dir", type=str, default=None, help="Path to the domain expert, None for default")

    args = parser.parse_args()

    instruction = """
    You are a data analysis assistant who strictly adheres to instructions. You have access to a Titanic passenger survival prediction model. It receives input in the following json format:

    {
        "X": {
            "Age": {
                "type": "int",
                "range": [0, 100]
            },
            "Fare": {
                "type": "float",
                "range": [0, 600]
            },
            "Parents/Children Aboard": {
                "type": "int",
                "range": [0, 10]
            },
            "Pclass": {
                "type": "category",
                "categories": [1, 2, 3]
            },
            "Sex": {
                "type": "category",
                "categories": ["male", "female"]
            },
            "Siblings/Spouses Aboard": {
                "type": "int",
                "range": [0, 10]
            }
        },
        "y": {
            "name": "Survived",
            "type": "bool"
        }
    }

    Your task is divided into three steps:

    Step 1 (this round): Generate one query to answer the user's question. Use the following format to generate queries:
    <<Age: [value] Fare: [value] Parents/Children Aboard: [value] Pclass: [value] Sex: [value] Siblings/Spouses Aboard: [value]>>
    Like this:
    <<Age: 20 Fare: 50 Parents/Children Aboard: 0 Pclass: 1 Sex: male Siblings/Spouses Aboard: 1>>

    Step 2 (this round): After generating the query, explicitly state that you are waiting for results:
    "I have generated the above queries and am waiting for the actual results in the next round of dialogue. I will not make any conclusions or assumptions until I receive the results."

    Step 3 (next round): In the next round of dialogue, I will provide the results for your queries. Then you should answer the user's question in only one word: 'Yes' or 'No'.

    Important notes:
    - In this round of dialogue, do not assume, generate, or infer any results.
    - Make only one query.
    - Do not provide any information or conclusions other than generating the query and the waiting statement.
    - Use <<>> symbols to enclose each query
    - Provide specific values for ALL fields. Do not use "Any" or ranges or skip any term.
    - In the next round, you will answer the user's question in only one word: 'Yes' or 'No'.

    Now, please address the user's question:
    """

    evaluate_baseline(args.mistral_models_path, args.model_name, args.dataset, args.max_new_tokens, args.batch_size,
                      args.n_instances, args.system_prompt, args.output_file, args.dataset_columns, args.expert_dir, instruction)

