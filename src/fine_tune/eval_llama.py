import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

import argparse
import json

from src.preprocess.DataScaler import TableScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.utils import get_collate_fn, get_dataset, get_class_tokens


@torch.no_grad()
def evaluate_llama(llama_models_path, model_name, dataset_type, system_prompt, model_state_dict_dir, batch_size,
                   n_instances, output_file, tune_lora, dp_type, epsilon, load_lora_only):

    if load_lora_only and not tune_lora:
        raise ValueError("LoRA can't be loaded if not using it.")

    if model_state_dict_dir is None:
        if dp_type is None:
            if not load_lora_only:
                model_state_dict_dir = f"model/llm/exp-lora-real_llama_{dataset_type}/model_final.pt"
            else:
                model_state_dict_dir = f"model/llm/exp-lora-real_llama_{dataset_type}"
        else:
            if not load_lora_only:
                model_state_dict_dir = f"model/llm/exp-lora-syn_llama_{dataset_type}_{dp_type}_{epsilon}/model_final.pt"
            else:
                model_state_dict_dir = f"model/llm/exp-lora-syn_llama_{dataset_type}_{dp_type}_{epsilon}"

    if output_file is None:
        os.makedirs(f"data/{dataset_type}/res", exist_ok=True)
        output_file = f"data/{dataset_type}/res/fine_tune_llama_test.csv"

    data_file = f"data/{dataset_type}/text/{dataset_type}_test_text.csv"
    dataset_json = f"src/dataset/{dataset_type}/{dataset_type}.json"
    dataset_info_json = f"src/dataset/dataset_info.json"
    with open(dataset_info_json, 'r') as f:
        dataset_info_json = json.load(f)

    if dataset_type not in dataset_info_json:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if system_prompt is None:
        system_prompt = dataset_info_json[dataset_type]['system_prompt']


    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=llama_models_path, torch_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=llama_models_path, torch_dtype=torch.bfloat16)

    # to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    embed_size = model.config.hidden_size

    class_tokens = get_class_tokens(dataset_type)

    print("Loading the model state dict...")

    if not load_lora_only:
        if tune_lora:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
            )

            model = get_peft_model(model, config)

        model.load_state_dict(torch.load(model_state_dict_dir, map_location='cpu'))
    else:
        model = PeftModel.from_pretrained(model, model_state_dict_dir)

    model = model.to(torch.bfloat16)
    model = model.to(torch.device('cuda'))

    print("Loading the dataset...")

    dataset = get_dataset(dataset_type, data_file, tokenizer=tokenizer,
                          dataset_json=dataset_json, n_instances=n_instances, system_prompt=system_prompt)

    custom_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=get_collate_fn(tokenizer.pad_token_id))

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
            features = batch['features'].to(torch.device('cuda')).to(torch.bfloat16)

            outputs = model(tokens, attention_mask=attn_mask, past_key_values=None, use_cache=False)

            logits = outputs.logits
            class_logits = logits[:, -1, class_tokens]
            probs = class_logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    print(all_preds)

    num_classes = len(class_tokens)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy on the test set: {accuracy:.4f}")

    results_df = pd.DataFrame({
        'ground_truth': all_labels,
        'predictions': all_preds,
    })
    if num_classes == 2:
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Weighted F1 Score on the test set: {f1:.4f}")

        y_bin = label_binarize(all_labels, classes=range(num_classes))
        if num_classes == 2:
            auc_roc = roc_auc_score(y_bin, all_probs[:, 1])
        else:
            auc_roc = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')
        print(f"AUC-ROC Score on the test set: {auc_roc:.4f}")

        results_df['weighted_f1'] = f1
        results_df['auc_roc'] = auc_roc

    for i in range(num_classes):
        results_df[f'prob_class_{i}'] = all_probs[:, i]

    results_df['accuracy'] = accuracy

    results_df.to_csv(output_file, index=False)
    print(f"Predictions, probabilities, and ground truth saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Domain llama model.')

    parser.add_argument("--llama_models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument('--system_prompt', type=str, default=None, help='System prompt for the model')
    parser.add_argument("--model_state_dict_dir", type=str, default=None, help="Path to save the model, None for default")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--n_instances", type=int, default=None, help="Number of instances to evaluate")
    parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions')
    parser.add_argument('-lora', '--tune_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--dp_type', type=str, default=None, help='Differential privacy type')
    parser.add_argument('--epsilon', type=str, default=None, help='Epsilon for DP')
    parser.add_argument('--load_lora_only', action='store_true', help='Load only the LoRA part of the model')


    args = parser.parse_args()

    print(args)

    evaluate_llama(args.llama_models_path, args.model_name, args.dataset, args.system_prompt,
                   args.model_state_dict_dir, args.batch_size,
                   args.n_instances, args.output_file, args.tune_lora, args.dp_type, args.epsilon, args.load_lora_only)



