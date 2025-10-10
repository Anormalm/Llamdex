import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data using PromptPATE')

    parser.add_argument("--mistral_models_path", type=str, default="model/llm", help="Path to the model cache")
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-v0.1",
                        help='Model name')
    parser.add_argument('--epsilon', type=float, nargs='+', required=True,
                        help='Privacy parameter epsilon')
    parser.add_argument('--delta', type=str, default='auto',
                        help='Privacy parameter delta. Use "auto" for 1/n, or specify a float value')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to generate')
    parser.add_argument('--n_teachers', type=int, default=50,
                        help='Number of teacher models')
    # parser.add_argument('--output_dir', type=str, default=None,
    #                     help='Output directory for synthetic data, default is None')
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='Confidence threshold for GNMax algorithm')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--n_instances', type=int, default=None,
                        help='Truncate to number of synthetic instances to generate')
    return parser.parse_args()

def get_delta(args, n):
    if args.delta == 'auto':
        return 1/n
    return float(args.delta)

def format_number(value):
    if isinstance(value, (int, float)):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip('0').rstrip('.')
    return str(value)

def create_instance_prompt(row, answer_column=None):
    prompt = ""
    for col in row.index:
        if col != answer_column:
            prompt += f"{col}: {row[col]}\n"
    if answer_column and answer_column in row:
        prompt += f"{answer_column}: {format_number(row[answer_column])}"
    return prompt


def batch_confident_gnmax(votes_batch, threshold, sigma1, sigma2, n_classes):
    class_votes = np.sum(votes_batch, axis=1) 

    noisy_max = np.max(class_votes, axis=1) + np.random.normal(0, sigma1, size=len(class_votes))
    
    confident_mask = noisy_max >= threshold
    
    results = np.full(len(votes_batch), None, dtype=object)
    
    if np.any(confident_mask):
        confident_votes = class_votes[confident_mask]
        noisy_votes = confident_votes + np.random.normal(0, sigma2, size=confident_votes.shape)
        results[confident_mask] = np.argmax(noisy_votes, axis=1)
    
    return results

def create_batch_prompts(instances_batch, context_instances, config):
    prompts = []
    target_column = config['y']['name']
    
    context_text = ""
    for _, instance in context_instances.iterrows():
        context_text += create_instance_prompt(instance) + "\n"
        context_text += f"{target_column}: {format_number(instance[target_column])}\n\n"
    
    for _, instance in instances_batch.iterrows():
        prompt = context_text + create_instance_prompt(instance)
        prompt += f"\n{target_column}:"
        prompts.append(prompt)
    
    return prompts

def generate_synthetic_data(train_df, test_df, config, args, num_classes):
    delta = args.delta
    print(f"Using delta = {delta} (1/n where n={len(train_df)})")
    
    epsilons = args.epsilon
    sigmas = []
    for epsilon in epsilons:
        epsilon = float(epsilon)
        sigma = np.sqrt(2 * np.log(1.25/delta)) / (epsilon * 0.5)
        sigmas.append(sigma)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                            cache_dir=args.mistral_models_path, 
                                            torch_dtype=torch.bfloat16,
                                            padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                cache_dir=args.mistral_models_path, 
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto")
    
    if tokenizer._pad_token is None:
        tokenizer._pad_token = tokenizer.unk_token
    
    model.eval()
    device = model.device

    class_token_ids = [tokenizer.encode(str(i))[0] for i in range(num_classes)]
    
    synthetic_data_list = [[] for _ in range(len(epsilons))]
    target_column = config['y']['name']
    
    batch_size = args.batch_size
    n_batches = (len(train_df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Generating synthetic data"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_df))
        target_instances = train_df.iloc[start_idx:end_idx]
        
        all_teacher_votes = np.zeros((len(target_instances), args.n_teachers, num_classes))
        
        for teacher_idx in range(args.n_teachers):
            context_instances = test_df.sample(n=1)
            
            prompts = create_batch_prompts(target_instances, context_instances, config)
            
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding='longest',
                truncation=False,
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :] 
                probs = torch.softmax(logits, dim=-1)
                
                class_logits = logits[:, class_token_ids]
                probs = torch.softmax(class_logits, dim=-1)
                
                votes = torch.argmax(probs, dim=1).cpu().numpy()
                all_teacher_votes[:, teacher_idx, :] = np.eye(num_classes)[votes]
        

        for i, sigma in enumerate(sigmas):
            aggregated_indices = batch_confident_gnmax(
                all_teacher_votes,
                args.threshold,
                sigma,
                sigma,
                num_classes
            )

            for instance_idx, class_idx in enumerate(aggregated_indices):
                if class_idx is not None:
                    synthetic_row = target_instances.iloc[instance_idx:instance_idx+1].copy()
                    synthetic_row[target_column] = float(class_idx)
                    synthetic_data_list[i].append(synthetic_row)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = []
    for i, synthetic_data in enumerate(synthetic_data_list):
        if not synthetic_data:
            print(f"Warning: No synthetic data generated for epsilon={epsilons[i]}")
            results.append(None)
        else:
            results.append(pd.concat(synthetic_data, ignore_index=True))
            print(f"Generated {len(results[-1])} samples for epsilon={epsilons[i]}")

    return results

def main():
    args = parse_args()

    output_dirs = [f"data/{args.dataset}/dpsyn/eps{epsilon}/prompt-pate" for epsilon in args.epsilon]

    if args.dataset not in ['nursery', 'titanic', 'bank_marketing', 'wine_quality']:
        raise ValueError(f"Invalid dataset type: {args.dataset}")
    
    for output_dir in output_dirs:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading configuration and data...")
    config = f"src/dataset/{args.dataset}/{args.dataset}.json"

    with open(config, 'r') as f:
        config = json.load(f)

    num_classes = 2 if config['y']['type'] == "bool" else len(config['y']['categories'])
    
    print("Generating synthetic data...")
    # Actually "train_df" means the unlabeled public data and test_df is used for prompt-pate training
    train_df = pd.read_csv(f"data/{args.dataset}/clean/syn_{args.dataset}_train_onehot.csv")
    test_df = pd.read_csv(f"data/{args.dataset}/clean/{args.dataset}_train_onehot.csv")

    if args.n_instances is not None:
        train_df = train_df.sample(n=args.n_instances)
    else:
        # sample same number of instances as real training data
        if len(train_df) > len(test_df):
            train_df = train_df.sample(n=len(test_df))

    args.delta = get_delta(args, len(train_df))
    print(f"Dataset size: {len(train_df)}")
    print(f"Using epsilon = {args.epsilon}, delta = {args.delta}")
    
    results = generate_synthetic_data(train_df, test_df, config, args, num_classes)
    
    output_paths = [f"{output_dir}/synthetic_data.csv" for output_dir in output_dirs]

    for i, synthetic_data in enumerate(results):
        if synthetic_data is not None:
            synthetic_data.to_csv(output_paths[i], index=False)
    
    print(f"Synthetic data saved to {output_paths}")
            

if __name__ == "__main__":
    main()