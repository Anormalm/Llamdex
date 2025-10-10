import os
import re
import numpy as np
import argparse
from collections import defaultdict

def process_file(file_path, dataset, dataset_accuracies):
    with open(file_path, 'r') as file:
        content = file.read()
        accuracy_matches = re.findall(r'Accuracy on the test set: ([\d.]+)', content)
        if len(accuracy_matches) > 1:
            raise Exception(f"Multiple accuracy matches found in file: {file_path}")
        elif len(accuracy_matches) == 1:
            accuracy = float(accuracy_matches[0])
            dataset_accuracies[dataset].append(accuracy)
        else:
            print(f"Warning: No accuracy match found in file: {file_path}")

def main(args):
    if args.type == 'lora_real':
        log_dir = '_log/log_add_exp/lora_real'
    # elif args.type == 'lora_syn_prompt-pate':
    #     log_dir = '_log/log_add_exp/lora_syn_prompt-pate'
    # elif args.type == 'lora_syn_seqpate':
    #     log_dir = '_log/log_add_exp/lora_syn_seqpate'
    elif args.type in ['llamdex', 'llamdex_zeropad', 'llamdex_nontranslate', 'llamdex_gbdt', 'llamdex_mlp_to_gbdt']:
        # log_dir = '_log/log_add_exp/llamdex'
        log_dir = 'log/llamdex'
    # elif args.type == 'ablation_token':
    #     log_dir = '_log/log_add_exp/ablation_token'
    elif args.type == 'original_mistral':
        log_dir = '_log/log_add_exp/original_mistral'
    elif args.type == 'short_prompt_baseline':
        log_dir = '_log/log_add_exp/short_prompt_baseline'
    else:
        raise ValueError(f"Invalid type: {args.type}")

    dataset_accuracies = defaultdict(lambda: defaultdict(list))

    model_type = args.model_type
    print(f"Analyzing logs for {args.type} with model type {model_type}")

    if args.type == 'lora_real':
        for filename in os.listdir(log_dir):
            if filename.startswith('eval_') and filename.endswith('.log'):
                dataset = re.search(r'eval_(.+)_' + model_type + '_seed', filename).group(1)
                process_file(os.path.join(log_dir, filename), dataset, dataset_accuracies[0])
    elif args.type in ['llamdex', 'llamdex_zeropad', 'llamdex_nontranslate', 'llamdex_gbdt', 'llamdex_mlp_to_gbdt']:
        for dataset_folder in os.listdir(log_dir):
            dataset_path = os.path.join(log_dir, dataset_folder)
            if os.path.isdir(dataset_path):
                for filename in os.listdir(dataset_path):
                    file_prefix = ''
                    if args.type == 'llamdex_zeropad':
                        file_prefix = f'eval_{model_type}_zeropad'
                    elif args.type == 'llamdex_nontranslate':
                        file_prefix = f'eval_{model_type}_nontranslate'
                    elif args.type == 'llamdex_gbdt':
                        file_prefix = f'eval_{model_type}_gbdt'
                    elif args.type == 'llamdex_mlp_to_gbdt':
                        file_prefix = f'eval_{model_type}_mlp_to_gbdt'
                    else:
                        file_prefix = f'eval_{model_type}_{dataset_folder}'

                    if filename.startswith(file_prefix) and filename.endswith('.log'):
                        layer_match = re.search(r'layer(\d+)', filename)
                        if layer_match:
                            layer = int(layer_match.group(1))
                            if args.layer is None or layer == args.layer:
                                process_file(os.path.join(dataset_path, filename), dataset_folder,
                                             dataset_accuracies[layer])

    elif args.type in ['original_mistral', 'prompt_baseline', 'short_prompt_baseline']:
        for dataset_folder in os.listdir(log_dir):
            dataset_path = os.path.join(log_dir, dataset_folder)
            if os.path.isdir(dataset_path):
                for filename in os.listdir(dataset_path):
                    if filename.startswith(f'eval_{model_type}_{dataset_folder}_seed') and filename.endswith('.log'):
                        process_file(os.path.join(dataset_path, filename), dataset_folder, dataset_accuracies[0])
    else:
        raise ValueError(f"Invalid type: {args.type}")

    print(f"Results for {args.type}")
    print("Dataset\tBest Layer\tMean Accuracy\tStd Deviation")
    print("--------------------------------------------------")

    if args.type in ['llamdex', 'llamdex_gbdt', 'llamdex_mlp_to_gbdt'] and args.layer is None:
        for dataset in dataset_accuracies[0].keys():  # Assuming all datasets are present in layer 0
            best_layer = max(dataset_accuracies.keys(),
                             key=lambda l: np.mean(dataset_accuracies[l][dataset]) if dataset in dataset_accuracies[l] else -float('inf'))
            accuracies = dataset_accuracies[best_layer][dataset]
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            print(f"{dataset}\t{best_layer}\t\t{mean_accuracy:.4f}\t\t{std_accuracy:.4f}")
    else:
        layer = args.layer if args.type == 'llamdex' else 0
        for dataset, accuracies in dataset_accuracies[layer].items():
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            print(f"{dataset}\t{layer}\t\t{mean_accuracy:.4f}\t\t{std_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze accuracy logs.")
    parser.add_argument('--type', type=str,
                        required=True, help="Type of experiment")
    parser.add_argument('--layer', type=int, help="Layer number for llamdex", default=None)
    parser.add_argument('--model_type', type=str, choices=['mistral', 'llama'], default='mistral', help="Model type")
    args = parser.parse_args()
    main(args)
