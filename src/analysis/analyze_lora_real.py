import os
import re
import numpy as np
import argparse
from collections import defaultdict

def main(args):
    if args.type == 'llamdex' and args.layer is None:
        raise ValueError("Layer must be specified for llamdex type")

    if args.type == 'lora_real':
        log_dir = '_log/lora_real'
    elif args.type == 'llamdex':
        log_dir = '_log/llamdex'
    elif args.type == 'original_mistral':
        log_dir = '_log/original_mistral'
    else:
        raise ValueError(f"Invalid type: {args.type}")

    dataset_accuracies = defaultdict(list)

    if args.type == 'lora_real':
        for filename in os.listdir(log_dir):
            if filename.startswith('eval_') and filename.endswith('.log'):
                dataset = re.search(r'eval_(.+)_mistral_seed', filename).group(1)
                process_file(os.path.join(log_dir, filename), dataset, dataset_accuracies)
    else:  # for llamdex and original_mistral
        for dataset_folder in os.listdir(log_dir):
            dataset_path = os.path.join(log_dir, dataset_folder)
            if os.path.isdir(dataset_path):
                for filename in os.listdir(dataset_path):
                    if args.type == 'llamdex':
                        if filename.startswith(f'eval_mistral_{dataset_folder}_layer{args.layer}_seed') and filename.endswith('.log'):
                            process_file(os.path.join(dataset_path, filename), dataset_folder, dataset_accuracies)
                    elif args.type == 'original_mistral':
                        if filename.startswith(f'eval_mistral_{dataset_folder}_seed') and filename.endswith('.log'):
                            process_file(os.path.join(dataset_path, filename), dataset_folder, dataset_accuracies)

    print(f"Results for {args.type}" + (f" (Layer {args.layer})" if args.type == 'llamdex' else ""))
    print("Dataset\tMean Accuracy\tStd Deviation")
    print("----------------------------------------")
    for dataset, accuracies in dataset_accuracies.items():
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        print(f"{dataset}\t{mean_accuracy:.4f}\t\t{std_accuracy:.4f}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze accuracy logs.")
    parser.add_argument('--type', type=str, choices=['lora_real', 'llamdex', 'original_mistral'],
                        required=True, help="Type of experiment")
    parser.add_argument('--layer', type=int, help="Layer number for llamdex", default=None)
    args = parser.parse_args()
    main(args)