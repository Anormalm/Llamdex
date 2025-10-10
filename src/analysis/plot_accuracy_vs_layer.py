import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def extract_info_from_filename(filename):
    match = re.search(r'eval_mistral_(\w+)_layer(\d+)_seed(\d+)\.log', filename)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None


def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        accuracy_match = re.search(r'Accuracy on the test set: ([\d.]+)', content)
        if accuracy_match:
            return float(accuracy_match.group(1))
    return None

dataset_to_label = {
#    'adult': 'adult',
    'bank_marketing': 'bank',
    'wine_quality': 'wine',
    'titanic': 'titanic',
    'nursery': 'nursery',
}

dataset_to_marker = {
#    'adult': 'o',
    'bank_marketing': 's',
    'wine_quality': 'D',
    'titanic': '^',
    'nursery': 'v',
}


def collect_data(log_dir):
    data = defaultdict(lambda: defaultdict(list))

    for dataset in os.listdir(log_dir):
        if dataset not in dataset_to_label:
            continue

        dataset_dir = os.path.join(log_dir, dataset)
        if os.path.isdir(dataset_dir):
            for filename in os.listdir(dataset_dir):
                dataset_name, layer, seed = extract_info_from_filename(filename)
                if dataset_name and layer is not None and seed is not None:
                    accuracy = process_file(os.path.join(dataset_dir, filename))
                    if accuracy is not None:
                        data[dataset_name][layer].append(accuracy)

    return data


def plot_results(data):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10, 6))

    for i, dataset in enumerate(sorted(data.keys())):
        color = f'C{i}'
        layers = sorted(data[dataset].keys())
        mean_accuracies = [np.mean(data[dataset][layer]) for layer in layers]
        std_accuracies = [np.std(data[dataset][layer]) for layer in layers]

        plt.errorbar(layers, mean_accuracies, yerr=std_accuracies, markersize=10,
                     label=dataset_to_label[dataset], fmt=f'-{dataset_to_marker[dataset]}', color=color)

    plt.xlabel('Inserted Layer')
    plt.ylabel('Accuracy')
    # plt.title('Llamdex Accuracy vs Layer for Different Datasets')
    # make legend horizontal
    plt.legend(fontsize='small', ncol=3, loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('_fig/llamdex-accuracy-vs-layer.png', dpi=600, bbox_inches='tight')
    plt.close()


def main():
    log_dir = '_log/llamdex'
    data = collect_data(log_dir)
    plot_results(data)
    print("Plot has been generated and saved as '_fig/llamdex_accuracy_vs_layer.png'.")


if __name__ == "__main__":
    main()
