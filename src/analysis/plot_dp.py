import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def extract_info_from_filename(filename):
    match = re.search(r'eval_(.+)_mistral_seed(\d+)_eps-?([\d.]+)\.log', filename)
    if match:
        return match.group(1), int(match.group(2)), float(match.group(3))
    return None, None, None


def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        accuracy_match = re.search(r'Accuracy on the test set: ([\d.]+)', content)
        if accuracy_match:
            return float(accuracy_match.group(1))
    return None

def process_file_dp_opt(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        accuracy_match = re.search(r'Final evaluation accuracy: ([\d.]+)%', content)
        if accuracy_match:
            return float(accuracy_match.group(1))
    return None


def collect_data(log_dir):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Process lora_syn_pate-gan, lora_syn_prompt-pate, lora_syn_seqpate, lora_syn_table_diffusion
    for method in ['lora_syn_pate-gan', 'lora_syn_prompt-pate', 'lora_syn_seqpate', 'lora_syn_table_diffusion']:
        # if method == 'lora_syn_pate-gan':
        #     method_dir = os.path.join(log_dir, method)
        # else:
        #     method_dir = os.path.join(log_dir, 'log_add_exp', method)
        method_dir = os.path.join(log_dir, method)
        if os.path.exists(method_dir):
            for filename in os.listdir(method_dir):
                dataset, seed, eps = extract_info_from_filename(filename)
                if dataset and seed is not None and eps is not None:
                    accuracy = process_file(os.path.join(method_dir, filename))
                    if accuracy is not None:
                        data[dataset][method][eps].append(accuracy)

    # Process prompt API
    prompt_file = os.path.join(log_dir, 'prompt_baseline', 'eval_mistral_dp_all.log')
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as file:
            for line in file:
                match = re.search(r'Dataset (\w+), eps ([\d.]+), seed (\d+) - accuracy ([\d.]+)', line)
                if match:
                    dataset, eps, seed, accuracy = match.groups()
                    data[dataset]['prompt_api'][float(eps)].append(float(accuracy))

    # Process Original LLM
    original_dir = os.path.join(log_dir, 'original_mistral')
    if os.path.exists(original_dir):
        for dataset in os.listdir(original_dir):
            dataset_dir = os.path.join(original_dir, dataset)
            if os.path.isdir(dataset_dir):
                for filename in os.listdir(dataset_dir):
                    if filename.startswith(f'eval_mistral_{dataset}_seed'):
                        accuracy = process_file(os.path.join(dataset_dir, filename))
                        if accuracy is not None:
                            data[dataset]['original_llm'][0].append(accuracy)  # Use 0 as a dummy epsilon value

    # # Process dp-opt
    # dp_opt_dir = os.path.join(log_dir, 'dp-opt')
    # if os.path.exists(dp_opt_dir):
    #     for dataset in os.listdir(dp_opt_dir):
    #         dataset_dir = os.path.join(dp_opt_dir, dataset)
    #         if os.path.isdir(dataset_dir):
    #             for filename in os.listdir(dataset_dir):
    #                 print(filename)
    #                 match = re.search(r'eval_dp_opt_(\w+)_seed(\d+)_eps([\d.]+)\.log', filename)
    #                 print(match)
    #                 if match:
    #                     dataset_name, seed, eps = match.groups()
    #                     accuracy = process_file_dp_opt(os.path.join(dataset_dir, filename))
    #                     if accuracy is not None:
    #                         data[dataset]['dp-opt'][float(eps)].append(accuracy)


    # Process Llamdex
    llamdex_dir = os.path.join(log_dir, 'dp_llamdex')
    if os.path.exists(llamdex_dir):
        for dataset in os.listdir(llamdex_dir):
            dataset_dir = os.path.join(llamdex_dir, dataset)
            if os.path.isdir(dataset_dir):
                llamdex_data = defaultdict(lambda: defaultdict(list))
                for filename in os.listdir(dataset_dir):
                    match = re.search(r'eval_mistral_(\w+)_layer(\d+)_eps-([\d.]+)_seed(\d+)\.log', filename)
                    if match:
                        dataset_name, layer, eps, seed = match.groups()
                        accuracy = process_file(os.path.join(dataset_dir, filename))
                        if accuracy is not None:
                            llamdex_data[float(eps)][int(layer)].append(accuracy)

                # Select the best layer for each epsilon
                for eps in llamdex_data:
                    best_layer = max(llamdex_data[eps], key=lambda l: np.mean(llamdex_data[eps][l]))
                    data[dataset]['llamdex'][eps] = llamdex_data[eps][best_layer]

    return data


def get_random_guess_accuracy(dataset):
    dataset_to_test_path = {
        # 'adult': "_data/adult/clean/adult_test.csv",
        'bank_marketing': "_data/bank_marketing/clean/bank_marketing_test.csv",
        'wine_quality': "_data/wine_quality/clean/wine_quality_test.csv",
        'titanic': "_data/titanic/clean/titanic_test.csv",
        'nursery': "_data/nursery/clean/nursery_test.csv"
    }

    dataset_info = json.load(open("src/dataset/dataset_info.json"))
    label_name = dataset_info[dataset]['answer_column']
    n_classes = dataset_info[dataset]['expert_output_size']

    if dataset not in dataset_to_test_path:
        raise ValueError(f"Unknown dataset: {dataset}")
    test_path = dataset_to_test_path[dataset]
    df = pd.read_csv(test_path)

    labels = df[label_name].values
    if n_classes == 1:
        random_preds = np.random.randint(0, 2, len(labels))
    else:
        random_preds_values = np.random.randint(0, n_classes, len(labels))
        random_preds = [chr(ord('A') + value) for value in random_preds_values]

    return np.mean(labels == random_preds), np.std(labels == random_preds)


method_to_name = {
    'lora_syn_dp-wgan': 'DP-WGAN',
    'lora_syn_pate-gan': 'PATE-GAN'
}

dataset_to_name = {
    # 'adult': 'adult',
    'bank_marketing': 'bank',
    'wine_quality': 'wine',
    'titanic': 'titanic',
    'nursery': 'nursery'
}


def plot_results(data):
    plt.rcParams.update({'font.size': 16})

    method_to_name = {
        'lora_syn_pate-gan': 'PATE-GAN',
        'lora_syn_prompt-pate': 'Prompt-PATE',
        'lora_syn_seqpate': 'SeqPATE',
        'lora_syn_table_diffusion': 'Table Diffusion',
        'prompt_api': 'Expert API',
        'original_llm': 'Original LLM',
        'llamdex': 'Llamdex'
    }

    dataset_to_name = {
        # 'adult': 'adult',
        'bank_marketing': 'bank',
        'wine_quality': 'wine',
        'titanic': 'titanic',
        'nursery': 'nursery'
    }

    method_styles = {
        'lora_syn_pate-gan': {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
        'prompt_api': {'color': 'red', 'linestyle': '--', 'marker': 's'},
        'llamdex': {'color': 'purple', 'linestyle': '-.', 'marker': '^'},
        'lora_syn_prompt-pate': {'color': 'orange', 'linestyle': ':', 'marker': 'D'},
        'lora_syn_seqpate': {'color': 'green', 'linestyle': '-.', 'marker': 'v'},
        'lora_syn_table_diffusion': {'color': 'cyan', 'linestyle': '-', 'marker': 'x'},
    }
    #     'lora_syn_pate-gan': {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
    #     'prompt_api': {'color': 'red', 'linestyle': '--', 'marker': 's'},
    #     'llamdex': {'color': 'purple', 'linestyle': '-.', 'marker': '^'}
    # }

    for dataset in data:
        if dataset not in dataset_to_name:
            continue

        plt.figure(figsize=(8, 6))

        eps_values = sorted(set(eps for method in data[dataset] for eps in data[dataset][method]))

        for method in data[dataset]:
            if method == 'original_llm':
                # Plot Original LLM as a horizontal line
                accuracies = data[dataset][method][0]
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                plt.axhline(y=mean_accuracy, color='green', linestyle=':', label='Original LLM')
                plt.fill_between([min(eps_values), max(eps_values)],
                                 [mean_accuracy - std_accuracy] * 2,
                                 [mean_accuracy + std_accuracy] * 2,
                                 color='green', alpha=0.1)
            else:
                mean_accuracies = []
                std_accuracies = []
                method_eps = []
                for eps in eps_values:
                    if eps in data[dataset][method]:
                        mean_accuracies.append(np.mean(data[dataset][method][eps]))
                        std_accuracies.append(np.std(data[dataset][method][eps]))
                        method_eps.append(eps)

                plt.errorbar(method_eps, mean_accuracies, yerr=std_accuracies,
                             label=method_to_name[method], capsize=5, markersize=8,
                             **method_styles[method], alpha=0.7)

        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

        plt.xlabel('ε')
        plt.ylabel('Accuracy')
        plt.title(f'{dataset_to_name[dataset]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.xscale('linear')
        custom_ticks = [0.5, 1, 2, 3, 4, 5]
        plt.xticks(custom_ticks, [f'{tick:.1f}' for tick in custom_ticks])
        plt.xlim(0.4, 5.1)

        plt.tight_layout()
        plt.savefig(f'_fig/{dataset}_accuracy.png', dpi=300)
        plt.close()


def print_results_table(data):
    print("\nResults Table for llamdex:")
    print("============================================")

    all_eps = set()
    for dataset in data:
        if 'llamdex' in data[dataset]:
            all_eps.update(data[dataset]['llamdex'].keys())
    all_eps = sorted(all_eps)

    eps_header = "|".join([f"ε={eps:.1f}" for eps in all_eps])
    print(f"Dataset|{eps_header}")
    print("-" * (len(eps_header) + 10))

    for dataset in sorted(data.keys()):
        if dataset not in dataset_to_name:
            continue

        if 'llamdex' in data[dataset]:
            results = []
            for eps in all_eps:
                if eps in data[dataset]['llamdex']:
                    acc = np.mean(data[dataset]['llamdex'][eps]) * 100
                    std = np.std(data[dataset]['llamdex'][eps]) * 100
                    results.append(f"${acc:.2f}\pm{std:.2f}$")
                else:
                    results.append("--")

            results_str = "|".join(results)
            print(f"{dataset_to_name[dataset]:<8}|{results_str}")

    print("============================================")
    #
    # print("\nResults Table for lora_syn_prompt-pate:")
    # print("============================================")
    #
    # all_eps = set()
    # for dataset in data:
    #     if 'lora_syn_prompt-pate' in data[dataset]:
    #         all_eps.update(data[dataset]['lora_syn_prompt-pate'].keys())
    # all_eps = sorted(all_eps)
    #
    # eps_header = " | ".join([f"ε={eps:.1f}" for eps in all_eps])
    # print(f"Dataset | {eps_header}")
    # print("-" * (len(eps_header) + 10))
    #
    # for dataset in sorted(data.keys()):
    #     if dataset not in dataset_to_name:
    #         continue
    #
    #     if 'lora_syn_prompt-pate' in data[dataset]:
    #         results = []
    #         for eps in all_eps:
    #             if eps in data[dataset]['lora_syn_prompt-pate']:
    #                 acc = np.mean(data[dataset]['lora_syn_prompt-pate'][eps])
    #                 results.append(f"{acc:.3f}")
    #             else:
    #                 results.append("--")
    #
    #         results_str = " | ".join(results)
    #         print(f"{dataset_to_name[dataset]:<8} | {results_str}")
    #
    # print("============================================")
    #
    # print("\nResults Table for lora_syn_seqpate:")
    # print("============================================")
    #
    # all_eps = set()
    # for dataset in data:
    #     if 'lora_syn_seqpate' in data[dataset]:
    #         all_eps.update(data[dataset]['lora_syn_seqpate'].keys())
    # all_eps = sorted(all_eps)
    #
    # eps_header = " | ".join([f"ε={eps:.1f}" for eps in all_eps])
    # print(f"Dataset | {eps_header}")
    # print("-" * (len(eps_header) + 10))
    #
    # for dataset in sorted(data.keys()):
    #     if dataset not in dataset_to_name:
    #         continue
    #
    #     if 'lora_syn_seqpate' in data[dataset]:
    #         results = []
    #         for eps in all_eps:
    #             if eps in data[dataset]['lora_syn_seqpate']:
    #                 acc = np.mean(data[dataset]['lora_syn_seqpate'][eps])
    #                 results.append(f"{acc:.3f}")
    #             else:
    #                 results.append("--")
    #
    #         results_str = " | ".join(results)
    #         print(f"{dataset_to_name[dataset]:<8} | {results_str}")
    #
    # print("============================================")

    # print("\nResults Table for lora_syn_table_diffusion:")
    # print("============================================")
    #
    # all_eps = set()
    # for dataset in data:
    #     if 'lora_syn_table_diffusion' in data[dataset]:
    #         all_eps.update(data[dataset]['lora_syn_table_diffusion'].keys())
    # all_eps = sorted(all_eps)
    #
    # eps_header = "|".join([f"ε={eps:.1f}" for eps in all_eps])
    # print(f"Dataset|{eps_header}")
    # print("|-" * (len(eps_header) + 1) + "|")
    #
    # for dataset in sorted(data.keys()):
    #     if dataset not in dataset_to_name:
    #         continue
    #
    #     if 'lora_syn_table_diffusion' in data[dataset]:
    #         results = []
    #         for eps in all_eps:
    #             if eps in data[dataset]['lora_syn_table_diffusion']:
    #                 acc = np.mean(data[dataset]['lora_syn_table_diffusion'][eps]) * 100
    #                 std = np.std(data[dataset]['lora_syn_table_diffusion'][eps]) * 100
    #                 results.append(f"${acc:.2f}\pm{std:.2f}$")
    #             else:
    #                 results.append("--")
    #
    #         results_str = "|".join(results)
    #         print(f"{dataset_to_name[dataset]:<8}|{results_str}")
    #
    # print("============================================")
    #
    #
    #
    # print("\nResults Table for dp-opt:")
    # print("============================================")
    #
    # all_eps = set()
    # for dataset in data:
    #     if 'dp-opt' in data[dataset]:
    #         all_eps.update(data[dataset]['dp-opt'].keys())
    # all_eps = sorted(all_eps)
    #
    # eps_header = " | ".join([f"ε={eps:.1f}" for eps in all_eps])
    # print(f"Dataset | {eps_header}")
    # print("-" * (len(eps_header) + 10))
    #
    # for dataset in sorted(data.keys()):
    #     if dataset not in dataset_to_name:
    #         continue
    #
    #     if 'dp-opt' in data[dataset]:
    #         results = []
    #         for eps in all_eps:
    #             if eps in data[dataset]['dp-opt']:
    #                 acc = np.mean(data[dataset]['dp-opt'][eps])
    #                 std = np.std(data[dataset]['dp-opt'][eps])
    #                 results.append(f"${acc:.2f}\pm{std:.2f}$")
    #             else:
    #                 results.append("--")
    #
    #         results_str = " | ".join(results)
    #         print(f"{dataset_to_name[dataset]:<8} | {results_str}")
    #
    # print("============================================")



def get_token_mapping_ablation_results(log_dir):
    datasets = ['titanic', 'wine_quality', 'bank_marketing', 'nursery']
    results = defaultdict(lambda: defaultdict(list))

    for dataset in datasets:
        path = os.path.join(log_dir, 'llamdex', dataset)
        if not os.path.exists(path):
            continue

        for seed in range(5): # 5 seeds
            filename = f'eval_mistral_nontranslate_{dataset}_layer0_seed{seed}.log'
            file_path = os.path.join(path, filename)

            if os.path.exists(file_path):
                accuracy = process_file(file_path)
                if accuracy is not None:
                    results[dataset][seed] = accuracy

    print("\nToken mapping ablation Study Results:")
    print("\n|Dataset|Accuracy(mean$\pm$std)|")
    print("|:--:|:--:|")

    for dataset in datasets:
        if results[dataset]:
            accuracies = list(results[dataset].values())
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"|{dataset}|${mean_acc * 100:.2f}\pm{std_acc * 100:.2f}$|")


def plot_num_tokens_ablation_results(log_dir):
    datasets = ['bank_marketing', 'nursery']
    tokens_list = [1, 5, 10, 20]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset in datasets:
        path = os.path.join(log_dir, 'log_add_exp', 'ablation_token', dataset)
        if not os.path.exists(path):
            continue

        for tokens in tokens_list:
            for seed in range(5): # 5 seeds
                filename = f'eval_mistral_{dataset}_layer0_tokens{tokens}_seed{seed}.log'
                file_path = os.path.join(path, filename)

                if os.path.exists(file_path):
                    accuracy = process_file(file_path)
                    if accuracy is not None:
                        results[dataset][tokens][seed] = accuracy

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for idx, dataset in enumerate(datasets):
        if not results[dataset]:
            continue

        ax = axs[idx]
        mean_accuracies = []
        std_accuracies = []

        for tokens in tokens_list:
            if results[dataset][tokens]:
                accuracies = list(results[dataset][tokens].values())
                mean_accuracies.append(np.mean(accuracies))
                std_accuracies.append(np.std(accuracies))
            else:
                mean_accuracies.append(np.nan)
                std_accuracies.append(np.nan)

        ax.errorbar(tokens_list, mean_accuracies, yerr=std_accuracies,
                    capsize=3, marker='o', linestyle='-', markersize=4, color='blue')

        ax.set_xlabel('Number of Tokens')
        if idx == 0:
            ax.set_ylabel('Accuracy')
        ax.set_title(dataset_to_name[dataset])
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(tokens_list)
        ax.set_xticklabels(tokens_list)
        ax.tick_params(axis='y')

        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

    plt.tight_layout()
    plt.savefig('_fig/num_tokens_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Combined plot for all datasets has been generated and saved.")


def get_num_tokens_ablation_results(log_dir):
    datasets = ['bank_marketing', 'nursery']
    tokens_list = [1, 5, 10, 20]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset in datasets:
        path = os.path.join(log_dir, 'log_add_exp', 'ablation_token', dataset)
        if not os.path.exists(path):
            continue

        for tokens in tokens_list:
            for seed in range(5):
                filename = f'eval_mistral_{dataset}_layer0_tokens{tokens}_seed{seed}.log'
                file_path = os.path.join(path, filename)

                if os.path.exists(file_path):
                    accuracy = process_file(file_path)
                    if accuracy is not None:
                        results[dataset][tokens][seed] = accuracy

    print("\nNumber of Tokens Ablation Study Results:")

    header = "|Dataset|" + "|".join([f"k={t}" for t in tokens_list]) + "|"
    print("\n" + header)
    print("|:--:|" + ":--:|" * len(tokens_list))

    for dataset in datasets:
        row = [dataset]
        for tokens in tokens_list:
            if results[dataset][tokens]:
                accuracies = list(results[dataset][tokens].values())
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                row.append(f"${mean_acc * 100:.2f}\pm{std_acc * 100:.2f}$")
            else:
                row.append("-")
        print("|" + "|".join(row) + "|")

def main():
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    log_dir = '_log'
    data = collect_data(log_dir)
    plot_results(data)
    print("Plots have been generated and saved.")
    # print_results_table(data)
    # get_token_mapping_ablation_results(log_dir)
    # get_num_tokens_ablation_results(log_dir)
    # plot_num_tokens_ablation_results(log_dir)


if __name__ == "__main__":
    main()
