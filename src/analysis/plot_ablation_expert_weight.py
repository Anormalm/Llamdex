""" (_log/ablation_expert_weight/nursery/layer0_weight0.0.log)
Namespace(mistral_models_path='model/llm', model_name='mistralai/Mistral-7B-Instruct-v0.3', dataset='nursery', expert_dir=None, encoder_dir=None, encoder_tokenizer=None, ffn_hidden_size=512, num_tokens=10, num_heads=8, dropout=0.0, use_norm=True, expert_input_size=None, expert_output_size=None, max_new_tokens=500, batch_size=16, n_instances=None, system_prompt=None, model_state_dict_dir='model/llm/exp_mistral_nursery_layer0_seed0/model_final.pt', output_file=None, mapping_hidden_size=32, expert_input_size_scaled=None, dataset_columns=None, tune_lora=False, baseline=False, layer=[0], direct_input=False, expert_weight=0.0)
Adding the custom expert to the model...
Loading the model state dict...
Loading the dataset...
Testing the model...
[3 3 3 ... 3 3 2]
Expert input loss: 0.2315
Accuracy on the test set: 0.3137
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Constants
DATASETS = ['bank_marketing', 'nursery', 'titanic', 'wine_quality']
DISPLAY_NAMES = {'bank_marketing': 'bank', 
                'nursery': 'nursery',
                'titanic': 'titanic', 
                'wine_quality': 'wine'}
COLORS = ['black', 'black', 'black', 'black']
LOG_DIR_BASE = '_log/ablation_expert_weight'
OUTPUT_DIR = '_fig/ablation_expert_weight'
FIGURE_SIZE = (10, 6)

# large font
plt.rcParams.update({'font.size': 22})

def extract_weight_from_filename(filename):
    """Extract weight value from log filename."""
    weight_match = re.search(r'weight(\d+\.\d+)', filename)
    return float(weight_match.group(1)) if weight_match else None

def extract_accuracy_from_content(content):
    """Extract accuracy value from log content."""
    acc_match = re.search(r'Accuracy on the test set: (\d+\.\d+)', content)
    return float(acc_match.group(1)) if acc_match else None

def get_data_points(dataset):
    """Get weight and accuracy data points for a dataset."""
    weights = []
    accuracies = []
    
    log_dir = os.path.join(LOG_DIR_BASE, dataset)
    if not os.path.exists(log_dir):
        return [], []
        
    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'):
            continue
            
        with open(os.path.join(log_dir, filename), 'r') as f:
            content = f.read()
            
        weight = extract_weight_from_filename(filename)
        accuracy = extract_accuracy_from_content(content)
        
        if weight is not None and accuracy is not None:
            weights.append(weight)
            accuracies.append(accuracy)
            
    return weights, accuracies

def plot_accuracy_vs_weight():
    """Create and save individual plots for each dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for dataset, color in zip(DATASETS, COLORS):
        weights, accuracies = get_data_points(dataset)
        
        if weights and accuracies:
            plt.figure(figsize=FIGURE_SIZE)
            
            # Sort by weights to get smooth line
            points = sorted(zip(weights, accuracies))
            weights, accuracies = zip(*points)
            
            plt.plot(weights, accuracies, 'o-', color=color, markersize=10)
            
            plt.xlabel('Expert Weight')
            plt.ylabel('Accuracy')
            plt.title(f'{DISPLAY_NAMES[dataset]}')
            plt.grid(True)
            plt.tight_layout()
            
            output_path = os.path.join(OUTPUT_DIR, f'{dataset}.png')
            plt.savefig(output_path)
            plt.close()

if __name__ == '__main__':
    plot_accuracy_vs_weight()
