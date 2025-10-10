import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.stats as stats
import glob

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.preprocess.DataScaler import TableScaler


class MICOStyleMIA:
    """
    MICO-style Membership Inference Attack for tabular data
    Implements multiple attack strategies used in the MICO competition
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.eps = 1e-8
    
    def extract_features(self, model, X, y):
        """
        Extract comprehensive features for MIA following MICO methodology
        """
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        features = []
        
        with torch.no_grad():
            # Forward pass
            outputs = model(X_tensor)
            
            # Handle different output formats
            if outputs.shape[1] == 1:  # Binary with single output
                probs_class1 = torch.sigmoid(outputs)
                probs = torch.cat([1-probs_class1, probs_class1], dim=1)
                outputs_2class = torch.cat([1-probs_class1, probs_class1], dim=1)
            else:
                outputs_2class = outputs
                probs = F.softmax(outputs_2class, dim=1)
            
            
            # Feature 1: Predicted probabilities for each class
            for i in range(probs.shape[1]):
                features.append(probs[:, i].cpu().numpy())
            
            # Feature 2: Loss (negative log-likelihood)
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            losses = criterion(outputs_2class, y_tensor)
            features.append(losses.cpu().numpy())
            
            # Feature 3: Confidence (max probability)
            confidences = torch.max(probs, dim=1)[0]
            features.append(confidences.cpu().numpy())
            
            # Feature 4: Modified Entropy
            entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=1)
            features.append(entropy.cpu().numpy())
            
            # Feature 5: Prediction correctness
            predictions = torch.argmax(outputs_2class, dim=1)
            correctness = (predictions == y_tensor).float()
            features.append(correctness.cpu().numpy())
            
            # Feature 6: Margin (difference between top two probabilities)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            if sorted_probs.shape[1] >= 2:
                margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            else:
                margins = sorted_probs[:, 0]  # For binary case
            features.append(margins.cpu().numpy())
            
            # Feature 7: Predicted probability for true class
            # For our 2-class case after conversion, we can use gather
            true_class_probs = probs.gather(1, y_tensor.unsqueeze(1)).squeeze(1)
            features.append(true_class_probs.cpu().numpy())
            
        return np.array(features).T  # Shape: (n_samples, n_features)
    
    def compute_hardness_scores(self, features_train_models, membership_train_models, 
                               features_target, model_idx, phase='dev'):
        """
        Compute hardness-based scores following MICO methodology
        """
        hardness_scores = []
        n_samples, n_features = features_target.shape
        
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                # Get feature value for target sample
                target_value = features_target[sample_idx, feature_idx]
                
                # Collect in-training and out-of-training scores from other models
                in_scores = []
                out_scores = []
                
                for other_model_idx in range(len(features_train_models)):
                    if phase == 'train' and other_model_idx == model_idx:
                        continue  # Skip self for training phase
                    
                    # In-training scores (members)
                    member_mask = membership_train_models[other_model_idx][:, sample_idx].astype(bool)
                    if np.any(member_mask):
                        in_scores.extend(features_train_models[other_model_idx][member_mask, sample_idx, feature_idx])
                    
                    # Out-of-training scores (non-members)
                    non_member_mask = ~member_mask
                    if np.any(non_member_mask):
                        out_scores.extend(features_train_models[other_model_idx][non_member_mask, sample_idx, feature_idx])
                
                # Handle edge cases
                if len(out_scores) == 0 or len(out_scores) == 1:
                    out_scores = in_scores
                
                if len(in_scores) == 0:
                    in_scores = out_scores
                
                # Compute statistics
                in_scores = np.array(in_scores)
                out_scores = np.array(out_scores)
                all_scores = np.concatenate([in_scores, out_scores])
                
                in_mean = np.mean(in_scores) if len(in_scores) > 0 else 0
                out_mean = np.mean(out_scores) if len(out_scores) > 0 else 0
                all_mean = np.mean(all_scores) if len(all_scores) > 0 else 0
                all_std = np.std(all_scores) if len(all_scores) > 0 else 1
                
                # MICO hardness formulas
                formula_0 = target_value - (in_mean + out_mean) / 2  # Relative to average
                formula_1 = target_value - in_mean  # Relative to in-training
                formula_2 = (target_value - all_mean) / (all_std + self.eps)  # Standardized
                
                hardness_scores.append([formula_0, formula_1, formula_2])
        
        return np.array(hardness_scores).reshape(n_samples, n_features, 3)
    
    def attack_loss_based(self, features):
        """Loss-based attack (MICO's primary method)"""
        # Feature index 2 is loss
        return -features[:, 2]  # Negative loss (higher loss = more likely member)
    
    def attack_confidence_based(self, features):
        """Confidence-based attack"""
        # Feature index 3 is confidence
        return features[:, 3]  # Higher confidence = more likely member
    
    def attack_entropy_based(self, features):
        """Entropy-based attack"""
        # Feature index 4 is modified entropy
        return -features[:, 4]  # Lower entropy = more likely member
    
    def attack_correctness_based(self, features):
        """Correctness-based attack"""
        # Feature index 5 is correctness
        return features[:, 5]  # Correct prediction = more likely member
    
    def attack_margin_based(self, features):
        """Margin-based attack"""
        # Feature index 6 is margin
        return features[:, 6]  # Higher margin = more likely member
    
    def attack_hardness_based(self, hardness_scores, formula_idx=0, feature_idx=2):
        """Hardness-based attack using MICO methodology"""
        return hardness_scores[:, feature_idx, formula_idx]


def load_titanic_data():
    """Load Titanic data"""
    train_path = "data/titanic/clean/titanic_train.csv"
    test_path = "data/titanic/clean/titanic_test.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Prepare features (sorted by column name)
    X_train = train_df.drop(columns=['Survived']).sort_index(axis=1).to_numpy()
    y_train = train_df['Survived'].to_numpy()
    X_test = test_df.drop(columns=['Survived']).sort_index(axis=1).to_numpy()
    y_test = test_df['Survived'].to_numpy()
    
    return X_train, y_train, X_test, y_test


def extract_llamdex_accuracy(epsilon=None):
    """
    Extract Llamdex accuracy from evaluation logs
    """
    if epsilon is None:
        # Baseline (no DP) case
        pattern = "_log/lora_real/eval_titanic_mistral_seed*.log"
    else:
        # DP case
        pattern = f"_log/dp_llamdex/titanic/eval_mistral_titanic_layer*_eps-{epsilon}_seed*.log"
    
    log_files = glob.glob(pattern)
    accuracies = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                match = re.search(r'Accuracy on the test set: ([0-9.]+)', content)
                if match:
                    accuracies.append(float(match.group(1)))
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    if accuracies:
        return np.mean(accuracies)
    else:
        return None


def comprehensive_mico_analysis():
    """
    Comprehensive MICO-style MIA analysis
    """
    print("MICO-STYLE MEMBERSHIP INFERENCE ATTACK ANALYSIS")
    print("="*60)
    
    # Load and scale data
    X_train, y_train, X_test, y_test = load_titanic_data()
    
    dataset_json_path = "src/dataset/titanic/titanic.json"
    scaler = TableScaler(dataset_json=dataset_json_path)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train size: {len(X_train_scaled)}, Test size: {len(X_test_scaled)}")
    
    # Initialize MICO attack
    mico = MICOStyleMIA(device='cpu')
    
    epsilon_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    all_results = {}
    
    # Attack methods
    attack_methods = {
        'loss': mico.attack_loss_based,
        'confidence': mico.attack_confidence_based,
        'entropy': mico.attack_entropy_based,
        'correctness': mico.attack_correctness_based,
        'margin': mico.attack_margin_based,
    }
    
    # Analyze baseline (no DP)
    print(f"\n{'='*50}")
    print("ANALYZING BASELINE MODEL (NO DP)")
    print(f"{'='*50}")
    
    baseline_model_path = "model/experts/titanic_mlp.pth"
    baseline_results = {}
    
    if os.path.exists(baseline_model_path):
        model = torch.load(baseline_model_path, map_location='cpu')
        
        # Extract features for members and non-members
        member_features = mico.extract_features(model, X_train_scaled, y_train)
        non_member_features = mico.extract_features(model, X_test_scaled, y_test)
        
        print(f"Feature matrix shape: {member_features.shape}")
        print(f"Features: [class_probs, loss, confidence, entropy, correctness, margin, true_class_prob]")
        
        # Evaluate each attack method
        for method_name, attack_func in attack_methods.items():
            member_scores = attack_func(member_features)
            non_member_scores = attack_func(non_member_features)
            
            # Combine scores and labels
            all_scores = np.concatenate([member_scores, non_member_scores])
            membership_labels = np.concatenate([np.ones(len(member_scores)), 
                                              np.zeros(len(non_member_scores))])
            
            # Calculate metrics
            fpr, tpr, _ = roc_curve(membership_labels, all_scores)
            attack_auc = auc(fpr, tpr)
            
            # Calculate TPR at 10% FPR (MICO's primary metric)
            target_fpr = 0.1
            tpr_at_10_fpr = tpr[np.where(fpr <= target_fpr)[0][-1]] if np.any(fpr <= target_fpr) else 0
            
            baseline_results[method_name] = {
                'auc': attack_auc,
                'tpr_at_10_fpr': tpr_at_10_fpr,
                'member_scores': member_scores,
                'non_member_scores': non_member_scores
            }
            
            print(f"{method_name.upper()} Attack:")
            print(f"  AUC: {attack_auc:.4f}")
            print(f"  TPR@10%FPR: {tpr_at_10_fpr:.4f}")
            print(f"  Member scores - Mean: {np.mean(member_scores):.4f}, Std: {np.std(member_scores):.4f}")
            print(f"  Non-member scores - Mean: {np.mean(non_member_scores):.4f}, Std: {np.std(non_member_scores):.4f}")
    
    all_results['baseline'] = baseline_results
    
    # Analyze DP models
    for eps in epsilon_values:
        print(f"\n{'='*50}")
        print(f"ANALYZING DP MODEL (ε = {eps})")
        print(f"{'='*50}")
        
        model_path = f"model/experts/titanic_mlp_eps-{eps}_delta-0.001.pth"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        model = torch.load(model_path, map_location='cpu')
        
        # Extract features
        member_features = mico.extract_features(model, X_train_scaled, y_train)
        non_member_features = mico.extract_features(model, X_test_scaled, y_test)
        
        eps_results = {}
        
        # Evaluate each attack method
        for method_name, attack_func in attack_methods.items():
            member_scores = attack_func(member_features)
            non_member_scores = attack_func(non_member_features)
            
            # Combine scores and labels
            all_scores = np.concatenate([member_scores, non_member_scores])
            membership_labels = np.concatenate([np.ones(len(member_scores)), 
                                              np.zeros(len(non_member_scores))])
            
            # Calculate metrics
            fpr, tpr, _ = roc_curve(membership_labels, all_scores)
            attack_auc = auc(fpr, tpr)
            
            # Calculate TPR at 10% FPR
            target_fpr = 0.1
            tpr_at_10_fpr = tpr[np.where(fpr <= target_fpr)[0][-1]] if np.any(fpr <= target_fpr) else 0
            
            eps_results[method_name] = {
                'auc': attack_auc,
                'tpr_at_10_fpr': tpr_at_10_fpr,
                'member_scores': member_scores,
                'non_member_scores': non_member_scores
            }
            
            print(f"{method_name.upper()} Attack:")
            print(f"  AUC: {attack_auc:.4f}")
            print(f"  TPR@10%FPR: {tpr_at_10_fpr:.4f}")
        
        all_results[f'eps_{eps}'] = eps_results
    
    return all_results


def create_privacy_utility_figure(results):
    """Create dual y-axis figure: Attack TPR vs Accuracy over epsilon"""
    plt.rcParams.update({'font.size': 12})
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Start from epsilon = 1.0
    epsilon_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    methods = ['loss', 'confidence', 'entropy', 'correctness', 'margin']
    
    # Collect TPR and accuracy data for DP models
    tpr_data = []
    acc_data = []
    eps_plot = []
    
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_tpr = np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
            tpr_data.append(avg_tpr)
            eps_plot.append(eps)
        else:
            tpr_data.append(None)
        
        acc = extract_llamdex_accuracy(eps)
        acc_data.append(acc)
    
    # Filter out None values for DP models
    valid_indices = [i for i, (tpr, acc) in enumerate(zip(tpr_data, acc_data)) if tpr is not None and acc is not None]
    eps_valid = [eps_plot[i] for i in valid_indices]
    tpr_valid = [tpr_data[i] for i in valid_indices]
    acc_valid = [acc_data[i] for i in valid_indices]
    
    # Get baseline values
    baseline_tpr = None
    baseline_acc = None
    if 'baseline' in results:
        baseline_tpr = np.mean([results['baseline'].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
    
    baseline_acc = extract_llamdex_accuracy(None)
    
    # Add infinity point for No DP
    if baseline_tpr is not None and baseline_acc is not None:
        # Use a large number to represent infinity on the plot
        inf_x = max(eps_valid) + 1.5  # Position infinity point to the right
        eps_valid.append(inf_x)
        tpr_valid.append(baseline_tpr)
        acc_valid.append(baseline_acc)
    
    # Plot TPR on left y-axis
    color1 = 'tab:red'
    ax1.set_xlabel('Privacy Budget (ε)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Attack TPR@10%FPR (Lower = Better Privacy)', color=color1, fontsize=14, fontweight='bold')
    
    # Plot all points connected (including No DP point)
    ax1.plot(eps_valid, tpr_valid, 'o-', color=color1, linewidth=3, markersize=8, label='Attack TPR@10%FPR')
    # Make the No DP point more prominent
    if baseline_tpr is not None:
        ax1.plot(eps_valid[-1], tpr_valid[-1], 'o', color=color1, markersize=12, markeredgecolor='black', markeredgewidth=2)
    
    # Set TPR y-axis range
    ax1.set_ylim(0, 0.2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Add dotted vertical line at No DP point
    if baseline_tpr is not None and baseline_acc is not None:
        ax1.axvline(x=eps_valid[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
    
    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Llamdex Accuracy (Higher = Better Utility)', color=color2, fontsize=14, fontweight='bold')
    
    # Plot all points connected (including No DP point)
    ax2.plot(eps_valid, acc_valid, 's-', color=color2, linewidth=3, markersize=8, label='Llamdex Accuracy')
    # Make the No DP point more prominent
    if baseline_acc is not None:
        ax2.plot(eps_valid[-1], acc_valid[-1], 's', color=color2, markersize=12, markeredgecolor='black', markeredgewidth=2)
    
    # Set accuracy y-axis range
    ax2.set_ylim(0.20, 0.80)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Customize x-axis
    x_ticks = eps_valid[:-1] + [eps_valid[-1]]  # All epsilon values plus infinity point
    x_labels = [str(int(eps)) if eps.is_integer() else str(eps) for eps in eps_valid[:-1]] + ['∞']
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    
    # Add "(no DP)" annotation at infinity point
    if baseline_tpr is not None and baseline_acc is not None:
        ax1.annotate('(no DP)', xy=(eps_valid[-1], tpr_valid[-1]), 
                    xytext=(eps_valid[-1], tpr_valid[-1] + (max(tpr_valid) - min(tpr_valid)) * 0.1),
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # Add title
    ax1.set_title('Privacy-Utility Tradeoff Analysis', fontsize=16, fontweight='bold', pad=20)
    
    # Add legends
    lines1 = [plt.Line2D([0], [0], color=color1, marker='o', markersize=8, linewidth=3)]
    lines2 = [plt.Line2D([0], [0], color=color2, marker='s', markersize=8, linewidth=3)]
    labels = ['Attack TPR@10%FPR', 'Llamdex Accuracy']
    ax1.legend(lines1 + lines2, labels, loc='lower left')
    
    # Set x-axis limits with some padding
    ax1.set_xlim(min(eps_valid) - 0.3, max(eps_valid) + 0.5)
    
    plt.tight_layout()
    plt.savefig('_fig/privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_mico_visualization(results):
    """Create MICO-style visualization with privacy-utility tradeoff table"""
    plt.rcParams.update({'font.size': 12})
    
    # Create figure with mixed layout: table at top, plots at bottom
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid spec for custom layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, height_ratios=[1, 2, 2], hspace=0.3, wspace=0.3)
    
    # Top panel: Summary table
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    
    # Create summary table data
    epsilon_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    methods = ['loss', 'confidence', 'entropy', 'correctness', 'margin']
    
    # Collect data for table
    table_data = []
    
    # Row 1: Attack AUC
    auc_row = ['Attack AUC']
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_auc = np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods])
            auc_row.append(f"{avg_auc:.3f}")
        else:
            auc_row.append("N/A")
    # Baseline AUC
    if 'baseline' in results:
        baseline_auc = np.mean([results['baseline'].get(method, {}).get('auc', 0.5) for method in methods])
        auc_row.append(f"{baseline_auc:.3f}")
    else:
        auc_row.append("N/A")
    table_data.append(auc_row)
    
    # Row 2: Attack TPR@10%FPR
    tpr_row = ['Attack TPR@10%FPR']
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_tpr = np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
            tpr_row.append(f"{avg_tpr:.3f}")
        else:
            tpr_row.append("N/A")
    # Baseline TPR
    if 'baseline' in results:
        baseline_tpr = np.mean([results['baseline'].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
        tpr_row.append(f"{baseline_tpr:.3f}")
    else:
        tpr_row.append("N/A")
    table_data.append(tpr_row)
    
    # Row 3: Llamdex Accuracy
    acc_row = ['Llamdex Accuracy']
    for eps in epsilon_values:
        acc = extract_llamdex_accuracy(eps)
        if acc is not None:
            acc_row.append(f"{acc:.3f}")
        else:
            acc_row.append("N/A")
    # Baseline accuracy
    baseline_acc = extract_llamdex_accuracy(None)
    if baseline_acc is not None:
        acc_row.append(f"{baseline_acc:.3f}")
    else:
        acc_row.append("N/A")
    table_data.append(acc_row)
    
    # Column headers
    col_headers = ['Metric'] + [f'ε={eps}' for eps in epsilon_values] + ['No DP (∞)']
    
    # Create table
    table = ax_table.table(cellText=table_data, 
                          colLabels=col_headers,
                          cellLoc='center',
                          loc='center',
                          colColours=['lightgray'] * len(col_headers))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color code cells based on values
    for i in range(1, len(table_data) + 1):
        for j in range(1, len(col_headers)):
            if table_data[i-1][j] != "N/A":
                val = float(table_data[i-1][j])
                if i == 1 or i == 2:  # AUC and TPR rows (lower is better for privacy)
                    if val < 0.5:
                        table[(i, j)].set_facecolor('lightgreen')
                    elif val > 0.55:
                        table[(i, j)].set_facecolor('lightcoral')
                else:  # Accuracy row (higher is better for utility)
                    if val > 0.7:
                        table[(i, j)].set_facecolor('lightgreen')
                    elif val < 0.6:
                        table[(i, j)].set_facecolor('lightcoral')
    
    ax_table.set_title('Privacy-Utility Tradeoff Analysis', fontsize=16, fontweight='bold', pad=20)
    
    # Extract data for plotting
    eps_values_plot = []
    method_aucs = {method: [] for method in methods}
    method_tpr_10 = {method: [] for method in methods}
    
    for key in sorted(results.keys()):
        if key == 'baseline':
            eps_val = float('inf')
        else:
            eps_val = float(key.split('_')[1])
        
        eps_values_plot.append(eps_val)
        
        for method in methods:
            if method in results[key]:
                method_aucs[method].append(results[key][method]['auc'])
                method_tpr_10[method].append(results[key][method]['tpr_at_10_fpr'])
            else:
                method_aucs[method].append(0.5)  # Random baseline
                method_tpr_10[method].append(0.1)  # Random at 10% FPR
    
    # Sort by epsilon
    sorted_indices = np.argsort(eps_values_plot)
    eps_values_plot = [eps_values_plot[i] for i in sorted_indices]
    for method in methods:
        method_aucs[method] = [method_aucs[method][i] for i in sorted_indices]
        method_tpr_10[method] = [method_tpr_10[method][i] for i in sorted_indices]
    
    eps_for_plot = [1000 if x == float('inf') else x for x in eps_values_plot]
    
    # Plot 1: Privacy-Utility Tradeoff
    ax1 = fig.add_subplot(gs[1, 0])
    
    # Extract accuracy data for plotting
    accuracy_data = []
    auc_data = []
    tpr_data = []
    eps_labels = []
    
    for eps in epsilon_values:
        acc = extract_llamdex_accuracy(eps)
        key = f'eps_{eps}'
        if key in results and acc is not None:
            accuracy_data.append(acc)
            auc_data.append(np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods]))
            tpr_data.append(np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods]))
            eps_labels.append(f'ε={eps}')
    
    # Add baseline
    baseline_acc = extract_llamdex_accuracy(None)
    if 'baseline' in results and baseline_acc is not None:
        accuracy_data.append(baseline_acc)
        auc_data.append(np.mean([results['baseline'].get(method, {}).get('auc', 0.5) for method in methods]))
        tpr_data.append(np.mean([results['baseline'].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods]))
        eps_labels.append('No DP')
    
    # Create scatter plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(accuracy_data)))
    scatter = ax1.scatter(auc_data, accuracy_data, c=colors, s=100, alpha=0.7)
    
    # Annotate points
    for i, label in enumerate(eps_labels):
        ax1.annotate(label, (auc_data[i], accuracy_data[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Attack AUC (Lower = Better Privacy)')
    ax1.set_ylabel('Llamdex Accuracy (Higher = Better Utility)')
    ax1.set_title('Privacy vs Utility Tradeoff')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUC vs Epsilon
    ax2 = fig.add_subplot(gs[1, 1])
    avg_aucs = []
    eps_vals = []
    
    for key in sorted(results.keys(), key=lambda x: float('inf') if x == 'baseline' else float(x.split('_')[1])):
        if key == 'baseline':
            eps_val = float('inf')
        else:
            eps_val = float(key.split('_')[1])
        
        eps_vals.append(eps_val)
        avg_auc = np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods])
        avg_aucs.append(avg_auc)
    
    eps_plot = [1000 if x == float('inf') else x for x in eps_vals]
    ax2.plot(eps_plot[:-1], avg_aucs[:-1], 'o-', label='DP Models', linewidth=2, markersize=8)
    ax2.axhline(y=avg_aucs[-1], color='red', linestyle='--', label='No DP Baseline', linewidth=2)
    ax2.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Random Guess')
    
    ax2.set_xlabel('Privacy Budget (ε)')
    ax2.set_ylabel('Average Attack AUC')
    ax2.set_title('Attack Success vs Privacy Budget')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: TPR@10%FPR vs Epsilon
    ax3 = fig.add_subplot(gs[1, 2])
    avg_tprs = []
    
    for key in sorted(results.keys(), key=lambda x: float('inf') if x == 'baseline' else float(x.split('_')[1])):
        avg_tpr = np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
        avg_tprs.append(avg_tpr)
    
    ax3.plot(eps_plot[:-1], avg_tprs[:-1], 's-', label='DP Models', linewidth=2, markersize=8)
    ax3.axhline(y=avg_tprs[-1], color='red', linestyle='--', label='No DP Baseline', linewidth=2)
    ax3.axhline(y=0.1, color='black', linestyle=':', alpha=0.5, label='Random Guess')
    
    ax3.set_xlabel('Privacy Budget (ε)')
    ax3.set_ylabel('Average TPR @ 10% FPR')
    ax3.set_title('MICO Primary Metric vs Privacy Budget')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Epsilon
    ax4 = fig.add_subplot(gs[2, 0])
    acc_vals = []
    eps_acc = []
    
    for eps in epsilon_values:
        acc = extract_llamdex_accuracy(eps)
        if acc is not None:
            acc_vals.append(acc)
            eps_acc.append(eps)
    
    baseline_acc = extract_llamdex_accuracy(None)
    
    ax4.plot(eps_acc, acc_vals, 'o-', label='DP Models', linewidth=2, markersize=8)
    if baseline_acc is not None:
        ax4.axhline(y=baseline_acc, color='green', linestyle='--', label='No DP Baseline', linewidth=2)
    
    ax4.set_xlabel('Privacy Budget (ε)')
    ax4.set_ylabel('Llamdex Accuracy')
    ax4.set_title('Model Utility vs Privacy Budget')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Method comparison
    ax5 = fig.add_subplot(gs[2, 1])
    x_pos = range(len(methods))
    width = 0.12
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, (key, results_dict) in enumerate(sorted(results.items(), key=lambda x: float('inf') if x[0] == 'baseline' else float(x[0].split('_')[1]))):
        if key == 'baseline':
            label = 'No DP'
        else:
            eps_val = key.split('_')[1]
            label = f'ε={eps_val}'
        
        aucs = [results_dict.get(method, {}).get('auc', 0.5) for method in methods]
        offset = (i - len(results)/2) * width
        ax5.bar([x + offset for x in x_pos], aucs, width, label=label, alpha=0.8)
    
    ax5.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
    ax5.set_xlabel('Attack Method')
    ax5.set_ylabel('AUC')
    ax5.set_title('Attack Success by Method')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([m.capitalize() for m in methods], rotation=45)
    ax5.legend(loc='lower left')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Privacy effectiveness
    ax6 = fig.add_subplot(gs[2, 2])
    
    if 'baseline' in results:
        baseline_auc_avg = np.mean([results['baseline'].get(m, {}).get('auc', 0.5) for m in methods])
        baseline_tpr_avg = np.mean([results['baseline'].get(m, {}).get('tpr_at_10_fpr', 0.1) for m in methods])
        
        privacy_gains_auc = []
        privacy_gains_tpr = []
        eps_vals_numeric = []
        
        for key in sorted(results.keys()):
            if key == 'baseline':
                continue
            
            eps_val = float(key.split('_')[1])
            eps_vals_numeric.append(eps_val)
            
            current_auc = np.mean([results[key].get(m, {}).get('auc', 0.5) for m in methods])
            current_tpr = np.mean([results[key].get(m, {}).get('tpr_at_10_fpr', 0.1) for m in methods])
            
            privacy_gains_auc.append((baseline_auc_avg - current_auc) / (baseline_auc_avg - 0.5) * 100 if baseline_auc_avg > 0.5 else 0)
            privacy_gains_tpr.append((baseline_tpr_avg - current_tpr) / (baseline_tpr_avg - 0.1) * 100 if baseline_tpr_avg > 0.1 else 0)
        
        ax6.plot(eps_vals_numeric, privacy_gains_auc, 'o-', label='AUC Privacy Gain', linewidth=2, markersize=8)
        ax6.plot(eps_vals_numeric, privacy_gains_tpr, 's-', label='TPR Privacy Gain', linewidth=2, markersize=8)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Privacy Budget (ε)')
        ax6.set_ylabel('Privacy Gain (%)')
        ax6.set_title('Privacy Protection Effectiveness')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('_fig/mico_style_mia_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_latex_table(results):
    """Create LaTeX format table"""
    print("\n" + "="*80)
    print("PRIVACY-UTILITY TRADEOFF ANALYSIS - LATEX TABLE")
    print("="*80)
    
    methods = ['loss', 'confidence', 'entropy', 'correctness', 'margin']
    epsilon_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Collect data
    auc_values = []
    tpr_values = []
    accuracy_values = []
    
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_auc = np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods])
            avg_tpr = np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
            auc_values.append(f"{avg_auc:.3f}")
            tpr_values.append(f"{avg_tpr:.3f}")
        else:
            auc_values.append("N/A")
            tpr_values.append("N/A")
        
        acc = extract_llamdex_accuracy(eps)
        if acc is not None:
            accuracy_values.append(f"{acc:.3f}")
        else:
            accuracy_values.append("N/A")
    
    # Baseline values
    baseline_auc = "N/A"
    baseline_tpr = "N/A"
    baseline_acc = "N/A"
    
    if 'baseline' in results:
        baseline_auc_val = np.mean([results['baseline'].get(method, {}).get('auc', 0.5) for method in methods])
        baseline_tpr_val = np.mean([results['baseline'].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
        baseline_auc = f"{baseline_auc_val:.3f}"
        baseline_tpr = f"{baseline_tpr_val:.3f}"
    
    baseline_acc_val = extract_llamdex_accuracy(None)
    if baseline_acc_val is not None:
        baseline_acc = f"{baseline_acc_val:.3f}"
    
    # Print LaTeX table
    print("\n\\begin{table}[h!]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("Metric & $\\varepsilon=1.0$ & $\\varepsilon=2.0$ & $\\varepsilon=3.0$ & $\\varepsilon=4.0$ & $\\varepsilon=5.0$ & No DP ($\\infty$) \\\\")
    print("\\hline")
    print(f"Attack AUC & {' & '.join(auc_values)} & {baseline_auc} \\\\")
    print("\\hline")
    print(f"Attack TPR@10\\%FPR & {' & '.join(tpr_values)} & {baseline_tpr} \\\\")
    print("\\hline")
    print(f"Llamdex Accuracy & {' & '.join(accuracy_values)} & {baseline_acc} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Privacy-Utility Tradeoff Analysis for Titanic Dataset}")
    print("\\label{tab:privacy_utility}")
    print("\\end{table}")

def create_mico_results_table(results):
    """Create both markdown and LaTeX tables"""
    print("\n" + "="*80)
    print("PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("="*80)
    
    methods = ['loss', 'confidence', 'entropy', 'correctness', 'margin']
    
    # Extract data for the summary table
    epsilon_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    print("\n### Markdown Table")
    print("| Metric | ε=1.0 | ε=2.0 | ε=3.0 | ε=4.0 | ε=5.0 | No DP (∞) |")
    print("|:-------|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|")
    
    # Attack AUC row
    auc_values = []
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_auc = np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods])
            auc_values.append(f"{avg_auc:.3f}")
        else:
            auc_values.append("N/A")
    
    # Baseline AUC
    baseline_auc = "N/A"
    if 'baseline' in results:
        baseline_auc = np.mean([results['baseline'].get(method, {}).get('auc', 0.5) for method in methods])
        baseline_auc = f"{baseline_auc:.3f}"
    
    print(f"| Attack AUC | {' | '.join(auc_values)} | {baseline_auc} |")
    
    # Attack TPR@10%FPR row
    tpr_values = []
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_tpr = np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
            tpr_values.append(f"{avg_tpr:.3f}")
        else:
            tpr_values.append("N/A")
    
    # Baseline TPR
    baseline_tpr = "N/A"
    if 'baseline' in results:
        baseline_tpr = np.mean([results['baseline'].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
        baseline_tpr = f"{baseline_tpr:.3f}"
    
    print(f"| Attack TPR@10%FPR | {' | '.join(tpr_values)} | {baseline_tpr} |")
    
    # Llamdex Accuracy row
    accuracy_values = []
    for eps in epsilon_values:
        acc = extract_llamdex_accuracy(eps)
        if acc is not None:
            accuracy_values.append(f"{acc:.3f}")
        else:
            accuracy_values.append("N/A")
    
    # Baseline accuracy
    baseline_acc = extract_llamdex_accuracy(None)
    if baseline_acc is not None:
        baseline_acc = f"{baseline_acc:.3f}"
    else:
        baseline_acc = "N/A"
    
    print(f"| Llamdex Accuracy | {' | '.join(accuracy_values)} | {baseline_acc} |")
    
    # Generate LaTeX table
    create_latex_table(results)
    
    print("\n### Detailed Analysis")
    print("\n#### AUC Results (by Method)")
    print("| Privacy (ε) | Loss | Confidence | Entropy | Correctness | Margin | Average |")
    print("|:-----------:|:----:|:----------:|:-------:|:-----------:|:------:|:-------:|")
    
    for key in sorted(results.keys(), key=lambda x: float('inf') if x == 'baseline' else float(x.split('_')[1])):
        if key == 'baseline':
            eps_str = "∞ (No DP)"
        else:
            eps_str = key.split('_')[1]
        
        aucs = []
        for method in methods:
            auc_val = results[key].get(method, {}).get('auc', 0.5)
            aucs.append(auc_val)
        
        avg_auc = np.mean(aucs)
        auc_strs = [f"{auc:.3f}" for auc in aucs]
        
        print(f"| {eps_str:<11} | {' | '.join(auc_strs)} | {avg_auc:.3f} |")
    
    print("\n#### TPR @ 10% FPR Results (by Method)")
    print("| Privacy (ε) | Loss | Confidence | Entropy | Correctness | Margin | Average |")
    print("|:-----------:|:----:|:----------:|:-------:|:-----------:|:------:|:-------:|")
    
    for key in sorted(results.keys(), key=lambda x: float('inf') if x == 'baseline' else float(x.split('_')[1])):
        if key == 'baseline':
            eps_str = "∞ (No DP)"
        else:
            eps_str = key.split('_')[1]
        
        tprs = []
        for method in methods:
            tpr_val = results[key].get(method, {}).get('tpr_at_10_fpr', 0.1)
            tprs.append(tpr_val)
        
        avg_tpr = np.mean(tprs)
        tpr_strs = [f"{tpr:.3f}" for tpr in tprs]
        
        print(f"| {eps_str:<11} | {' | '.join(tpr_strs)} | {avg_tpr:.3f} |")
    
    print("\n### Key Insights:")
    print("- **AUC = 0.5**: Random guessing performance")
    print("- **TPR@10%FPR = 0.1**: Random guessing at 10% false positive rate")
    print("- **Lower values**: Better privacy protection")
    print("- **Higher Llamdex Accuracy**: Better utility")
    print("- **MICO uses TPR@10%FPR** as the primary evaluation metric")
    
    # Best and worst methods
    if 'baseline' in results:
        baseline_results = results['baseline']
        method_scores = {method: baseline_results.get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods}
        best_method = max(method_scores.keys(), key=lambda x: method_scores[x])
        worst_method = min(method_scores.keys(), key=lambda x: method_scores[x])
        
        print(f"\n### Method Analysis (Baseline):")
        print(f"- **Most effective attack**: {best_method.capitalize()} (TPR@10%FPR: {method_scores[best_method]:.3f})")
        print(f"- **Least effective attack**: {worst_method.capitalize()} (TPR@10%FPR: {method_scores[worst_method]:.3f})")


def create_demo_combined_figure():
    """Create a demo figure with sample data for testing"""
    # Sample data for demonstration
    sample_results = {
        'eps_0.5': {
            'loss': {'auc': 0.489, 'tpr_at_10_fpr': 0.101},
            'confidence': {'auc': 0.488, 'tpr_at_10_fpr': 0.100},
            'entropy': {'auc': 0.490, 'tpr_at_10_fpr': 0.102}
        },
        'eps_1.0': {
            'loss': {'auc': 0.508, 'tpr_at_10_fpr': 0.059},
            'confidence': {'auc': 0.507, 'tpr_at_10_fpr': 0.058},
            'entropy': {'auc': 0.509, 'tpr_at_10_fpr': 0.060}
        },
        'eps_2.0': {
            'loss': {'auc': 0.507, 'tpr_at_10_fpr': 0.095},
            'confidence': {'auc': 0.506, 'tpr_at_10_fpr': 0.094},
            'entropy': {'auc': 0.508, 'tpr_at_10_fpr': 0.096}
        },
        'baseline': {
            'loss': {'auc': 0.521, 'tpr_at_10_fpr': 0.130},
            'confidence': {'auc': 0.520, 'tpr_at_10_fpr': 0.129},
            'entropy': {'auc': 0.522, 'tpr_at_10_fpr': 0.131}
        }
    }
    
    # Mock the extract_llamdex_accuracy function for demo
    def mock_extract_llamdex_accuracy(eps):
        if eps is None:
            return 0.755
        elif eps == 0.5:
            return 0.653
        elif eps == 1.0:
            return 0.757
        elif eps == 2.0:
            return 0.769
        else:
            return None
    
    # Temporarily replace the function
    global extract_llamdex_accuracy
    original_func = extract_llamdex_accuracy
    extract_llamdex_accuracy = mock_extract_llamdex_accuracy
    
    try:
        create_combined_table_plot(sample_results)
        print("Demo combined figure created: '_fig/privacy_utility_combined.png'")
    finally:
        # Restore original function
        extract_llamdex_accuracy = original_func


def main():
    print("Starting MICO-Style MIA Analysis...")
    
    # Ensure output directory exists
    os.makedirs('_fig', exist_ok=True)
    
    # Run comprehensive MICO-style analysis
    results = comprehensive_mico_analysis()
    
    # Create the new privacy-utility figure
    print("\nCreating privacy-utility tradeoff figure...")
    create_privacy_utility_figure(results)
    
    # Create results table
    print("\nGenerating MICO-style results table...")
    create_mico_results_table(results)
    
    print(f"\nAnalysis complete! Generated:")
    print(f"  - LaTeX table (printed above)")
    print(f"  - '_fig/privacy_utility_tradeoff.png' (dual y-axis plot)")
    
    # Save results for future reference
    with open('_log/mia_analysis/mico_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = {}
            for method, method_results in value.items():
                serializable_results[key][method] = {
                    'auc': float(method_results['auc']),
                    'tpr_at_10_fpr': float(method_results['tpr_at_10_fpr'])
                }
        json.dump(serializable_results, f, indent=2)
    
    return results


# Removed old table figure function - now using LaTeX table output instead


# Removed old combined plot function - now using dual y-axis figure instead

def create_combined_table_plot(results):
    """Create combined visualization with table and key plots"""
    plt.rcParams.update({'font.size': 11})
    
    fig = plt.figure(figsize=(16, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)
    
    # Top panel: Summary table (spans all columns)
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    
    # Create summary table data
    epsilon_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    methods = ['loss', 'confidence', 'entropy', 'correctness', 'margin']
    
    # Collect data for table
    table_data = []
    
    # Row 1: Attack AUC
    auc_row = ['Attack AUC']
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_auc = np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods])
            auc_row.append(f"{avg_auc:.3f}")
        else:
            auc_row.append("N/A")
    # Baseline AUC
    if 'baseline' in results:
        baseline_auc = np.mean([results['baseline'].get(method, {}).get('auc', 0.5) for method in methods])
        auc_row.append(f"{baseline_auc:.3f}")
    else:
        auc_row.append("N/A")
    table_data.append(auc_row)
    
    # Row 2: Attack TPR@10%FPR
    tpr_row = ['Attack TPR@10%FPR']
    for eps in epsilon_values:
        key = f'eps_{eps}'
        if key in results:
            avg_tpr = np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
            tpr_row.append(f"{avg_tpr:.3f}")
        else:
            tpr_row.append("N/A")
    # Baseline TPR
    if 'baseline' in results:
        baseline_tpr = np.mean([results['baseline'].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
        tpr_row.append(f"{baseline_tpr:.3f}")
    else:
        tpr_row.append("N/A")
    table_data.append(tpr_row)
    
    # Row 3: Llamdex Accuracy
    acc_row = ['Llamdex Accuracy']
    for eps in epsilon_values:
        acc = extract_llamdex_accuracy(eps)
        if acc is not None:
            acc_row.append(f"{acc:.3f}")
        else:
            acc_row.append("N/A")
    # Baseline accuracy
    baseline_acc = extract_llamdex_accuracy(None)
    if baseline_acc is not None:
        acc_row.append(f"{baseline_acc:.3f}")
    else:
        acc_row.append("N/A")
    table_data.append(acc_row)
    
    # Column headers
    col_headers = ['Metric'] + [f'ε={eps}' for eps in epsilon_values] + ['No DP (∞)']
    
    # Create table
    table = ax_table.table(cellText=table_data, 
                          colLabels=col_headers,
                          cellLoc='center',
                          loc='center',
                          colColours=['lightgray'] * len(col_headers))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)
    
    # Color code cells based on values
    for i in range(1, len(table_data) + 1):
        for j in range(1, len(col_headers)):
            if table_data[i-1][j] != "N/A":
                val = float(table_data[i-1][j])
                if i == 1 or i == 2:  # AUC and TPR rows (lower is better for privacy)
                    if val < 0.5:
                        table[(i, j)].set_facecolor('lightgreen')
                    elif val > 0.55:
                        table[(i, j)].set_facecolor('lightcoral')
                else:  # Accuracy row (higher is better for utility)
                    if val > 0.7:
                        table[(i, j)].set_facecolor('lightgreen')
                    elif val < 0.6:
                        table[(i, j)].set_facecolor('lightcoral')
    
    ax_table.set_title('Privacy-Utility Tradeoff Analysis', fontsize=16, fontweight='bold', pad=15)
    
    # Bottom plots
    # Plot 1: Privacy vs Utility Scatter
    ax1 = fig.add_subplot(gs[1, 0])
    
    # Extract data for plotting
    accuracy_data = []
    auc_data = []
    eps_labels = []
    colors = []
    
    for i, eps in enumerate(epsilon_values):
        acc = extract_llamdex_accuracy(eps)
        key = f'eps_{eps}'
        if key in results and acc is not None:
            accuracy_data.append(acc)
            auc_data.append(np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods]))
            eps_labels.append(f'ε={eps}')
            colors.append(plt.cm.viridis(i / len(epsilon_values)))
    
    # Add baseline
    baseline_acc = extract_llamdex_accuracy(None)
    if 'baseline' in results and baseline_acc is not None:
        accuracy_data.append(baseline_acc)
        auc_data.append(np.mean([results['baseline'].get(method, {}).get('auc', 0.5) for method in methods]))
        eps_labels.append('No DP')
        colors.append('red')
    
    # Create scatter plot
    scatter = ax1.scatter(auc_data, accuracy_data, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Annotate points
    for i, label in enumerate(eps_labels):
        ax1.annotate(label, (auc_data[i], accuracy_data[i]), xytext=(8, 8), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random AUC')
    ax1.set_xlabel('Attack AUC (Lower = Better Privacy)', fontweight='bold')
    ax1.set_ylabel('Llamdex Accuracy (Higher = Better Utility)', fontweight='bold')
    ax1.set_title('Privacy vs Utility Tradeoff', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Attack Success vs Epsilon
    ax2 = fig.add_subplot(gs[1, 1])
    
    # Extract epsilon and AUC data
    eps_plot = []
    auc_plot = []
    tpr_plot = []
    
    for key in sorted(results.keys(), key=lambda x: float('inf') if x == 'baseline' else float(x.split('_')[1])):
        if key == 'baseline':
            continue
        eps_val = float(key.split('_')[1])
        eps_plot.append(eps_val)
        auc_plot.append(np.mean([results[key].get(method, {}).get('auc', 0.5) for method in methods]))
        tpr_plot.append(np.mean([results[key].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods]))
    
    # Add baseline as horizontal lines
    if 'baseline' in results:
        baseline_auc_val = np.mean([results['baseline'].get(method, {}).get('auc', 0.5) for method in methods])
        baseline_tpr_val = np.mean([results['baseline'].get(method, {}).get('tpr_at_10_fpr', 0.1) for method in methods])
        ax2.axhline(y=baseline_auc_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label='No DP (AUC)')
        ax2.axhline(y=baseline_tpr_val, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='No DP (TPR)')
    
    ax2.plot(eps_plot, auc_plot, 'o-', color='blue', linewidth=2, markersize=8, label='Attack AUC')
    ax2.plot(eps_plot, tpr_plot, 's-', color='green', linewidth=2, markersize=8, label='Attack TPR@10%FPR')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (AUC)')
    ax2.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5, label='Random (TPR)')
    
    ax2.set_xlabel('Privacy Budget (ε)', fontweight='bold')
    ax2.set_ylabel('Attack Success Rate', fontweight='bold')
    ax2.set_title('Attack Success vs Privacy Budget', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Utility vs Epsilon
    ax3 = fig.add_subplot(gs[1, 2])
    
    # Extract accuracy data
    eps_acc_plot = []
    acc_vals = []
    
    for eps in epsilon_values:
        acc = extract_llamdex_accuracy(eps)
        if acc is not None:
            eps_acc_plot.append(eps)
            acc_vals.append(acc)
    
    baseline_acc_val = extract_llamdex_accuracy(None)
    
    ax3.plot(eps_acc_plot, acc_vals, 'o-', color='purple', linewidth=3, markersize=10, label='DP Models')
    if baseline_acc_val is not None:
        ax3.axhline(y=baseline_acc_val, color='red', linestyle='--', linewidth=3, alpha=0.8, label='No DP Baseline')
    
    ax3.set_xlabel('Privacy Budget (ε)', fontweight='bold')
    ax3.set_ylabel('Llamdex Accuracy', fontweight='bold')
    ax3.set_title('Model Utility vs Privacy Budget', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Set y-axis limits for better visualization
    if acc_vals:
        y_min = min(acc_vals) - 0.05
        y_max = max(acc_vals) + 0.05
        if baseline_acc_val is not None:
            y_min = min(y_min, baseline_acc_val - 0.05)
            y_max = max(y_max, baseline_acc_val + 0.05)
        ax3.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('_fig/privacy_utility_combined.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results = main()