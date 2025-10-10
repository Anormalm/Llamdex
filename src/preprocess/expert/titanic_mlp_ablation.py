import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.ops import MLP
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from src.preprocess.DataScaler import TableScaler
from src.preprocess.expert.train import train_binary_cls_nn


def train_titanic_mlp_ablation(dataset_name, age_bound, scaler=None, epsilon=None, delta=None, device='cuda', use_age_specific_scaler=True, layer=None):
    """
    Train MLP on Titanic ablation dataset with specific age range
    
    Args:
        dataset_name: Base dataset name (e.g., "syn_titanic_ablation")
        age_bound: Age upper bound for ablation (e.g., 200, 1000, 5000)
        scaler: Data scaler to use
        epsilon: Differential privacy parameter
        delta: Differential privacy parameter
        device: Device to train on
        use_age_specific_scaler: Whether to use age-specific configuration
        layer: Layer information for model naming
    """
    print(f"Training MLP on {dataset_name}_age{age_bound} dataset (age range: [0, {age_bound}])")
    
    # Construct file paths for ablation datasets
    train_path = f"data/titanic/ablation/{dataset_name}_age{age_bound}_train.csv"
    test_path = f"data/titanic/ablation/{dataset_name}_age{age_bound}_test.csv"
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    # Load data
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    # Prepare features and labels (values sorted by column name)
    X_train = train_df.drop(columns=['Survived']).sort_index(axis=1).to_numpy()
    y_train = train_df['Survived'].to_numpy()
    X_test = test_df.drop(columns=['Survived']).sort_index(axis=1).to_numpy()
    y_test = test_df['Survived'].to_numpy()
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Age range restriction: [0, {age_bound}] years")

    # Force age-specific scaler for ablation studies
    if use_age_specific_scaler:
        # Use age-specific scaler for proper normalization
        print(f"Using age-specific scaler for range [0, {age_bound}]...")
        age_specific_json_path = f"src/dataset/titanic/titanic_ablation_age{age_bound}.json"
        if os.path.exists(age_specific_json_path):
            from src.preprocess.DataScaler import TableScaler
            age_scaler = TableScaler(dataset_json=age_specific_json_path)
            X_train = age_scaler.transform(X_train)
            X_test = age_scaler.transform(X_test)
            print(f"✅ Applied age-specific scaling for range [0, {age_bound}]")
        else:
            print(f"⚠️  Age-specific JSON not found: {age_specific_json_path}")
            print("Using data without scaling")
    elif scaler is not None:
        print("Applying provided data scaling...")
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Train the model
    print("Training MLP model...")
    model = train_binary_cls_nn(X_train, y_train, X_test, y_test, epsilon=epsilon, delta=delta, device=device)

    # Construct save path (no layer info - expert model is layer-agnostic)
    if epsilon is not None and delta is not None:
        save_model_path = f"model/experts/{dataset_name}_age{age_bound}_mlp_eps-{epsilon}_delta-{delta}.pth"
    else:
        save_model_path = f"model/experts/{dataset_name}_age{age_bound}_mlp.pth"
    
    # Save the model
    os.makedirs(f"model/experts", exist_ok=True)
    torch.save(model, save_model_path)
    print(f"Model saved to {save_model_path}")
    
    return model, save_model_path


def main():
    parser = argparse.ArgumentParser(description='Train MLP models for Titanic ablation study')
    parser.add_argument('--dataset_name', type=str, default='syn_titanic_ablation',
                        help='Base dataset name (default: syn_titanic_ablation)')
    parser.add_argument('--age_bounds', type=int, nargs='+', default=[200, 1000, 5000],
                        help='Age upper bounds for ablation study (default: [200, 1000, 5000])')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (default: cuda)')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Differential privacy epsilon parameter')
    parser.add_argument('--delta', type=float, default=None,
                        help='Differential privacy delta parameter')
    parser.add_argument('--use_scaler', action='store_true', default=False,
                        help='Whether to use original data scaler (default: False)')
    parser.add_argument('--use_age_specific_scaler', action='store_true', default=True,
                        help='Whether to use age-specific scaler (default: True)')
    parser.add_argument('--layer', type=int, default=30,
                        help='Layer number for model identification (default: 30)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TITANIC ABLATION STUDY - MLP EXPERT TRAINING")
    print("="*60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Age bounds: {args.age_bounds}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print(f"Use original scaler: {args.use_scaler}")
    print(f"Use age-specific scaler: {args.use_age_specific_scaler}")
    if args.epsilon is not None:
        print(f"Differential Privacy: ε={args.epsilon}, δ={args.delta}")
    print("="*60)
    
    # Initialize scaler
    scaler = None
    if args.use_scaler:
        print("Initializing data scaler...")
        titanic_root_dir = f"data/titanic/"
        dataset_json_path = f"src/dataset/titanic/titanic.json"
        scaler = TableScaler(dataset_json=dataset_json_path)
        print("✅ Data scaler initialized")
    
    # Train models for each age bound
    successful_training = 0
    failed_training = 0
    trained_models = []
    
    for age_bound in args.age_bounds:
        try:
            print(f"\n🔄 Training model for age range [0, {age_bound}]...")
            model, save_path = train_titanic_mlp_ablation(
                dataset_name=args.dataset_name,
                age_bound=age_bound,
                scaler=scaler,
                epsilon=args.epsilon,
                delta=args.delta,
                device=args.device,
                use_age_specific_scaler=True
            )
            
            successful_training += 1
            trained_models.append((age_bound, save_path))
            print(f"✅ Successfully trained model for age range [0, {age_bound}]")
            
        except Exception as e:
            print(f"❌ Failed to train model for age range [0, {age_bound}]: {str(e)}")
            failed_training += 1
            continue
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total age ranges: {len(args.age_bounds)}")
    print(f"Successfully trained: {successful_training}")
    print(f"Failed training: {failed_training}")
    
    if trained_models:
        print("\nTrained Models:")
        for age_bound, model_path in trained_models:
            model_size = os.path.getsize(model_path) / (1024*1024)  # MB
            print(f"  ✅ Age [0, {age_bound}]: {model_path} ({model_size:.1f} MB)")
    
    if failed_training == 0:
        print("\n🎉 All models trained successfully!")
    else:
        print(f"\n⚠️  {failed_training} models failed to train.")
    
    print("="*60)


if __name__ == '__main__':
    main()

# Example usage:
# python src/preprocess/expert/titanic_mlp_ablation.py
# python src/preprocess/expert/titanic_mlp_ablation.py --age_bounds 20 40 60
# python src/preprocess/expert/titanic_mlp_ablation.py --epsilon 1.0 --delta 1e-3 