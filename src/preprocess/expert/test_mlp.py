import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.ops import MLP
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm

from src.preprocess.DataScaler import TableScaler
# from train import train_multi_cls_nn # No need to import


class CustomLabelEncoder:
    def __init__(self, labels):
        self.labels = labels
        self.classes_ = np.array(labels)

    def fit(self, y):
        return self

    def transform(self, y):
        return np.array([self.labels.index(item) if item in self.labels else -1 for item in y])

    def inverse_transform(self, y):
        return np.array([self.labels[item] if 0 <= item < len(self.labels) else None for item in y])

def evaluate_wine_quality_mlp(model_path, dataset_name, scaler=None, device='cuda'):
    """
    Loads a pretrained MLP model and evaluates its test accuracy on the specified wine quality dataset.

    Args:
        model_path (str): Path to the pretrained model.
        dataset_name (str): Name of the dataset to evaluate (e.g., "wine_quality" or "syn_wine_quality").
        scaler (TableScaler, optional): TableScaler object for data scaling. If provided, data will be scaled.
        device (str, optional): Device for evaluation ('cuda' or 'cpu'). Defaults to 'cuda'.
    """

    print(f"Evaluating MLP on {dataset_name} dataset")
    test_path = f"data/wine_quality/clean/{dataset_name}_test.csv"
    test_df = pd.read_csv(test_path, header=0)

    # 1. Load the model
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()  # Set the model to evaluation mode

    # 2. Prepare the data (exactly the same as during training)
    all_labels = list('ABCDEFGHIJK')
    le = CustomLabelEncoder(all_labels)  # Use the same labels as during training
    test_df['quality'] = le.transform(test_df['quality'])

    X_test = test_df.drop(columns=['quality']).sort_index(axis=1).to_numpy()
    y_test = test_df['quality'].to_numpy()

    if scaler is not None:
        X_test = scaler.transform(X_test)

    # 3. Create DataLoader
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Same batch_size as training, no shuffle

    # 4. Evaluate the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    test_correct = 0
    test_total = 0
    # test_loss = 0 # No need to calculate the loss
    criterion = nn.CrossEntropyLoss() # Need to define the loss function

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for X_batch, y_batch in tqdm(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.reshape(-1, 1)
            y_pred = model(X_batch)
            y_batch = y_batch.flatten() # Need to flatten y_batch for multi-class classification
            # loss = criterion(y_pred, y_batch) # No need to calculate the loss

            test_correct += y_pred.argmax(dim=1).eq(y_batch).sum().item()
            test_total += y_batch.size(0)

    test_acc = test_correct / test_total
    # print(f"Test Accuracy: {test_acc:.4f}") # Don't print the result here.
    return test_acc

if __name__ == '__main__':
    wine_quality_root_dir = f"data/wine_quality/"
    dataset_json_path = f"src/dataset/wine_quality/wine_quality.json"
    scaler = TableScaler(dataset_json=dataset_json_path)

     # Store accuracies for different epsilons
    accuracies = {}

    # Evaluate the non-DP model
    model_path = "model/experts/wine_quality_mlp.pth"
    if os.path.exists(model_path):
        acc = evaluate_wine_quality_mlp(model_path, "wine_quality", scaler)
        accuracies["non-DP"] = f"{acc*100:.2f}"
    else:
        print(f"Warning: Non-DP model not found at {model_path}")
        accuracies["non-DP"] = "N/A"


    # Evaluate DP models
    delta = 1e-3
    for epsilon in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:  # Removed 6.0
        model_path_dp = f"model/experts/wine_quality_mlp_eps-{epsilon}_delta-{delta}.pth"
        if os.path.exists(model_path_dp):
            acc = evaluate_wine_quality_mlp(model_path_dp, "wine_quality", scaler)
            accuracies[f"eps-{epsilon}"] = f"{acc*100:.2f}"
        else:
            print(f"Warning: DP model not found at {model_path_dp}")
            accuracies[f"eps-{epsilon}"] = "N/A"

    # Create Markdown table
    print("Dataset | ε=0.5 | ε=1 | ε=2 | ε=3 | ε=4 | ε=5 | Non-DP")
    print("------- | ----- | --- | --- | --- | --- | --- | ------")
    print(
        f"Wine Quality | {accuracies.get('eps-0.5', 'N/A')} | {accuracies.get('eps-1.0', 'N/A')} | {accuracies.get('eps-2.0', 'N/A')} | {accuracies.get('eps-3.0', 'N/A')} | {accuracies.get('eps-4.0', 'N/A')} | {accuracies.get('eps-5.0', 'N/A')} | {accuracies.get('non-DP', 'N/A')}")