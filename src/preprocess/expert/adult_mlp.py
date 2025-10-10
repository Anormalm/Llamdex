import os
import json

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


def train_adult_mlp(dataset_name, scaler=None, epsilon=None, delta=None, device='cuda'):
    use_dp = epsilon is not None and delta is not None

    print(f"Training MLP on {dataset_name} dataset")
    train_path = f"data/adult/clean/{dataset_name}_train.csv"
    test_path = f"data/adult/clean/{dataset_name}_test.csv"
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    # values sorted by column name
    X_train = train_df.drop(columns=['salary>50K']).sort_index(axis=1).to_numpy()
    y_train = train_df['salary>50K'].to_numpy()
    X_test = test_df.drop(columns=['salary>50K']).sort_index(axis=1).to_numpy()
    y_test = test_df['salary>50K'].to_numpy()

    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Train the model
    model = train_binary_cls_nn(X_train, y_train, X_test, y_test, epsilon=epsilon, delta=delta, device=device)

    # Save the model
    if use_dp:
        save_model_path = f"model/experts/{dataset_name}_mlp_eps-{epsilon}_delta-{delta}.pth"
    else:
        save_model_path = f"model/experts/{dataset_name}_mlp.pth"
    os.makedirs(f"model/experts", exist_ok=True)
    torch.save(model, save_model_path)
    print(f"Model saved to {save_model_path}")


if __name__ == '__main__':
    adult_root_dir = f"data/adult/"
    dataset_json_path = f"src/dataset/adult/adult.json"
    scaler = TableScaler(dataset_json=dataset_json_path)

    # train on clean data
    train_adult_mlp("adult", scaler)
    # train on synthetic data
    train_adult_mlp("syn_adult", scaler)

    # train on clean data with differential privacy
    delta = 1e-5        # around 1/|train_data|
    for epsilon in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        train_adult_mlp("adult", scaler, epsilon=epsilon, delta=delta, device='cuda')


