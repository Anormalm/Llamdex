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
from train import train_multi_cls_nn


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

def train_nursery_mlp(dataset_name, scaler=None, epsilon=None, delta=None):
    use_dp = epsilon is not None and delta is not None

    label = 'evaluation'

    print(f"Training MLP on {dataset_name} dataset")
    train_path = f"data/nursery/clean/{dataset_name}_train.csv"
    test_path = f"data/nursery/clean/{dataset_name}_test.csv"
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    # Encode the Age_Group labels
    all_labels = list('ABCD')  # Assuming Age_Group has categories A through C
    le = CustomLabelEncoder(all_labels)
    train_df[label] = le.transform(train_df[label])
    test_df[label] = le.transform(test_df[label])

    # values sorted by column name
    X_train = train_df.drop(columns=[label]).sort_index(axis=1).to_numpy()
    y_train = train_df[label].to_numpy()
    X_test = test_df.drop(columns=[label]).sort_index(axis=1).to_numpy()
    y_test = test_df[label].to_numpy()

    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    num_classes = len(le.classes_)
    print(num_classes)

    # Train the model (Use 60 epochs because 10 epochs is not enough for the model to converge)
    model = train_multi_cls_nn(X_train, y_train, X_test, y_test, n_classes=num_classes,
                               epsilon=epsilon, delta=delta, n_epochs=20)

    # Save the model
    # if use_dp:
    #     save_model_path = f"model/experts/{dataset_name}_mlp_eps-{epsilon}_delta-{delta}.pth"
    # else:
    #     save_model_path = f"model/experts/{dataset_name}_mlp.pth"
    # os.makedirs(f"model/experts", exist_ok=True)
    # torch.save(model, save_model_path)
    # print(f"Model saved to {save_model_path}")

if __name__ == '__main__':
    nursery_root_dir = f"data/nursery/"
    dataset_json_path = f"src/dataset/nursery/nursery.json"
    scaler = TableScaler(dataset_json=dataset_json_path)

    # # train on clean data
    # train_nursery_mlp("nursery", scaler)
    # # train on synthetic data
    # train_nursery_mlp("syn_nursery", scaler)

    # train on clean data with differential privacy
    delta = 1e-4
    for epsilon in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        train_nursery_mlp("nursery", scaler, epsilon=epsilon, delta=delta)
