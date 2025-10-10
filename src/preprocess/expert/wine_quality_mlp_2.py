import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.ops import MLP
import pandas as pd
import numpy as np
import os
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

# --- Define the Model ---
class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes, dropout_rate=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_channels[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_channels[0]))
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_channels) - 1):
            layers.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_channels[i+1]))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_channels[-1], num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def train_wine_quality_mlp(dataset_name, scaler=None, epsilon=None, delta=None,
                             learning_rate=1e-3, batch_size=64, num_epochs=50,
                             hidden_channels=[400, 200], dropout_rate=0.2):
    use_dp = epsilon is not None and delta is not None

    print(f"Training MLP on {dataset_name} dataset")
    train_path = f"data/wine_quality/clean/{dataset_name}_train.csv"
    test_path = f"data/wine_quality/clean/{dataset_name}_test.csv"
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    # Encode the quality labels
    all_labels = list('ABCDEFGHIJK')
    le = CustomLabelEncoder(all_labels)
    train_df['quality'] = le.transform(train_df['quality'])
    test_df['quality'] = le.transform(test_df['quality'])

    # values sorted by column name
    X_train = train_df.drop(columns=['quality']).sort_index(axis=1).to_numpy()
    y_train = train_df['quality'].to_numpy()
    X_test = test_df.drop(columns=['quality']).sort_index(axis=1).to_numpy()
    y_test = test_df['quality'].to_numpy()

    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    num_classes = len(le.classes_)
    print(num_classes)

    # --- Training Loop ---
    def train_loop(model, train_loader, test_loader, optimizer, criterion, num_epochs, device):
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

            # Validation Step
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(test_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # --- Initialize Model, Optimizer, Criterion, and DataLoaders ---
    model = SimpleMLP(X_train.shape[1], hidden_channels, num_classes, dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Train the Model ---
    train_loop(model, train_loader, test_loader, optimizer, criterion, num_epochs, device='cuda')

    # Save the model
    if use_dp:
        save_model_path = f"model/experts/{dataset_name}_mlp_eps-{epsilon}_delta-{delta}.pth"
    else:
        save_model_path = f"model/experts/{dataset_name}_mlp.pth"
    os.makedirs(f"model/experts", exist_ok=True)
    torch.save(model, save_model_path)
    print(f"Model saved to {save_model_path}")

if __name__ == '__main__':
    wine_quality_root_dir = f"data/wine_quality/"
    dataset_json_path = f"src/dataset/wine_quality/wine_quality.json"
    scaler = TableScaler(dataset_json=dataset_json_path)

    # train on clean data
    train_wine_quality_mlp("wine_quality", scaler,
                             learning_rate=1e-3, batch_size=256, num_epochs=100,
                             hidden_channels=[150, 100, 50], dropout_rate=0.0)
    # train on synthetic data
    train_wine_quality_mlp("syn_wine_quality", scaler,
                           learning_rate=1e-3, batch_size=256, num_epochs=100,
                           hidden_channels=[150, 100, 50], dropout_rate=0.0)
    #
    # # train on clean data with differential privacy
    # delta = 1e-3  # around 1/|train_data|
    # for epsilon in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
    #     train_wine_quality_mlp("wine_quality", scaler, epsilon=epsilon, delta=delta)