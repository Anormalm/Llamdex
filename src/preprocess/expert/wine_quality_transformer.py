import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import your custom scaler
from src.preprocess.DataScaler import TableScaler  # Make sure DataScaler.py is in src/preprocess


# Define the Transformer model for wine quality classification
class WineQualityTransformer(nn.Module):
    def __init__(self, num_features, num_classes, num_layers=1, num_heads=2, dropout=0.1, hidden_dim=64):
        super(WineQualityTransformer, self).__init__()
        # Embed the input features to a higher dimensional space (d_model=64)
        self.embedding = nn.Linear(num_features, hidden_dim)
        # Transformer encoder; using batch_first=True so input shape is (batch_size, seq_length, features)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        # Classification layer that maps from the encoder's output to the number of classes
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x


# Load and preprocess the dataset
def load_data(train_path, test_path, table_scaler):
    # Read CSV files for train and test sets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Define a mapping for quality labels. Adjust this dictionary as per your dataset.
    quality_mapping = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
        'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10
    }
    train_df['quality'] = train_df['quality'].map(quality_mapping)
    test_df['quality'] = test_df['quality'].map(quality_mapping)

    # Separate features and labels
    X_train = train_df.drop(columns=['quality']).values
    y_train = train_df['quality'].values
    X_test = test_df.drop(columns=['quality']).values
    y_test = test_df['quality'].values

    # Scale the features using your custom TableScaler
    X_train = table_scaler.transform(X_train)
    X_test = table_scaler.transform(X_test)

    print("Sample scaled features (first 5 rows):", X_train[:5])
    print("Sample labels (first 5):", y_train[:5])

    return X_train, y_train, X_test, y_test


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your TableScaler with the dataset JSON configuration
dataset_json_path = "src/dataset/wine_quality/wine_quality.json"  # Update this path if needed
table_scaler = TableScaler(dataset_json=dataset_json_path)

# Paths to the training and testing CSV files
train_path = "data/wine_quality/clean/wine_quality_train.csv"
test_path = "data/wine_quality/clean/wine_quality_test.csv"
X_train, y_train, X_test, y_test = load_data(train_path, test_path, table_scaler)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Determine the number of input features and classes dynamically
num_features = X_train.shape[1]
num_classes = 11

# Initialize the model, loss function, and optimizer
model = WineQualityTransformer(num_features, num_classes).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_params = count_parameters(model)
print(f"The number of trainable parameters in the model is: {num_params}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Training phase
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_predictions / total_samples
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Evaluation phase on the test set
        model.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Calculate test accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_predictions / total_samples
        print(
            f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")


# Train the model for 30 epochs
train(model, train_loader, test_loader, criterion, optimizer, epochs=30, device=device)