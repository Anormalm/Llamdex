import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# Define the CNN model
class WineQualityCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(WineQualityCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (num_features // 4), 128)  # Adjusted for pooling
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load data
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine train and test for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], axis=0)

    # Encode 'type' column
    le = LabelEncoder()
    combined_df['type'] = le.fit_transform(combined_df['type'])

    # Convert 'quality' to categorical type
    combined_df['quality'] = combined_df['quality'].astype('category')

    # Encode 'quality' using cat.codes
    combined_df['quality'] = combined_df['quality'].cat.codes

    # Split back into train and test
    train_df = combined_df.iloc[:len(train_df)]
    test_df = combined_df.iloc[len(train_df):]

    X_train = train_df.drop(columns=['quality']).values
    y_train = train_df['quality'].values
    X_test = test_df.drop(columns=['quality']).values
    y_test = test_df['quality'].values

    return X_train, y_train, X_test, y_test

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_path = "data/wine_quality/clean/wine_quality_train.csv"
test_path = "data/wine_quality/clean/wine_quality_test.csv"
X_train, y_train, X_test, y_test = load_data(train_path, test_path)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model parameters
num_features = X_train.shape[1]
num_classes = len(np.unique(np.concatenate((y_train.numpy(), y_test.numpy()))))  # Dynamically determine num_classes

# Initialize model, loss, and optimizer
model = WineQualityCNN(num_features, num_classes).to(device)
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

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Evaluate on test set
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
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Train the model
train(model, train_loader, test_loader, criterion, optimizer, epochs=30, device=device)