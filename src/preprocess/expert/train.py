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

import opacus
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from tqdm import tqdm

from src.preprocess.DataScaler import TableScaler


def train_nn(X_train, y_train, X_test, y_test, task='binary',
             device='cuda', n_epochs=10, hidden_channels=None, lr=1e-4, batch_size=64,
             epsilon=None, delta=None, sigma=None, n_classes=1):
    assert task in ['bin-cls', 'reg', 'multi-cls'], \
        f"Invalid task: {task}, must be one of ['bin-cls', 'reg', 'multi-cls']"

    use_dp = (epsilon is not None and delta is not None) or sigma is not None

    if hidden_channels is None:
        hidden_channels = [400, 200, n_classes]

    # Train the model using GPU
    model = nn.Sequential(
        MLP(in_channels=X_train.shape[1], hidden_channels=hidden_channels, norm_layer=nn.LayerNorm),
        nn.Sigmoid()
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # determine type of y
    if task == 'bin-cls':
        y_dtype = torch.float32
        criterion = nn.BCELoss()
    elif task == 'multi-cls':
        y_dtype = torch.long
        criterion = nn.CrossEntropyLoss()
    elif task == 'reg':
        y_dtype = torch.float32
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Invalid task: {task}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=y_dtype))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=y_dtype))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if use_dp:
        validator = ModuleValidator()
        validator.validate(model, strict=True)

        privacy_engine = opacus.PrivacyEngine()
        if sigma is None:
            # Use epsilon-delta to determine sigma
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model, optimizer=optimizer, data_loader=train_loader,
                epochs=n_epochs, target_epsilon=epsilon, target_delta=delta, max_grad_norm=1.0
            )
        else:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model, optimizer=optimizer, data_loader=train_loader,
                noise_multiplier=sigma, max_grad_norm=1.0
            )
        print(f"Using sigma = {optimizer.noise_multiplier}, Max grad norm = {optimizer.max_grad_norm}"
              f"Target epsilon = {epsilon}, Target delta = {delta}")

    for epoch in range(n_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        if use_dp:
            epoch_data_loader = BatchMemoryManager(data_loader=train_loader,
                                                   max_physical_batch_size=batch_size,
                                                   optimizer=optimizer).__enter__()
        else:
            epoch_data_loader = train_loader

        for X_batch, y_batch in tqdm(epoch_data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.reshape(-1, 1)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            if task == 'multi-cls':
                y_batch = y_batch.flatten()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            if task == 'bin-cls':
                train_correct += (y_pred > 0.5).eq(y_batch).sum().item()
            elif task == 'multi-cls':
                train_correct += y_pred.argmax(dim=1).eq(y_batch).sum().item()
            elif task == 'reg':
                train_loss += loss.item() * y_batch.size(0)  # sum of loss
            else:
                raise ValueError(f"Invalid task: {task}")

            train_total += y_batch.size(0)
        train_acc = train_correct / train_total
        train_loss /= train_total

        if use_dp:
            epsilon = privacy_engine.get_epsilon(delta)
            print(f"Epoch {epoch}: Epsilon: {epsilon:.2f}, Delta: {delta}")

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.reshape(-1, 1)
            y_pred = model(X_batch)
            if task == 'multi-cls':
                y_batch = y_batch.flatten()
            loss = criterion(y_pred, y_batch)

            if task == 'bin-cls':
                test_correct += (y_pred > 0.5).eq(y_batch).sum().item()
            elif task == 'multi-cls':
                test_correct += y_pred.argmax(dim=1).eq(y_batch).sum().item()
            elif task == 'reg':
                test_loss += loss.item() * y_batch.size(0)
            else:
                raise ValueError(f"Invalid task: {task}")

            test_total += y_batch.size(0)
        test_acc = test_correct / test_total
        test_loss /= test_total

        if task == 'reg':
            print(f"Epoch {epoch}: Train MSE loss: {train_loss:.4f}, Test MSE loss: {test_loss:.4f}")
        else:
            print(f"Epoch {epoch}: Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    if use_dp:
        # Remove the addition hook module in model
        model = model._module

    return model


def train_binary_cls_nn(X_train, y_train, X_test, y_test, **kwargs):
    return train_nn(X_train, y_train, X_test, y_test, task='bin-cls', **kwargs)


def train_multi_cls_nn(X_train, y_train, X_test, y_test, **kwargs):
    return train_nn(X_train, y_train, X_test, y_test, task='multi-cls', **kwargs)


def train_reg_nn(X_train, y_train, X_test, y_test, **kwargs):
    return train_nn(X_train, y_train, X_test, y_test, task='reg', **kwargs)
