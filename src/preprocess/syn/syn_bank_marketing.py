import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.datasets import make_classification
from tqdm import tqdm
from torchvision.ops import MLP
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import random
from tqdm import tqdm
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import make_classification_from_json


def syn_bank_marketing(dataset_json, n_samples=1000, random_state=0, cat_na_ratio=0.1, test_ratio=0.2):
    syn_bank_marketing_table = make_classification_from_json(n_samples=n_samples, dataset_json=dataset_json, random_state=random_state)
    cat_columns = [col for col in syn_bank_marketing_table.columns if syn_bank_marketing_table[col].dtype == 'category']

    # Set some entries to be nan (missing values or not applicable)
    for col in cat_columns:
        syn_bank_marketing_table.loc[syn_bank_marketing_table.sample(frac=cat_na_ratio).index, col] = np.nan

    # split the dataset into train and test
    test_size = int(len(syn_bank_marketing_table) * test_ratio)
    syn_bank_marketing_test = syn_bank_marketing_table[:test_size]
    syn_bank_marketing_train = syn_bank_marketing_table[test_size:]

    return syn_bank_marketing_train, syn_bank_marketing_test


if __name__ == '__main__':
    bank_marketing_clean_root = 'data/bank_marketing/clean'
    os.makedirs(bank_marketing_clean_root, exist_ok=True)
    bank_marketing_json_path = 'src/dataset/bank_marketing/bank_marketing.json'
    syn_bank_marketing_train, syn_bank_marketing_test = syn_bank_marketing(bank_marketing_json_path, n_samples=10000, random_state=0,
                                                                           cat_na_ratio=0.1, test_ratio=0.2)
    syn_bank_marketing_train.to_csv(f'{bank_marketing_clean_root}/syn_bank_marketing_train.csv', index=False)
    syn_bank_marketing_test.to_csv(f'{bank_marketing_clean_root}/syn_bank_marketing_test.csv', index=False)
    print("Synthetic bank marketing dataset saved.")