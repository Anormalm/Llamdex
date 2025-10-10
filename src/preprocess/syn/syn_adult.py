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

# Feature columns without label term
# range: [min, max]
# Categories without '?' choice


# def refine_table(input_file: str, output_file: str):
#     """
#     Refine the adult dataset by removing the prefixes from the columns. Turn the one-hot encoded columns into a single
#     column with the category. For example, if the columns are "workclass_Private", "workclass_Self-emp-not-inc",
#     "workclass_Without-pay", etc., then the new column will be "workclass" with values "Private", "Self-emp-not-inc",
#     "Without-pay", etc.
#     Args:
#         input_file: The input CSV file.
#         output_file: The output CSV file.
#
#     Returns: new CSV file with refined columns in pandas DataFrame.
#
#     """
#     df = pd.read_csv(input_file)
#     new_df = pd.DataFrame()
#
#     processed_prefixes = set()
#
#     for col in df.columns:
#         if "_" in col:
#             prefix = col.split('_')[0]
#             if prefix in processed_prefixes:
#                 continue
#             related_cols = [c for c in df.columns if c.startswith(prefix + '_')]
#             related_categories = [c.split('_')[1] for c in related_cols]
#             print(", ".join(f'"{item}"' for item in related_categories))
#             for _, row in df[related_cols].iterrows():
#                 true_cols = [c for c in related_cols if row[c]]
#                 if len(true_cols) > 1:
#                     raise ValueError(f"Row {row.name} has {len(true_cols)} true columns")
#                 if prefix not in new_df.columns:
#                     new_df[prefix] = pd.Series()
#                 processed_prefixes.add(prefix)
#                 new_df.at[row.name, prefix] = true_cols[0].split('_')[1] if len(true_cols) == 1 else np.nan
#         else:
#             new_df[col] = df[col]
#
#     new_df.to_csv(output_file, index=False)
#     return new_df


def syn_adult(dataset_json, n_samples=1000, random_state=0, cat_na_ratio=0.1, test_ratio=0.2):
    syn_adult_table = make_classification_from_json(n_samples=n_samples, dataset_json=dataset_json, random_state=random_state)
    cat_columns = [col for col in syn_adult_table.columns if syn_adult_table[col].dtype == 'category']

    # Set some entries to be nan (missing values or not applicable)
    for col in cat_columns:
        syn_adult_table.loc[syn_adult_table.sample(frac=cat_na_ratio).index, col] = np.nan

    # Check some contradictions (to be checked)
    # syn_adult_table = adult_correct_contradictions(syn_adult_table)

    # split the dataset into train and test
    test_size = int(len(syn_adult_table) * test_ratio)
    syn_adult_test = syn_adult_table[:test_size]
    syn_adult_train = syn_adult_table[test_size:]

    return syn_adult_train, syn_adult_test


if __name__ == '__main__':
    adult_clean_root = 'data/adult/clean'
    os.makedirs(adult_clean_root, exist_ok=True)
    adult_json_path = 'src/dataset/adult/adult.json'
    syn_adult_train, syn_adult_test = syn_adult(adult_json_path, n_samples=10000, random_state=0,
                                                cat_na_ratio=0.1, test_ratio=0.2)
    syn_adult_train.to_csv(f'{adult_clean_root}/syn_adult_train.csv', index=False)
    syn_adult_test.to_csv(f'{adult_clean_root}/syn_adult_test.csv', index=False)
    print("Synthetic adult dataset saved.")




