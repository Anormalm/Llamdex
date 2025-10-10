import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import make_classification_from_json


def syn_abalone(dataset_json, n_samples=1000, random_state=0, cat_na_ratio=0.1, test_ratio=0.2):
    syn_abalone_table = make_classification_from_json(n_samples=n_samples, dataset_json=dataset_json,
                                                      random_state=random_state, n_classes=3)

    cat_columns = [col for col in syn_abalone_table.columns if (syn_abalone_table[col].dtype == 'category' and col != 'Age_Group')]
    float_columns = [col for col in syn_abalone_table.columns if syn_abalone_table[col].dtype == 'float64']

    # Round float columns to 4 decimal places
    for col in float_columns:
        syn_abalone_table[col] = syn_abalone_table[col].round(4)

    # Set some entries to be nan (missing values or not applicable)
    for col in cat_columns:
        syn_abalone_table.loc[syn_abalone_table.sample(frac=cat_na_ratio).index, col] = np.nan

    # split the dataset into train and test
    test_size = int(len(syn_abalone_table) * test_ratio)
    syn_abalone_test = syn_abalone_table[:test_size]
    syn_abalone_train = syn_abalone_table[test_size:]

    return syn_abalone_train, syn_abalone_test


if __name__ == '__main__':
    abalone_clean_root = 'data/abalone/clean'
    os.makedirs(abalone_clean_root, exist_ok=True)
    abalone_json_path = 'src/dataset/abalone/abalone.json'

    syn_abalone_train, syn_abalone_test = syn_abalone(abalone_json_path, n_samples=10000,
                                                      random_state=0, cat_na_ratio=0.1, test_ratio=0.2)

    syn_abalone_train.to_csv(f'{abalone_clean_root}/syn_abalone_train.csv', index=False)
    syn_abalone_test.to_csv(f'{abalone_clean_root}/syn_abalone_test.csv', index=False)
    print("Synthetic abalone dataset saved.")

    # Print some information about the generated dataset
    print("\nTrain dataset shape:", syn_abalone_train.shape)
    print("Test dataset shape:", syn_abalone_test.shape)
    print("\nTrain dataset columns:", syn_abalone_train.columns.tolist())
    print("\nTrain dataset info:")
    print(syn_abalone_train.info())
    print("\nTrain dataset 'Age_Group' distribution:")
    print(syn_abalone_train['Age_Group'].value_counts(normalize=True))

