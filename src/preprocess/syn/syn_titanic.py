import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import make_classification_from_json

def syn_titanic(dataset_json, n_samples=1000, random_state=0, cat_na_ratio=0.1, test_ratio=0.2):
    syn_titanic_table = make_classification_from_json(n_samples=n_samples, dataset_json=dataset_json, random_state=random_state)
    cat_columns = [col for col in syn_titanic_table.columns if syn_titanic_table[col].dtype == 'category']

    # Set some entries to be nan (missing values or not applicable)
    for col in cat_columns:
        syn_titanic_table.loc[syn_titanic_table.sample(frac=cat_na_ratio).index, col] = np.nan

    # Check some contradictions (to be implemented if needed)
    # syn_titanic_table = titanic_correct_contradictions(syn_titanic_table)

    # split the dataset into train and test
    test_size = int(len(syn_titanic_table) * test_ratio)
    syn_titanic_test = syn_titanic_table[:test_size]
    syn_titanic_train = syn_titanic_table[test_size:]

    return syn_titanic_train, syn_titanic_test

if __name__ == '__main__':
    titanic_clean_root = 'data/titanic/clean'
    os.makedirs(titanic_clean_root, exist_ok=True)
    titanic_json_path = 'src/dataset/titanic/titanic.json'
    syn_titanic_train, syn_titanic_test = syn_titanic(titanic_json_path, n_samples=10000, random_state=0,
                                                      cat_na_ratio=0.1, test_ratio=0.2)
    syn_titanic_train.to_csv(f'{titanic_clean_root}/syn_titanic_train.csv', index=False)
    syn_titanic_test.to_csv(f'{titanic_clean_root}/syn_titanic_test.csv', index=False)
    print("Synthetic Titanic dataset saved.")