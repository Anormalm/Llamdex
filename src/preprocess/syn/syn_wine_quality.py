import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import make_classification_from_json


def syn_wine_quality(dataset_json, n_samples=1000, random_state=0, cat_na_ratio=0.1, test_ratio=0.2):
    syn_wine_quality_table = make_classification_from_json(n_samples=n_samples, dataset_json=dataset_json,
                                                           random_state=random_state, n_classes=11)

    cat_columns = [col for col in syn_wine_quality_table.columns if (syn_wine_quality_table[col].dtype == 'category' and col != 'quality')]
    float_columns = [col for col in syn_wine_quality_table.columns if syn_wine_quality_table[col].dtype == 'float64']

    # Round float columns to 4 decimal places
    for col in float_columns:
        syn_wine_quality_table[col] = syn_wine_quality_table[col].round(4)

    # Set some entries to be nan (missing values or not applicable)
    for col in cat_columns:
        syn_wine_quality_table.loc[syn_wine_quality_table.sample(frac=cat_na_ratio).index, col] = np.nan

    # split the dataset into train and test
    test_size = int(len(syn_wine_quality_table) * test_ratio)
    syn_wine_quality_test = syn_wine_quality_table[:test_size]
    syn_wine_quality_train = syn_wine_quality_table[test_size:]

    return syn_wine_quality_train, syn_wine_quality_test


if __name__ == '__main__':
    wine_quality_clean_root = 'data/wine_quality/clean'
    os.makedirs(wine_quality_clean_root, exist_ok=True)
    wine_quality_json_path = 'src/dataset/wine_quality/wine_quality.json'

    syn_wine_quality_train, syn_wine_quality_test = syn_wine_quality(wine_quality_json_path, n_samples=10000,
                                                                     random_state=0, cat_na_ratio=0.1, test_ratio=0.2)

    syn_wine_quality_train.to_csv(f'{wine_quality_clean_root}/syn_wine_quality_train.csv', index=False)
    syn_wine_quality_test.to_csv(f'{wine_quality_clean_root}/syn_wine_quality_test.csv', index=False)
    print("Synthetic wine quality dataset saved.")