import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.dataset.synthetic import make_classification_from_json


def syn_nursery(dataset_json, n_samples=1000, random_state=0, cat_na_ratio=0.1, test_ratio=0.2,
                n_classes=4):
    syn_nursery_table = make_classification_from_json(n_samples=n_samples, dataset_json=dataset_json,
                                                      random_state=random_state, n_classes=n_classes)

    # Set some entries to be nan (missing values or not applicable)
    cat_columns = [col for col in syn_nursery_table.columns
                   if syn_nursery_table[col].dtype == 'category' and col != 'evaluation']
    print(f"{cat_columns=}")
    for col in cat_columns:
        syn_nursery_table.loc[syn_nursery_table.sample(frac=cat_na_ratio).index, col] = np.nan

    test_size = int(len(syn_nursery_table) * test_ratio)
    syn_test = syn_nursery_table[:test_size]
    syn_train = syn_nursery_table[test_size:]

    return syn_train, syn_test

if __name__ == '__main__':
    nursery_clean_root = 'data/nursery/clean'
    os.makedirs(nursery_clean_root, exist_ok=True)
    nursery_json_path = 'src/dataset/nursery/nursery.json'

    syn_nursery_train, syn_nursery_test = syn_nursery(nursery_json_path, n_samples=10000,
                                                        random_state=0, cat_na_ratio=0.1, test_ratio=0.2, n_classes=4)

    syn_nursery_train.to_csv(f'{nursery_clean_root}/syn_nursery_train.csv', index=False)
    syn_nursery_test.to_csv(f'{nursery_clean_root}/syn_nursery_test.csv', index=False)
    print("Synthetic nursery dataset saved.")



