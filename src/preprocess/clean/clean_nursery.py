import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

"""
   The hierarchical model ranks nursery-school applications according
   to the following concept structure:

   NURSERY            Evaluation of applications for nursery schools
   . EMPLOY           Employment of parents and child's nursery
   . . parents        Parents' occupation
   . . has_nurs       Child's nursery
   . STRUCT_FINAN     Family structure and financial standings
   . . STRUCTURE      Family structure
   . . . form         Form of the family
   . . . children     Number of children
   . . housing        Housing conditions
   . . finance        Financial standing of the family
   . SOC_HEALTH       Social and health picture of the family
   . . social         Social conditions
   . . health         Health conditions

   Attribute Values:
   parents        usual, pretentious, great_pret
   has_nurs       proper, less_proper, improper, critical, very_crit
   form           complete, completed, incomplete, foster
   children       1, 2, 3, more
   housing        convenient, less_conv, critical
   finance        convenient, inconv
   social         non-prob, slightly_prob, problematic
   health         recommended, priority, not_recom
   
   Class Distribution (number of instances per class)

   class        N         N[%]
   ------------------------------
   not_recom    4320   (33.333 %)
   recommend       2   ( 0.015 %)
   very_recom    328   ( 2.531 %)
   priority     4266   (32.917 %)
   spec_prior   4044   (31.204 %)

"""

def clean_nursery(raw_dir: str, dataset_json: str | dict) -> (pd.DataFrame, pd.DataFrame):
    if isinstance(dataset_json, str):
        dataset_json = json.load(open(dataset_json, 'r'))

    original_columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'evaluation']
    categorical_columns = [col for col, info in dataset_json['X'].items() if info['type'] == 'category']

    data_df = pd.read_csv(f"{raw_dir}/nursery.data", header=None, names=original_columns)
    print(f"Data shape: {data_df.shape}")

    # trim all the string columns
    data_df = data_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # remove rows with label 'recommend' as it has only 2 instances
    data_df = data_df[data_df['evaluation'] != 'recommend']

    # encode label
    label_mapping = {'not_recom': 'D', 'very_recom': 'C', 'priority': 'B', 'spec_prior': 'A'}
    data_df['evaluation'] = data_df['evaluation'].map(label_mapping)

    # sort columns by name
    data_df = data_df.reindex(sorted(data_df.columns), axis=1)

    # split data into train and test
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=0)

    return train_df, test_df


if __name__ == '__main__':
    nursery_root = os.path.join(os.path.dirname(__file__), "../../../data/nursery/")
    train_df, test_df = clean_nursery(f"{nursery_root}/raw",
                                        f"src/dataset/nursery/nursery.json")
    assert np.all(train_df.columns == test_df.columns)
    print(train_df.columns)

    print(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

    os.makedirs(f"{nursery_root}/clean", exist_ok=True)
    if os.path.exists(f"{nursery_root}/clean/nursery_train.csv") and os.path.exists(f"{nursery_root}/clean/nursery_test.csv"):
        input("Data already exists. Press Enter to overwrite...")

    train_df.to_csv(f"{nursery_root}/clean/nursery_train.csv", index=False)
    test_df.to_csv(f"{nursery_root}/clean/nursery_test.csv", index=False)

    print(f"Data saved to {nursery_root}/clean/nursery_train.csv and {nursery_root}/clean/nursery_test.csv")

