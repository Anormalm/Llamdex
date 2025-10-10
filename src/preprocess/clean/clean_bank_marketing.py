import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

"""
Label: y - has the client subscribed a term deposit? (binary: "yes","no")

1 - age (numeric)
2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
"blue-collar","self-employed","retired","technician","services")
3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
4 - education (categorical: "unknown","secondary","primary","tertiary")
5 - default: has credit in default? (binary: "yes","no")
6 - balance: average yearly balance, in euros (numeric)
7 - housing: has housing loan? (binary: "yes","no")
8 - loan: has personal loan? (binary: "yes","no")
9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
10 - day: last contact day of the month (numeric)
11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
12 - duration: last contact duration, in seconds (numeric)
13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
15 - previous: number of contacts performed before this campaign and for this client (numeric)
16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
"""

def clean_bank_marketing(raw_dir: str, dataset_json: str | dict) -> (pd.DataFrame, pd.DataFrame):
    if isinstance(dataset_json, str):
        dataset_json = json.load(open(dataset_json, 'r'))

    categorical_columns = [col for col, info in dataset_json['X'].items() if info['type'] == 'category']

    # We only use bank-full.csv but not bank.csv, and do train-test split
    full_data_df = pd.read_csv(f"{raw_dir}/bank-full.csv", sep=';')

    print(f"Full data shape: {full_data_df.shape}")
    assert len(full_data_df.columns) == 17  # 16 features + 1 label

    full_data_df = full_data_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    for col in categorical_columns:
        full_data_df[col] = full_data_df[col].replace('unknown', np.nan)

    full_data_df['subscribed'] = full_data_df['y'].apply(lambda x: 1 if x == 'yes' else 0)
    full_data_df.drop(columns=['y'], inplace=True)

    # sort columns by name
    full_data_df = full_data_df.reindex(sorted(full_data_df.columns), axis=1)

    train_df, test_df = train_test_split(full_data_df, test_size=0.2, random_state=42)

    return train_df, test_df

if __name__ == '__main__':
    bank_marketing_root = os.path.join(os.path.dirname(__file__), "../../../data/bank_marketing/")
    train_df, test_df = clean_bank_marketing(f"{bank_marketing_root}/raw",
                                             f"src/dataset/bank_marketing/bank_marketing.json")
    assert np.all(train_df.columns == test_df.columns)
    print(train_df.columns)

    print(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

    os.makedirs(f"{bank_marketing_root}/clean", exist_ok=True)
    if os.path.exists(f"{bank_marketing_root}/clean/bank_marketing_train.csv") and os.path.exists(
            f"{bank_marketing_root}/clean/bank_marketing_test.csv"):
        input("Data already exists. Press Enter to overwrite...")

    train_df.to_csv(f"{bank_marketing_root}/clean/bank_marketing_train.csv", index=False)
    test_df.to_csv(f"{bank_marketing_root}/clean/bank_marketing_test.csv", index=False)

    print(
        f"Data saved to {bank_marketing_root}/clean/bank_marketing_train.csv and {bank_marketing_root}/clean/bank_marketing_test.csv")