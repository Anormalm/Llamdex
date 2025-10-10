import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

"""
Label: Survived (0 = No, 1 = Yes)

Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
Name: Passenger's name
Sex: Sex
Age: Age in years
Siblings/Spouses Aboard: Number of siblings / spouses aboard the Titanic
Parents/Children Aboard: Number of parents / children aboard the Titanic
Fare: Passenger fare
"""

def clean_titanic(raw_dir: str, dataset_json: str | dict):
    if isinstance(dataset_json, str):
        dataset_json = json.load(open(dataset_json, 'r'))
    
    original_columns = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings/Spouses Aboard', 
                        'Parents/Children Aboard', 'Fare']

    categorical_columns = [col for col, info in dataset_json['X'].items() if info['type'] == 'category']

    data_df = pd.read_csv(f"{raw_dir}/titanic.csv")

    data_df = data_df.drop('Name', axis=1)

    print(f"Data shape: {data_df.shape}")
    assert len(data_df.columns) == 7  # 6 features + 1 label

    # Trim all the string columns
    data_df = data_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Sort columns by name
    data_df = data_df.reindex(sorted(data_df.columns), axis=1)

    # Ensure 'Survived' is the last column
    cols = list(data_df.columns)
    cols.remove('Survived')
    cols.append('Survived')
    data_df = data_df[cols]

    # Split the data into train and test sets
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['Survived'])

    return train_df, test_df

if __name__ == '__main__':
    titanic_root = os.path.join(os.path.dirname(__file__), "../../../data/titanic/")
    train_df, test_df = clean_titanic(f"{titanic_root}/raw", f"src/dataset/titanic/titanic.json")
    
    print(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")
    print(f"Train columns: {train_df.columns}")
    print(f"Test columns: {test_df.columns}")

    # Save the cleaned data
    os.makedirs(f"{titanic_root}/clean", exist_ok=True)
    if os.path.exists(f"{titanic_root}/clean/titanic_train.csv") and os.path.exists(f"{titanic_root}/clean/titanic_test.csv"):
        input("Data already exists. Press Enter to overwrite...")

    train_df.to_csv(f"{titanic_root}/clean/titanic_train.csv", index=False)
    test_df.to_csv(f"{titanic_root}/clean/titanic_test.csv", index=False)

    print(f"Data saved to {titanic_root}/clean/titanic_train.csv and {titanic_root}/clean/titanic_test.csv")