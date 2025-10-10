import os
import pandas as pd
import numpy as np
import json

"""
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data): 
12 - quality (score of A to K)
"""

def score_to_letter(score):
    return chr(ord('A') + int(score))

def clean_wine_quality(raw_dir: str, dataset_json: str | dict) -> (pd.DataFrame, pd.DataFrame):
    if isinstance(dataset_json, str):
        dataset_json = json.load(open(dataset_json, 'r'))

    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
               'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
               'alcohol', 'quality']

    # Read both red and white wine datasets
    red_wine_df = pd.read_csv(f"{raw_dir}/winequality-red.csv", sep=';')
    white_wine_df = pd.read_csv(f"{raw_dir}/winequality-white.csv", sep=';')

    # Add a 'type' column to distinguish between red and white wines
    red_wine_df['type'] = 'red'
    white_wine_df['type'] = 'white'

    # Combine the datasets
    wine_df = pd.concat([red_wine_df, white_wine_df], ignore_index=True)

    print(f"Combined data shape: {wine_df.shape}")
    assert len(wine_df.columns) == 13  # 11 features + 1 quality + 1 type

    print("Missing values:")
    print(wine_df.isnull().sum())

    wine_df['quality'] = wine_df['quality'].apply(score_to_letter)

    # Sort columns by name
    wine_df = wine_df.reindex(sorted(wine_df.columns), axis=1)

    # Ensure 'quality' is the last column
    cols = list(wine_df.columns)
    cols.remove('quality')
    cols.append('quality')
    wine_df = wine_df[cols]

    wine_df = wine_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(0.8 * len(wine_df))
    train_df = wine_df[:train_size]
    test_df = wine_df[train_size:]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


if __name__ == '__main__':
    wine_root = os.path.join(os.path.dirname(__file__), "../../../data/wine_quality/")
    train_df, test_df = clean_wine_quality(f"{wine_root}/raw", f"src/dataset/wine_quality/wine_quality.json")
    assert np.all(train_df.columns == test_df.columns)
    print(train_df.columns)

    # Save the cleaned data
    os.makedirs(f"{wine_root}/clean", exist_ok=True)
    if os.path.exists(f"{wine_root}/clean/wine_quality_train.csv") and os.path.exists(
            f"{wine_root}/clean/wine_quality_test.csv"):
        input("Data already exists. Press Enter to overwrite...")

    train_df.to_csv(f"{wine_root}/clean/wine_quality_train.csv", index=False)
    test_df.to_csv(f"{wine_root}/clean/wine_quality_test.csv", index=False)

    print(f"Data saved to {wine_root}/clean/wine_quality_train.csv and {wine_root}/clean/wine_quality_test.csv")