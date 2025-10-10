import os
import pandas as pd
import numpy as np
import json


def clean_abalone(raw_dir: str, dataset_json: str | dict) -> (pd.DataFrame, pd.DataFrame, dict):
    if isinstance(dataset_json, str):
        dataset_json = json.load(open(dataset_json, 'r'))

    columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
               'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

    # Read the abalone dataset
    abalone_df = pd.read_csv(f"{raw_dir}/abalone.data", names=columns)

    print(f"Data shape: {abalone_df.shape}")
    assert len(abalone_df.columns) == 9  # 8 features + 1 Rings

    print("Missing values:")
    print(abalone_df.isnull().sum())

    # Convert 'Rings' to Age and then to Age_Group
    abalone_df['Age'] = abalone_df['Rings'] + 1.5
    age_groups, bins = pd.qcut(abalone_df['Age'], q=3, labels=['A', 'B', 'C'], retbins=True)

    # Get the age ranges for each group
    age_group_ranges = {group: f"{bins[i]:.2f} - {bins[i + 1]:.2f}"
                        for i, group in enumerate(['A', 'B', 'C'])}

    abalone_df['Age_Group'] = age_groups

    # Drop 'Rings' and 'Age' columns
    abalone_df = abalone_df.drop(['Rings', 'Age'], axis=1)

    # Sort columns by name
    abalone_df = abalone_df.reindex(sorted(abalone_df.columns), axis=1)

    # Ensure 'Age_Group' is the last column
    cols = list(abalone_df.columns)
    cols.remove('Age_Group')
    cols.append('Age_Group')
    abalone_df = abalone_df[cols]

    abalone_df = abalone_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(0.8 * len(abalone_df))
    train_df = abalone_df[:train_size]
    test_df = abalone_df[train_size:]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df, age_group_ranges


if __name__ == '__main__':
    abalone_root = os.path.join(os.path.dirname(__file__), "../../../data/abalone/")
    train_df, test_df, age_group_ranges = clean_abalone(f"{abalone_root}/raw", f"src/dataset/abalone/abalone.json")
    assert np.all(train_df.columns == test_df.columns)
    print(train_df.columns)

    # Save the cleaned data
    os.makedirs(f"{abalone_root}/clean", exist_ok=True)
    if os.path.exists(f"{abalone_root}/clean/abalone_train.csv") and os.path.exists(
            f"{abalone_root}/clean/abalone_test.csv"):
        input("Data already exists. Press Enter to overwrite...")

    train_df.to_csv(f"{abalone_root}/clean/abalone_train.csv", index=False)
    test_df.to_csv(f"{abalone_root}/clean/abalone_test.csv", index=False)

    print(f"Data saved to {abalone_root}/clean/abalone_train.csv and {abalone_root}/clean/abalone_test.csv")

    # Print distribution of Age_Group
    print("\nAge_Group distribution:")
    print("Train set: ", train_df['Age_Group'].value_counts(normalize=True))
    print("Test set: ", test_df['Age_Group'].value_counts(normalize=True))

    # Print Age_Group ranges
    print("\nAge_Group ranges:")
    for group, age_range in age_group_ranges.items():
        print(f"{group}: {age_range}")