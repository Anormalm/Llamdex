import os
import pandas as pd
import numpy as np
import json

"""
Label: >50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""


def clean_adult(raw_dir: str, dataset_json: str | dict) -> (pd.DataFrame, pd.DataFrame):
    if isinstance(dataset_json, str):
        dataset_json = json.load(open(dataset_json, 'r'))
    original_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'salary']

    categorical_columns = [col for col, info in dataset_json['X'].items() if info['type'] == 'category']

    train_data_df = pd.read_csv(f"{raw_dir}/adult.data", header=None, names=original_columns)
    test_data_df = pd.read_csv(f"{raw_dir}/adult.test", header=None, skiprows=1, names=original_columns)

    print(f"Train data shape: {train_data_df.shape}, Test data shape: {test_data_df.shape}")
    assert len(train_data_df.columns) == len(test_data_df.columns) == 15  # 14 features + 1 label

    # trim all the string columns
    train_data_df = train_data_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    test_data_df = test_data_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # if '?' is present in numerical columns, remove the rows
    # replace '?' with np.nan if categorical, and with np.nan if continuous
    for col in train_data_df.columns:
        if col in categorical_columns:
            train_data_df[col] = train_data_df[col].replace('?', np.nan)
            test_data_df[col] = test_data_df[col].replace('?', np.nan)
        else:
            train_data_df = train_data_df[train_data_df[col] != '?']
            test_data_df = test_data_df[test_data_df[col] != '?']

    # encode label
    train_data_df['salary>50K'] = train_data_df['salary'].apply(lambda x: 1 if x == '>50K' else 0)
    test_data_df['salary>50K'] = test_data_df['salary'].apply(lambda x: 1 if x == '>50K.' else 0)  # test data has '.' at the end

    train_data_df.drop(columns=['salary'], inplace=True)
    test_data_df.drop(columns=['salary'], inplace=True)

    # sort columns by name
    train_data_df = train_data_df.reindex(sorted(train_data_df.columns), axis=1)
    test_data_df = test_data_df.reindex(sorted(test_data_df.columns), axis=1)

    # # encode categorical columns; strictly follow the order in description
    # train_data_dummy_df = pd.get_dummies(train_data_df, columns=categorical_columns)
    # test_data_dummy_df = pd.get_dummies(test_data_df, columns=categorical_columns)
    # test_data_dummy_df['native-country_Holand-Netherlands'] = 0     # add missing column in test data
    #
    # # sort columns by name
    # train_data_dummy_df = train_data_dummy_df.reindex(sorted(train_data_dummy_df.columns), axis=1)
    # test_data_dummy_df = test_data_dummy_df.reindex(sorted(test_data_dummy_df.columns), axis=1)
    #
    # # convert all columns to float
    # train_data_dummy_df = train_data_dummy_df.astype(float)

    return train_data_df, test_data_df


if __name__ == '__main__':
    adult_root = os.path.join(os.path.dirname(__file__), "../../../data/adult/")
    train_df, test_df = clean_adult(f"{adult_root}/raw", f"src/dataset/adult/adult.json")
    assert np.all(train_df.columns == test_df.columns)
    print(train_df.columns)

    # Save the cleaned data
    os.makedirs(f"{adult_root}/clean", exist_ok=True)
    if os.path.exists(f"{adult_root}/clean/adult_train.csv") and os.path.exists(f"{adult_root}/clean/adult_test.csv"):
        input("Data already exists. Press Enter to overwrite...")

    train_df.to_csv(f"{adult_root}/clean/adult_train.csv", index=False)
    test_df.to_csv(f"{adult_root}/clean/adult_test.csv", index=False)

    print(f"Data saved to {adult_root}/clean/train.csv and {adult_root}/clean/test.csv")
