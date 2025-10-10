import os
import json

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold  # Import RandomizedSearchCV and RepeatedStratifiedKFold

from src.preprocess.DataScaler import TableScaler
def train_titanic_gbdt(dataset_name, scaler=None, epsilon=None, delta=None):
    # delta is NOT used in this function
    use_dp = epsilon is not None and delta is not None

    print(f"Training GBDT on {dataset_name} dataset")
    train_path = f"data/titanic/clean/{dataset_name}_train.csv"
    test_path = f"data/titanic/clean/{dataset_name}_test.csv"
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    # Encode the Survived labels
    le = LabelEncoder()
    train_df['Survived'] = le.fit_transform(train_df['Survived'])
    test_df['Survived'] = le.transform(test_df['Survived'])

    # values sorted by column name
    X_train = train_df.drop(columns=['Survived']).sort_index(axis=1).to_numpy()
    y_train = train_df['Survived'].to_numpy()
    X_test = test_df.drop(columns=['Survived']).sort_index(axis=1).to_numpy()
    y_test = test_df['Survived'].to_numpy()

    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    num_classes = len(le.classes_)  # Get number of classes from the fitted encoder
    print(f"Number of classes: {num_classes}")

    if use_dp:
        raise NotImplementedError("Differential Privacy for XGBoost is not yet supported")

    else:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            eta=0.1,
            max_depth=50,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='gpu_hist',
            n_estimators=50,
            use_label_encoder=False,
            enable_categorical=False,
            device='cuda:0'
        )
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if use_dp:
        save_model_path = f"model/experts/{dataset_name}_gbdt_eps-{epsilon}_delta-{delta}.json"
    else:
        save_model_path = f"model/experts/{dataset_name}_gbdt.json"
    os.makedirs(f"model/experts", exist_ok=True)
    model.save_model(save_model_path)
    print(f"Model saved to {save_model_path}")


if __name__ == '__main__':
    titanic_root_dir = f"data/titanic/"
    dataset_json_path = f"src/dataset/titanic/titanic.json"
    scaler = TableScaler(dataset_json=dataset_json_path)

    # train on synthetic data
    train_titanic_gbdt("syn_titanic", scaler)
    # train on clean data
    train_titanic_gbdt("titanic", scaler)
