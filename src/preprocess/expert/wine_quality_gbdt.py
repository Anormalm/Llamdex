import os
import json

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold  # Import RandomizedSearchCV and RepeatedStratifiedKFold

from src.preprocess.DataScaler import TableScaler
def train_wine_quality_gbdt(dataset_name, scaler=None, epsilon=None, delta=None):
    # delta is NOT used in this function
    use_dp = epsilon is not None and delta is not None

    print(f"Training GBDT on {dataset_name} dataset")
    train_path = f"data/wine_quality/clean/{dataset_name}_train.csv"
    test_path = f"data/wine_quality/clean/{dataset_name}_test.csv"
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    # Encode the quality labels
    le = LabelEncoder()
    train_df['quality'] = le.fit_transform(train_df['quality'])
    test_df['quality'] = le.transform(test_df['quality'])

    # values sorted by column name
    X_train = train_df.drop(columns=['quality']).sort_index(axis=1).to_numpy()
    y_train = train_df['quality'].to_numpy()
    X_test = test_df.drop(columns=['quality']).sort_index(axis=1).to_numpy()
    y_test = test_df['quality'].to_numpy()

    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    num_classes = len(le.classes_)  # Get number of classes from the fitted encoder
    print(f"Number of classes: {num_classes}")

    if use_dp:
        # model = xgb.XGBClassifier(
        #     objective='multi:softmax',
        #     num_class=num_classes,
        #     eval_metric='merror',
        #     eta=0.1,
        #     max_depth=6,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     privacy_budget=epsilon,  # Controls epsilon
        #     tree_method='gpu_hist',
        #     process_type='default',
        #     grow_policy='depthwise',
        #     max_bin=16,
        #     n_estimators=100,
        #     enable_categorical=False,
        #     device='cuda:0'
        # )
        # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
        raise NotImplementedError("Differential Privacy for XGBoost is not yet supported")

    else:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            eval_metric='merror',
            eta=0.1,
            max_depth=20,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='gpu_hist',
            n_estimators=30,
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
    wine_quality_root_dir = f"data/wine_quality/"
    dataset_json_path = f"src/dataset/wine_quality/wine_quality.json"
    scaler = TableScaler(dataset_json=dataset_json_path)

    # train on synthetic data
    train_wine_quality_gbdt("syn_wine_quality", scaler)
    # train on clean data
    train_wine_quality_gbdt("wine_quality", scaler)
