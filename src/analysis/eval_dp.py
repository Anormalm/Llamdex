import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Callable
from sklearn.metrics import roc_auc_score

from src.preprocess.DataScaler import TableScaler


def load_dp_file(eps='1', method='pate-gan', dataset='adult', label='salary>50K', data_dir='data'):
    data_file = f"{data_dir}/{dataset}/dpsyn/eps{eps}/{method}/synthetic_data.csv"
    df = pd.read_csv(data_file, index_col=0)
    X = df.drop(columns=[label]).values
    y = df[label].values
    return X, y


def load_test_file(dataset='adult', data_dir='data', label='salary>50K',
                   label_map:Callable=None):
    data_file = f"{data_dir}/{dataset}/clean/{dataset}_test.csv"
    df = pd.read_csv(data_file)
    X = df.drop(columns=[label]).values
    y = df[label]
    if label_map is not None:
        y = y.map(label_map).values
    else:
        y = y.values
    return X, y


def eval_dp(dataset='adult', eps='1', method='pate-gan', label='salary>50K',
            n_estimators=100, n_classes=2):
    X_train, y_train = load_dp_file(eps=eps, method=method, dataset=dataset, label=label)
    if n_classes > 2:
        label_map = lambda x: ord(x) - ord('A')
    else:
        label_map = None
    X_test, y_test = load_test_file(dataset=dataset, label=label,
                                    label_map=label_map)

    if n_classes == 2:
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=8, objective='binary:logistic',
                                  device='cuda:6', max_bin=256)
        model.fit(X_train, y_train)

        # scale the test data
        scaler = TableScaler(dataset_json=f"src/dataset/{dataset}/{dataset}.json")
        X_test_scale = scaler.transform(X_test)
        y_pred = model.predict(X_test_scale)
        accuracy = np.mean(y_pred == y_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_scale)[:, 1])
        print(f"Dataset {dataset}, method {method}, eps {eps} - accuracy {accuracy}, auc {auc}")

    else:
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=8, objective='multi:softmax',
                                  num_class=n_classes, device='cuda:6', max_bin=256)
        model.fit(X_train, y_train)

        # scale the test data
        scaler = TableScaler(dataset_json=f"src/dataset/{dataset}/{dataset}.json")
        X_test_scale = scaler.transform(X_test)
        y_pred = model.predict(X_test_scale)
        accuracy = np.mean(y_pred == y_test)
        print(f"Dataset {dataset}, method {method}, eps {eps} - accuracy {accuracy}")


if __name__ == '__main__':
    for method in ['dp-wgan', 'pate-gan']:
        for dataset, label, n_classes in [
                               ('adult', 'salary>50K', 2),
                               ('titanic', 'Survived', 2),
                               ('bank_marketing', 'subscribed', 2),
                               ('wine_quality', 'quality', 11),
                               ('nursery', 'evaluation', 4)]:
            for eps in ['1', '2', '3', '4', '5']:
                eval_dp(dataset=dataset, eps=eps, method=method, label=label, n_classes=n_classes,
                        n_estimators=100)




