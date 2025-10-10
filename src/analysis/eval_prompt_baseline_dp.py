import numpy as np
import pandas as pd

from src.preprocess.DataScaler import TableScaler
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import csv


def load_prompt_result_file(dataset, seed, dataset_json=None, data_dir='data'):
    data_file = f"{data_dir}/{dataset}/res/baseline_seed{seed}.csv"
    df = pd.read_csv(data_file)
    df = df.drop(columns=[col for col in df.columns if col.startswith('expert_response_')])

    def analyze_feature(feature_name, value):
        feature_info = dataset_json['X'].get(feature_name)
        if feature_info is None:
            return np.nan

        if feature_info['type'] == 'category':
            categories = feature_info['categories']
            if categories and isinstance(categories[0], (int, float)):
                try:
                    converted_value = int(value) if isinstance(categories[0], int) else float(value)
                    return converted_value if converted_value in categories else np.nan
                except ValueError:
                    return np.nan
            else:
                return value if value in categories else np.nan

        elif feature_info['type'] in ['int', 'float']:
            try:
                value = float(value)
                min_val, max_val = feature_info['range']
                if min_val <= value <= max_val:
                    return int(value) if feature_info['type'] == 'int' else value
                else:
                    return 0
            except ValueError:
                return 0
        else:
            return np.nan

    for col in df.columns:
        if col not in ['prediction', 'label']:
            df[col] = df[col].apply(lambda x: analyze_feature(col, x))

    X = df.drop(columns=['prediction', 'label'])
    y = df['label'].values
    return X, y


def eval_prompt_baseline_dp(dataset, eps, delta, seed, n_classes, batch_size=4):
    dataset_json = json.load(open(f"src/dataset/{dataset}/{dataset}.json"))
    X_test, y_test = load_prompt_result_file(dataset=dataset, seed=seed, dataset_json=dataset_json)

    scaler = TableScaler(dataset_json=dataset_json)
    X_test = torch.tensor(scaler.transform(np.array(X_test, dtype=object)), dtype=torch.bfloat16)

    expert = torch.load(f"model/experts/{dataset}_mlp_eps-{eps}_delta-{delta}.pth", map_location='cuda')
    expert = expert.to(torch.bfloat16).eval()

    test_dataset = TensorDataset(X_test, torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.cuda()
            y_pred_batch = expert(X_batch).detach().cpu().float().numpy()

            if n_classes == 2:
                y_pred_batch = (y_pred_batch > 0.5).astype(int).flatten()
            else:
                y_pred_batch = np.argmax(y_pred_batch, axis=1)

            all_preds.extend(y_pred_batch)
            all_true.extend(y_batch.numpy())

    accuracy = accuracy_score(all_true, all_preds)
    print(f"Dataset {dataset}, eps {eps}, seed {seed} - accuracy {accuracy:.4f}")


if __name__ == '__main__':
    results = []
    datasets = [
        ('adult', '1e-05', 2),
        ('titanic', '0.001', 2),
        ('bank_marketing', '1e-05', 2),
        ('wine_quality', '0.001', 11),
        ('nursery', '0.0001', 4)
    ]
    epsilons = ['0.5', '1.0', '2.0', '3.0', '4.0', '5.0']
    seeds = [0, 1, 2, 3, 4]

    for dataset, delta, n_classes in datasets:
        for eps in epsilons:
            for seed in seeds:
                eval_prompt_baseline_dp(dataset=dataset, eps=eps, delta=delta, seed=seed, n_classes=n_classes)
