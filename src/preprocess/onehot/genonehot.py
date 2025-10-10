import os
import pandas as pd
import numpy as np

from src.dataset.utils import TableScaler


process_datasets = {
    'titanic': {
        'dataset_json_path': 'src/dataset/titanic/titanic.json',
        'train_data_path': 'data/titanic/clean/titanic_train.csv',
        'test_data_path': 'data/titanic/clean/titanic_test.csv',
    },
    'adult': {
        'dataset_json_path': 'src/dataset/adult/adult.json',
        'train_data_path': 'data/adult/clean/adult_train.csv',
        'test_data_path': 'data/adult/clean/adult_test.csv',
    },
    'wine_quality': {
        'dataset_json_path': 'src/dataset/wine_quality/wine_quality.json',
        'train_data_path': 'data/wine_quality/clean/wine_quality_train.csv',
        'test_data_path': 'data/wine_quality/clean/wine_quality_test.csv',
        'label_map': lambda x: ord(x) - ord('A'),
    },
    'bank_marketing': {
        'dataset_json_path': 'src/dataset/bank_marketing/bank_marketing.json',
        'train_data_path': 'data/bank_marketing/clean/bank_marketing_train.csv',
        'test_data_path': 'data/bank_marketing/clean/bank_marketing_test.csv',
    },
    'abalone': {
        'dataset_json_path': 'src/dataset/abalone/abalone.json',
        'train_data_path': 'data/abalone/clean/abalone_train.csv',
        'test_data_path': 'data/abalone/clean/abalone_test.csv',
        'label_map': lambda x: ord(x) - ord('A'),
    },
    'nursery': {
        'dataset_json_path': 'src/dataset/nursery/nursery.json',
        'train_data_path': 'data/nursery/clean/nursery_train.csv',
        'test_data_path': 'data/nursery/clean/nursery_test.csv',
        'label_map': lambda x: ord(x) - ord('A'),
    },
}

if __name__ == '__main__':
    for dataset_name, dataset_info in process_datasets.items():
        scaler = TableScaler(dataset_json=dataset_info['dataset_json_path'])
        train_data_path = dataset_info['train_data_path']
        test_data_path = dataset_info['test_data_path']

        train_df = pd.read_csv(train_data_path, header=0)
        test_df = pd.read_csv(test_data_path, header=0)
        train_df_scaled = scaler.transform(train_df, label_map=dataset_info.get('label_map', None))
        test_df_scaled = scaler.transform(test_df, label_map=dataset_info.get('label_map', None))

        train_output_path = train_data_path.split('.')[0] + '_onehot.csv'
        test_output_path = test_data_path.split('.')[0] + '_onehot.csv'
        train_df_scaled.to_csv(train_output_path, index=False)
        test_df_scaled.to_csv(test_output_path, index=False)
        print(f"Saved one-hot encoded data to {train_output_path} and {test_output_path}")

