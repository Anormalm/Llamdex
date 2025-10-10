import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import MinMaxScaler
from torch import nn


class OneHotScaler:
    def __init__(self, categories: list[str], basename: str = None):
        """
        One-hot encode the categorical values
        Args:
            categories: The list of categories. Example: ['A', 'B', 'C']
            basename: The basename of the column. Example 'education' for 'education_Bachelors', 'education_HS-grad'
                      This is used
        """
        self.categories = categories
        self.cat_index = {cat: i for i, cat in enumerate(categories)}
        self.basename = basename

    def transform(self, value: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self._transform_value, 1, value)

    def _transform_value(self, value):
        onehot = np.zeros(len(self.categories))
        if isinstance(value, np.ndarray):
            value = value.item()

        if isinstance(value, float) and np.isnan(value):
            return onehot

        onehot[self.cat_index[value]] = 1
        return onehot

    def inverse_transform(self, onehot: list[int]) -> str:
        return self.categories[onehot.index(1)]

    @property
    def category_full_names(self):
        if self.basename is None:
            return self.categories
        return [f"{self.basename}_{cat}" for cat in self.categories]

    @property
    def tensor_map(self):
        return [nn.Identity() for _ in self.categories]


class TensorMinMaxScaler:
    def __init__(self, feature_range=(0, 1), clip_outliers=True):
        self.min_val, self.max_val = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.min_ = None
        self.clip_outliers = clip_outliers

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            self.data_min_ = torch.min(X, dim=0)[0]
            self.data_max_ = torch.max(X, dim=0)[0]
        else:
            self.data_min_ = np.min(X, axis=0)
            self.data_max_ = np.max(X, axis=0)

        # Handle the case where min == max
        diff = self.data_max_ - self.data_min_
        diff[diff == 0] = 1

        self.scale_ = (self.max_val - self.min_val) / diff
        self.min_ = self.min_val - self.data_min_ * self.scale_
        return self

    def fit_minmax(self, min_val, max_val):
        self.data_min_ = min_val
        self.data_max_ = max_val
        # Handle the case where min == max
        diff = self.data_max_ - self.data_min_
        if diff == 0:
            diff = 1

        self.scale_ = (self.max_val - self.min_val) / diff
        self.min_ = self.min_val - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        if self.clip_outliers:
            # Clip outliers to training range before scaling
            if isinstance(X, torch.Tensor):
                X_clipped = torch.clamp(X, min=self.data_min_, max=self.data_max_)
            else:
                X_clipped = np.clip(X, self.data_min_, self.data_max_)
            return X_clipped * self.scale_ + self.min_
        else:
            return X * self.scale_ + self.min_

    def inverse_transform(self, X_scaled):
        X_original = (X_scaled - self.min_) / self.scale_
        if self.clip_outliers:
            # Clip to original range after inverse transform
            if isinstance(X_original, torch.Tensor):
                X_original = torch.clamp(X_original, min=self.data_min_, max=self.data_max_)
            else:
                X_original = np.clip(X_original, self.data_min_, self.data_max_)
        return X_original

    @property
    def tensor_map(self):
        def _transform(X: torch.Tensor):
            if isinstance(self.scale_, np.ndarray):
                scale_ = torch.tensor(self.scale_).to(X.device)
                min_ = torch.tensor(self.min_).to(X.device)
                data_min_ = torch.tensor(self.data_min_).to(X.device)
                data_max_ = torch.tensor(self.data_max_).to(X.device)
            else:
                scale_ = self.scale_.to(X.device)
                min_ = self.min_.to(X.device)
                data_min_ = self.data_min_.to(X.device)
                data_max_ = self.data_max_.to(X.device)
            
            if self.clip_outliers:
                X_clipped = torch.clamp(X, min=data_min_, max=data_max_)
                return X_clipped * scale_ + min_
            else:
                return X * scale_ + min_
        return _transform



class TableScaler:
    def __init__(self, dataset_json: dict | str):
        """
        Scaler for the input of domain experts
        Args:
            dataset_json: A dictionary containing the dataset or the path of the json object. Examples:
            {
                "X": {
                    "age": {"type": "int", "range": [17, 100]},
                    "capital-gain": {"type": "int", "range": [0, 100000]},
                    "capital-loss": {"type": "int", "range": [0, 100000]},
                    "education-num": {"type": "int", "range": [1, 16]},
                    "education": {
                        "type": "category",
                        "categories": [
                            "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm",
                            "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool",
                            "Prof-school", "Some-college"
                        ]
                    },
                    "fnlwgt": {"type": "int", "range": [0, 2000000]},
                    "hours-per-week": {"type": "int", "range": [0, 100]},
                    "marital-status": {
                        "type": "category",
                        "categories": [
                            "Divorced", "Married-AF-spouse", "Married-civ-spouse",
                            "Married-spouse-absent", "Never-married", "Separated",
                            "Widowed"
                        ]
                    },
                    "native-country": {
                        "type": "category",
                        "categories": [
                            "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
                            "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
                            "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary",
                            "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico",
                            "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland",
                            "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand",
                            "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"
                        ]
                    },
                    "occupation": {
                        "type": "category",
                        "categories": [
                            "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
                            "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service",
                            "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support",
                            "Transport-moving"
                        ]
                    },
                    "race": {
                        "type": "category",
                        "categories": [
                            "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"
                        ]
                    },
                    "relationship": {
                        "type": "category",
                        "categories": [
                            "Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried",
                            "Wife"
                        ]
                    },
                    "sex": {"type": "category", "categories": ["Female", "Male"]},
                    "workclass": {
                        "type": "category",
                        "categories": [
                            "Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc",
                            "Self-emp-not-inc", "State-gov", "Without-pay"
                        ]
                    }
                },
                "y": {"name": "salary>50K", "type": "bool"}
            }

            columns: The ordered columns of the dataset to be scaled.
        """
        if isinstance(dataset_json, str):
            with open(dataset_json, 'r') as f:
                dataset_json = json.load(f)
        self.dataset_json = dataset_json
        self.columns = list(sorted(dataset_json['X'].keys()))
        self.scalers = []
        for column in self.columns:
            assert column in dataset_json['X']

            column_info = dataset_json['X'][column]
            if column_info['type'] in ['int', 'float']:
                scaler = TensorMinMaxScaler(feature_range=(0, 1))
                scaler.fit([[column_info['range'][0]], [column_info['range'][1]]])  # set the range
                self.scalers.append(scaler)
            elif column_info['type'] == 'category':
                # convert to onehot
                categories = column_info['categories']
                scaler = OneHotScaler(categories, basename=column)
                self.scalers.append(scaler)
            else:
                raise ValueError(f"Unsupported type: {column_info['type']}")
        self.feature_size = sum([len(column_info['categories']) if column_info['type'] == 'category' else 1
                                 for column_info in dataset_json['X'].values()])

    def transform(self, vec: np.ndarray | pd.DataFrame, label_map=None) -> np.ndarray | pd.DataFrame:
        """
        Transform a feature vector to [0, 1] range and one-hot encoding.

        If the input is a DataFrame, the output will also be a DataFrame. The column is based on the vec.columns,
        labels are ignored. Missing columns is permitted, but new columns raise an error.

        if the input is a numpy array, the output will also be a numpy array. The order of the columns must match
        the order of the columns in self.columns.
        Args:
            vec: (batch_size, len(columns))
            label_map: A function mapping the label to the index of the one-hot encoding
        Returns: (batch_size, feature_size)
        """
        if isinstance(vec, np.ndarray):
            onehot_scaled_vec = []
            for i, scaler in enumerate(self.scalers):
                onehot_scaled_vec.append(scaler.transform(vec[:, i].reshape(-1, 1)))
            return np.concatenate(onehot_scaled_vec, axis=1).astype(np.float32)
        elif isinstance(vec, pd.DataFrame):
            onehot_scaled_vec = []
            onehot_columns = []
            for column in vec.columns:
                if column == self.dataset_json['y']['name']:
                    onehot_columns.append(column)
                    if label_map is not None:
                        onehot_scaled_vec.append(np.vectorize(label_map)(vec[column].values).reshape(-1, 1).astype(np.int32))
                    else:
                        onehot_scaled_vec.append(vec[column].values.reshape(-1, 1))
                    continue
                if column not in self.columns:
                    raise ValueError(f"Column {column} not found in the dataset")

                i = self.columns.index(column)
                onehot_scaled_vec.append(self.scalers[i].transform(vec[column].values.reshape(-1, 1)))
                if isinstance(self.scalers[i], OneHotScaler):
                    onehot_columns += self.scalers[i].category_full_names
                else:
                    onehot_columns.append(column)
            return pd.DataFrame(np.concatenate(onehot_scaled_vec, axis=1), columns=onehot_columns)

    def inverse_transform(self, scaled_vec: np.ndarray) -> np.ndarray:
        """
        Inverse transform a feature vector to the original range and one-hot encoding
        Args:
            scaled_vec: (batch_size, feature_size) tensor
        Returns: (batch_size, len(columns)) tensor
        """
        if scaled_vec.shape[1] != self.feature_size:
            raise ValueError(f"Feature size mismatch: expected {self.feature_size}, got {scaled_vec.size}")

        vec = []
        start_idx = 0
        for i, scaler in enumerate(self.scalers):
            if self.dataset_json['X'][self.columns[i]]['type'] == 'category':
                vec.append(scaler.inverse_transform(scaled_vec[:, start_idx: start_idx + len(scaler.categories)]))
                start_idx += len(scaler.categories)
            else:
                vec.append(scaler.inverse_transform(scaled_vec[:, start_idx: start_idx + 1]))
                start_idx += 1
        return np.concatenate(vec, axis=1)

    @property
    def tensor_map(self):
        def _tensor_map(vec: torch.Tensor):
            onehot_scaled_vec = []
            start_idx = 0
            for i, scaler in enumerate(self.scalers):
                tensor_map_i = scaler.tensor_map
                if self.dataset_json['X'][self.columns[i]]['type'] == 'category':
                    onehot_scaled_vec.append(vec[:, start_idx: start_idx + len(scaler.categories)])
                    start_idx += len(scaler.categories)
                else:
                    onehot_scaled_vec.append(tensor_map_i(vec[:, start_idx: start_idx + 1]))
                    start_idx += 1
            res = torch.cat(onehot_scaled_vec, dim=1)
            assert res.shape[1] == vec.shape[1]
            return res
        return _tensor_map
