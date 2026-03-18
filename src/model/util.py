import torch
import torch.nn as nn
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency for legacy tabular experts
    xgb = None


class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes, dropout_rate=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_channels[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_channels[0]))
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_channels) - 1):
            layers.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_channels[i+1]))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_channels[-1], num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class XGBoostModule(nn.Module):
    def __init__(self, model_path):
        super(XGBoostModule, self).__init__()
        if xgb is None:
            raise ImportError("xgboost is required to load XGBoostModule but is not installed in the current environment.")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

    def forward(self, x):
        pre_device = x.device
        pre_type = x.dtype
        x = x.cpu().detach().float().numpy()
        x = self.model.predict_proba(x)
        return torch.tensor(x, dtype=pre_type, device=pre_device)


class SwiGLU(nn.Module):
    def __init__(self, input_size, ffn_hidden_size, output_size, dropout=0.0):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.linear2 = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.linear3 = nn.Linear(ffn_hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear3(torch.nn.functional.silu(self.dropout(self.linear1(x))) * self.linear2(x))
