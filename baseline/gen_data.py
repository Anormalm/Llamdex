from src.dataset.synthetic import make_classification_from_json
import torch
import json
from src.preprocess.DataScaler import TableScaler
import pandas as pd

if __name__ == "__main__":

    dataset_json_dir = "src/dataset/adult/adult.json"
    model_dir = "model/experts/adult_mlp.pth"
    output_dir = "data/adult/baseline/adult_pred_mlp.csv"

    with open(dataset_json_dir, "r") as f:
        adult_columns = json.load(f)

    df = make_classification_from_json(n_samples=20000, dataset_json=adult_columns, class_sep=0.5, random_state=42)
    X = df.drop(columns=["salary>50K"]).sort_index(axis=1).to_numpy()

    scaler = TableScaler(adult_columns)
    X = scaler.transform(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_dir, map_location="cpu")
    model.to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_pred = model(X_tensor)
    y_pred = (y_pred > 0.5).cpu().flatten().numpy().astype(int)

    df["salary>50K"] = y_pred
    df.to_csv(output_dir, index=False)

