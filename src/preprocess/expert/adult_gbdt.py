import os
import pickle

import xgboost
import pandas as pd
import numpy as np

def train_adult_gbdt(adult_clean_root: str):
    train_path = os.path.join(adult_clean_root, "adult_train.csv")
    test_path = os.path.join(adult_clean_root, "adult_test.csv")
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    X_train = train_df.drop(columns=['salary>50K']).to_numpy()
    y_train = train_df['salary>50K'].to_numpy()
    X_test = test_df.drop(columns=['salary>50K']).to_numpy()
    y_test = test_df['salary>50K'].to_numpy()

    # Train the model using GPU
    model = xgboost.XGBClassifier(n_estimators=50, max_depth=4, device='cuda:1',
                                  objective='binary:logistic')
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    return model


if __name__ == '__main__':
    project_root = os.path.join(os.path.dirname(__file__), "../../../")
    model = train_adult_gbdt(f"{project_root}/data/adult/clean")
    os.makedirs(f"{project_root}/model/experts", exist_ok=True)
    with open(f"{project_root}/model/experts/adult_gbdt.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {project_root}/model/experts/adult_gbdt.pkl")

