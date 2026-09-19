import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(data_dir):
    base_path = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_path, data_dir)

    print(f"[INFO] Membaca data dari: {target_dir}")
    X_train = pd.read_csv(os.path.join(target_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(target_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(target_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(target_dir, 'y_test.csv')).values.ravel()

    # Training Model
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Logging Param & Metric ke MLflow
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    # Logging Artefak Model dengan memperbolehkan skops trusted types
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print(f"[SUCCESS] Training Selesai. Accuracy: {acc:.4f}")

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'churn_preprocessing'
    train_model(data_dir)