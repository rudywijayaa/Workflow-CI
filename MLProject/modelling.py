import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(data_dir):
    # 1. Inisialisasi DagsHub & MLflow Remote Tracking
    dagshub.init(repo_owner='rudywijayaa', repo_name='Eksperimen_SML_Preprocessing_Rudy-Wijaya', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rudywijayaa/Eksperimen_SML_Preprocessing_Rudy-Wijaya.mlflow")
    mlflow.set_experiment("Baseline_Model_Churn")

    # 2. Penanganan Path Data
    if not os.path.isabs(data_dir):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_dir = os.path.join(base_path, data_dir)
        if not os.path.exists(target_dir):
            target_dir = os.path.abspath(data_dir)
    else:
        target_dir = data_dir

    X_train = pd.read_csv(os.path.join(target_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(target_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(target_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(target_dir, 'y_test.csv')).values.ravel()

    # 3. Training & Logging
    with mlflow.start_run() as run:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log Params & Metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # Log Model Eksplisit ke Remote DagsHub Artifact Store
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        run_id = run.info.run_id
        print(f"[SUCCESS] Training Selesai. Accuracy: {acc:.4f} | RUN_ID: {run_id}")

        # Otomatis Pass RUN_ID ke GitHub Actions Environment
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a") as f:
                f.write(f"RUN_ID={run_id}\n")

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'churn_preprocessing'
    train_model(data_dir)