import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(data_dir):
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    is_mlflow_run = "MLFLOW_RUN_ID" in os.environ

    # Hanya inisialisasi DagsHub / set_experiment jika dijalankan secara manual (bukan lewat `mlflow run`)
    if not is_mlflow_run:
        if not is_ci:
            import dagshub
            token = os.getenv("DAGSHUB_USER_TOKEN")
            if token:
                os.environ["DAGSHUB_USER_TOKEN"] = token
                os.environ["MLFLOW_TRACKING_USERNAME"] = "rudywijayaa"
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token

            dagshub.init(
                repo_owner='rudywijayaa',
                repo_name='Eksperimen_SML_Preprocessing_Rudy-Wijaya',
                mlflow=True
            )
        mlflow.set_experiment("Baseline_Model_Churn")

    # Pastikan path dataset mengarah ke lokasi relatif file modelling.py
    base_path = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_path, data_dir)

    X_train = pd.read_csv(os.path.join(target_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(target_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(target_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(target_dir, 'y_test.csv')).values.ravel()

    # mlflow.start_run() akan otomatis menyambung ke parent run yang dibuat oleh `mlflow run`
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

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:5]
        )

        print(f"[SUCCESS] Training Selesai. Accuracy: {acc:.4f} | RUN_ID: {run.info.run_id}")

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'churn_preprocessing'
    train_model(data_dir)