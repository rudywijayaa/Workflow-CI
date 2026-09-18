import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.sklearn
import dagshub

print("[INFO] Memulai script Baseline Modelling...")

repo_owner = "rudywijayaa"
repo_name = "Eksperimen_SML_Preprocessing_Rudy-Wijaya"

# 1. Autentikasi DagsHub secara Ekstrem-Aman (CI/CD Friendly)
dagshub_token = (
    os.getenv("DAGSHUB_USER_TOKEN")
    or os.getenv("MLFLOW_TRACKING_PASSWORD")
    or os.getenv("DAGSHUB_TOKEN")
)

if dagshub_token:
    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
    try:
        import dagshub.auth
        dagshub.auth.add_app_token(dagshub_token)
    except Exception as e:
        print(f"[WARNING] Gagal menambahkan app token ke DagsHub auth: {e}")

# Always set Tracking URI langsung ke MLflow
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

try:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
except Exception as e:
    print(f"[WARNING] DagsHub init interaktif dilewati (Menggunakan Direct MLflow Auth): {e}")

mlflow.set_experiment("Baseline_Model_Churn")

# 2. Load Dataset
X_train = pd.read_csv("churn_preprocessing/X_train.csv")
X_test = pd.read_csv("churn_preprocessing/X_test.csv")
y_train = pd.read_csv("churn_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("churn_preprocessing/y_test.csv").values.ravel()

# Buat folder lokal sementara untuk menyimpan artefak sebelum diunggah
os.makedirs("artifacts", exist_ok=True)

# 3. Training & Logging
with mlflow.start_run(run_name="Baseline_RandomForest", nested=True) as run:
    print(f"[INFO] Running Run ID: {run.info.run_id}")

    n_estimators = 100
    max_depth = 10
    random_state = 42

    # Log Parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Train Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Predictions & Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # --- ARTEFAK 1: MLflow Model Standard (Menghasilkan file MLmodel untuk build-docker) ---
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )

    # --- ARTEFAK 2: Model Joblib Manual (Dioper ke extra_artifacts agar tidak menimpa folder model) ---
    model_path = "artifacts/model.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="extra_artifacts")

    # --- ARTEFAK 3: Plot Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Baseline Model")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="extra_artifacts")

    # --- ARTEFAK 4: Classification Report JSON ---
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_path = "artifacts/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    mlflow.log_artifact(report_path, artifact_path="extra_artifacts")

    print("[SUCCESS] Baseline model dan seluruh artefak berhasil diunggah ke DagsHub!")