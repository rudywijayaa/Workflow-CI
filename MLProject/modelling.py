import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi DagsHub MLflow Tracking
dagshub.init(repo_owner='rudywijayaa', repo_name='Eksperimen_SML_Preprocessing_Rudy-Wijaya', mlflow=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_env = os.getenv("CSV_URL", "churn_preprocessing/X_train.csv")
if csv_env.startswith("MLProject/"):
    csv_env = csv_env.replace("MLProject/", "", 1)

csv_url = os.path.join(BASE_DIR, csv_env)
y_url = csv_url.replace("X_train.csv", "y_train.csv")

X_train = pd.read_csv(csv_url)
y_train = pd.read_csv(y_url)

# Eksekusi MLflow Run
with mlflow.start_run(run_name="Baseline_RandomForest") as run:
    current_run_id = run.info.run_id
    print(f"[INFO] Running MLflow Run ID: {current_run_id}")

    # Otomatis simpan RUN_ID ke GitHub Actions Environment jika sedang running di CI
    github_env = os.getenv("GITHUB_ENV")
    if github_env:
        with open(github_env, "a") as f:
            f.write(f"RUN_ID={current_run_id}\n")

    n_estimators = 100
    max_depth = 10
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )
    model.fit(X_train, y_train.values.ravel())

    predictions = model.predict(X_train)
    acc = accuracy_score(y_train, predictions)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)

    # Log Model ke DagsHub Remote Artifact Store dengan izin skops_trusted_types
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print(f"[SUCCESS] Training selesai dan artefak terunggah untuk Run ID: {current_run_id}")