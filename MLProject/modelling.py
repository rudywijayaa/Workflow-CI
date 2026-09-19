import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dynamic BASE_DIR (folder tempat modelling.py berada)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ambil env CSV_URL dari CI/CD, fallback ke folder lokal
csv_env = os.getenv("CSV_URL", "churn_preprocessing/X_train.csv")
if csv_env.startswith("MLProject/"):
    csv_env = csv_env.replace("MLProject/", "", 1)

csv_url = os.path.join(BASE_DIR, csv_env)
y_url = csv_url.replace("X_train.csv", "y_train.csv")

# Baca Data
X_train = pd.read_csv(csv_url)
y_train = pd.read_csv(y_url)

target_var = os.getenv("TARGET_VAR", "Exited")

# Set nama run sesuai DagsHub (Baseline_RandomForest)
with mlflow.start_run(run_name="Baseline_RandomForest"):
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

    # Log Parameter & Metric
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)

    # Log Model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print("[SUCCESS] Training Baseline RandomForest selesai!")