import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dynamic BASE_DIR (folder tempat modelling.py berada)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ambil env CSV_URL dari CI/CD, jika tidak ada pakai fallback ke folder lokal
csv_env = os.getenv("CSV_URL", "churn_preprocessing/X_train.csv")

# Jika env masih membawa prefix "MLProject/", kita bersihkan
if csv_env.startswith("MLProject/"):
    csv_env = csv_env.replace("MLProject/", "", 1)

csv_url = os.path.join(BASE_DIR, csv_env)
y_url = csv_url.replace("X_train.csv", "y_train.csv")

# Baca Data
X_train = pd.read_csv(csv_url)
y_train = pd.read_csv(y_url)

target_var = os.getenv("TARGET_VAR", "Exited")

with mlflow.start_run():
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)
    acc = accuracy_score(y_train, predictions)

    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print("Training selesai dan model berhasil disimpan!")