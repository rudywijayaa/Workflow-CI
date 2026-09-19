import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Baca environment variable dari YAML (atau fallback ke path lokal)
csv_url = os.getenv("CSV_URL", "MLProject/churn_preprocessing/X_train.csv")
target_var = os.getenv("TARGET_VAR", "Exited")

# Ubah path target untuk y_train (sesuaikan dengan struktur foldermu)
y_url = csv_url.replace("X_train.csv", "y_train.csv")

X_train = pd.read_csv(csv_url)
y_train = pd.read_csv(y_url)

# 2. Mulai MLflow run lokal
with mlflow.start_run():
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)
    acc = accuracy_score(y_train, predictions)
    
    # Logging metrics & parameters
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)

    # 3. Log model wajib dengan artifact_path="model"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )

print("Pelatihan model berhasil dan tersimpan di mlruns lokal!")