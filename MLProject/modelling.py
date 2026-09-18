import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub

print("[INFO] Memulai script Baseline Modelling...")

# 1. Inisialisasi DagsHub MLflow Tracking
dagshub.init(repo_owner='rudywijayaa', repo_name='Eksperimen_SML_Preprocessing_Rudy-Wijaya', mlflow=True)

# 2. Load Data dari folder churn_preprocessing
X_train = pd.read_csv('churn_preprocessing/X_train.csv')
X_test = pd.read_csv('churn_preprocessing/X_test.csv')
y_train = pd.read_csv('churn_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('churn_preprocessing/y_test.csv').values.ravel()

# 3. Set Nama Eksperimen MLflow
mlflow.set_experiment("Baseline_Model_Churn")

# 4. Training & Logging Baseline Model
with mlflow.start_run(run_name="Baseline_RandomForest"):
    n_estimators = 100
    max_depth = 10
    random_state = 42

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log Model ke MLflow dengan skops_trusted_types
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("[SUCCESS] Baseline model berhasil dilatih dan di-log ke MLflow!")