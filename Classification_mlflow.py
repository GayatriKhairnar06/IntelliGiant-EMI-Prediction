# classification_mlflow.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("=== Step 1: Prerequisites ===")

# === Step 2: Load Data ===
x_train = pd.read_csv("data/x_train_class.csv")
x_val = pd.read_csv("data/x_val_class.csv")
x_test = pd.read_csv("data/x_test_class.csv")
y_train = pd.read_csv("data/y_train_class.csv")
y_val = pd.read_csv("data/y_val_class.csv")
y_test = pd.read_csv("data/y_test_class.csv")

# Ensure 1D
y_train = y_train.iloc[:, 0]
y_val = y_val.iloc[:, 0]
y_test = y_test.iloc[:, 0]

x_full = pd.concat([x_train, x_val])
y_full = pd.concat([y_train, y_val])

print(f"✅ Data Loaded | Train: {x_full.shape}, Test: {x_test.shape}")

# === Step 3: Define Models ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=150, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    #"SVC": SVC(probability=True, kernel='rbf', random_state=42),
    "XGBoostClassifier": XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
}

# === Step 4: Start MLflow Experiment ===
mlflow.set_experiment("EMI_Eligibility_Classification")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\n🚀 Training {name}...")
        model.fit(x_full, y_full)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        # ✅ Log parameters & metrics to MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # ✅ Log model artifact
        mlflow.sklearn.log_model(model, name)

print("🎯 All models logged successfully! View results in MLflow UI.")
