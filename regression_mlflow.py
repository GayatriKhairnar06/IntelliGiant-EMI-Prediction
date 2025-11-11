# regression_mlflow.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

print("=== Step 1: Prerequisites ===")

# -------------------------------
# Step 2: Load Data
# -------------------------------
x_train = pd.read_csv("data/x_train_reg.csv")
x_val = pd.read_csv("data/x_val_reg.csv")
x_test = pd.read_csv("data/x_test_reg.csv")
y_train = pd.read_csv("data/y_train_reg.csv")
y_val = pd.read_csv("data/y_val_reg.csv")
y_test = pd.read_csv("data/y_test_reg.csv")

y_train = y_train.iloc[:, 0]
y_val = y_val.iloc[:, 0]
y_test = y_test.iloc[:, 0]

x_full = pd.concat([x_train, x_val])
y_full = pd.concat([y_train, y_val])

print(f"✅ Data Loaded | Train: {x_full.shape}, Test: {x_test.shape}")

# -------------------------------
# Step 3: Define Models
# -------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=150, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
    "XGBoostRegressor": XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        eval_metric="rmse", random_state=42
    ),
}

# -------------------------------
# Step 4: Start MLflow Experiment
# -------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # connect to MLflow server
mlflow.set_experiment("EMI_Amount_Regression")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"\n🚀 Training {model_name}...")

        # Train model
        model.fit(x_full, y_full)
        y_pred = model.predict(x_test)

        # Evaluate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{model_name} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

        # Log metrics and parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # ✅ Log model artifact (required for registry)
        mlflow.sklearn.log_model(model, model_name)

print("\n🎯 All regression models logged successfully! View results in MLflow UI.")
