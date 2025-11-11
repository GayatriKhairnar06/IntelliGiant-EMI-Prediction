# regression_models.py

print("=== Step 1: Prerequisites ===")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np

print("=== Step 2: Load Data ===")
# Load regression data
x_train = pd.read_csv("data/x_train_reg.csv")
x_val = pd.read_csv("data/x_val_reg.csv")
x_test = pd.read_csv("data/x_test_reg.csv")

y_train = pd.read_csv("data/y_train_reg.csv")
y_val = pd.read_csv("data/y_val_reg.csv")
y_test = pd.read_csv("data/y_test_reg.csv")

print(f"✅ Data Loaded | Train: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}")

# Combine train + validation
x_full = pd.concat([x_train, x_val])
y_full = pd.concat([y_train, y_val])

print("\n=== Step 3: Model Training ===")

# 1️⃣ Linear Regression
lr = LinearRegression()
lr.fit(x_full, y_full)
y_pred_lr = lr.predict(x_test)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression R²: {r2_lr:.4f}")

# 2️⃣ Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_full, np.ravel(y_full))
y_pred_rf = rf.predict(x_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest R²: {r2_rf:.4f}")

best_model = rf if r2_rf > r2_lr else lr
best_name = "RandomForestRegressor" if r2_rf > r2_lr else "LinearRegression"

print(f"\n✅ Best Model: {best_name} (R²: {max(r2_lr, r2_rf):.4f})")

print("\n=== Step 4: Error Metrics ===")
y_pred_best = best_model.predict(x_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred_best):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_best):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, f"models/{best_name}_max_emi.pkl")
print(f"💾 Model saved as models/{best_name}_max_emi.pkl")
