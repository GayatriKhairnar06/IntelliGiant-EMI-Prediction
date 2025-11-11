# regression_model_registry.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import sys

# =====================================================
# Step 1: Connect to MLflow Tracking Server
# =====================================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# =====================================================
# Step 2: Load Regression Experiment
# =====================================================
experiment_name = "EMI_Amount_Regression"
experiment = client.get_experiment_by_name(experiment_name)

if not experiment:
    print(f"❌ Experiment '{experiment_name}' not found in MLflow!")
    sys.exit(1)

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if runs.empty:
    print("❌ No regression runs found in MLflow.")
    sys.exit(1)

# =====================================================
# Step 3: Detect Metric Automatically
# =====================================================
possible_metrics = [
    "metrics.r2", "metrics.R2", "metrics.r2_score", "metrics.R2_SCORE",
    "metrics.rmse", "metrics.RMSE",
    "metrics.mse", "metrics.MSE",
    "metrics.mae", "metrics.MAE"
]

metric_to_use = None
for m in possible_metrics:
    if m in runs.columns:
        metric_to_use = m
        break

if not metric_to_use:
    print("❌ No recognized metric (R2, RMSE, MSE, MAE) found in MLflow runs.")
    print("Available columns:", list(runs.columns))
    sys.exit(1)

ascending = False if "R2" in metric_to_use or "r2" in metric_to_use else True
runs_sorted = runs.sort_values(metric_to_use, ascending=ascending)
best_run = runs_sorted.iloc[0]

metric_name = metric_to_use.split(".")[-1]
metric_value = best_run[metric_to_use]
print(f"\n🏆 Best model based on {metric_name.upper()} = {metric_value:.4f}")

run_id = best_run["run_id"]
run_name = best_run["tags.mlflow.runName"]
model_name = f"EMI_{run_name}_Model"

# =====================================================
# Step 4: Detect the Correct Artifact Path
# =====================================================
# List artifacts to find the correct model directory name
try:
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    if not artifacts:
        raise Exception("No artifacts found for this run.")
    artifact_path = artifacts[0]  # usually the model folder name
    print(f"📦 Found model artifact folder: '{artifact_path}'")
except Exception as e:
    print(f"⚠️ Could not find model artifact path: {e}")
    sys.exit(1)

# =====================================================
# Step 5: Register the Model
# =====================================================
try:
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{artifact_path}",
        name=model_name
    )
    print(f"✅ Successfully registered model '{model_name}' (Version {result.version})")
except Exception as e:
    print(f"⚠️ Model registration failed: {e}")
    sys.exit(1)

# =====================================================
# Step 6: Promote to Production
# =====================================================
try:
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🚀 Model '{model_name}' promoted to Production successfully!")
except Exception as e:
    print(f"⚠️ Could not promote model to Production: {e}")
