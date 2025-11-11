# model_register.py

import mlflow
from mlflow.tracking import MlflowClient

# === Step 1: Connect to MLflow tracking URI ===
mlflow.set_tracking_uri("file:///D:/Project/intelligent/mlruns")
client = MlflowClient()

# === Step 2: Choose your experiment ===
experiment_name = "EMI_Eligibility_Classification"
experiment = client.get_experiment_by_name(experiment_name)

if not experiment:
    print(f"⚠️ Experiment '{experiment_name}' not found.")
    exit()

# === Step 3: Fetch all runs and sort by Accuracy ===
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

if not runs:
    print("⚠️ No runs found for this experiment.")
    exit()

# === Step 4: Display top models ===
print("\n🏆 Top Model Runs:")
for i, run in enumerate(runs[:5], start=1):
    print(f"{i}. {run.data.params['model_name']} - Acc: {run.data.metrics['accuracy']:.4f} | Run ID: {run.info.run_id}")

# === Step 5: Choose a model to register ===
choice = int(input("\nEnter the number of the model to register (1–5): ")) - 1
selected_run = runs[choice]
model_name = selected_run.data.params["model_name"]

model_uri = f"runs:/{selected_run.info.run_id}/{model_name}"
registry_name = f"EMI_{model_name}_Model"

# === Step 6: Register model ===
try:
    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
    print(f"\n✅ Model '{registry_name}' registered successfully as version {mv.version}!")
except Exception as e:
    print(f"⚠️ Model registration failed: {e}")
