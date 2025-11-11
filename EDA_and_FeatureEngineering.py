# EDA_and_FeatureEngineering.py

print("=== STEP 2: Exploratory Data Analysis (EDA) ===")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
print("below 2 lines will make sure mlflow ui could show visuals ")
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("EMI_Eligibility_Exploration")

# Create folder for EDA visuals
os.makedirs("eda_visuals", exist_ok=True)

# ------------------------------
# STEP 2A — Load & Prepare Data
# ------------------------------
df_scaled = pd.read_csv("final_preprocessed_dataset.csv")

# Recreate EMI eligibility column from one-hot encoded fields
if 'emi_eligibility_Not_Eligible' in df_scaled.columns:
    df_scaled['emi_eligibility'] = df_scaled.apply(
        lambda row: 'Not_Eligible' if row['emi_eligibility_Not_Eligible'] else
                    ('High_Risk' if row.get('emi_eligibility_High_Risk', False) else 'Eligible'),
        axis=1
    )

# Make a copy to unscale numeric data for visual interpretability
df = df_scaled.copy()

# These were standardized before — convert back using mean/std from your dataset (approximation)
# If original unscaled dataset is not available, visualize relative values instead
scaled_cols = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'current_emi_amount', 'credit_score',
    'bank_balance', 'emergency_fund', 'requested_amount',
    'requested_tenure', 'max_monthly_emi'
]

print("\n--- Dataset Overview ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# ------------------------------
# STEP 2B — EDA Visualizations
# ------------------------------
print("\n📊 Generating EDA Visualizations...")

# 1️⃣ EMI Eligibility Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='emi_eligibility', data=df)
plt.title("EMI Eligibility Distribution")
plt.savefig("eda_visuals/emi_eligibility_distribution.png")
plt.close()

# 2️⃣ EMI Eligibility across Lending Scenarios
emi_scenario_cols = [col for col in df.columns if col.startswith('emi_scenario_')]
emi_scenario_df = df.melt(id_vars='emi_eligibility', value_vars=emi_scenario_cols,
                          var_name='Scenario', value_name='Active')
emi_scenario_df = emi_scenario_df[emi_scenario_df['Active']]
emi_scenario_df['Scenario'] = emi_scenario_df['Scenario'].str.replace('emi_scenario_', '')

plt.figure(figsize=(8,5))
sns.countplot(x='Scenario', hue='emi_eligibility', data=emi_scenario_df)
plt.title("EMI Eligibility Across Lending Scenarios")
plt.xticks(rotation=45)
plt.savefig("eda_visuals/emi_eligibility_vs_scenario.png")
plt.close()

# 3️⃣ Correlation Heatmap
financial_vars = [
    'monthly_salary','monthly_rent','current_emi_amount','credit_score',
    'bank_balance','emergency_fund','requested_amount','max_monthly_emi'
]
plt.figure(figsize=(10,6))
sns.heatmap(df[financial_vars].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Financial Variables")
plt.savefig("eda_visuals/financial_correlation_heatmap.png")
plt.close()

# 4️⃣ Age Distribution by EMI Eligibility
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='age', hue='emi_eligibility', kde=True, bins=25)
plt.title("Age Distribution by EMI Eligibility")
plt.savefig("eda_visuals/age_vs_emi_eligibility.png")
plt.close()

# 5️⃣ Employment Type vs EMI Eligibility
employment_cols = [col for col in df.columns if col.startswith('employment_type_')]
employment_df = df.melt(id_vars='emi_eligibility', value_vars=employment_cols,
                        var_name='EmploymentType', value_name='Active')
employment_df = employment_df[employment_df['Active']]
employment_df['EmploymentType'] = employment_df['EmploymentType'].str.replace('employment_type_', '')

plt.figure(figsize=(8,5))
sns.countplot(x='EmploymentType', hue='emi_eligibility', data=employment_df)
plt.title("EMI Eligibility Across Employment Types")
plt.savefig("eda_visuals/employment_vs_emi_eligibility.png")
plt.close()

print("✅ EDA Visuals Saved in 'eda_visuals/' Folder!")

# ------------------------------
# STEP 3 — Feature Engineering
# ------------------------------
print("\n=== STEP 3: Feature Engineering ===")

# 1️⃣ Derived Financial Ratios
df['debt_to_income_ratio'] = df['current_emi_amount'] / (df['monthly_salary'] + 1e-6)
df['expense_to_income_ratio'] = df['other_monthly_expenses'] / (df['monthly_salary'] + 1e-6)
df['affordability_ratio'] = df['max_monthly_emi'] / (df['monthly_salary'] + 1e-6)

# 2️⃣ Risk Scoring
df['employment_stability_score'] = np.where(df['years_of_employment'] >= 5, 1, 0)
df['credit_risk_score'] = pd.cut(df['credit_score'],
                                 bins=[-np.inf, 500, 700, np.inf],
                                 labels=[0,1,2]).astype(int)

# 3️⃣ Interaction Features
df['income_credit_interaction'] = df['monthly_salary'] * df['credit_score']
df['loan_request_ratio'] = df['requested_amount'] / (df['bank_balance'] + 1e-6)

# 4️⃣ Encode Categorical Variables (if needed)
cat_cols = ['gender','house_type']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 5️⃣ Scale Numeric Columns for Modeling
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['float64','int64']).columns
df[num_cols] = scaler.fit_transform(df[num_cols])

# 6️⃣ Save Feature-Engineered Dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/feature_engineered_dataset.csv", index=False)

print("💾 Feature engineering complete! Saved as 'data/feature_engineered_dataset.csv'")
print("📁 EDA visuals saved in 'eda_visuals/' — ready for MLflow or report inclusion.")

print("Below Code Helps User to Show Visuals i Mlflow")
# ======================================================
# 📦 Step 6: Log EDA Visuals and Feature Data to MLflow
# ======================================================

import mlflow
import os

# (Optional) Set your MLflow tracking URI if using local server
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define folder where visuals are saved
eda_folder = "eda_visuals"
os.makedirs(eda_folder, exist_ok=True)

# Example: Save all your plots like this
# plt.savefig(os.path.join(eda_folder, "distribution_plot.png"))

# Start a dedicated EDA tracking run
with mlflow.start_run(run_name="EDA_Analysis"):
    print("\n📊 Logging EDA artifacts to MLflow...")

    # Log entire visuals folder (all saved PNGs, etc.)
    mlflow.log_artifacts(eda_folder, artifact_path="EDA_Plots")

    # If your cleaned dataset is stored
    if os.path.exists("final_preprocessed_dataset.csv"):
        mlflow.log_artifact("final_preprocessed_dataset.csv")

    # If your feature-engineered dataset is stored
    if os.path.exists("data/feature_engineered_dataset.csv"):
        mlflow.log_artifact("data/feature_engineered_dataset.csv")

    print("✅ EDA visuals and datasets successfully logged to MLflow!")

print("\n🎯 Run 'mlflow ui' and open http://127.0.0.1:5000 to explore EDA results visually.")

