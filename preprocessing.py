print("---Step 1: Prerequisite---")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

print("---Step 2: Dataset Loading---")
df = pd.read_csv("emi_prediction_dataset.csv", low_memory=False)
print(df.head())
print(df.columns)
print("Dataset loaded")

# Step 3: Missing / Duplicate check
print("----Step 3: Check for Missing or Duplicated Data------")
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Step 4: Handle missing values
print("\n---- Step 4: Handling Missing Values ------")
df['education'] = df['education'].fillna(df['education'].mode()[0])
numeric_cols = ['monthly_rent', 'credit_score', 'bank_balance', 'emergency_fund']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())
print("✅ Missing values handled successfully!")

#Convert to numeric
numeric_cols = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'current_emi_amount', 'credit_score',
    'bank_balance', 'emergency_fund', 'requested_amount', 'requested_tenure',
    'max_monthly_emi'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print("✅ All numeric columns converted to float successfully.")
# ---  Final NaN Handling ---
print("\n--- : Final Missing Value Handling ---")
# Fill any remaining NaN with median (for numeric) or mode (for categorical)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)
print("✅ All remaining NaN values handled successfully.")

# Step 5: Outlier Handling
print("\n---- Step 5: Outlier Detection & Treatment ------")
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))
print("✅ Outliers treated using IQR method.")

# Step 6: Feature Scaling
print("\n---- Step 6: Feature Scaling ------")
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
joblib.dump(scaler, "scaler.pkl")
print("✅ Feature scaling complete!")

# Step 7: Encoding
print("\n---- Step 7: Encoding Categorical Columns ------")
label_encoder = LabelEncoder()
binary_cols = ['gender', 'existing_loans', 'house_type']
for col in binary_cols:
    df[col] = label_encoder.fit_transform(df[col])

multi_cat_cols = [
    'marital_status', 'education', 'employment_type', 'company_type',
    'emi_scenario', 'emi_eligibility'
]
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Encoding complete! New shape:", df.shape)
print("Remaining NaN count:", df.isnull().sum().sum())

# Save final dataset
df.to_csv("final_preprocessed_dataset.csv", index=False)
print("💾 Final preprocessed dataset saved successfully!")
