"""
data_splitter.py
----------------
This script splits the preprocessed dataset into train, validation, and test sets
for both classification (emi_eligibility) and regression (max_monthly_emi) tasks.
"""

print("--- Step 1: Prerequisites ---")
import pandas as pd
from sklearn.model_selection import train_test_split

print("--- Step 2: Load Preprocessed Dataset ---")
df = pd.read_csv("final_preprocessed_dataset.csv")
print("✅ Dataset loaded successfully! Shape:", df.shape)

# Identify target variables
print("--- Step 3: Define Targets ---")
# Regression target
y_reg = df['max_monthly_emi']
# Classification target: pick columns starting with 'emi_eligibility_'
# --- Classification Target ---
class_cols = [col for col in df.columns if col.startswith('emi_eligibility_')]
if len(class_cols) > 0:
    # pick only one dummy column as y (1 = Eligible, 0 = Not_Eligible)
    y_class = df[class_cols[0]]
else:
    raise ValueError("❌ Classification target columns (emi_eligibility_) not found after encoding.")


# Input features (drop both target sets)
x = df.drop(columns=['max_monthly_emi'] + class_cols)

print(f"Feature matrix: {x.shape}, Regression target: {y_reg.shape}, Classification target: {y_class.shape}")

# Step 4: Split for regression
print("\n--- Step 4: Splitting Data for Regression ---")
x_train_reg, x_temp_reg, y_train_reg, y_temp_reg = train_test_split(x, y_reg, test_size=0.3, random_state=42)
x_val_reg, x_test_reg, y_val_reg, y_test_reg = train_test_split(x_temp_reg, y_temp_reg, test_size=0.5, random_state=42)

# Step 5: Split for classification
print("\n--- Step 5: Splitting Data for Classification ---")
x_train_class, x_temp_class, y_train_class, y_temp_class = train_test_split(x, y_class, test_size=0.3, random_state=42)
x_val_class, x_test_class, y_val_class, y_test_class = train_test_split(x_temp_class, y_temp_class, test_size=0.5, random_state=42)

# Step 6: Save all splits
print("\n--- Step 6: Saving Split Data ---")
x_train_reg.to_csv("data/x_train_reg.csv", index=False)
x_val_reg.to_csv("data/x_val_reg.csv", index=False)
x_test_reg.to_csv("data/x_test_reg.csv", index=False)
y_train_reg.to_csv("data/y_train_reg.csv", index=False)
y_val_reg.to_csv("data/y_val_reg.csv", index=False)
y_test_reg.to_csv("data/y_test_reg.csv", index=False)

x_train_class.to_csv("data/x_train_class.csv", index=False)
x_val_class.to_csv("data/x_val_class.csv", index=False)
x_test_class.to_csv("data/x_test_class.csv", index=False)
y_train_class.to_csv("data/y_train_class.csv", index=False)
y_val_class.to_csv("data/y_val_class.csv", index=False)
y_test_class.to_csv("data/y_test_class.csv", index=False)

print("💾 All splits saved successfully inside 'data/' folder!")

print("\n✅ Data splitting completed:")
print("  • Regression → Train/Val/Test shapes:",
      x_train_reg.shape, x_val_reg.shape, x_test_reg.shape)
print("  • Classification → Train/Val/Test shapes:",
      x_train_class.shape, x_val_class.shape, x_test_class.shape)
