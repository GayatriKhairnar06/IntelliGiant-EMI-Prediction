# app.py — IntelliGiant EMI Prediction (feature-safe, order-matching)
# Save/replace your existing app.py with this file.

import streamlit as st

# ------------------ MUST BE FIRST STREAMLIT COMMAND ------------------
st.set_page_config(page_title="IntelliGiant EMI Prediction", page_icon="💰", layout="centered")

# ------------------ imports (after page config) ------------------
import joblib
import pandas as pd
import numpy as np
import traceback

# ------------------ Helpers ------------------
def get_feature_names(model):
    """
    Try to extract feature names expected by a model.
    Tries sklearn attribute `feature_names_in_`, then XGBoost's booster feature names.
    Returns list or None.
    """
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        # XGBoost sklearn wrapper may store _feature_names or underlying booster
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            fn = booster.feature_names
            if fn:
                return list(fn)
        # fallback
        return None
    except Exception:
        return None

def one_hot_match(feature_name, user_values):
    """
    Given a feature_name like 'education_Post Graduate' or 'emi_scenario_Vehicle EMI',
    check user_values (dict) to decide if it should be 1 or 0.
    """
    # split on first underscore
    if "_" not in feature_name:
        return None
    base, cat = feature_name.split("_", 1)
    base = base.strip().lower()
    cat = cat.strip().lower()

    # common mapping keys in user inputs (lowercase)
    if base in user_values:
        val = user_values[base]
        # if user val is categorical string
        if isinstance(val, str):
            return 1 if val.strip().lower() == cat else 0
        # if user val is boolean/int
        if isinstance(val, (int, float, np.integer, np.floating)):
            # probably not one-hot; return None to let caller handle numeric
            return None
    # Also handle patterns where feature_name uses spaces or hyphens vs user value
    # e.g., 'house_type_Owned' vs user_values['house_type']='Owned'
    for k, v in user_values.items():
        if isinstance(v, str) and base == k:
            return 1 if v.strip().lower() == cat else 0
    return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# ------------------ Load models ------------------
model_load_error = None
clf_model = None
reg_model = None
try:
    clf_model = joblib.load("EMI_LogisticRegression_Model.pkl")
    reg_model = joblib.load("EMI_XGBoostRegressor_Model.pkl")
    model_status = "loaded"
except Exception as e:
    model_load_error = traceback.format_exc()
    model_status = f"error: {e}"

# ------------------ Page header ------------------
st.title("💸 IntelliGiant: EMI Eligibility & Prediction Platform")
if model_status == "loaded":
    st.success("✅ Models loaded successfully!")
else:
    st.error("❌ Error loading models. See details below.")
    st.code(model_load_error)

st.write("Analyze your financial profile to check EMI eligibility and estimate EMI amount.")

# ------------------ User Input UI ------------------
st.header("📋 Enter Applicant Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education", ["Graduate", "High School", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Self-employed"])
    company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Small", "Startup"])
with col2:
    monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 2000000, 50000, step=1000)
    years_of_employment = st.number_input("Years of Employment", 0, 40, 3)
    house_type = st.selectbox("House Type", ["Owned", "Rented"])
    monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000)
    family_size = st.number_input("Family Size", 1, 10, 4)
    dependents = st.number_input("Dependents", 0, 10, 1)

st.header("💰 Financial Information")
school_fees = st.number_input("School Fees (₹)", 0, 100000, 5000)
college_fees = st.number_input("College Fees (₹)", 0, 200000, 0)
travel_expenses = st.number_input("Travel Expenses (₹)", 0, 50000, 2000)
groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 100000, 8000)
other_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 50000, 3000)
existing_loans = st.selectbox("Any Existing Loans?", ["No", "Yes"])
current_emi = st.number_input("Current EMI (₹)", 0, 100000, 0)
credit_score = st.number_input("Credit Score", 300, 900, 650)
bank_balance = st.number_input("Bank Balance (₹)", 0, 10000000, 20000)
emergency_fund = st.number_input("Emergency Fund (₹)", 0, 1000000, 10000)
emi_scenario = st.selectbox(
    "EMI Scenario",
    ["Home Appliances EMI", "Vehicle EMI", "Education EMI", "Personal Loan EMI"]
)
requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 5000000, 200000, step=1000)
requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 24)

# ------------------ Build a user_values dict for easy matching ------------------
user_values = {
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "education": education,
    "employment_type": employment_type,
    "company_type": company_type,
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "house_type": house_type,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_expenses,
    "existing_loans": 1 if existing_loans == "Yes" else 0,
    "current_emi_amount": current_emi,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "emi_scenario": emi_scenario,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
}

# ------------------ Determine required features (from models) ------------------
required_features_clf = get_feature_names(clf_model) if clf_model is not None else None
required_features_reg = get_feature_names(reg_model) if reg_model is not None else None

# Prefer classifier feature list for matching if present, else reg, else union
if required_features_clf:
    required_features = required_features_clf
elif required_features_reg:
    required_features = required_features_reg
else:
    required_features = None

if required_features is None:
    st.warning("Could not detect feature names from models. The app will attempt a best-effort mapping.")
    # fallback to the columns you previously used in the app (best-effort)
    required_features = [
        "age", "gender", "marital_status_Single", "education_High School",
        "education_Post Graduate", "education_Professional", "monthly_salary",
        "employment_type_Private", "employment_type_Self-employed", "years_of_employment",
        "company_type_MNC", "company_type_Mid-size", "company_type_Small", "company_type_Startup",
        "house_type", "monthly_rent", "family_size", "dependents", "school_fees",
        "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
        "emergency_fund", "emi_scenario_Education EMI", "emi_scenario_Home Appliances EMI",
        "emi_scenario_Personal Loan EMI", "emi_scenario_Vehicle EMI",
        "requested_amount", "requested_tenure"
    ]

# ------------------ Construct input row matching required_features in order ------------------
input_row = {}
missing_filled = []
unmapped = []

for feat in required_features:
    # exact direct numeric match?
    key_lower = feat.strip().lower()
    if key_lower in user_values:
        # direct value present (careful with categorical strings vs numeric)
        input_row[feat] = user_values[key_lower]
        continue

    # common straightforward keys mapping (e.g. 'gender' might be stored as 'gender' numeric)
    if feat.strip().lower() == "gender":
        input_row[feat] = 1 if gender == "Male" else 0
        continue

    # try numeric-suffixed names like 'monthly_salary' etc.
    if feat in user_values:
        input_row[feat] = user_values[feat]
        continue

    # try one-hot style matching like 'education_Post Graduate'
    oh = one_hot_match(feat, user_values)
    if oh is not None:
        input_row[feat] = oh
        continue

    # Some training pipelines use prefixes or different separators like '-'
    # Try replacing '-' with '_' and retry
    alt = feat.replace("-", "_")
    if alt.lower() in user_values:
        input_row[feat] = user_values[alt.lower()]
        continue

    # handle cases like 'marital_status_Married' or 'marital_status_Single'
    if "_" in feat:
        base = feat.split("_", 1)[0].strip().lower()
        if base in user_values and isinstance(user_values[base], str):
            # set 1 if category matches
            input_row[feat] = 1 if user_values[base].strip().lower() == feat.split("_", 1)[1].strip().lower() else 0
            continue

    # handle boolean-like names stored as 0/1 e.g., 'existing_loans'
    if feat.lower() in ["existing_loans", "existingloan", "has_existing_loans"]:
        input_row[feat] = 1 if existing_loans == "Yes" else 0
        continue

    # if none matched, set default 0 (safe) and note it
    input_row[feat] = 0
    missing_filled.append(feat)

# Convert any numpy types and ensure numeric where possible
for k, v in input_row.items():
    # if v is a string that should be numeric, try to convert where reasonable
    if isinstance(v, str):
        # leave strings for models expecting string (rare) else attempt to convert
        try:
            nv = float(v)
            input_row[k] = nv
        except Exception:
            # keep as-is (some models accept categorical strings if pipeline exists)
            input_row[k] = v
    else:
        input_row[k] = v

# Build DataFrame with columns in required order
input_df = pd.DataFrame([input_row], columns=required_features)

# Show a compact preview and warning if many features had to be filled
with st.expander("Input features preview (matched to model)"):
    st.dataframe(input_df.T.rename(columns={0: "value"}))

if missing_filled:
    st.warning(f"{len(missing_filled)} feature(s) were not directly mappable from UI and were set to 0. Example: {missing_filled[:5]}")
    if st.checkbox("Show which features were auto-filled with 0"):
        st.write(missing_filled)

# ------------------ Prediction ------------------
if st.button("🔍 Predict EMI Eligibility and Amount"):
    if model_status != "loaded":
        st.error("Models not loaded. Cannot predict.")
    else:
        try:
            # Ensure both models accept the same feature order. If not, reorder accordingly
            # For classifier:
            clf_features = get_feature_names(clf_model) or required_features
            reg_features = get_feature_names(reg_model) or required_features

            X_clf = input_df.copy()
            X_reg = input_df.copy()

            # Reorder/clamp columns for each model if needed
            if clf_features:
                # keep only required columns for clf in same order
                clf_cols = [c for c in clf_features if c in X_clf.columns]
                missing = [c for c in clf_features if c not in X_clf.columns]
                if missing:
                    st.warning(f"Classifier expects {len(missing)} columns missing from input — they'll be treated as 0: {missing[:5]}")
                # add missing columns as zeros
                for c in missing:
                    X_clf[c] = 0
                X_clf = X_clf[clf_features]

            if reg_features:
                reg_cols = [c for c in reg_features if c in X_reg.columns]
                missing_r = [c for c in reg_features if c not in X_reg.columns]
                if missing_r:
                    st.warning(f"Regressor expects {len(missing_r)} columns missing from input — they'll be treated as 0: {missing_r[:5]}")
                for c in missing_r:
                    X_reg[c] = 0
                X_reg = X_reg[reg_features]

            # predict
            eligibility_pred = clf_model.predict(X_clf)[0]
            eligibility_label = "Eligible" if int(eligibility_pred) == 1 else "Not Eligible"

            emi_pred = reg_model.predict(X_reg)[0]

            st.subheader("📊 Prediction Results")
            st.write(f"### ✅ EMI Eligibility: **{eligibility_label}**")
            st.write(f"### 💵 Predicted EMI Amount: **₹{emi_pred:,.2f}**")

        except Exception as e:
            st.error("⚠️ Error during prediction. See details below.")
            st.exception(e)
