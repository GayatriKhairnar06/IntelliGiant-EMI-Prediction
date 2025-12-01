# app.py — IntelliGiant EMI Prediction (UI like screenshots, exact feature order)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------------------
# MUST be the FIRST Streamlit command
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliGiant EMI Eligibility Prediction System",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------------------------
# Minimal CSS to mimic the screenshot look (wide, subtle cards)
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* page background */
    .reportview-container .main {
        background-color: #ffffff;
    }
    /* header fonts */
    .big-title { font-size: 34px; font-weight: 800; color: #1f2937; }
    .subtitle { font-size: 16px; color: #4b5563; margin-bottom: 12px; }
    /* card-like inputs (give padding so inputs look spaced) */
    .stTextInput, .stNumberInput, .stSelectbox {
        padding: 6px 0;
    }
    /* tabs styling (small tweak) */
    .css-1avcm0n e1fqkh3o0 { padding: 0; }
    /* footer */
    .ig-footer { color: #6b7280; font-size: 13px; margin-top: 25px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Load models (cached resource) — adjust paths if your models are in a subfolder
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("EMI_LogisticRegression_Model.pkl")
    reg = joblib.load("EMI_XGBoostRegressor_Model.pkl")
    return clf, reg

try:
    clf_model, reg_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = e

# ---------------------------------------------------------------------
# Header (left aligned like screenshot)
# ---------------------------------------------------------------------
st.markdown("<div class='big-title'>🏦 IntelliGiant EMI Eligibility Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict your EMI Eligibility and EMI Amount using Machine Learning</div>", unsafe_allow_html=True)

# show small tabs
tab1, tab2 = st.tabs(["📋 EMI Eligibility (Classification)", "💰 EMI Amount Prediction (Regression)"])

# ---------------------------------------------------------------------
# Helper: build ordered DataFrame using model.feature_names_in_
# ---------------------------------------------------------------------
def build_ordered_input_df(values_dict, feature_names):
    """Return a DataFrame with columns ordered exactly as feature_names.
       Missing keys in values_dict will be filled with 0."""
    ordered = []
    for f in feature_names:
        ordered.append(values_dict.get(f, 0))
    df = pd.DataFrame([ordered], columns=list(feature_names))
    return df

# ---------------------------------------------------------------------
# TAB 1: Classification (EMI Eligibility)
# ---------------------------------------------------------------------
with tab1:
    st.markdown("#### 📝 Input Details for EMI Eligibility Check")
    # three columns layout similar to screenshot
    c1, c2, c3 = st.columns([1.4, 1.4, 1.4])

    with c1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1, key="cls_age")
        gender = st.selectbox("Gender", ["Male", "Female"], key="cls_gender")
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3, step=1, key="cls_family")
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1, step=1, key="cls_dependents")

    with c2:
        monthly_salary = st.number_input("Monthly Salary", min_value=0, max_value=2000000, value=50000, step=1000, key="cls_salary")
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=40, value=5, step=1, key="cls_years")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1, key="cls_credit")
        bank_balance = st.number_input("Bank Balance", min_value=0, max_value=10000000, value=50000, step=1000, key="cls_bank")

    with c3:
        requested_amount = st.number_input("Requested Loan Amount", min_value=0, max_value=5000000, value=200000, step=1000, key="cls_req_amt")
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, max_value=360, value=24, step=1, key="cls_req_ten")
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1, step=1, key="cls_exist")
        house_type = st.selectbox("House Type", ["Owned", "Rented"], key="cls_house")

    st.markdown("")  # spacing

    # Build dictionary of inputs using the exact feature names used during training
    # (these are the names you printed with feature_names_in_)
    # We'll map them to values; any missing features will be set to 0 later.
    cls_values = {
        "age": age,
        "gender": 1 if gender == "Male" else 0,                 # original used 1/0
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "house_type": 1 if house_type == "Owned" else 0,
        "monthly_rent": 0,             # not provided in UI here; keep default 0 to match training
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": 0,
        "college_fees": 0,
        "travel_expenses": 0,
        "groceries_utilities": 0,
        "other_monthly_expenses": 0,
        "existing_loans": 1 if existing_loans > 0 else 0,
        "current_emi_amount": 0,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": 0,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        # dummy columns encoded as in your training
        "marital_status_Single": 1 if st.session_state.get("cls_marital", "Married") == "Single" else 0,
        "education_High School": 0,
        "education_Post Graduate": 0,
        "education_Professional": 0,
        "employment_type_Private": 0,
        "employment_type_Self-employed": 0,
        "company_type_MNC": 0,
        "company_type_Mid-size": 0,
        "company_type_Small": 0,
        "company_type_Startup": 0,
        "emi_scenario_Education EMI": 0,
        "emi_scenario_Home Appliances EMI": 0,
        "emi_scenario_Personal Loan EMI": 0,
        "emi_scenario_Vehicle EMI": 0
    }

    # If models failed to load show message
    if not models_loaded:
        st.error(f"Model load error: {load_error}")
    else:
        # Ensure DataFrame columns are ordered exactly as model expects
        feature_names_clf = clf_model.feature_names_in_
        X_cls = build_ordered_input_df(cls_values, feature_names_clf)

        # Button and prediction
        if st.button("🔎 Check EMI Eligibility", key="btn_check_elig"):
            try:
                pred = clf_model.predict(X_cls)[0]
                if pred == 1:
                    st.success("✅ EMI Eligibility: Eligible")
                else:
                    st.error("❌ EMI Eligibility: Not Eligible")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------------------------------------------------------------
# TAB 2: Regression (EMI Amount Prediction)
# ---------------------------------------------------------------------
with tab2:
    st.markdown("#### 🧾 Input Details for EMI Amount Prediction")
    r1, r2, r3 = st.columns([1.4, 1.4, 1.4])

    with r1:
        age_r = st.number_input("Age", min_value=18, max_value=70, value=30, step=1, key="reg_age")
        monthly_salary_r = st.number_input("Monthly Salary", min_value=0, max_value=2000000, value=50000, step=1000, key="reg_salary")
        years_of_employment_r = st.number_input("Years of Employment", min_value=0, max_value=40, value=5, step=1, key="reg_years")

    with r2:
        requested_amount_r = st.number_input("Requested Amount", min_value=0, max_value=5000000, value=200000, step=1000, key="reg_req_amt")
        requested_tenure_r = st.number_input("Requested Tenure (months)", min_value=1, max_value=360, value=24, step=1, key="reg_req_ten")
        credit_score_r = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1, key="reg_credit")

    with r3:
        existing_loans_r = st.number_input("Existing Loans", min_value=0, max_value=10, value=1, step=1, key="reg_exist")
        bank_balance_r = st.number_input("Bank Balance", min_value=0, max_value=10000000, value=50000, step=1000, key="reg_bank")
        family_size_r = st.number_input("Family Size", min_value=1, max_value=10, value=3, step=1, key="reg_family")

    st.markdown("")  # spacing

    # Build regression dictionary (keys matching training)
    reg_values = {
        "age": age_r,
        "gender": 0,   # keep 0/1; UI didn't include gender here in screenshot, default 0
        "monthly_salary": monthly_salary_r,
        "years_of_employment": years_of_employment_r,
        "house_type": 0,
        "monthly_rent": 0,
        "family_size": family_size_r,
        "dependents": 0,
        "school_fees": 0,
        "college_fees": 0,
        "travel_expenses": 0,
        "groceries_utilities": 0,
        "other_monthly_expenses": 0,
        "existing_loans": 1 if existing_loans_r > 0 else 0,
        "current_emi_amount": 0,
        "credit_score": credit_score_r,
        "bank_balance": bank_balance_r,
        "emergency_fund": 0,
        "requested_amount": requested_amount_r,
        "requested_tenure": requested_tenure_r,
        # dummies (set to 0 by default)
        "marital_status_Single": 0,
        "education_High School": 0,
        "education_Post Graduate": 0,
        "education_Professional": 0,
        "employment_type_Private": 0,
        "employment_type_Self-employed": 0,
        "company_type_MNC": 0,
        "company_type_Mid-size": 0,
        "company_type_Small": 0,
        "company_type_Startup": 0,
        "emi_scenario_Education EMI": 0,
        "emi_scenario_Home Appliances EMI": 0,
        "emi_scenario_Personal Loan EMI": 0,
        "emi_scenario_Vehicle EMI": 0
    }

    if not models_loaded:
        st.error(f"Model load error: {load_error}")
    else:
        feature_names_reg = reg_model.feature_names_in_
        X_reg = build_ordered_input_df(reg_values, feature_names_reg)

        if st.button("💡 Predict EMI Amount", key="btn_predict_emi"):
            try:
                emi_val = reg_model.predict(X_reg)[0]
                st.info(f"📌 Predicted EMI Amount: ₹{emi_val:,.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------------------------------------------------------------
# Footer (small)
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("<div class='ig-footer'>🚀 Powered by MLflow, Streamlit, and Scikit-learn | Developed by Gayatri Khairnar 💙</div>", unsafe_allow_html=True)
