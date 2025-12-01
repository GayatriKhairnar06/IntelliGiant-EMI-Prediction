import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------------------
# Load Models Safely (NO path changes, works with your structure)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("EMI_LogisticRegression_Model.pkl")
    reg = joblib.load("EMI_XGBoostRegressor_Model.pkl")
    return clf, reg

clf_model, reg_model = load_models()

# ---------------------------------------------------------------------
# UI CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliGiant EMI Prediction System",
    layout="wide",
    page_icon="💳"
)

# ---------------------------------------------------------------------
# CSS for Beautiful Dashboard UI
# ---------------------------------------------------------------------
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stTabs [role="tab"] {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin-right: 10px;
            font-weight: 600;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
        }
        .big-title {
            font-size: 32px; 
            font-weight: 800; 
            color: #222;
        }
        .sub {
            font-size: 17px;
            margin-bottom: 20px;
            color: #444;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: #6c757d;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown("<div class='big-title'>💳 IntelliGiant EMI Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Predict EMI Eligibility & EMI Amount using Machine Learning</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TAB SYSTEM
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["🏦 EMI Eligibility (Classification)", "📊 EMI Amount Prediction (Regression)"])

# **************************************************************************************
# TAB 1 — CLASSIFICATION
# **************************************************************************************
with tab1:

    st.markdown("### 📝 Input Details for EMI Eligibility Check")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_size = st.number_input("Family Size", 1, 20, 3)

    with col2:
        monthly_salary = st.number_input("Monthly Salary", 0, 1000000, 50000)
        years_of_employment = st.number_input("Years of Employment", 0, 50, 5)
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        bank_balance = st.number_input("Bank Balance", 0, 5000000, 50000)

    with col3:
        requested_amount = st.number_input("Requested Loan Amount", 0, 20000000, 200000)
        requested_tenure = st.number_input("Requested Tenure (months)", 1, 360, 24)
        existing_loans = st.number_input("Existing Loans", 0, 20, 1)
        house_type = st.selectbox("House Type", ["Owned", "Rented"])

    dependents = st.number_input("Dependents", 0, 10, 1)

    # ---------------- Build input row EXACTLY matching your model ----------------
    input_dict = {
        "age": age,
        "gender": gender,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "house_type": house_type,
        "monthly_rent": 0,   # <-- You had this in training (kept same)
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": 0,
        "college_fees": 0,
        "travel_expenses": 0,
        "groceries_utilities": 0,
        "other_monthly_expenses": 0,
        "existing_loans": existing_loans,
        "current_emi_amount": 0,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": 0,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
    }

    # Convert DF
    X = pd.DataFrame([input_dict])

    # Dummy columns **exactly matching model training**
    needed_cols = list(clf_model.feature_names_in_)
    for col in needed_cols:
        if col not in X:
            X[col] = 0
    X = X[needed_cols]

    if st.button("Check EMI Eligibility"):
        pred = clf_model.predict(X)[0]
        if pred == 1:
            st.success("✅ You are Eligible for EMI")
        else:
            st.error("❌ You are Not Eligible for EMI")

# **************************************************************************************
# TAB 2 — REGRESSION
# **************************************************************************************
with tab2:

    st.markdown("### 🧮 Input Details for EMI Amount Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        monthly_salary = st.number_input("Monthly Salary", 0, 1000000, 50000)
        years_of_employment = st.number_input("Years of Employment", 0, 50, 5)
        credit_score = st.number_input("Credit Score", 300, 900, 650)

    with col2:
        requested_amount = st.number_input("Requested Amount", 0, 20000000, 200000)
        requested_tenure = st.number_input("Requested Tenure (months)", 1, 360, 24)
        existing_loans = st.number_input("Existing Loans", 0, 20, 1)
        bank_balance = st.number_input("Bank Balance", 0, 5000000, 50000)

    family_size = st.number_input("Family Size", 1, 20, 3)

    # Build regression input row exactly like training
    input_reg = {
        "age": age,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "existing_loans": existing_loans,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "family_size": family_size,
    }

    Xr = pd.DataFrame([input_reg])

    needed_cols_reg = list(reg_model.feature_names_in_)
    for col in needed_cols_reg:
        if col not in Xr:
            Xr[col] = 0
    Xr = Xr[needed_cols_reg]

    if st.button("Predict EMI Amount"):
        emi_pred = reg_model.predict(Xr)[0]
        st.info(f"📌 **Predicted EMI Amount:** ₹{emi_pred:,.2f}")

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown("<div class='footer'>Powered by MLflow, Streamlit & Scikit-learn | Developed by Gayatri Khairnar 💙</div>", unsafe_allow_html=True)
