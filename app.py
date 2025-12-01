# app.py — IntelliGiant EMI Prediction
# Deployment-ready version (no MLflow dependency)

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------------------------------------
# Step 1: Load trained models (stored locally)
# -----------------------------------------------------------
st.set_page_config(
    page_title="IntelliGiant EMI Prediction",
    page_icon="💰",
    layout="centered"
)

try:
    clf_model = joblib.load("EMI_LogisticRegression_Model.pkl")
    reg_model = joblib.load("EMI_XGBoostRegressor_Model.pkl")
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")

# -----------------------------------------------------------
# Step 2: Streamlit Page Setup
# -----------------------------------------------------------
st.title("💸 IntelliGiant: EMI Eligibility & Prediction Platform")
st.write("Analyze your financial profile to check EMI eligibility and estimate EMI amount.")

# -----------------------------------------------------------
# Step 3: Collect User Inputs
# -----------------------------------------------------------
st.header("📋 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox(
        "Education",
        ["Graduate", "High School", "Post Graduate", "Professional"]
    )
    employment_type = st.selectbox("Employment Type", ["Private", "Self-employed"])
    company_type = st.selectbox(
        "Company Type",
        ["MNC", "Mid-size", "Small", "Startup"]
    )

with col2:
    monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 2000000, 50000)
    years_of_employment = st.number_input("Years of Employment", 0, 40, 3)
    house_type = st.selectbox("House Type", ["Owned", "Rented"])
    monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000)
    family_size = st.number_input("Family Size", 1, 10, 4)
    dependents = st.number_input("Dependents", 0, 5, 1)

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

requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 5000000, 200000)
requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 24)

# -----------------------------------------------------------
# Step 4: Preprocess Input Data
# -----------------------------------------------------------
input_data = pd.DataFrame({
    "age": [age],
    "gender": [1 if gender == "Male" else 0],
    "marital_status_Single": [1 if marital_status == "Single" else 0],
    "education_High School": [1 if education == "High School" else 0],
    "education_Post Graduate": [1 if education == "Post Graduate" else 0],
    "education_Professional": [1 if education == "Professional" else 0],
    "monthly_salary": [monthly_salary],
    "employment_type_Private": [1 if employment_type == "Private" else 0],
    "employment_type_Self-employed": [1 if employment_type == "Self-employed" else 0],
    "years_of_employment": [years_of_employment],
    "company_type_MNC": [1 if company_type == "MNC" else 0],
    "company_type_Mid-size": [1 if company_type == "Mid-size" else 0],
    "company_type_Small": [1 if company_type == "Small" else 0],
    "company_type_Startup": [1 if company_type == "Startup" else 0],
    "house_type": [1 if house_type == "Owned" else 0],
    "monthly_rent": [monthly_rent],
    "family_size": [family_size],
    "dependents": [dependents],
    "school_fees": [school_fees],
    "college_fees": [college_fees],
    "travel_expenses": [travel_expenses],
    "groceries_utilities": [groceries_utilities],
    "other_monthly_expenses": [other_expenses],
    "existing_loans": [1 if existing_loans == "Yes" else 0],
    "current_emi_amount": [current_emi],
    "credit_score": [credit_score],
    "bank_balance": [bank_balance],
    "emergency_fund": [emergency_fund],
    "emi_scenario_Education EMI": [1 if emi_scenario == "Education EMI" else 0],
    "emi_scenario_Home Appliances EMI": [1 if emi_scenario == "Home Appliances EMI" else 0],
    "emi_scenario_Personal Loan EMI": [1 if emi_scenario == "Personal Loan EMI" else 0],
    "emi_scenario_Vehicle EMI": [1 if emi_scenario == "Vehicle EMI" else 0],
    "requested_amount": [requested_amount],
    "requested_tenure": [requested_tenure],
})

# -----------------------------------------------------------
# Step 5: Prediction
# -----------------------------------------------------------
if st.button("🔍 Predict EMI Eligibility and Amount"):
    try:
        # Classification prediction
        eligibility_pred = clf_model.predict(input_data)[0]
        eligibility_label = "Eligible" if eligibility_pred == 1 else "Not Eligible"

        # Regression prediction (EMI amount)
        emi_pred = reg_model.predict(input_data)[0]

        st.subheader("📊 Prediction Results")
        st.write(f"### ✅ EMI Eligibility: {eligibility_label}")
        st.write(f"### 💵 Predicted EMI Amount: ₹{emi_pred:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# -----------------------------------------------------------
# Step 6: Footer
# -----------------------------------------------------------
st.markdown("---")
st.caption("🚀 Developed by Gayatri Khairnar | IntelliGiant EMI Prediction Platform")
