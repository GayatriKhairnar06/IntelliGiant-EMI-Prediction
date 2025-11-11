# app.py
import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
import numpy as np
import os

# =========================================================
# Step 1: MLflow Tracking Setup
# =========================================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow server must be running

CLASSIFICATION_MODEL_NAME = "EMI_LogisticRegression_Model"
REGRESSION_MODEL_NAME = "EMI_XGBoostRegressor_Model"

# =========================================================
# Step 2: Load Models from MLflow Registry
# =========================================================
@st.cache_resource
def load_models():
    try:
        clf_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{CLASSIFICATION_MODEL_NAME}/Production"
        )
    except Exception as e:
        clf_model = None
        st.error(f"Error loading classification model: {e}")

    try:
        reg_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{REGRESSION_MODEL_NAME}/Production"
        )
    except Exception as e:
        reg_model = None
        st.error(f"Error loading regression model: {e}")

    return clf_model, reg_model


clf_model, reg_model = load_models()

# =========================================================
# Step 3: Streamlit UI Setup
# =========================================================
st.set_page_config(page_title="Intelligent EMI Eligibility System", layout="wide")
st.title("🏦 IntelliGiant EMI Eligibility Prediction System")
st.markdown("### Predict your EMI Eligibility and EMI Amount using Machine Learning")

tabs = st.tabs(["📊 EMI Eligibility (Classification)", "💰 EMI Amount Prediction (Regression)"])

# =========================================================
# 🧠 TAB 1 — Classification
# =========================================================
with tabs[0]:
    st.subheader("📋 Input Details for EMI Eligibility Check")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_size = st.number_input("Family Size", 1, 10, 3)
        dependents = st.number_input("Dependents", 0, 5, 1)
    with col2:
        monthly_salary = st.number_input("Monthly Salary", 10000, 200000, 50000)
        years_of_employment = st.number_input("Years of Employment", 0, 40, 5)
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        bank_balance = st.number_input("Bank Balance", 0, 1000000, 50000)
    with col3:
        requested_amount = st.number_input("Requested Loan Amount", 10000, 1000000, 200000)
        requested_tenure = st.number_input("Requested Tenure (months)", 6, 60, 24)
        existing_loans = st.number_input("Existing Loans", 0, 10, 1)
        house_type = st.selectbox("House Type", ["Owned", "Rented", "Company Provided"])

    # 🧾 Prepare Input Data
    if st.button("🔍 Check EMI Eligibility"):
        try:
            # ✅ Load feature template from training data
            template = pd.read_csv("data/x_train_class.csv")
            template = template.iloc[0:1].copy()

            # Fill user inputs
            template["age"] = age
            template["gender"] = 1 if gender == "Male" else 0
            template["monthly_salary"] = monthly_salary
            template["years_of_employment"] = years_of_employment
            template["family_size"] = family_size
            template["dependents"] = dependents
            template["credit_score"] = credit_score
            template["bank_balance"] = bank_balance
            template["requested_amount"] = requested_amount
            template["requested_tenure"] = requested_tenure
            template["existing_loans"] = existing_loans
            template["house_type"] = 1 if house_type == "Owned" else 0

            # Fill all missing columns with 0
            input_data = template.fillna(0)

            if clf_model:
                eligibility_pred = clf_model.predict(input_data)[0]
                label = "✅ Eligible" if eligibility_pred == 1 else "❌ Not Eligible"
                st.success(f"**Predicted EMI Eligibility:** {label}")
            else:
                st.warning("Classification model not loaded.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# =========================================================
# 💰 TAB 2 — Regression
# =========================================================
with tabs[1]:
    st.subheader("📋 Input Details for EMI Amount Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_r = st.number_input("Age", 18, 70, 30, key="age_r")
        monthly_salary_r = st.number_input("Monthly Salary", 10000, 200000, 50000, key="salary_r")
        years_of_employment_r = st.number_input("Years of Employment", 0, 40, 5, key="emp_r")
    with col2:
        requested_amount_r = st.number_input("Requested Amount", 10000, 1000000, 200000, key="req_amt_r")
        requested_tenure_r = st.number_input("Requested Tenure (months)", 6, 60, 24, key="req_tenure_r")
        credit_score_r = st.number_input("Credit Score", 300, 900, 650, key="cred_r")
    with col3:
        existing_loans_r = st.number_input("Existing Loans", 0, 10, 1, key="loan_r")
        bank_balance_r = st.number_input("Bank Balance", 0, 1000000, 50000, key="bank_r")
        family_size_r = st.number_input("Family Size", 1, 10, 3, key="fam_r")

    if st.button("💡 Predict EMI Amount"):
        try:
            # ✅ Load feature template from training data
            template_r = pd.read_csv("data/x_train_reg.csv")
            template_r = template_r.iloc[0:1].copy()

            template_r["age"] = age_r
            template_r["monthly_salary"] = monthly_salary_r
            template_r["years_of_employment"] = years_of_employment_r
            template_r["requested_amount"] = requested_amount_r
            template_r["requested_tenure"] = requested_tenure_r
            template_r["credit_score"] = credit_score_r
            template_r["existing_loans"] = existing_loans_r
            template_r["bank_balance"] = bank_balance_r
            template_r["family_size"] = family_size_r

            input_data_r = template_r.fillna(0)

            if reg_model:
                emi_amount = reg_model.predict(input_data_r)[0]
                st.success(f"💰 **Predicted EMI Amount:** ₹{emi_amount:,.2f}")
            else:
                st.warning("Regression model not loaded.")
        except Exception as e:
            st.error(f"Error during regression prediction: {e}")

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("🚀 Powered by MLflow, Streamlit, and Scikit-learn | Developed by Gayatri Khairnar 💙")
