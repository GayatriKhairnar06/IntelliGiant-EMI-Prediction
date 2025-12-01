# app.py — IntelliGiant EMI Prediction (Streamlit Deployment)

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------------------------------------
# Step 1: Load trained models (stored locally)
# -----------------------------------------------------------
try:
    clf_model = joblib.load("models/EMI_LogisticRegression_Model.pkl")
    reg_model = joblib.load("models/EMI_XGBoostRegressor_Model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    st.success("Models loaded successfully 🎉")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# -----------------------------------------------------------
# Step 2: Page configuration
# -----------------------------------------------------------
st.set_page_config(page_title="IntelliGiant EMI Prediction", page_icon="💰", layout="centered")

# -----------------------------------------------------------
# Step 3: UI Header
# -----------------------------------------------------------
st.title("💰 IntelliGiant EMI Prediction")
st.write("Predict EMI Eligibility & EMI Amount using Machine Learning.")

# -----------------------------------------------------------
# Step 4: Input Form
# -----------------------------------------------------------
st.header("📄 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 80, 25)
    income = st.number_input("Monthly Income (₹)", 5000, 500000, 50000)
    savings = st.number_input("Monthly Savings (₹)", 0, 200000, 5000)
    credit_score = st.number_input("Credit Score", 300, 900, 650)

with col2:
    expenses = st.number_input("Monthly Expenses (₹)", 1000, 300000, 20000)
    existing_loans = st.number_input("Existing Loan Amount (₹)", 0, 1000000, 0)
    dependents = st.number_input("Number of Dependents", 0, 10, 1)
    loan_term = st.number_input("Loan Term (months)", 3, 240, 36)

# -----------------------------------------------------------
# Step 5: Create DataFrame
# -----------------------------------------------------------
input_dict = {
    "age": age,
    "income": income,
    "savings": savings,
    "credit_score": credit_score,
    "expenses": expenses,
    "existing_loans": existing_loans,
    "dependents": dependents,
    "loan_term": loan_term
}

input_df = pd.DataFrame([input_dict])

# Apply scaler
scaled_input = scaler.transform(input_df)

# -----------------------------------------------------------
# Step 6: Prediction Button
# -----------------------------------------------------------
if st.button("Predict EMI"):
    # Classification (Eligibility)
    eligibility_pred = clf_model.predict(scaled_input)[0]
    eligibility_prob = clf_model.predict_proba(scaled_input)[0][1]

    # Regression (Amount Prediction)
    emi_amount = reg_model.predict(scaled_input)[0]

    # -------------------------------------------------------
    # Step 7: Display Results
    # -------------------------------------------------------
    st.subheader("📌 Prediction Results")

    if eligibility_pred == 1:
        st.success(f"✅ Customer is Eligible for EMI (Confidence: {eligibility_prob*100:.2f}%)")
        st.info(f"💵 Recommended EMI Amount: **₹{emi_amount:,.2f}**")
    else:
        st.error(f"❌ Customer is NOT Eligible for EMI (Confidence: {(1-eligibility_prob)*100:.2f}%)")
        st.warning("💡 Improve credit score, increase savings, or reduce expenses.")


# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.write("---")
st.caption("Built with ❤️ by Gayatri • IntelliGiant EMI Prediction System")
