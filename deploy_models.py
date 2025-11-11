# deploy_model.py
import streamlit as st
import pandas as pd
import mlflow.pyfunc
import mlflow
import os

# -------------------------------------------------------
# 1️⃣  Connect to MLflow Tracking Server
# -------------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

CLASS_MODEL_NAME = "EMI_LogisticRegression_Model"   # your classification model
REG_MODEL_NAME = "EMI_Regression_Model"              # regression model (if you register later)

st.set_page_config(page_title="💼 EMI Intelligent Predictor", layout="wide")

st.title("💰 Intelligent EMI Prediction System")
st.write("🔹 Predict EMI Eligibility & Monthly EMI Amount using MLflow Registered Models")

# -------------------------------------------------------
# 2️⃣  Select Mode
# -------------------------------------------------------
mode = st.radio("Select Prediction Type", ["Classification (Eligibility)", "Regression (EMI Amount)"])

# -------------------------------------------------------
# 3️⃣  Load Selected Model from MLflow
# -------------------------------------------------------
try:
    if mode.startswith("Classification"):
        model = mlflow.pyfunc.load_model(f"models:/{CLASS_MODEL_NAME}/Production")
        st.success(f"✅ Loaded classification model: {CLASS_MODEL_NAME}")
    else:
        model = mlflow.pyfunc.load_model(f"models:/{REG_MODEL_NAME}/Production")
        st.success(f"✅ Loaded regression model: {REG_MODEL_NAME}")
except Exception as e:
    st.error(f"⚠️ Unable to load model from MLflow: {e}")
    st.stop()

# -------------------------------------------------------
# 4️⃣  Load Feature Template (from your final_preprocessed_dataset.csv)
# -------------------------------------------------------
if os.path.exists("final_preprocessed_dataset.csv"):
    df_template = pd.read_csv("final_preprocessed_dataset.csv")
    feature_names = df_template.drop(columns=["emi_eligibility_High_Risk", "emi_eligibility_Not_Eligible"], errors='ignore').columns.tolist()
else:
    st.error("❌ 'final_preprocessed_dataset.csv' not found. Please ensure it’s in your project folder.")
    st.stop()

# -------------------------------------------------------
# 5️⃣  Input Section (auto-generated from dataset columns)
# -------------------------------------------------------
st.sidebar.header("📋 Applicant Details")

# We'll take a sample row as reference for UI
sample = df_template.sample(1).iloc[0]
user_inputs = {}

for col in feature_names:
    val = sample[col]
    if isinstance(val, (float, int)):
        user_inputs[col] = st.sidebar.number_input(f"{col}", value=float(val))
    elif isinstance(val, bool):
        user_inputs[col] = st.sidebar.selectbox(f"{col}", [False, True], index=int(val))
    else:
        user_inputs[col] = st.sidebar.text_input(f"{col}", value=str(val))

input_df = pd.DataFrame([user_inputs])

# -------------------------------------------------------
# 6️⃣  Run Prediction
# -------------------------------------------------------
if st.button("🚀 Predict"):
    try:
        pred = model.predict(input_df)

        st.subheader("📊 Prediction Result")
        if mode.startswith("Classification"):
            result = "✅ Eligible" if int(pred[0]) == 1 else "❌ Not Eligible"
            st.success(f"Prediction: {result}")
        else:
            st.info(f"Predicted EMI Amount: ₹{round(pred[0], 2)}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------------------------------
# 7️⃣  Display Input Preview
# -------------------------------------------------------
with st.expander("🔍 View Processed Input Data"):
    st.dataframe(input_df)
