import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load the trained model
# ---------------------------
model = joblib.load("telco_churn_model.pkl")

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("Telco Customer Churn Prediction")
st.write("Fill the fields below to predict if a customer will churn.")

# ---------------------------
# User Inputs
# ---------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], help="1 if customer is senior citizen, else 0")
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=70.0)
Speed = st.number_input("Internet Speed (Mbps)", min_value=0.0, max_value=1000.0, value=50.0)
DataAllowance = st.number_input("Data Allowance (GB)", min_value=0.0, max_value=10000.0, value=100.0)
TenureGroup = st.selectbox("Tenure Group", ["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"])

# ---------------------------
# Make Prediction
# ---------------------------
if st.button("Predict Churn"):
    # Create dataframe
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "Speed": Speed,
        "DataAllowance": DataAllowance,
        "TenureGroup": TenureGroup
    }])

    # Predict
    churn_prob = model.predict_proba(input_df)[:, 1][0]
    churn_class = model.predict(input_df)[0]

    # Display
    st.subheader("Prediction Results")
    st.write(f"Churn Probability: **{churn_prob:.2f}**")
    st.write(f"Predicted Class: **{'Churn' if churn_class == 1 else 'No Churn'}**")
