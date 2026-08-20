"""
streamlit_app.py — Interactive churn prediction UI
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from src.config import ONNX_PATH, NUM_COLS, CAT_COLS

st.set_page_config(page_title="Customer Churn Prediction", page_icon=":", layout="wide")
st.title("Customer Churn Prediction")

session = ort.InferenceSession(str(ONNX_PATH))


def predict(data: dict) -> tuple[int, float]:
    df = pd.DataFrame([data])
    input_dict = {}
    for col in NUM_COLS:
        input_dict[col] = df[col].values.astype(np.float32).reshape(-1, 1)
    for col in CAT_COLS:
        input_dict[col] = df[col].values.astype(object).reshape(-1, 1)
    classes, prob_dicts = session.run(None, input_dict)
    proba = np.array([[v for v in d.values()] for d in prob_dicts])
    return int(classes[0]), float(proba[0][1])


col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    st.subheader("Services")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.subheader("Billing")
col3, col4, col5 = st.columns(3)

with col3:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with col4:
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
with col5:
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])

col6, col7 = st.columns(2)
with col6:
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
with col7:
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)

if st.button("Predict Churn", type="primary", use_container_width=True):
    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    churn, probability = predict(data)

    st.divider()

    if churn == 1:
        st.error(f"**Will Churn** (probability: {probability:.1%})")
    else:
        st.success(f"**Will Not Churn** (probability: {probability:.1%})")
