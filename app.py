# app.py
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# This must be the first Streamlit command
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ---- File paths (built from the folder that contains app.py) ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "results.json"

# Must match the training columns and their order (see columns_needed in results.json)
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

YES_NO = ["No", "Yes"]
INTERNET_ADDON = ["No", "Yes", "No internet service"]

# Dropdown choices for each text column
OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": YES_NO,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Ready-made customers for the demo buttons
SAMPLES = {
    "High-risk example": {
        "gender": "Female", "SeniorCitizen": "Yes", "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    },
    "Low-risk example": {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
    },
}


# ---- Loading functions ----
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_report():
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    return {}


def load_sample(name):
    st.session_state.update(SAMPLES[name])


def select(name, label):
    return st.selectbox(label, OPTIONS[name], key=name)


def number(name, label, **kwargs):
    return st.number_input(label, key=name, **kwargs)


# Set default form values once, so the sample buttons can overwrite them
for name, value in SAMPLES["Low-risk example"].items():
    st.session_state.setdefault(name, value)

model = load_model()
report = load_report()
threshold = report.get("cutoff", 0.5)

# ---- Sidebar: model performance ----
with st.sidebar:
    st.header("Model performance")
    if report:
        test = report["test_results"]
        st.caption(f"Model: {report['best_model'].replace('_', ' ').title()}")
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", test["roc_auc"])
        c2.metric("F1", test["f1"])
        c1.metric("Precision", test["precision"])
        c2.metric("Recall", test["recall"])
        st.caption(f"Decision cut-off: {threshold}")
        st.caption("Scores come from a hidden test set the model never trained on.")
    else:
        st.info("Run src/model_selection.py first to create reports/results.json.")

# ---- Main page ----
st.title("Customer Churn Predictor")
st.write("Estimate how likely a customer is to leave, for one customer or a whole file.")

tab_single, tab_batch = st.tabs(["Single customer", "Batch upload"])

with tab_single:
    b1, b2 = st.columns(2)
    with b1:
        st.button("Load high-risk example", on_click=load_sample, args=("High-risk example",))
    with b2:
        st.button("Load low-risk example", on_click=load_sample, args=("Low-risk example",))

    st.subheader("Account and demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        select("gender", "Gender")
        select("SeniorCitizen", "Senior citizen")
    with col2:
        select("Partner", "Partner")
        select("Dependents", "Dependents")
    with col3:
        number("tenure", "Tenure (months)", min_value=0, max_value=100)

    st.subheader("Services")
    col1, col2 = st.columns(2)
    with col1:
        select("PhoneService", "Phone service")
        select("MultipleLines", "Multiple lines")
        select("InternetService", "Internet service")
        select("OnlineSecurity", "Online security")
        select("OnlineBackup", "Online backup")
        select("DeviceProtection", "Device protection")
    with col2:
        select("TechSupport", "Tech support")
        select("StreamingTV", "Streaming TV")
        select("StreamingMovies", "Streaming movies")

    st.subheader("Billing")
    col1, col2 = st.columns(2)
    with col1:
        select("Contract", "Contract")
        select("PaperlessBilling", "Paperless billing")
        select("PaymentMethod", "Payment method")
    with col2:
        number("MonthlyCharges", "Monthly charges", min_value=0.0)
        number("TotalCharges", "Total charges", min_value=0.0)

    if st.button("Predict churn", type="primary"):
        input_df = pd.DataFrame([{col: st.session_state[col] for col in FEATURE_COLUMNS}])
        probability = model.predict_proba(input_df)[0][1]

        st.metric("Churn probability", f"{probability:.1%}")
        if probability >= threshold:
            st.error("High risk: this customer is likely to churn. Consider a retention offer.")
        else:
            st.success("Low risk: this customer is likely to stay.")

with tab_batch:
    st.write("Upload a CSV with the same columns as the training data. The `Churn` column is optional and ignored.")
    uploaded = st.file_uploader("Customer CSV", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in data.columns]

        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            X = data[FEATURE_COLUMNS].copy()
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

            probabilities = model.predict_proba(X)[:, 1]
            result = data.copy()
            result["churn_probability"] = probabilities.round(3)
            result["predicted_churn"] = (probabilities >= threshold).map({True: "Yes", False: "No"})

            high_risk = int((probabilities >= threshold).sum())
            st.metric("Customers flagged as high risk", f"{high_risk} of {len(result)}")

            st.dataframe(
                result.sort_values("churn_probability", ascending=False).head(50),
                use_container_width=True,
            )
            st.download_button(
                "Download predictions",
                result.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )