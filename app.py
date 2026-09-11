import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write(
    "This app predicts whether a customer is likely to **churn** based on their account "
    "and usage details. Fill in the customer information below and click **Predict Churn**."
)


# ---------------------------------------------------------------
# Load trained pipeline (preprocessing + model), trained in model_training.ipynb
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Could not find `churn_pipeline.pkl`. Run the notebook first to train and save the model.")
    st.stop()


# ---------------------------------------------------------------
# Collect user inputs for every feature the model was trained on
# ---------------------------------------------------------------
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    region = st.selectbox("Region", ["East", "West", "North", "South"])
    tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=840.0, step=10.0)
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Bank transfer", "Credit card", "Mailed check"],
    )

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    num_support_calls = st.number_input("Number of Support Calls", min_value=0, max_value=50, value=1, step=1)
    late_payments_last_year = st.number_input(
        "Late Payments (last year)", min_value=0, max_value=24, value=0, step=1
    )
    avg_monthly_usage_gb = st.slider("Avg. Monthly Usage (GB)", 0.0, 1000.0, 200.0, step=5.0)

st.divider()

# ---------------------------------------------------------------
# Predict
# ---------------------------------------------------------------
if st.button("Predict Churn", type="primary"):
    try:
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "region": region,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "num_support_calls": num_support_calls,
            "late_payments_last_year": late_payments_last_year,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
        }])

        prediction = model.predict(input_df)[0]
        result_label = "Yes" if prediction == 1 else "No"

        if result_label == "Yes":
            st.error(f"### Prediction: **{result_label}** — this customer is likely to churn.")
        else:
            st.success(f"### Prediction: **{result_label}** — this customer is likely to stay.")

        # Bonus: prediction probability / confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            st.write("**Prediction confidence:**")
            c1, c2 = st.columns(2)
            c1.metric("No (stay)", f"{proba[0]:.1%}")
            c2.metric("Yes (churn)", f"{proba[1]:.1%}")

    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

st.caption("Model trained in `model_training.ipynb` using a Scikit-Learn preprocessing + classification pipeline.")
