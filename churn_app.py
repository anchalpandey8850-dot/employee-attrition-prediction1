
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Cancellation Prediction",
    page_icon="📊"
)

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_pipeline.joblib")

model = load_model()

st.title("Customer Cancellation Prediction")
st.write("Enter customer details to predict the likelihood of cancellation.")

with st.form("customer_form"):
    Territory = st.text_input("Territory", "North")
    DeviceType = st.text_input("Device Type", "iPhone")
    Promotion = st.selectbox("Promotion", ["yes", "no"])
    HasRenewed = st.selectbox("Has Renewed", ["yes", "no"])
    PlanType = st.text_input("Plan Type", "Red Plan")

    BaseFee = st.number_input("Base Fee", min_value=0.0, value=50.0)
    AdditionalFeatureSpend = st.number_input(
        "Additional Feature Spend", min_value=0.0, value=10.0
    )
    NumberOfPenalties = st.number_input(
        "Number of Penalties", min_value=0.0, value=0.0
    )
    CurrentPeriodUsage = st.number_input(
        "Current Period Usage", min_value=0.0, value=100.0
    )
    PriorPeriodUsage = st.number_input(
        "Prior Period Usage", min_value=0.0, value=100.0
    )
    ServiceRating = st.number_input(
        "Service Rating", min_value=0.0, value=4.0
    )
    ServiceTickets = st.number_input(
        "Service Tickets", min_value=0.0, value=0.0
    )
    StartMonth = st.text_input("Start Month", "Jan")
    StartWeek = st.text_input("Start Week", "Week 1")
    CustomerTenure = st.number_input(
        "Customer Tenure", min_value=0.0, value=12.0
    )

    submitted = st.form_submit_button("Predict Cancellation")

if submitted:
    customer = pd.DataFrame([{
        "Territory": Territory,
        "DeviceType": DeviceType,
        "Promotion": Promotion,
        "HasRenewed": HasRenewed,
        "PlanType": PlanType,
        "BaseFee": BaseFee,
        "AdditionalFeatureSpend": AdditionalFeatureSpend,
        "NumberOfPenalties": NumberOfPenalties,
        "CurrentPeriodUsage": CurrentPeriodUsage,
        "PriorPeriodUsage": PriorPeriodUsage,
        "ServiceRating": ServiceRating,
        "ServiceTickets": ServiceTickets,
        "StartMonth": StartMonth,
        "StartWeek": StartWeek,
        "CustomerTenure": CustomerTenure
    }])

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])

    st.subheader("Prediction Result")
    st.write("Cancellation Prediction:", "Yes" if prediction == 1 else "No")
    st.metric("Cancellation Probability", f"{probability:.1%}")
