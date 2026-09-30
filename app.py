import streamlit as st
import pandas as pd
import joblib

# Load trained XGBoost model
model = joblib.load("churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="centered"
)

# Main heading
st.title("🤖 AI-Powered Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn "
    "using an XGBoost machine learning model."
)

st.divider()

# Customer information
st.subheader("👤 Customer Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    value=10
)

usage_frequency = st.number_input(
    "Usage Frequency",
    min_value=0,
    value=12
)

support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    value=7
)

payment_delay = st.number_input(
    "Payment Delay (days)",
    min_value=0,
    value=22
)

total_spend = st.number_input(
    "Total Spend",
    min_value=0.0,
    value=450.0
)

last_interaction = st.number_input(
    "Last Interaction (days)",
    min_value=0,
    value=15
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Premium", "Standard"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

st.divider()

# Prediction button
# Prediction button
if st.button("🔮 Predict Churn"):

    # Convert categorical values into numbers
    gender_male = 1 if gender == "Male" else 0

    subscription_premium = (
        1 if subscription_type == "Premium" else 0
    )

    subscription_standard = (
        1 if subscription_type == "Standard" else 0
    )

    contract_monthly = (
        1 if contract_length == "Monthly" else 0
    )

    contract_quarterly = (
        1 if contract_length == "Quarterly" else 0
    )

    # Create customer DataFrame
    customer = pd.DataFrame([{
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage_frequency,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction,
        "Gender_Male": gender_male,
        "Subscription Type_Premium": subscription_premium,
        "Subscription Type_Standard": subscription_standard,
        "Contract Length_Monthly": contract_monthly,
        "Contract Length_Quarterly": contract_quarterly
    }])

    # Make prediction
    prediction = model.predict(customer)[0]

    # Get churn probability
    probability = model.predict_proba(customer)[0][1]

    # Prediction result
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to Churn")
    else:
        st.success("✅ Prediction: Customer is likely to Stay")

    # Display probability
    st.write(
        "Churn Probability:",
        round(probability * 100, 2),
        "%"
    )

    # Probability progress bar
    st.progress(float(probability))

    # Risk level
    if probability < 0.30:
        st.success("🟢 Risk Level: Low")
    elif probability < 0.70:
        st.warning("🟡 Risk Level: Medium")
    else:
        st.error("🔴 Risk Level: High")

    st.write("Customer information collected successfully!")
    st.divider()

st.subheader("📌 About This Project")

st.write(
    "This AI-powered customer churn prediction system uses "
    "machine learning to predict whether a customer is likely "
    "to leave a service."
)

st.write(
    "The system uses an XGBoost classification model trained "
    "on customer behavior and subscription data. It provides "
    "a churn prediction, churn probability, and risk level."
)

st.caption(
    "Developed as a BTech Computer Science & Engineering project."
)