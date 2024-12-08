import streamlit as st
import pandas as pd
import numpy as np
from predict_helper import predict
from predict_helper import prepare_training_model
# Set up the layout with 3 columns per row
st.title("Loan and Credit Analysis")

# Target ratio
target_ratio = 4.5
tolerance = 0.1  # Allow a small deviation from the target ratio
# Row 1
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", min_value=18, max_value=72, value=25)
with col2:
    loan = st.number_input("Loan Amount (USD)", min_value=0, value=10000, step=10, format="%d")
with col3:
    income = st.number_input("Income (USD)", min_value=0, value=5000, step=10, format="%d")

# Row 2
col4, col5, col6 = st.columns(3)
with col4:
    # Calculate Loan/Income Ratio only if income is not 0
    if income != 0:
        loan_to_income = round(loan / income, 2)
        st.metric("Loan/Income Ratio", value=loan_to_income)

        # Check if the ratio is higher than target_ratio
        if loan_to_income > target_ratio:
            st.warning(f"Please adjust the Loan or Income to get closer to {target_ratio}.")
    else:
        st.warning("Income cannot be zero. Please enter a valid income.")
with col5:
    avg_dpd_delinquent = st.number_input(
        "Avg DPD/Delinquent Ratio", min_value=0.0, step=0.1, max_value=10.0, value=3.3
    )
with col6:
    number_of_open_accounts = st.number_input(
        "Number of Open Accounts", min_value=1, max_value=4, step=1, value=2
    )

# Row 3
col7, col8, col9 = st.columns(3)
with col7:
    credit_utilization_ratio = st.number_input(
        "Credit Utilization (%)", min_value=0, step=5, max_value=100, value=70
    )
with col8:
    loan_tenure_months = st.number_input(
        "Loan Tenure (Months)", min_value=0, step=1, value=36, max_value = 59
    )
with col9:
    delinquent_to_loan = st.number_input(
        "Delinquent/Loan Month (%)", min_value=0, step=10, value=40, max_value=100
    )

# Row 4
col10, col11, col12 = st.columns(3)
with col10:
    loan_purpose = st.selectbox(
        "Loan Purpose", options=["Home", "Education", "Personal", "Auto"]
    )
with col11:
    residence_type = st.selectbox(
        "Residence Type", options=["Mortgage", "Owned", "Rented"]
    )
with col12:
    loan_type = st.selectbox("Loan Type", options=["Secured", "Unsecured"])


input_data = {
        "Age": age,
        "Income": income,
        "Loan Amount": loan,
        "Loan/Income Ratio": loan_to_income,
        "Avg DPD/Delinquent Ratio": avg_dpd_delinquent,
        "Number of Open Accounts": number_of_open_accounts,
        "Credit Utilization (%)": credit_utilization_ratio,
        "Loan Tenure (Months)": loan_tenure_months,
        "Delinquent/Loan Month (%)": delinquent_to_loan,
        "Loan Purpose": loan_purpose,
        "Residence Type": residence_type,
        "Loan Type": loan_type,
    }


if st.button('Predict'):
    default_probabilities, credit_score, rating = predict(input_data)
    st.success(f"Probabilities: {default_probabilities:.2%}")
    st.success(f"Credit Score: {credit_score}")
    st.success(f"Rating: {rating}")
