import streamlit as st
import pandas as pd
import pickle

# =========================
# Load trained artifacts
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
feature_order = pickle.load(open("feature_order.pkl", "rb"))
ohe_cols = pickle.load(open("ohe_cols.pkl", "rb"))

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Loan Approval System", layout="centered")

st.title("Loan Approval Prediction System")
st.write("Enter applicant details to check loan approval status")

# =========================
# User Inputs
# =========================
age = st.number_input("Age", min_value=18, max_value=70)
income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10)
loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360)
savings = st.number_input("Savings Amount", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
collateral = st.number_input("Collateral Value", min_value=0)

credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
dti = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0)

gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Married", "Single"])
purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
employer = st.selectbox("Employer Category", ["Government", "Private", "Self"])

# =========================
# Prediction
# =========================
if st.button("Predict Loan Approval"):
    input_dict = {
        "Age": age,
        "Applicant_Income": income,
        "Coapplicant_Income": coapplicant_income,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Loan_Term": loan_term,
        "Savings": savings,
        "Loan_Amount": loan_amount,
        "Collateral_Value": collateral,
        "DTI_Ratio_sq": dti ** 2,
        "Credit_Score_sq": credit_score ** 2,
        "Gender": gender,
        "Education_Level": education,
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": property_area,
        "Employer_Category": employer
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical columns
    encoded = encoder.transform(input_df[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(ohe_cols)
    )

    # Drop categorical columns and combine
    input_df = input_df.drop(columns=ohe_cols)
    final_df = pd.concat([input_df, encoded_df], axis=1)

    # Enforce same feature order as training
    final_df = final_df.reindex(columns=feature_order, fill_value=0)

    # Scale and predict
    scaled = scaler.transform(final_df)
    prediction = model.predict(scaled)

    # Output
    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

