# Loan Approval Prediction System

A supervised machine learning web application that predicts whether a loan application will be approved or rejected based on applicant financial and demographic details.

## Features
- End-to-end ML pipeline (preprocessing → feature engineering → model training)
- Logistic Regression model for interpretable predictions
- Interactive Streamlit web interface
- Real-time loan approval prediction
- Feature consistency enforced between training and inference

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Logistic Regression

## Input Parameters
- Applicant & Coapplicant Income
- Credit Score
- Debt-to-Income Ratio
- Loan Amount & Term
- Savings & Existing Loans
- Demographic & Employment Details

## Live Demo
👉 _Add Streamlit link here_

## 📁 Project Structure
Loan_Approval_System/
│── app.py
│── train_model.py
│── loan_approval_data.csv
│── model.pkl
│── scaler.pkl
│── encoder.pkl
│── ohe_cols.pkl
│── feature_order.pkl
│── requirements.txt


## How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
