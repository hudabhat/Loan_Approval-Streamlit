import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("loan_approval_data.csv")

# Drop ID
data = data.drop("Applicant_ID", axis=1)

# Separate columns
numerical_cols = data.select_dtypes(include=["number"]).columns
categorical_cols = data.select_dtypes(include=["object"]).columns

# Imputation
num_imp = SimpleImputer(strategy="mean")
cat_imp = SimpleImputer(strategy="most_frequent")

data[numerical_cols] = num_imp.fit_transform(data[numerical_cols])
data[categorical_cols] = cat_imp.fit_transform(data[categorical_cols])

# Encoding
le = LabelEncoder()
data["Loan_Approved"] = le.fit_transform(data["Loan_Approved"])

ohe_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category"
]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(data[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))

data = data.drop(columns=ohe_cols)
df = pd.concat([data, encoded_df], axis=1)

# Feature engineering
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

X = df.drop(columns=["Loan_Approved", "DTI_Ratio", "Credit_Score"])
y = df["Loan_Approved"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save artifacts
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(ohe, open("encoder.pkl", "wb"))
pickle.dump(ohe_cols, open("ohe_cols.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("feature_order.pkl", "wb"))

print("Model training complete. Files saved.")
