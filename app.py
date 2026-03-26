import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("cs_students.csv")

# -------------------------------
# Data Preprocessing (same as notebook)
# -------------------------------

# Drop unwanted columns (if exist)
df = df.drop(columns=['Student_ID', 'Name'], errors='ignore')

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# -------------------------------
# Split Data
# -------------------------------


X = df.drop("Future Career", axis=1)
y = df["Future Career"]
# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = GradientBoostingClassifier()
model.fit(X_scaled, y)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🎯 CS Students Career Path Predictor")

st.write("Fill student details:")

# Create inputs dynamically (VERY IMPORTANT - matches dataset)
input_data = {}

for col in X.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    else:
        input_data[col] = st.selectbox(f"{col}", le_dict[col].classes_)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

for col in input_df.columns:
    if col in le_dict:
        unseen = set(input_df[col]) - set(le_dict[col].classes_)
        if unseen:
            print(f"{col} has unseen values: {unseen}")


# Scale input
input_scaled = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Future Career"):
    prediction = model.predict(input_scaled)

    # Decode output
    career_label = le_dict['Future Career'].inverse_transform(prediction)

    st.success(f"🎯 Predicted Future Career: {career_label[0]}")