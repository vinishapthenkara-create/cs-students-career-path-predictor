import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("career_model.pkl")

# -----------------------------
# App Title
# -----------------------------
st.title("CS Students Career Path Predictor")

st.write("Predict the future career path of CS students based on their academic profile.")

# -----------------------------
# User Inputs
# -----------------------------
gpa = st.slider("GPA", 0.0, 10.0, 7.0)

age = st.slider("Age", 18, 30, 22)

gender = st.selectbox("Gender", ["Male", "Female"])

major = st.selectbox("Major", ["Computer Science", "Data Science", "AI", "Cyber Security"])

domain = st.selectbox("Interested Domain",
                      ["Web Development",
                       "Machine Learning",
                       "Data Science",
                       "Cyber Security",
                       "Cloud Computing"])

# -----------------------------
# Encoding Inputs
# -----------------------------
gender_val = 1 if gender == "Male" else 0

major_dict = {
    "Computer Science":0,
    "Data Science":1,
    "AI":2,
    "Cyber Security":3
}

domain_dict = {
    "Web Development":0,
    "Machine Learning":1,
    "Data Science":2,
    "Cyber Security":3,
    "Cloud Computing":4
}

major_val = major_dict[major]
domain_val = domain_dict[domain]

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Career"):

    input_data = pd.DataFrame([[age,gpa,gender_val,major_val,domain_val]],
                              columns=["Age","GPA","Gender","Major","Interested Domain"])

    prediction = model.predict(input_data)

    st.success(f"Predicted Career Path: {prediction[0]}")