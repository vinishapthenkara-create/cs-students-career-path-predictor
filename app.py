import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("career_model.pkl")

st.set_page_config(page_title="Career Predictor", layout="centered")

st.title("🎯 CS Students Career Path Predictor")
st.write("Fill student details to predict career path")

# -----------------------------
# USER INPUTS (ALL 11 FEATURES)
# -----------------------------
age = st.number_input("Age", 18, 30)
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)

internships = st.number_input("Internships", 0, 10)
projects = st.number_input("Projects", 0, 20)
certifications = st.number_input("Certifications", 0, 10)

coding_skills = st.selectbox("Coding Skills", ["Low", "Medium", "High"])
communication = st.selectbox("Communication Skills", ["Low", "Medium", "High"])
aptitude = st.selectbox("Aptitude", ["Low", "Medium", "High"])
problem_solving = st.selectbox("Problem Solving", ["Low", "Medium", "High"])
teamwork = st.selectbox("Teamwork", ["Low", "Medium", "High"])
leadership = st.selectbox("Leadership", ["Low", "Medium", "High"])

# -----------------------------
# ENCODING (same as training)
# -----------------------------
map_val = {"Low": 0, "Medium": 1, "High": 2}

coding_skills = map_val[coding_skills]
communication = map_val[communication]
aptitude = map_val[aptitude]
problem_solving = map_val[problem_solving]
teamwork = map_val[teamwork]
leadership = map_val[leadership]

# -----------------------------
# CREATE INPUT DATAFRAME
# IMPORTANT: SAME ORDER AS TRAINING
# -----------------------------
input_data = pd.DataFrame([[
    age, cgpa, internships, projects,
    certifications, coding_skills,
    communication, aptitude,
    problem_solving, teamwork, leadership
]], columns=[
    'age', 'cgpa', 'internships', 'projects',
    'certifications', 'coding_skills',
    'communication', 'aptitude',
    'problem_solving', 'teamwork', 'leadership'
])

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Career Path"):
    try:
        prediction = model.predict(input_data)
        st.success(f"🎯 Predicted Career Path: {prediction[0]}")

    except Exception as e:
        st.error("❌ Error occurred!")
        st.write("Error:", e)
        st.write("Input shape:", input_data.shape)
        st.write("Model expects:", model.n_features_in_)