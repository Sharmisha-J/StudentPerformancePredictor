# ==============================
# LEVEL 3: Streamlit Web App
# ==============================

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("Student Performance Predictor")

st.write("Enter student details:")

# Inputs
hours = st.slider("Study Hours", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
score = st.slider("Previous Score", 0, 100, 60)

# Predict
if st.button("Predict"):
    data = np.array([[hours, attendance, score]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("PASS")
    else:
        st.error("FAIL")
