import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('salary_model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💼 Employee Salary Prediction")
st.write("Enter your experience and skills to predict the salary.")

# Input fields
experience = st.slider("Years of Experience", 0, 20, 2)
test_score = st.slider("Test Score (out of 100)", 0, 100, 60)
interview_score = st.slider("Interview Score (out of 10)", 0, 10, 7)

# Predict button
if st.button("Predict Salary"):
    features = np.array([[experience, test_score, interview_score]])
    prediction = model.predict(features)[0]
    st.success(f"💰 Predicted Salary: ₹ {round(prediction, 2)}")
