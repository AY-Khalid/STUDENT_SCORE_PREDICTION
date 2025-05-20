import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('lasso_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('feature_columns.pkl')

st.title("Student Exam Score Predictor (Lasso Regression)")

# Manual input fields
def user_input():
    st.header("Enter Student Information")

    inputs = {}

    inputs['Hours_Studied'] = st.number_input("Hours Studied", 0, 20)
    inputs['Attendance'] = st.slider("Attendance (%)", 0, 100)
    inputs['Sleep_Hours'] = st.number_input("Sleep Hours", 0, 12)
    inputs['Previous_Scores'] = st.number_input("Previous Scores", 0, 100)
    inputs['Tutoring_Sessions'] = st.number_input("Tutoring Sessions", 0, 20)
    inputs['Physical_Activity'] = st.slider("Physical Activity (hours/week)", 0, 20)

    # Sample categorical selections 
    inputs['Parental_Involvement_High'] = int(st.selectbox("Parental Involvement", ['Low', 'Medium', 'High']) == 'High')
    inputs['Access_to_Resources_High'] = int(st.selectbox("Access to Resources", ['Low', 'Medium', 'High']) == 'Yes')
    inputs['Internet_Access_Yes'] = int(st.selectbox("Internet Access", ['No', 'Yes']) == 'Yes')
    inputs['Distance_from_Home_Near'] = int(st.selectbox("Distance From Home", ['Far', 'Moderate', 'Near']) == 'Near')
    inputs['Extracurricular_Activities_Yes'] = int(st.selectbox("Extracurricular Activies", ['No', 'Yes']) == 'Yes')
    inputs['Gender_Male'] = int(st.selectbox("Gender", ['Female', 'Male']) == 'Male')

    # Fill 0 for all features first
    data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Set values
    for key, value in inputs.items():
        if key in data.columns:
            data.at[0, key] = value

    return data

input_df = user_input()

# Prediction
if st.button("Predict Exam Score"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Exam Score: {prediction:.2f}")