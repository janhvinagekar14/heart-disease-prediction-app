import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("heart_model.pkl")

prediction = model.predict(data)

st.title("Heart Disease Prediction")

# Inputs (MATCH EXACT ORDER)

age = st.number_input("Age")
restingbp = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fastingbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
maxhr = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Oldpeak")

sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])

cp = st.selectbox("Chest Pain Type (encoded)", [0,1,2,3])
restecg = st.selectbox("Resting ECG (encoded)", [0,1,2])
exercise_angina = st.selectbox("Exercise Angina (0=No,1=Yes)", [0,1])
st_slope = st.selectbox("ST Slope (0=Up,1=Flat,2=Down)", [0,1,2])

# Prediction
if st.button("Predict"):
    data = np.array([[age, restingbp, chol, fastingbs, maxhr, oldpeak,
                      sex, cp, restecg, exercise_angina, st_slope]])

    prediction = model.predict(data)
    prob = model.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error(f" High Risk ({prob:.2f})")
    else:
        st.success(f" Low Risk ({prob:.2f})")