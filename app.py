import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("heart_model.pkl")



st.title("Heart Disease Prediction")

# Inputs (MATCH EXACT ORDER)




fastingbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
maxhr = st.number_input("Max Heart Rate")


sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])


exercise_angina = st.selectbox("Exercise Angina (0=No,1=Yes)", [0,1])
st_slope = st.selectbox("ST Slope (0=Up,1=Flat,2=Down)", [0,1,2])

# Prediction
if st.button("Predict"):
    data = np.array([[age,  fastingbs, maxhr, 
                      sex,  exercise_angina, st_slope]])

    prediction = model.predict(data)
    prob = model.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error(f" High Risk ({prob:.2f})")
    else:
        st.success(f" Low Risk ({prob:.2f})")
