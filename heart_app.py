import streamlit as st
import pickle
import numpy as np

# Load the model
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💓 Heart Disease Prediction App")

st.write("Enter your health information below:")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
cholesterol = st.number_input("Cholesterol Level (mg/dl)", value=200)
fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"])
fasting_bs = 1 if fasting_bs == "Yes" else 0
max_hr = st.number_input("Maximum Heart Rate Achieved", value=150)
oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0)

sex = st.radio("Sex", ["Male", "Female"])
sex_m = 1 if sex == "Male" else 0

cp_type = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
cp_ata = cp_nap = cp_ta = 0
if cp_type == "ATA":
    cp_ata = 1
elif cp_type == "NAP":
    cp_nap = 1
elif cp_type == "TA":
    cp_ta = 1

resting_ecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
ecg_normal = ecg_st = 0
if resting_ecg == "Normal":
    ecg_normal = 1
elif resting_ecg == "ST":
    ecg_st = 1

exercise_angina = st.radio("Exercise-Induced Angina", ["Yes", "No"])
angina = 1 if exercise_angina == "Yes" else 0

st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])
slope_flat = slope_up = 0
if st_slope == "Flat":
    slope_flat = 1
elif st_slope == "Up":
    slope_up = 1

# Create input array
input_data = np.array([[
    age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak,
    sex_m, cp_ata, cp_nap, cp_ta,
    ecg_normal, ecg_st, angina,
    slope_flat, slope_up
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"💔 High Risk of Heart Disease\n🔢 Probability: {round(probability * 100, 2)}%")
    else:
        st.success(f"💚 No Heart Disease Detected\n🔢 Probability: {round(probability * 100, 2)}%")
