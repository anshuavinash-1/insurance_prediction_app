import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Insurance Charges Prediction App")

st.write("Enter user details to predict insurance charges")

# Inputs
age = st.number_input("Age", min_value=1, max_value=100)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
children = st.number_input("Number of Children", min_value=0, max_value=10)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert inputs
sex = 1 if sex == "female" else 0
smoker = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Final input array
input_data = np.array([[age, sex, bmi, children, smoker,
                        region_northwest, region_southeast, region_southwest]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Charges"):
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Insurance Charges: {prediction[0]:.2f}")