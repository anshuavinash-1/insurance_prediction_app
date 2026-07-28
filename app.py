import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Insurance Charges Prediction")

age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Children", 0, 10)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

sex = 1 if sex == "female" else 0
smoker = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

input_data = np.array([[age, sex, bmi, children, smoker,
                        region_northwest, region_southeast, region_southwest]])

if st.button("Predict"):
    pred = model.predict(input_data)
    st.success(f"Predicted Charges: {pred[0]:.2f}")