import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------
# Load Data & Model
# -------------------------
df = pd.read_csv("insurance.csv")
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Title
# -------------------------
st.title("💰 Insurance Charges Prediction Dashboard")

# -------------------------
# 📊 DATASET INSIGHTS
# -------------------------
st.header("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Charges", f"₹ {df['charges'].mean():,.0f}")
col2.metric("Max Charges", f"₹ {df['charges'].max():,.0f}")
col3.metric("Min Charges", f"₹ {df['charges'].min():,.0f}")

# -------------------------
# 📈 CHARTS
# -------------------------
st.subheader("Charges Distribution")
st.bar_chart(df['charges'])

st.subheader("Smoker vs Non-Smoker Charges")
smoker_avg = df.groupby("smoker")["charges"].mean()
st.bar_chart(smoker_avg)

# -------------------------
# 🧾 USER INPUT
# -------------------------
st.header("🧾 Enter User Details")

age = st.number_input("Age", 1, 100)
#sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Children", 0, 10)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# -------------------------
# ENCODING (MATCH TRAINING)
# -------------------------
#sex = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

#input_data = np.array([[age, sex, bmi, children, smoker_val,region_northwest, region_southeast, region_southwest]])


input_data = np.array([[age, bmi, children, smoker_val,
                        region_northwest, region_southeast, region_southwest]])
# -------------------------
# 🔮 PREDICTION + INSIGHTS
# -------------------------
if st.button("Predict"):

    pred = model.predict(input_data)[0]

    st.success(f"💰 Predicted Charges: ₹ {pred:,.2f}")

    # -------------------------
    # ⚠️ RISK LEVEL
    # -------------------------
    if pred < 10000:
        risk = "🟢 Low"
    elif pred < 30000:
        risk = "🟡 Medium"
    else:
        risk = "🔴 High"

    st.subheader(f"Risk Level: {risk}")

    # -------------------------
    # 🔍 WHY THIS PREDICTION
    # -------------------------
    st.subheader("🔍 Key Drivers")

    if smoker_val == 1:
        st.write("🚬 Smoking significantly increases charges")

    if bmi > 30:
        st.write("⚖️ High BMI increases health risk")

    if age > 50:
        st.write("👴 Age contributes to higher insurance cost")

    if children > 2:
        st.write("👨‍👩‍👧 More dependents slightly increase cost")

    # -------------------------
    # 📊 COMPARISON WITH AVERAGE
    # -------------------------
    st.subheader("📊 Comparison with Average User")

    avg_charge = df['charges'].mean()
    diff = pred - avg_charge

    if diff > 0:
        st.write(f"⬆️ ₹ {diff:,.0f} higher than average")
    else:
        st.write(f"⬇️ ₹ {abs(diff):,.0f} lower than average")

    # -------------------------
    # 📈 USER PERCENTILE
    # -------------------------
    percentile = (df['charges'] < pred).mean() * 100
    st.write(f"📍 You are in top {percentile:.1f}% of charges")

    # -------------------------
    # 📉 FEATURE IMPORTANCE
    # -------------------------
    st.subheader("📉 Feature Importance")

    feature_names = [
        "age", "bmi", "children", "smoker",
        "region_northwest", "region_southeast", "region_southwest"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df)
    st.bar_chart(importance_df.set_index("Feature"))