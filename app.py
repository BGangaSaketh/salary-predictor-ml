import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# Load model
model = joblib.load("model/salary_model.pkl")

# Title
st.title("💼 Salary Prediction App")
st.subheader("Predict salary based on years of experience")

st.write("---")

# Input
years = st.slider(
    "Select Years of Experience",
    min_value=0.0,
    max_value=20.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(np.array([[years]]))

    st.success(
        f"Estimated Salary: ₹ {prediction[0]:,.2f}"
    )

    if years < 2:
        st.info("Entry-level salary range")
    elif years < 5:
        st.info("Mid-level professional salary range")
    else:
        st.info("Senior-level salary range")

st.write("---")
st.caption("Built using Python, Scikit-learn, and Streamlit")