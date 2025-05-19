import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('vehicle_maintenance_model.pkl')

st.title("Vehicle Maintenance Prediction")
st.write("Predict the number of hours until a vehicle needs maintenance.")

# Input fields
building = st.number_input("Building (encoded integer)", min_value=0)
department = st.number_input("Department (encoded integer)", min_value=0)
vehicle = st.number_input("Vehicle (numeric ID)")
odometer = st.number_input("Odometer reading")
priority = st.selectbox("Priority", options=[1, 2, 3])

# Predict button
if st.button("Predict Maintenance Time"):
    features = np.array([[building, department, vehicle, odometer, priority]])
    prediction = model.predict(features)
    st.success(f"⏱ Estimated time to maintenance: **{prediction[0]:.2f} hours**")
