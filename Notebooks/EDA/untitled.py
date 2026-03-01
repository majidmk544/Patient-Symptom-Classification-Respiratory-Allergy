import streamlit as st
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "disease_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("🩺 Patient Disease Prediction System")

st.write("Enter symptom values below:")

# Create input dynamically
num_features = model.n_features_in_
user_input = []

for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    user_input.append(val)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    disease = encoder.inverse_transform(prediction)

    st.success(f"Predicted Disease: {disease[0]}")