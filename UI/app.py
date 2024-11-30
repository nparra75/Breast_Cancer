import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os


# Define the relative path to the model folder
model_folder = "./Model/"
# Load the saved model, scaler, and selected features
best_mlp = joblib.load(os.path.join(model_folder, 'best_mlp_model.pkl'))
scaler = joblib.load(os.path.join(model_folder, 'scaler.pkl'))

# Set Streamlit page configuration
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")

# App title
st.title("🩺  Breast Cancer Prediction App")
st.write("""
This application predicts the likelihood of a patient having breast cancer based on the data entered..
""")

# User input for features

#st.write("👩‍⚕️ Enter Patient Details")
# User Input Features
st.sidebar.header("👩‍⚕️ Enter Patient Details")

radius = st.sidebar.slider("Radius", 0, 30, 0, step=1)
perimeter = st.sidebar.slider("Perimeter", 0, 200, 0, step=1)
area = st.sidebar.slider("Area", 0, 2500, 0, step=1)
concavity = st.sidebar.slider("Concavity", 0.0, 0.500, 0.0, step=0.01)
concave_points = st.sidebar.slider("Concave Points", 0.0, 0.200, 0.0, step=0.01)
worst_radius = st.sidebar.slider("Worst Radius", 7.93, 36.04, 0.0, step=0.01)
worst_perimeter = st.sidebar.slider("Worst Perimeter", 50.41, 251.2, 0.0, step=0.01)
worst_area = st.sidebar.slider("Worst Area", 185.2, 4254.0, 0.0, step=0.01)
worst_concavity = st.sidebar.slider("Worst Concavity", 0.0, 1.252, 0.0, step=0.01)
worst_concave_points = st.sidebar.slider("Worst Concave Points", 0.0, 0.291, 0.0, step=0.01)

data = {
    'radius': radius,
    'perimeter': perimeter,
    'area': area,
    'concavity': concavity,
    'concave points': concave_points,
    'worst radius': worst_radius,
    'worst perimeter': worst_perimeter,
    'worst area': worst_area,
    'worst concavity': worst_concavity,
    'worst concave points': worst_concave_points
}

input_df = pd.DataFrame(data, index=[0])

# Display user input
st.subheader("👤 Patient Input Features")
st.write(input_df)

# Predict button
if st.button("Predict"):
    # Convert user input to numpy array
    user_data = np.array(input_df).reshape(1, -1)
    
    # Scale the input data
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    prediction = best_mlp.predict(user_data_scaled)
    prediction_prob = best_mlp.predict_proba(user_data_scaled)
    
    # Display results
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {prediction_prob.max() * 100:.2f}%")