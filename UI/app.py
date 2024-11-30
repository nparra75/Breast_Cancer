import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os


# Define the relative path to the model folder
model_folder = "./Model/"
# Load the saved model, scaler, and selected features
best_mlp = joblib.load(os.path.join(model_folder, 'best_mlp_model.pkl'))
selected_features = joblib.load(os.path.join(model_folder, 'selected_features.pkl'))
scaler = joblib.load(os.path.join(model_folder, 'scaler.pkl'))

# App title
st.title("Breast Cancer Prediction App")

# User input for features
user_data = []
st.write("👩‍⚕️ Enter Patient Details")


for feature in selected_features:
    value = st.number_input(f"{feature}", step=0.1)
    user_data.append(value)

# Display the entered data
if st.button("Display Entered Data"):
    user_data_array = np.array(user_data).reshape(1, -1)
    st.write("You entered the following data:")
    st.dataframe(pd.DataFrame(user_data_array, columns=selected_features))

# Predict button
if st.button("Predict"):
    # Convert user input to numpy array
    user_data = np.array(user_data).reshape(1, -1)
    
    # Scale the input data
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    prediction = best_mlp.predict(user_data_scaled)
    prediction_prob = best_mlp.predict_proba(user_data_scaled)
    
    # Display results
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {prediction_prob.max() * 100:.2f}%")