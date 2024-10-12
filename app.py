import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load or create a dummy model
model_file = 'trained_aqi_model.pkl'

# Check if the model file exists
try:
    model = joblib.load(model_file)
    # st.write("Model loaded successfully.")
except FileNotFoundError:
    # Dummy training data for the model (replace this with your actual training data)
    X_dummy = pd.DataFrame({
        'Temperature (°C)': np.random.uniform(15, 35, 100),
        'Humidity (%)': np.random.uniform(30, 90, 100),
        'CO2 Level (ppm)': np.random.uniform(300, 800, 100)
    })
    y_dummy = np.random.uniform(0, 300, 100)  # Dummy AQI values

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_dummy, y_dummy)

    # Save the trained model
    joblib.dump(model, model_file)
    st.write("Model trained and saved successfully.")

# Streamlit app
st.title("AQI Prediction App")

# Initialize session state for random data
if 'random_data' not in st.session_state:
    st.session_state.random_data = None

# Button to generate random data
if st.button("Generate Random Data"):
    # Generate random values
    st.session_state.random_data = {
        'temperature': np.random.uniform(15, 35),
        'humidity': np.random.uniform(30, 90),
        'co2_level': np.random.uniform(300, 800)
    }
    st.success(f"Random Data Generated: Temperature={st.session_state.random_data['temperature']:.2f}°C, "
               f"Humidity={st.session_state.random_data['humidity']:.2f}%, "
               f"CO2 Level={st.session_state.random_data['co2_level']:.2f} ppm")

# Input fields for user input, filled with random data if available
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, 
                               value=st.session_state.random_data['temperature'] if st.session_state.random_data else 25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, 
                            value=st.session_state.random_data['humidity'] if st.session_state.random_data else 60.0)
co2_level = st.number_input("CO2 Level (ppm)", min_value=0.0, max_value=2000.0, 
                             value=st.session_state.random_data['co2_level'] if st.session_state.random_data else 400.0)

# Button to predict AQI
if st.button("Predict AQI"):
    input_data = pd.DataFrame({
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
        'CO2 Level (ppm)': [co2_level]
    })
    predicted_aqi = model.predict(input_data)
    st.write(f"Predicted AQI: {predicted_aqi[0]:.2f}")
