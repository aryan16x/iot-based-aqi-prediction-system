import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for an IoT air quality monitoring system
num_samples = 1000

# Features
temperature = np.random.uniform(15, 35, num_samples)  # Temperature in Celsius
humidity = np.random.uniform(20, 90, num_samples)  # Humidity in percentage
co2_level = np.random.uniform(300, 1000, num_samples)  # CO2 levels in ppm

# Target variable: Air Quality Index (AQI)
# Assume AQI is based on temperature, humidity, and CO2 level in a simplified manner
aqi = 0.5 * temperature + 0.3 * humidity + 0.2 * co2_level / 10  # Simplified formula

# Generate the dataset
iot_data = pd.DataFrame({
    'Temperature (°C)': temperature,
    'Humidity (%)': humidity,
    'CO2 Level (ppm)': co2_level,
    'Air Quality Index (AQI)': aqi
})

# Display the first few rows of the dataset
print(iot_data.head())

# Save the DataFrame as a CSV file
iot_data.to_csv('iot_data.csv', index=False)  # Save without the index
