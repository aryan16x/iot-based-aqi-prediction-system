import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Import joblib for saving the model

# Step 1: Load the data
data = pd.read_csv('iot_data.csv')

# Step 2: Preprocess the data
# Check for missing values
print(data.isnull().sum())

# Assuming there are no missing values, we can proceed
# Define features and target variable
X = data[['Temperature (°C)', 'Humidity (%)', 'CO2 Level (ppm)']]
y = data['Air Quality Index (AQI)']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Step 6: Save the trained model
model_file = 'trained_aqi_model.pkl'
joblib.dump(model, model_file)
print(f'Model saved to {model_file}')

# Step 7: Make predictions
# Example prediction
example_data = pd.DataFrame({
    'Temperature (°C)': [25],
    'Humidity (%)': [60],
    'CO2 Level (ppm)': [400]
})
predicted_aqi = model.predict(example_data)
print(f'Predicted AQI: {predicted_aqi[0]}')
