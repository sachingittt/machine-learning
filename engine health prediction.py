import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Example engine dataset
data = {
    "temperature": [300, 320, 340, 360, 380, 400],
    "vibration": [0.02, 0.03, 0.04, 0.06, 0.08, 0.10],
    "pressure": [30, 32, 35, 40, 45, 50],
    "health_score": [95, 90, 85, 75, 60, 50]   # target (engine health)
}

df = pd.DataFrame(data)

# Features and target
X = df[["temperature", "vibration", "pressure"]]
y = df["health_score"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Predicted Health:", predictions)
print("Actual Health:", y_test.values)

# Error
error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)
