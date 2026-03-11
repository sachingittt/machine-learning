# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data (X = input, y = output)
X = np.array([[1], [2], [3], [4], [5]])   # features
y = np.array([2, 4, 6, 8, 10])            # labels

# Create model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict new value
prediction = model.predict([[6]])

print("Prediction for input 6:", prediction)
