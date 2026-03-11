import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data
# Hours studied
X = np.array([[1], [2], [3], [4], [5]])

# Pass (1) or Fail (0)
y = np.array([0, 0, 0, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict result for 3.5 hours study
prediction = model.predict([[3.5]])

print("Prediction (Pass=1, Fail=0):", prediction)
