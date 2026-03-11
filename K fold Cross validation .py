from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

# Create KFold object
kfold = KFold(n_splits=5)

model = LinearRegression()

for train_index, test_index in kfold.split(X):
    
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Test model
    score = model.score(X_test, y_test)
    
    print("Accuracy:", score)
