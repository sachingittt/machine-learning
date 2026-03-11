import pandas as pd

# Create dataset
data = {
    "Name": ["A", "B", "C", "D"],
    "Gender": ["Male", "Female", "Female", "Male"]
}

df = pd.DataFrame(data)

# Create dummy variables
dummy = pd.get_dummies(df["Gender"])

print(dummy)
