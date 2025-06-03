import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv("test_data.csv")  # Make sure this file is in your working directory

# Drop non-numeric or irrelevant columns
df_model = df.drop(columns=["Date"])
X = df_model.drop(columns=["cpbl"])
y = df_model["cpbl"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Create a DataFrame to show coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

# Print the coefficients
print("Coefficients affecting CPBL:\n")
print(coefficients)
