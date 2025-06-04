import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the CSV file
df = pd.read_csv("test_data.csv")  # Ensure this file is in your working directory

# Prepare features (X) and target (y)
df_model = df.drop(columns=["Date"])
X = df_model.drop(columns=["cpbl"])
y = df_model["cpbl"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate polynomial features of degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
feature_names = poly.get_feature_names_out(X.columns)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Generate the polynomial equation
equation = f"cpbl = {model.intercept_:.3f}"
for name, coef in zip(feature_names, model.coef_):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} {abs(coef):.3f}*{name}"

# Print the equation (optionally, write to file)
print("Polynomial Regression Equation for cpbl:\n")
print(equation)
