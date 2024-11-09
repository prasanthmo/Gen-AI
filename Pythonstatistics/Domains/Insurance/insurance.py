import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('insurance.csv')
print(df.info())  # Get a quick overview of the dataset
print(df.describe())  # Summary statistics for numerical columns
print(df.isnull().sum())
plt.boxplot(x=df['bmi'])
plt.show()

sex_mapping = {'female': 0, 'male': 1}  # Assuming male is higher than female
smoker_mapping = {'no': 0, 'yes': 1}  # Assuming non-smoker is lower than smoker
region_mapping = {
    'northeast': 0,
    'southeast': 1,
    'southwest': 2,
    'northwest': 3
}  # Assuming a specific order for regions

# Apply the mappings
df['sex'] = df['sex'].map(sex_mapping)
df['smoker'] = df['smoker'].map(smoker_mapping)
df['region'] = df['region'].map(region_mapping)
print(df.head())

sns.pairplot(df)  # Use 'smoker' as the hue to color the points
plt.title('Pairplot of Features')
plt.show()
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
# Define independent variables (X) and the dependent variable (y)
X = df[['sex', 'smoker', 'region', 'age', 'bmi']]
y = df['expenses']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display the coefficients and intercept with two decimal places
print(f"Coefficients: {[f'{coef:.2f}' for coef in model.coef_]}")
print(f"Intercept: {model.intercept_:.2f}")
# Scatter plot of actual vs predicted expenses
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.xlabel('Actual Expenses')
plt.ylabel('Predicted Expenses')
plt.title('Actual vs Predicted Expenses')
plt.show()
