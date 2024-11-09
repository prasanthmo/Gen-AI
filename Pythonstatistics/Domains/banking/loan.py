import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Loan_default.csv')
print(df.head())  # View the first few rows
print(df.info())  # Check column types and null values

print(df.describe())
pd.set_option('display.max_columns',None)

print(df.describe())
 #Calculate the correlation matrix including only numeric variables
correlation_matrix = df.select_dtypes(include='number').corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()