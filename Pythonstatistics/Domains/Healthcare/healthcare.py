import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and inspect data
df = pd.read_csv('Healthcare-Diabetes.csv')
print(df.head())  # View the first few rows
print(df.info())  # Check column types and null values
print(df.describe())  # Get summary statistics

# Step 2: Replace zero values with the mean of non-zero values in specified columns
cols_to_replace = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']

for col in cols_to_replace:
    mean_value = df[df[col] != 0][col].mean()  # Mean of non-zero values
    df[col].replace(0, mean_value, inplace=True)

# Print updated columns and description
print("\nUpdated 'Glucose', 'BloodPressure', 'Insulin', 'BMI':\n")
print(df[['Glucose', 'BloodPressure', 'Insulin', 'BMI']])
print("\nDescription of the dataset after replacement:\n")
print(df.describe())

# Step 3: Box plots for 'Glucose', 'BloodPressure', 'Insulin', and 'BMI'
columns_to_plot = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
for col, color in zip(columns_to_plot, colors):
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color=color)
    plt.title(f'Box plot of {col}')
    plt.show()

# Step 4: Correlation Matrix
plt.figure(figsize=(8, 6))
correlation_matrix = df[cols_to_replace + ['Outcome']].corr()  # Include 'Outcome' if it's the target variable
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Step 5: Logistic Regression Model
# Assuming 'Outcome' column is the target variable for diabetes prediction
X = df[['Glucose', 'BloodPressure', 'Insulin', 'BMI']]
y = df['Outcome']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train logistic regression model
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Model Accuracy: {:.2f}%".format(accuracy * 100))

