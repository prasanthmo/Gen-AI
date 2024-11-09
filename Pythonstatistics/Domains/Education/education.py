#analysis of student performance with GPA:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


# Load your dataset into a pandas DataFrame (replace 'your_dataset.csv' with your file path)
df = pd.read_csv('Student_performance_data _.csv')

#code to show all columns clearly 
pd.set_option('display.max_columns', None)


# Columns of interest
columns = ['Age', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular']

# 1. Handling missing values: Identifying missing values in each column
missing_values = df[columns].isnull().sum()
print("Missing values in each column:\n", missing_values)

# 2. Identifying rows where all values are zero in the selected columns
rows_all_zero = df.loc[(df[columns] == 0).all(axis=1)]
print("\nRows where all values are zero:\n", rows_all_zero)

# 3. Providing descriptive statistics for the selected columns
descriptive_stats = df[columns].describe()
print("\nDescriptive statistics for the selected columns:\n", descriptive_stats)

# 4. Creating a histogram for GPA
plt.figure(figsize=(8, 6))
sns.histplot(df['GPA'], bins=20, kde=True, color='skyblue')  # Assuming 'GPA' is the column name
plt.title('Histogram of GPA')
plt.xlabel('GPA')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Create subplots for each column
columns = ['StudyTimeWeekly', 'Absences', 'ParentalSupport']

# Generate separate boxplots for each column
for i, col in enumerate(columns, 1):
    plt.subplot(2, 2, i)  # 2 rows, 2 columns
    sns.boxplot(data=df, y=col, color='lightblue')
    plt.title(f'Boxplot for {col}')
    plt.ylabel(col)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# 5. Creating scatter plots for GPA against other factors
plt.figure(figsize=(15, 10))

# Scatter plot for GPA vs StudyTimeWeekly
plt.subplot(2, 2, 1)
sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df, color='blue', alpha=0.6)
plt.title('GPA vs Study Time Weekly')
plt.xlabel('Study Time Weekly (hours)')
plt.ylabel('GPA')

# Scatter plot for GPA vs Absences
plt.subplot(2, 2, 2)
sns.scatterplot(x='Absences', y='GPA', data=df, color='orange', alpha=0.6)
plt.title('GPA vs Absences')
plt.xlabel('Number of Absences')
plt.ylabel('GPA')

# Scatter plot for GPA vs ParentalSupport
plt.subplot(2, 2, 3)
sns.scatterplot(x='ParentalSupport', y='GPA', data=df, color='green', alpha=0.6)
plt.title('GPA vs Parental Support')
plt.xlabel('Parental Support Level')
plt.ylabel('GPA')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the scatter plots
plt.show()

# Columns of interest for correlation analysis
columns_of_interest = ['GPA', 'StudyTimeWeekly', 'Absences', 'ParentalSupport', 'Tutoring', 'Extracurricular']

#  Calculate the correlation matrix
correlation_matrix = df[columns_of_interest].corr()

#  Print the correlation matrix
print("\nCorrelation Matrix:\n", correlation_matrix)

# 6. Visualize the correlation matrix with a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Selecting relevant columns
X = df[['StudyTimeWeekly', 'Absences', 'ParentalSupport','Tutoring']]  # Independent variables
y = df['GPA']  # Dependent variable

# 1. Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluate the model
coefficients = model.coef_
intercept = model.intercept_

# Print the results
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Make predictions and plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predited GPA')
plt.xlabel('Actual GPA')
plt.ylabel('Predited GPA')
plt.grid()
plt.show()