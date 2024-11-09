import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the CSV file into a DataFrame
df = pd.read_csv('taxidata.csv')

# Display the first few rows of the DataFrame
print("Initial Data:")
print(df.head())

# Get a concise summary of the DataFrame
print("\nData Summary:")
print(df.info())
pd.set_option('display.max_columns', None)

# Generate descriptive statistics for all columns except specified ones
excluded_columns = ['payment_type', 'tip_amount', 'imp_surcharge', 'pickup_location_id', 'dropoff_location_id']
eda_columns = df.drop(columns=excluded_columns, errors='ignore')  # Drop excluded columns

print("\nDescriptive Statistics for Selected Columns:")
print(eda_columns.describe(include='all'))  # include='all' to describe all columns, including non-numeric

# EDA: Box plots for outlier detection
num_columns = len(eda_columns.columns)
num_plots = 2  # Number of box plots per window

# Loop through the columns and create box plots in separate windows
for i in range(0, num_columns, num_plots):
    plt.figure(figsize=(12, 5))  # Create a new figure for each set of plots
    for j in range(num_plots):
        if i + j < num_columns:  # Check to avoid index out of range
            plt.subplot(1, num_plots, j + 1)  # Create a grid of subplots
            sns.boxplot(x=eda_columns.iloc[:, i + j])  # Use iloc to select the column
            plt.title(f'Box Plot of {eda_columns.columns[i + j]}')
            plt.xlabel(eda_columns.columns[i + j])
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plots

# Create a heatmap for the correlation of the remaining columns
plt.figure(figsize=(12, 8))
correlation_matrix = eda_columns.corr()  # Calculate the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Heatmap of Correlation Matrix')
plt.show()  # Display the heatmap

# Prepare data for linear regression
# Select features (excluding the target variable 'total_amount')
X = eda_columns.drop(columns=['total_amount'], errors='ignore')  # Features
y = eda_columns['total_amount']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Line for perfect prediction
plt.title('Actual vs Predicted Total Amount')
plt.xlabel('Actual Total Amount')
plt.ylabel('Predicted Total Amount')
plt.xlim([0, y.max()])  # Set limits for x-axis
plt.ylim([0, y.max()])  # Set limits for y-axis
plt.grid()
plt.show()  # Display the plot
