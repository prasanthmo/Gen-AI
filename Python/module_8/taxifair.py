import pandas as pd

# Load the taxi data (update the path if needed)
file_path = 'C:/Users/mohan/GenAi/Python/module_8/taxi_rides_2000.csv'  # Use forward slashes
taxi_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(taxi_data.head())

# Display the data types of each column
print("\nData Types:")
print(taxi_data.dtypes)

# Filter numeric columns only for correlation matrix calculation
numeric_columns = taxi_data.select_dtypes(include=['float64', 'int64'])

# Check if there are any numeric columns
if numeric_columns.empty:
    print("No numeric columns available for correlation matrix.")
else:
    # Calculate the correlation matrix
    corr_matrix = numeric_columns.corr()

    # Display the correlation matrix
    print("\nCorrelation Matrix:")
    print(corr_matrix)
