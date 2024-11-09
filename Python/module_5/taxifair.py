import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('data.csv')

# Remove any rows where trip distance is zero to avoid division by zero
df = df[df['trip_distance'] > 0]

# Calculate fare per mile
df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']

# Display first few rows to check the calculation
print(df[['fare_amount', 'trip_distance', 'fare_per_mile']].head())

# Plot the distribution of fare per mile
plt.figure(figsize=(10, 6))
sns.histplot(df['fare_per_mile'], bins=50, kde=True, color='blue')  # Histogram with density curve
plt.title('Distribution of Fare per Mile')
plt.xlabel('Fare per Mile (USD)')
plt.ylabel('Frequency')
plt.xlim(0, 10)  # Limiting to reasonable fare per mile range (adjust based on data)
plt.show()
