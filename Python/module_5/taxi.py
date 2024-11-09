import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('data.csv', low_memory=False)

# Convert 'tpep_pickup_datetime' to datetime and extract hour
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['hour'] = df['tpep_pickup_datetime'].dt.hour

# Calculate hourly counts
hourly_counts = df['hour'].value_counts().sort_index()

# Create the bar plot
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, hue=hourly_counts.index, palette='coolwarm', legend=False)

plt.title('Hourly Distribution of Taxi Pickups')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Pickups')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

plt.show()

# Convert 'pickup_datetime' to datetime format
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

# Extract the hour from the 'pickup_datetime' column
df['hour'] = df['tpep_pickup_datetime'].dt.hour

# Group by the 'hour' column and count the number of rides in each hour
hourly_counts = df['hour'].value_counts().sort_index()

# Print total trips per hour
print("Total trips for each hour:")
for hour, trips in hourly_counts.items():
    print(f"Hour {hour}: {trips} trips")

# Plot the hourly distribution of rides
plt.figure(figsize=(10, 6))
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='coolwarm')

# Title and labels
plt.title('Hourly Distribution of Taxi Rides (Bar Plot)')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Rides per Day')

# Show the plot
plt.show()
