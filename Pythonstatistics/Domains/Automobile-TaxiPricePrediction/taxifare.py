import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('taxidata.csv')
print(df.head())  # View the first few rows
print(df.info())  # Check column types and null values
pd.set_option('display.max_columns',None)
print(df.describe())
# Create subplots for each column
import matplotlib.pyplot as plt
import seaborn as sns

# Columns for boxplots
mns = ['trip_distance', 'rate_code', 'payment_type', 'extra', 'mta_tax', 'tolls_amount', 'imp_surcharge', 'hour_of_day', 'trip_duration']

# Calculate the grid size
num_plots = len(mns)
rows = (num_plots // 2) + (num_plots % 2)  # 2 columns, so calculate rows accordingly

plt.figure(figsize=(12, rows * 3))  # Adjust figure size

# Generate separate box plots for each column
for i, col in enumerate(mns, 1):
    plt.subplot(rows, 2, i)  # `rows` rows, 2 columns
    sns.boxplot(data=df, y=col, color='lightblue')
    plt.title(f'Boxplot for {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

for col in mns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, y=col, color='lightblue')
    plt.title(f'Boxplot for {col}')
    plt.ylabel(col)
    plt.show()


# Add 'fare_amount' to the columns list
mns = ['trip_distance', 'rate_code', 'payment_type', 'extra', 'mta_tax', 'tolls_amount', 'imp_surcharge', 'hour_of_day', 'trip_duration', 'fare_amount']

# Select only the columns in `mns` from the DataFrame
df_selected = df[mns]

# Calculate the correlation matrix
correlation_matrix = df_selected.corr()

# Display the correlation matrix
print("Correlation Matrix:\n", correlation_matrix)

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Selected Features")
plt.show()

# Step 1: Define the features and target variable
X = df[mns].drop('fare_amount', axis=1)  # Features
y = df['fare_amount']                    # Target variable

# Step 2: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Scale the features (useful for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 5: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
# Predict on test set
y_pred = model.predict(X_test_scaled)

# Step 3: Add predictions to test DataFrame for visualization
df_test = X_test.copy()
df_test['actual_fare'] = y_test.values
df_test['predicted_fare'] = y_pred
 

# Include latitude and longitude if available in your data
print(df.columns)  # Check the column names in the DataFrame
# Attempt to access 'pickup_latitude' and 'pickup_longitude'
# If they don't exist, you'll need to adjust the column names accordingly
df_test['pickup_latitude'] = df.loc[df_test.index, 'pickup_latitude']  # This line may cause an error if 'pickup_latitude' does not exist
df_test['pickup_longitude'] = df.loc[df_test.index, 'pickup_longitude']  # This line may cause an error if 'pickup_longitude' does not exist

# Step 4: Initialize a map (e.g., centered around an average pickup location)
average_lat = df_test['pickup_latitude'].mean()
average_lon = df_test['pickup_longitude'].mean()
m = folium.Map(location=[average_lat, average_lon], zoom_start=12)

# Step 5: Add markers for each point in the test data with actual and predicted fare
for _, row in df_test.iterrows():
    folium.Marker(
        location=[row['pickup_latitude'], row['pickup_longitude']],
        popup=f"Actual Fare: ${row['actual_fare']:.2f}, Predicted Fare: ${row['predicted_fare']:.2f}",
        icon=folium.Icon(color="blue" if row['actual_fare'] <= row['predicted_fare'] else "red")
    ).add_to(m)

# Display the map
m.save("fare_prediction_map.html")
