import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
# Download latest version
#path = kagglehub.dataset_download("anshtanwar/metro-interstate-traffic-volume")

#print("Path to dataset files:", path)
#path = kagglehub.dataset_download("anshtanwar/metro-interstate-traffic-volume")

#print("Path to dataset files:", path)
#csv_file = None
#for file in os.listdir(path):

   # if file.endswith('.csv'):
       # csv_file = os.path.join(path, file)
      #  break

#if csv_file is None:
   # print("No CSV file found in the directory.")
#else:
    #print(f"Found CSV file: {csv_file}")


df = pd.read_csv('Trafficdata.csv')
df.head(2)

rows, columns = df.shape

print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")

print(df.info())

non_null_holiday = df[df['holiday'].notnull()]
non_null_holiday
df['holiday'] = df['holiday'].fillna('No Holiday')
df.head()
for column in df.columns:
    null_count = df[column].isnull().sum()
    print(f"Number of null values in '{column}': {null_count}")

df['date_time']=pd.to_datetime(df['date_time'],format='mixed')
df['day'] = df['date_time'].dt.day_name()
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['hour'] = df['date_time'].dt.hour
#df['date'] = df['date_time'].dt.date

df.drop('date_time',axis=1,inplace = True)
df.head()
df['temp']=df['temp'].apply(lambda x: (x- 273.15) * 9/5 + 32)
df.head()

df_year = df.groupby('year')['traffic_volume'].mean()

df_year = df_year.reset_index()

# Now you can plot with Plotly Express, and the x-axis will follow the days_order
fig = px.line(df_year, x='year', y='traffic_volume', title='Average Traffic Volume based on Year')

# Customize the plot
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Average Traffic Volume',
    xaxis_tickmode='array',  # Use array mode for ticks
    xaxis_tickvals=list(range(2012, 2018)),  # Set the tick values from 2012 to 2018
)
fig.show()
# Define the order for the days of the week
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Group the data by 'day' and calculate the mean of 'traffic_volume'
df_day = df.groupby('day')['traffic_volume'].mean()

# Reset the index to make 'day' a column
df_day = df_day.reset_index()

# Convert the 'day' column to a categorical type with the specified order
df_day['day'] = pd.Categorical(df_day['day'], categories=days_order, ordered=True)

# Sort the DataFrame by 'day' to ensure it's in the correct order
df_day = df_day.sort_values('day')

# Now you can plot with Plotly Express, and the x-axis will follow the days_order
fig = px.line(df_day, x='day', y='traffic_volume', title='Average Traffic Volume based on the Day of the Week')

# Customize the plot
fig.update_layout(
    xaxis_title='Day of the Week',
    yaxis_title='Average Traffic Volume',
    xaxis_tickangle=-45  # Rotate x-axis labels for better visibility
)

# Show the interactive plot
fig.show()
# Show the interactive plot
df_hour = df.groupby('hour')['traffic_volume'].mean()

df_hour = df_hour.reset_index()

# Now you can plot with Plotly Express, and the x-axis will follow the days_order
fig = px.line(df_hour, x='hour', y='traffic_volume', title='Average Traffic Volume based on Time of the Day')

# Customize the plot
fig.update_layout(
    xaxis_title='Time Of the Day',
    yaxis_title='Average Traffic Volume',
    xaxis_tickmode='array',  # Use array mode for ticks
    xaxis_tickvals=list(range(0, 24)),  # Set the tick values from 0 to 23

)
df['holiday'] = df['holiday'].apply(lambda x: 'No Holiday' if x=='No Holiday' else 'Holiday')
sns.boxplot(x='holiday', y='traffic_volume', data=df)

# Customize the plot
plt.title('Traffic Volume Boxplot for Holidays vs Non-Holidays')
plt.xlabel('Holiday Category')
plt.ylabel('Traffic Volume')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Show the plot
plt.tight_layout()
plt.show()
print(df.describe())
fig, axes = plt.subplots(nrows=int(len(df.columns) / 4), ncols=4, figsize=(15, 10))

# Iterate over each column (feature)
for i, column in enumerate(df.columns):
    row = i // 4
    col = i % 4

    # Check if the column contains numerical data
    if pd.api.types.is_numeric_dtype(df[column]):
        axes[row, col].boxplot(df[column])
        axes[row, col].set_title(column)
    else:
        # Handle non-numerical columns (e.g., categorical)
        # You could skip them or use a different visualization
        print(f"Skipping boxplot for non-numerical column: {column}")

# Adjust spacing between subplots
plt.tight_layout()
plt.show()
sns.pairplot(df)

# Show the plot
plt.show()
print(df['temp'].value_counts().sort_index())
print(df['rain_1h'].value_counts().sort_index())
print(df['snow_1h'].value_counts().sort_index())
df.describe()
print(df.shape)
#drop the rows which has temp=0
rows_to_drop = df[df['temp'] < -50].index
print(len(rows_to_drop))
df = df.drop(rows_to_drop)
print(df.shape)
df.head()
df_weather=df[['temp', 'rain_1h', 'snow_1h', 'clouds_all','month','year','traffic_volume']]
#print(df_weather.head())
sns.heatmap(df_weather.corr(), annot=True)
plt.show()
df_weather=df[['temp', 'rain_1h', 'snow_1h', 'clouds_all','month','year','traffic_volume']]
#print(df_weather.head())

df.drop(['weather_description'],axis=1,inplace=True)
x_df=df.drop(['traffic_volume'],axis=1)
y_df=df['traffic_volume']
df.info()
label_encoder = LabelEncoder()

df['weather_main'] = label_encoder.fit_transform(df['weather_main'])
df['day'] = label_encoder.fit_transform(df['day'])
df['holiday'] = label_encoder.fit_transform(df['holiday'])

# Display the label mappings for each feature
'''weather_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
day_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
holiday_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))'''
print(df.head())
print(df.info())
X=df.drop(['traffic_volume'],axis=1)
y=df['traffic_volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train,  y_train)
# Print the model coefficients
intercept = model.intercept_
coefficients = model.coef_

print("Intercept:", intercept)
print("Model Coefficients:", coefficients)


equation = f"y = {intercept:.2f}"

# Loop through the coefficients and features to build the equation
for i, coef in enumerate(coefficients):
    equation += f" + ({coef:.2f}) * X_{i+1}"  # X_{i+1} represents the feature index

# Print the linear equation
print("Linear Regression Equation:")
print(equation)
y_pred = model.predict(X_test)

# Calculate accuracy (R² score) and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"Mean Squared Error: {mse}")
df_comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Print the DataFrame
print(df_comparison)
# Create a scatter plot
plt.scatter(y_test, y_pred)

# Add a 45-degree line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

# Set labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')

# Show the plot
plt.show()