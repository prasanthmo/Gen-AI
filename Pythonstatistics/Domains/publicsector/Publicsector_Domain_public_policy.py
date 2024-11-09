import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset into a DataFrame
file_path = r"C:\Users\lahar\Gen-AI\AP_Industrial_Policy_Dataset.csv"
df = pd.read_csv(file_path)

# Code to show all columns clearly
pd.set_option('display.max_columns', None)

# Check for missing values in each column
missing_values = df.isnull().sum()

# Display the number of missing values for each column
print("Missing Values in Each Column:")
print(missing_values[missing_values > 0])

# Display total number of rows and columns
print(f"\nTotal Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

# Display DataFrame info
print("\nDataFrame Info:")
print(df.info())

# Display the first few rows of the DataFrame
print("\nDataFrame Head:")
print(df.head())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())  # Include all columns, even non-numeric

# Plotting boxplots for numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Create boxplots for each numeric column
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)  # Adjust the number of rows and columns based on the number of numeric columns
    sns.boxplot(data=df, y=col, color='lightblue')
    plt.title(f'Boxplot for {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

sns.set(style="whitegrid")

# Create the bar plot with 'Year' as hue
plt.figure(figsize=(10, 8))
sns.barplot(x='Industry Name', y='Employment Rate (%)', hue='Year', data=df, palette="Set2", errorbar=None)

# Customize the plot
plt.title('Employment Rate by Industry with Year as Hue', fontsize=18)
plt.xlabel('Industry Name', fontsize=14)
plt.ylabel('Employment Rate (%)', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='Year', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

def my_autopct(pct):
    total = industry_investment.sum()
    val = pct * total / 100
    return f'{pct:.1f}%\n({val:.0f} M INR)'  # Format percentage and actual value
# Aggregate the data by 'Industry Name' to get the total investment for each industry
industry_investment = df.groupby('Industry Name')['Investment (in Million INR)'].sum()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(industry_investment, labels=industry_investment.index, autopct=my_autopct, startangle=90, colors=sns.color_palette("Set2"))

# Add a title
plt.title('Total Investment by Industry', fontsize=16)

# Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.axis('equal')

# Show the pie chart
plt.show()

# Aggregate the data by 'Industry Name' to get the total investment for each industry
industry_investment = df.groupby('Industry Name')['Energy Consumption (in GWh)'].sum()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(industry_investment, labels=industry_investment.index, autopct=my_autopct, startangle=90, colors=sns.color_palette("Set2"))

# Add a title
plt.title('Energy Consumption (in GWh) by Industry', fontsize=16)

# Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.axis('equal')

# Show the pie chart
plt.show()

# Group the data by 'Industry Name' and 'Year' and sum the 'Government Incentives'
df_grouped = df.groupby(['Industry Name', 'Year'])['Government Incentives (in Million INR)'].sum().reset_index()

# Create a pivot table to reshape the data for plotting
df_pivot = df_grouped.pivot(index='Year', columns='Industry Name', values='Government Incentives (in Million INR)')
# Create the stacked bar chart
ax = df_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))

# Annotate the bars with values
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.2f}', (x + width/2, y + height/2), ha='center', va='center')

plt.title('Government Spending by Industry and Year')
plt.xlabel('Year')
plt.ylabel('Government Incentives (in Million INR)')
plt.xticks(rotation=45)
plt.legend(title='Industry Name')
plt.show()