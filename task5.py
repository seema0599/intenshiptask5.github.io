#CREDIT CARD FRAUD DETECTION

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# Load the dataset
df = pd.read_csv('creditcard.csv')
print(df)

# Check for missing values and basic dataset info
print(f"Dataset shape: {df.shape}")
print(df.describe())
print(df.dtypes)
print(f"Missing values:\n{df.isnull().sum()}")

# Count the number of fraudulent and legitimate transactions
transaction_counts = df['Class'].value_counts()
print("Number of legitimate transactions:", transaction_counts[0])
print("Number of fraudulent transactions:", transaction_counts[1])

# Calculate the percentage of fraudulent transactions
total_transactions = len(df)
transaction_counts = df['Class'].value_counts()
fraudulent_transactions = transaction_counts[1]
non_fraudulent_transactions = transaction_counts[0]
percentage_fraudulent = (fraudulent_transactions / total_transactions) * 100
print(f"Percentage of fraudulent transactions: {percentage_fraudulent:.2f}%")

# Data for the pie chart
labels = ['Non-Fraudulent', 'Fraudulent']
sizes = [non_fraudulent_transactions, fraudulent_transactions]
plt.pie(sizes, labels=labels, autopct='%2.2f%%', startangle=140, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of Fraudulent Transactions')
plt.show()

# Calculate descriptive statistics for the 'Amount' column
min_amount = df['Amount'].min()
max_amount = df['Amount'].max()
mean_amount = df['Amount'].mean()
median_amount = df['Amount'].median()

# Print the statistics
print(f"Minimum amount: {min_amount:.2f}")
print(f"Maximum amount: {max_amount:.2f}")
print(f"Mean amount: {mean_amount:.2f}")
print(f"Median amount: {median_amount:.2f}")


# Find the maximum transaction amount
max_amount = df['Amount'].max()
max_amount_row = df[df['Amount'] == max_amount]
is_fraudulent = max_amount_row['Class'].values[0]
print(f"The maximum transaction amount is {max_amount}, and it is {'fraudulent' if is_fraudulent else 'legitimate'}.")

import matplotlib.pyplot as plt
import seaborn as sns

#Bar Chart Showing the Count of Fraudulent vs. Legitimate Transactions¶
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Count of Fraudulent vs. Legitimate Transactions')
plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()

#Heatmap to Visualize the Correlation Between Numerical Features
correlation_matrix=df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix,cmap='coolwarm',vmin=-1,vmax=1,annot=False,linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Ensure there are no infinite values in the 'Time' column
df['Time'].replace([float('inf'), -float('inf')], float('nan'), inplace=True)

# Splitting the data
class_0 = df.loc[df['Class'] == 0]["Time"]
class_1 = df.loc[df['Class'] == 1]["Time"]

# Creating a Seaborn KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(class_0, label='Not Fraud', fill=True, common_norm=False)
sns.kdeplot(class_1, label='Fraud', fill=True, common_norm=False)

# Adding titles and labels
plt.title('Credit Card Transactions Time Density Plot')
plt.xlabel('Time [s]')
plt.ylabel('Density')
plt.legend()
plt.show()

#Hour with the Most Frequent Fraudulent Transactions
# Convert the 'Time' column to hours
df['transaction_hour'] = (df['Time'] // 3600) % 24

# Assuming 'Class' column indicates fraudulent transactions (1 for fraud, 0 for non-fraud)
df['is_fraudulent'] = df['Class']

# Separate the fraudulent and non-fraudulent transactions
fraudulent_df = df[df['is_fraudulent'] == 1]
non_fraudulent_df = df[df['is_fraudulent'] == 0]
# Aggregate by hour for fraudulent and non-fraudulent transactions
fraudulent_by_hour = fraudulent_df.groupby('transaction_hour').size()
non_fraudulent_by_hour = non_fraudulent_df.groupby('transaction_hour').size()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(fraudulent_by_hour.index - 0.2, fraudulent_by_hour.values, width=0.4, color='red', alpha=0.7, label='Fraudulent')
plt.bar(non_fraudulent_by_hour.index + 0.2, non_fraudulent_by_hour.values, width=0.4, color='blue', alpha=0.7, label='Non-Fraudulent')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.title('Transactions by Hour of Day')
plt.xticks(range(0, 24))  # Set x-ticks for hours
plt.legend()
plt.tight_layout()
plt.show()

# Convert the 'Time' column to hours
df['transaction_hour'] = (df['Time'] // 3600) % 24

# Assuming 'Class' column indicates fraudulent transactions (1 for fraud, 0 for non-fraud)
df['is_fraudulent'] = df['Class']

# Separate the fraudulent transactions
fraudulent_df = df[df['is_fraudulent'] == 1]

# Aggregate by hour for fraudulent transactions
fraudulent_by_hour = fraudulent_df.groupby('transaction_hour').size()
# Find the hour with the most frequent fraudulent transactions
most_frequent_hour = fraudulent_by_hour.idxmax()
most_frequent_count = fraudulent_by_hour.max()

# Print the result
print(f"The hour with the most frequent fraudulent transactions is: {most_frequent_hour}:00")
print(f"Number of fraudulent transactions during this hour: {most_frequent_count}")

# Plotting the distribution of fraudulent transactions by hour
plt.figure(figsize=(10, 6))
plt.bar(fraudulent_by_hour.index, fraudulent_by_hour.values, color='red', alpha=0.7)
plt.xlabel('Hour of Day')
plt.ylabel('Number of Fraudulent Transactions')
plt.title('Fraudulent Transactions by Hour of Day')
plt.xticks(range(0, 24))  # Set x-ticks for hours
plt.tight_layout()
plt.show()

# Convert the 'Time' column to hours
df['transaction_hour'] = (df['Time'] // 3600) % 24

# Replace inf values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Calculate required statistics for each hour
hourly_stats = df.groupby('transaction_hour').agg({
    'Amount': ['sum', 'mean', 'min', 'max', 'median']
}).reset_index()

# Rename columns for easier access
hourly_stats.columns = ['Hour', 'Total', 'Mean', 'Min', 'Max', 'Median']
# Separate fraudulent and non-fraudulent transactions
fraudulent_df = df[df['Class'] == 1]
non_fraudulent_df = df[df['Class'] == 0]

# Calculate required statistics for fraudulent transactions
fraudulent_stats = fraudulent_df.groupby('transaction_hour').agg({
    'Amount': ['sum', 'mean', 'min', 'max', 'median']
}).reset_index()
fraudulent_stats.columns = ['Hour', 'Total', 'Mean', 'Min', 'Max', 'Median']

# Calculate required statistics for non-fraudulent transactions
non_fraudulent_stats = non_fraudulent_df.groupby('transaction_hour').agg({
    'Amount': ['sum', 'mean', 'min', 'max', 'median']
}).reset_index()
non_fraudulent_stats.columns = ['Hour', 'Total', 'Mean', 'Min', 'Max', 'Median']

# Set up the plotting environment
sns.set(style="whitegrid")

# Plot Total Amount
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
sns.lineplot(ax=ax1, x='Hour', y='Total', data=non_fraudulent_stats, label='Non-Fraudulent')
sns.lineplot(ax=ax2, x='Hour', y='Total', data=fraudulent_stats, label='Fraudulent', color='red')
ax1.set_title('Total Amount of Non-Fraudulent Transactions by Hour')
ax2.set_title('Total Amount of Fraudulent Transactions by Hour')
plt.suptitle('Total Amount of Transactions by Hour')
plt.show()

# Plot Maximum Amount
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
sns.lineplot(ax=ax1, x='Hour', y='Max', data=non_fraudulent_stats, label='Non-Fraudulent')
sns.lineplot(ax=ax2, x='Hour', y='Max', data=fraudulent_stats, label='Fraudulent', color='red')
ax1.set_title('Maximum Amount of Non-Fraudulent Transactions by Hour')
ax2.set_title('Maximum Amount of Fraudulent Transactions by Hour')
plt.suptitle('Maximum Amount of Transactions by Hour')
plt.show()

# Create a temporary DataFrame with 'Amount' and 'Class' columns
tmp = df[['Amount', 'Class']].copy()

# Separate the data into non-fraudulent and fraudulent transactions
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']

# Display summary statistics
print("Non-Fraudulent Transactions:",class_0.describe())
print("\nFraudulent Transactions:",class_1.describe())

# Filter data for fraudulent transactions
fraud = df.loc[df['Class'] == 1]
print("data for fraudulent transactions:",fraud)

# Create the scatter plot using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=fraud['Time'],y=fraud['Amount'],color='red',alpha=0.5)
plt.title('Amount of Fraudulent Transactions')
plt.xlabel('Time [s]')
plt.ylabel('Amount')
plt.show()




