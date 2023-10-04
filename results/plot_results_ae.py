import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from the CSV file
df = pd.read_csv('data.csv')

# Remove rows with missing values (None in Confidence Difference)
df = df.dropna(subset=['confidence_difference'])

# Extract the 'Confidence Difference' column as a Series
data = df['confidence_difference']

# Descriptive statistics
series = pd.Series(data)
print(series.describe())

data = [x for x in data if x is not None]  # Cleans empty entries out
series = pd.Series(data)
print(series.describe())

# histogram
plt.hist(data, bins=10, edgecolor='black')
plt.xlabel('Confidence Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Confidence Difference')
plt.grid(axis='y')
plt.show()

# boxpot
sns.boxplot(y=data)
plt.ylabel('Confidence Difference')
plt.title('Boxplot of Confidence Difference')
plt.show()

# linechart
plt.plot(data)
plt.xlabel('Data Point Index')
plt.ylabel('Confidence Difference')
plt.title('Line Chart of Confidence Difference')
plt.grid(axis='both')
plt.show()

# tables
df = pd.DataFrame(data, columns=['Confidence Difference'])
print(df)

# heatmap
correlation_matrix = np.corrcoef(data, data)
sns.heatmap(correlation_matrix, annot=True)
plt.title('Heatmap of Confidence Difference')
plt.show()
