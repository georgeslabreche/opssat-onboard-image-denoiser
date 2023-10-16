import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from the CSV file
df = pd.read_csv('csv/results_classification-AE-FPN-50-short.csv')

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
plt.figure(figsize=(8, 6), facecolor='white')
plt.hist(data, bins=10, edgecolor='black', color='#1f77b4', alpha=0.5) 
plt.xlabel('Confidence Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Confidence Difference')
plt.grid(color='red', linestyle='--', linewidth=0.5, axis='both', which='both')
plt.savefig('figures/AE/FPN-50/classification_confidence_distribution/classification_confidence_distribution_histogram.svg', format='svg')
plt.clf()

# boxplot
plt.figure(figsize=(8, 6), facecolor='white')
sns.boxplot(y=data)
plt.ylabel('Confidence Difference')
plt.title('Boxplot of Confidence Difference')
plt.grid(color='red', linestyle='--', linewidth=0.5, axis='both', which='both')
plt.savefig('figures/AE/FPN-50/classification_confidence_distribution/classification_confidence_distribution_boxplot.svg', format='svg')
plt.clf()

# linechart
plt.figure(figsize=(8, 6), facecolor='white')
plt.plot(data, color='#1f77b4')
plt.xlabel('Data Point Index')
plt.ylabel('Confidence Difference')
plt.title('Line Chart of Confidence Difference')
plt.grid(color='red', linestyle='--', linewidth=0.5, axis='both', which='both')
plt.savefig('figures/AE/FPN-50/classification_confidence_distribution/classification_confidence_distribution_linechart.svg', format='svg')
plt.clf()

# tables
df = pd.DataFrame(data, columns=['Confidence Difference'])
print(df)

# heatmap
plt.figure(figsize=(8, 6), facecolor='white')
correlation_matrix = np.corrcoef(data, data)
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.title('Heatmap of Confidence Difference')
plt.grid(color='red', linestyle='--', linewidth=0.5, axis='both', which='both')
plt.savefig('figures/AE/FPN-50/classification_confidence_distribution/classification_confidence_distribution_heatmap.svg', format='svg')
plt.clf()
