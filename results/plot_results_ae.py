import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# todo: read from csv file instead
#       make sure that the csv file also contains the classification results (for context)
data = [-0.169742, -0.06547, -0.04404, -0.037052, -0.050134, -0.310529, -0.159407, 0.046612, 0.044578, -0.357572, -0.024214, -0.035141, -0.167083, -0.006572, 0.023848, -0.105729, -0.10302, -0.110709, 0.004851, -0.081153, -0.009501, 0.07843, None, 0.141651, -0.232892, None, None, -0.09453, -0.009037, -0.003863, None, 0.055157]

# descriptive statistics
data = [x for x in data if x is not None]  # Removing None values
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
