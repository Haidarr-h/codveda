import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use inline plots if you're in Jupyter Notebook
# %matplotlib inline

# Step 1: Load data
# You can replace this with your own dataset (e.g., pd.read_csv('your_data.csv'))
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

print("🔍 First 5 rows of the dataset:")
print(df.head())

print("\n📈 Summary Statistics:")
print(df.describe())  # Includes count, mean, std, min, max, etc.

# Plot histogram for each numerical feature
df.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histogram of Features")
plt.tight_layout()
plt.show()

# One boxplot for each numeric column
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of Numeric Features")
plt.xticks(rotation=45)
plt.show()

# Only use numeric columns
corr_matrix = df.select_dtypes(include=np.number).corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print("\n📝 EDA Insights Summary:")

# Top correlated features
print("Top Correlations:")
correlation_pairs = corr_matrix.unstack().sort_values(ascending=False)
top_correlations = correlation_pairs[(correlation_pairs < 1.0) & (correlation_pairs > 0.5)]
print(top_correlations)

# Class distribution (if classification task)
if 'species' in df.columns:
    print("\nClass Distribution:")
    print(df['species'].value_counts())
