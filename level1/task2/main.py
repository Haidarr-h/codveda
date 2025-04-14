!pip install pandas scikit-learn

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

# Step 1: Simulate a raw dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', None],
    'Age': [25, 30, 35, 40, 200, np.nan, 28], 
    'Gender': ['Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female'],
    'Salary': [50000, 60000, 55000, np.nan, 120000, 62000, 58000], 
    'Department': ['HR', 'Engineering', 'HR', 'Marketing', 'Engineering', 'Engineering', 'HR']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Step 3: Remove Outliers (we'll define an outlier as values beyond 3 standard deviations)
def remove_outliers(df, column):
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] >= mean - 3*std) & (df[column] <= mean + 3*std)]

df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Salary')

# Step 4: Convert Categorical to Numeric
# Example using One-Hot Encoding for 'Department'
df = pd.get_dummies(df, columns=['Department'])

# Example using Label Encoding for 'Gender'
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

# Step 5: Normalize or Standardize
scaler = StandardScaler()  # Or MinMaxScaler()

# We’ll scale only the numeric columns
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

df
