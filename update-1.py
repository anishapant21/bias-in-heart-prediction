import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

df = pd.read_csv('./dataset/heart_disease_uci.csv')

df = df.drop(['id', 'dataset'], axis=1)

new_column_names = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG',
    'thalch': 'Max Heart Rate',
    'exang': 'Exercise-Induced Angina',
    'oldpeak': 'ST Depression',
    'slope': 'Slope of ST',
    'ca': 'Number of Major Vessels',
    'thal': 'Thalassemia',
    'num': 'Diagnosis'
}

df = df.rename(columns=new_column_names)

df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 0 if x == 0 else 1)

print("Missing values before removal:")
print(df.isnull().sum())
print(f"Original dataset shape: {df.shape}")

df = df.dropna()
print("\nDataset shape after removing missing values:")
print(df.shape)

numerical_features = []
categorical_features = []

for column in df.columns:
    if column == 'Diagnosis':
        continue
        
    if df[column].dtype == 'object' or df[column].nunique() < 10:
        categorical_features.append(column)
    else:
        numerical_features.append(column)

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Set up the age groups
df['Age Group'] = pd.cut(df['Age'], bins=[29, 40, 50, 60, 70, 100], 
                        labels=["30s", "40s", "50s", "60s", "70+"])

# Create gender-age intersectional groups
df['Gender_Age_Group'] = df['Sex'].astype(str) + "_" + df['Age Group'].astype(str)

# Print the distribution of our demographic groups
print("\nGender distribution:")
print(df['Sex'].value_counts())

print("\nAge group distribution:")
print(df['Age Group'].value_counts())

print("\nIntersectional group distribution:")
print(df['Gender_Age_Group'].value_counts())

