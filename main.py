import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('./dataset/heart_disease_uci.csv')

# 2. Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# 3. Imputation Strategy
# Impute numerical columns with median
numerical_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.median()))

# Impute categorical columns with mode
categorical_cols = ['restecg', 'exang', 'slope', 'ca', 'thal']
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Drop columns with excessive missing values
df.drop(columns=['ca', 'thal'], inplace=True)

# 4. Verify the missing values after imputation
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# 5. Expanded Descriptive Analysis
print("\nDataset Overview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Categorical variable distributions
for col in ['sex', 'cp', 'restecg', 'exang', 'slope', 'num']:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

# 6. Visual Exploration
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="age", hue="sex", kde=True, palette="coolwarm")
plt.title("Age Distribution by Gender")
plt.show()

# Heart disease outcome distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="num", hue="num", palette="viridis", legend=False)
plt.title("Heart Disease Diagnosis (0 = No Disease, 1+ = Disease)")
plt.show()

# Box plots for key numerical variables
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(data=df, x='num', y=col, hue='num', palette='viridis', legend=False)
    plt.title(f"{col} Distribution by Heart Disease Status")
plt.tight_layout()
plt.show()

# Violin plots to explore distributions
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x='sex', y='chol', hue='num', split=True, palette='muted')
plt.title("Cholesterol Levels by Gender and Disease Outcome")
plt.show()

# Correlation heatmap - FIX: Create a copy of the dataframe with numeric columns only
# Encode categorical variables for correlation analysis
df_numeric = df.copy()

# One-hot encode string columns or convert to numeric
if 'sex' in df_numeric.columns and df_numeric['sex'].dtype == 'object':
    df_numeric['sex'] = df_numeric['sex'].map({'Male': 1, 'Female': 0})

if 'exang' in df_numeric.columns and df_numeric['exang'].dtype == 'object':
    df_numeric['exang'] = df_numeric['exang'].map({True: 1, False: 0})

# Get all categorical columns that need to be one-hot encoded
cat_columns = df_numeric.select_dtypes(include=['object']).columns.tolist()
for col in cat_columns:
    # Skip one-hot encoding for columns with too many unique values
    if df_numeric[col].nunique() < 10:  # Reasonable threshold
        # Create dummy variables
        dummies = pd.get_dummies(df_numeric[col], prefix=col, drop_first=True)
        # Add dummy variables to the dataframe
        df_numeric = pd.concat([df_numeric, dummies], axis=1)
    # Drop the original categorical column
    df_numeric.drop(columns=[col], inplace=True)

# Now calculate correlation with numeric data only
plt.figure(figsize=(10, 8))
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 7. Analyze Potential Bias
# Gender-based outcome breakdown
gender_outcome = df.groupby(['sex', 'num']).size().unstack().fillna(0)
gender_outcome.plot(kind='bar', stacked=True, colormap='viridis')
plt.title("Heart Disease Outcome by Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Age group breakdown
df['age_group'] = pd.cut(df['age'], bins=[29, 40, 50, 60, 70, 100], labels=["30s", "40s", "50s", "60s", "70+"])
age_outcome = df.groupby(['age_group', 'num']).size().unstack().fillna(0)
age_outcome.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title("Heart Disease Outcome by Age Group")
plt.ylabel("Count")
plt.show()

# Explore cholesterol bias by gender
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='sex', y='chol', hue='num', palette='Set2')
plt.title("Cholesterol Levels by Gender and Heart Disease Status")
plt.show()

# Chest pain type comparison
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='cp', hue='num', palette='pastel')
plt.title("Chest Pain Type vs Heart Disease")
plt.show()

# Resting ECG analysis
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='restecg', hue='num', palette='muted')
plt.title("Resting ECG Results vs Heart Disease")
plt.show()