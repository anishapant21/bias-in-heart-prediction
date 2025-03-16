import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#1. Exploratory Data Analysis
# Load the dataset
df = pd.read_csv('./dataset/heart_disease_uci.csv')

# Display basic info
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check gender and age distributions
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="age", hue="sex", kde=True, palette="coolwarm")
plt.title("Age Distribution by Gender")
plt.show()

# Heart disease outcome distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="num", palette="viridis")
plt.title("Heart Disease Diagnosis (0 = No Disease, 1+ = Disease)")
plt.show()


# Gender-based outcome breakdown
gender_outcome = df.groupby(['sex', 'num']).size().unstack()
gender_outcome.plot(kind='bar', stacked=True, colormap='viridis')
plt.title("Heart Disease Outcome by Gender")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Female", "Male"], rotation=0)
plt.show()

# Age group breakdown
df['age_group'] = pd.cut(df['age'], bins=[29, 40, 50, 60, 70, 100], labels=["30s", "40s", "50s", "60s", "70+"])
age_outcome = df.groupby(['age_group', 'num']).size().unstack()
age_outcome.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title("Heart Disease Outcome by Age Group")
plt.ylabel("Count")
plt.show()
