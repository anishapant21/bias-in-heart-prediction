import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure diagrams folder exists
os.makedirs('diagrams', exist_ok=True)

# 1. Load the dataset
df = pd.read_csv('./dataset/heart_disease_uci.csv')

# 2. Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# 3. Imputation Strategy
numerical_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.median()))

categorical_cols = ['restecg', 'exang', 'slope', 'ca', 'thal']
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

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

for col in ['sex', 'cp', 'restecg', 'exang', 'slope', 'num']:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

# 6. Visual Exploration
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="age", hue="sex", kde=True, palette="coolwarm")
plt.title("Age Distribution by Gender")
plt.savefig('diagrams/age_distribution_by_gender.png')
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="num", hue="num", palette="viridis", legend=False)
plt.title("Heart Disease Diagnosis (0 = No Disease, 1+ = Disease)")
plt.savefig('diagrams/heart_disease_diagnosis.png')
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(data=df, x='num', y=col, hue='num', palette='viridis', legend=False)
    plt.title(f"{col} Distribution by Heart Disease Status")
plt.tight_layout()
plt.savefig('diagrams/numerical_variable_distributions.png')
plt.show()

plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x='sex', y='chol', hue='num', split=True, palette='muted')
plt.title("Cholesterol Levels by Gender and Disease Outcome")
plt.savefig('diagrams/cholesterol_gender_outcome.png')
plt.show()

# Correlation heatmap
df_numeric = df.copy()

if 'sex' in df_numeric.columns and df_numeric['sex'].dtype == 'object':
    df_numeric['sex'] = df_numeric['sex'].map({'Male': 1, 'Female': 0})

if 'exang' in df_numeric.columns and df_numeric['exang'].dtype == 'object':
    df_numeric['exang'] = df_numeric['exang'].map({True: 1, False: 0})

cat_columns = df_numeric.select_dtypes(include=['object']).columns.tolist()
for col in cat_columns:
    if df_numeric[col].nunique() < 10:
        dummies = pd.get_dummies(df_numeric[col], prefix=col, drop_first=True)
        df_numeric = pd.concat([df_numeric, dummies], axis=1)
    df_numeric.drop(columns=[col], inplace=True)

plt.figure(figsize=(10, 8))
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig('diagrams/correlation_heatmap.png')
plt.show()

# 7. Analyze Potential Bias
# Gender breakdown
gender_outcome = df.groupby(['sex', 'num']).size().unstack().fillna(0)
gender_outcome.plot(kind='bar', stacked=True, colormap='viridis')
plt.title("Heart Disease Outcome by Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.savefig('diagrams/gender_outcome.png')
plt.show()

# Age group breakdown
df['age_group'] = pd.cut(df['age'], bins=[29, 40, 50, 60, 70, 100], labels=["30s", "40s", "50s", "60s", "70+"])
age_outcome = df.groupby(['age_group', 'num']).size().unstack().fillna(0)
age_outcome.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title("Heart Disease Outcome by Age Group")
plt.ylabel("Count")
plt.savefig('diagrams/age_group_outcome.png')
plt.show()

# Intersectional Analysis: Gender + Age Groups
df['gender_age_group'] = df['sex'].astype(str) + "_" + df['age_group'].astype(str)
intersectional_outcome = df.groupby(['gender_age_group', 'num']).size().unstack().fillna(0)
intersectional_outcome.plot(kind='bar', stacked=True, colormap='plasma')
plt.title("Heart Disease Outcome by Gender and Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig('diagrams/gender_age_intersection.png')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prepare data for multiclass classification
X = df.drop(columns=['num'])
y = df['num']  # Keep original multiclass labels (0, 1, 2, 3, 4)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Train the Multiclass Logistic Regression Model (softmax regression)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 3. Evaluate the model
y_pred = model.predict(X_test)

print("\nOverall Classification Report (Multiclass):")
print(classification_report(y_test, y_pred))

print("\nOverall Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='viridis', values_format='d')
plt.title("Confusion Matrix - Multiclass Logistic Regression")
plt.savefig('diagrams/multiclass_confusion_matrix.png')
plt.show()

# 4. Subgroup Analysis for Multiclass
def evaluate_multiclass_subgroup(data, subgroup_name):
    """ Evaluate a subgroup using the trained multiclass model """
    X_sub = scaler.transform(data.drop(columns=['num']))
    y_sub = data['num']
    y_pred_sub = model.predict(X_sub)

    print(f"\nClassification Report for {subgroup_name} (Multiclass):")
    print(classification_report(y_sub, y_pred_sub))

# Men vs. Women
evaluate_multiclass_subgroup(df[df['sex'] == 1], "Men")
evaluate_multiclass_subgroup(df[df['sex'] == 0], "Women")

# Age Groups
evaluate_multiclass_subgroup(df[df['age_group'] == "30s"], "Young (30s)")
evaluate_multiclass_subgroup(df[df['age_group'] == "50s"], "Middle-aged (50s)")
evaluate_multiclass_subgroup(df[df['age_group'] == "70+"], "Older (70+)")

# Intersectional Groups
evaluate_multiclass_subgroup(df[df['gender_age_group'] == "1_30s"], "Young Men (30s)")
evaluate_multiclass_subgroup(df[df['gender_age_group'] == "0_30s"], "Young Women (30s)")
evaluate_multiclass_subgroup(df[df['gender_age_group'] == "1_70+"], "Older Men (70+)")
evaluate_multiclass_subgroup(df[df['gender_age_group'] == "0_70+"], "Older Women (70+)")
