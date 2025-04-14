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

# Create age groups and intersectional groups
df['Age Group'] = pd.cut(df['Age'], bins=[29, 40, 50, 60, 70, 100], 
                        labels=["30s", "40s", "50s", "60s", "70+"])
df['Gender_Age_Group'] = df['Sex'].astype(str) + "_" + df['Age Group'].astype(str)

# Print the counts to see what we're working with
print("\nGender distribution:")
print(df['Sex'].value_counts())

print("\nAge group distribution:")
print(df['Age Group'].value_counts())

print("\nIntersectional group distribution:")
print(df['Gender_Age_Group'].value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def analyze_coefficients_for_subgroup(df, subgroup_name, subgroup_condition):
    """
    Train a logistic regression model on a specific subgroup and analyze coefficients
    """
    # Filter the dataframe for the subgroup
    subgroup_df = df[subgroup_condition]
    
    print(f"\n{subgroup_name} sample size: {len(subgroup_df)}")
    
    # Skip if we don't have enough samples (adjust this threshold if needed)
    if len(subgroup_df) < 10:
        print(f"Skipping {subgroup_name} due to insufficient samples")
        return None
    
    # Prepare features and target
    X = subgroup_df.drop(['Diagnosis', 'Age Group', 'Gender_Age_Group'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # Handle categorical features
    y = subgroup_df['Diagnosis']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(C=5.0, solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # Get coefficients
    coeffs = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })
    
    # Sort by absolute coefficient value
    coeffs = coeffs.sort_values('Coefficient', key=abs, ascending=False)
    
    print(f"Top 5 coefficients for {subgroup_name}:")
    print(coeffs.head(5))
    
    return coeffs

# First, let's check what values are actually in the Sex column
print("\nUnique values in Sex column:", df['Sex'].unique())

# Define the gender conditions
male_condition = df['Sex'] == 'Male'
female_condition = df['Sex'] == 'Female'

# Check that our conditions actually select rows
print("\nMales selected:", male_condition.sum())
print("Females selected:", female_condition.sum())

# Analyze by gender
print("\n===== Gender-based Coefficient Analysis =====")
male_coeffs = analyze_coefficients_for_subgroup(df, "Male", male_condition)
female_coeffs = analyze_coefficients_for_subgroup(df, "Female", female_condition)

# Analyze by age groups
print("\n===== Age-based Coefficient Analysis =====")
age_coeffs = {}
for age_group in ["40s", "50s", "60s"]:  # Focus on groups with more samples
    age_coeffs[age_group] = analyze_coefficients_for_subgroup(
        df, f"Age {age_group}", df['Age Group'] == age_group
    )

# Adjust intersectional group definitions based on what we find
intersectional_groups = [
    ("Male 50s", (male_condition) & (df['Age Group'] == "50s")),
    ("Male 40s", (male_condition) & (df['Age Group'] == "40s")),
    ("Female 50s", (female_condition) & (df['Age Group'] == "50s"))
]

# Verify the group sizes
for name, condition in intersectional_groups:
    print(f"{name} size: {condition.sum()}")

intersect_coeffs = {}
for group_name, condition in intersectional_groups:
    intersect_coeffs[group_name] = analyze_coefficients_for_subgroup(
        df, group_name, condition
    )
import matplotlib.pyplot as plt
import seaborn as sns

# Create a function for coefficient visualization
def visualize_coefficient_comparison(group1_name, group1_coeffs, group2_name, group2_coeffs, top_n=5):
    """Create a visualization comparing coefficients between two demographic groups"""
    if group1_coeffs is None or group2_coeffs is None:
        print(f"Cannot visualize - one of the groups has insufficient data")
        return None
    
    # Get top features from both groups
    top_features = set(group1_coeffs.head(top_n)['Feature'].tolist() + 
                     group2_coeffs.head(top_n)['Feature'].tolist())
    
    # Create comparison dataframe
    comparison_data = []
    for feature in top_features:
        g1_row = group1_coeffs[group1_coeffs['Feature'] == feature]
        g2_row = group2_coeffs[group2_coeffs['Feature'] == feature]
        
        g1_coef = g1_row['Coefficient'].values[0] if len(g1_row) > 0 else 0
        g2_coef = g2_row['Coefficient'].values[0] if len(g2_row) > 0 else 0
        
        # Simplify feature names for better display
        feature_display = feature.replace('_', ' ').replace('Thalassemia ', '')
        
        comparison_data.append({
            'Feature': feature_display,
            group1_name: g1_coef,
            group2_name: g2_coef,
            'Difference': abs(g1_coef - g2_coef)
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('Difference', ascending=False)
    
    # Reshape data for plotting
    plot_data = comparison_df.melt(id_vars='Feature', 
                                 value_vars=[group1_name, group2_name],
                                 var_name='Group', value_name='Coefficient')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', hue='Group', data=plot_data)
    
    plt.title(f'Feature Importance Comparison: {group1_name} vs {group2_name}', fontsize=16)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Coefficient Value (Impact on Heart Disease Prediction)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.legend(title='Group')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'coefficient_comparison_{group1_name}_{group2_name}.png', dpi=300)
    plt.show()
    
    return comparison_df

# Now visualize the comparisons we're interested in
# 1. Gender comparison
visualize_coefficient_comparison("Male", male_coeffs, "Female", female_coeffs)

# 2. Age comparison (40s vs 60s)
visualize_coefficient_comparison("Age 40s", age_coeffs["40s"], "Age 60s", age_coeffs["60s"])

# 3. Intersectional comparison (Male 50s vs Female 50s)
visualize_coefficient_comparison("Male 50s", intersect_coeffs["Male 50s"], 
                              "Female 50s", intersect_coeffs["Female 50s"])


def visualize_coefficient_differences(group1_name, group1_coeffs, group2_name, group2_coeffs, top_n=8):
    """Create a visualization showing the magnitude of coefficient differences"""
    comparison_df = visualize_coefficient_comparison(group1_name, group1_coeffs, 
                                                   group2_name, group2_coeffs, 
                                                   top_n=15)  # Get more features
    
    if comparison_df is None:
        return
    
    # Take top N differences
    diff_df = comparison_df[['Feature', 'Difference']].sort_values('Difference', ascending=False).head(top_n)
    
    # Create bar chart of differences
    plt.figure(figsize=(10, 8))
    bars = sns.barplot(x='Difference', y='Feature', data=diff_df, palette='viridis')
    
    # Add value labels to bars
    for bar in bars.patches:
        bars.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.2f}', 
                ha='left', va='center')
    
    plt.title(f'Features with Largest Coefficient Differences\n{group1_name} vs {group2_name}', fontsize=16)
    plt.xlabel('Absolute Difference in Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'coefficient_differences_{group1_name}_{group2_name}.png', dpi=300)
    plt.show()

# Visualize the differences
visualize_coefficient_differences("Male", male_coeffs, "Female", female_coeffs)
visualize_coefficient_differences("Age 40s", age_coeffs["40s"], "Age 60s", age_coeffs["60s"])
visualize_coefficient_differences("Male 50s", intersect_coeffs["Male 50s"], 
                               "Female 50s", intersect_coeffs["Female 50s"])