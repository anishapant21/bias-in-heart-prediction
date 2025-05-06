import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer

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
df['Age Group'] = pd.cut(df['Age'], bins=[29, 50, 60, 100], 
                        labels=["30s-40s", "50s", "60+"])

# Create gender-age intersectional groups
df['Gender_Age_Group'] = df['Sex'].astype(str) + "_" + df['Age Group'].astype(str)

# Print the distribution of our demographic groups
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
for age_group in ["30s-40s", "50s", "60+"]:  # Focus on groups with more samples
    age_coeffs[age_group] = analyze_coefficients_for_subgroup(
        df, f"Age {age_group}", df['Age Group'] == age_group
    )

# Adjust intersectional group definitions based on what we find
intersectional_groups = [
    ("Male 30s-40s", (male_condition) & (df['Age Group'] == "30s-40s")),
    ("Male 50s", (male_condition) & (df['Age Group'] == "50s")),
    ("Male 60+", (male_condition) & (df['Age Group'] == "60+")),
    ("Female 30s-40s", (female_condition) & (df['Age Group'] == "30s-40s")),
    ("Female 50s", (female_condition) & (df['Age Group'] == "50s")),
    ("Female 60+", (female_condition) & (df['Age Group'] == "60+"))

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
    # plt.show()
    
    return comparison_df

# Now visualize the comparisons we're interested in
# 1. Gender comparison
visualize_coefficient_comparison("Male", male_coeffs, "Female", female_coeffs)

# 2. Age comparison (40s vs 60s)
visualize_coefficient_comparison("Age 40s", age_coeffs["30s-40s"], "Age 60s", age_coeffs["60+"])

# 3. Intersectional comparison (Male 50s vs Female 50s)
visualize_coefficient_comparison("Male 50s", intersect_coeffs["Male 50s"], 
                              "Female 50s", intersect_coeffs["Female 50s"])


# ------------------------------------- SINGLE MODEL --------------------------------------------------------------------------------

# Import necessary libraries for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ------------------------------------- SINGLE MODEL WITH CROSS-VALIDATION --------------------------------------------------------------------------------

# Import necessary libraries for model evaluation and cross-validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV

# Prepare data for modeling
X = df.drop(['Diagnosis', 'Age Group', 'Gender_Age_Group'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
y = df['Diagnosis']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Using stratified k-fold to maintain class balance

# Define model
model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)

# Perform cross-validation and get scores
print("\n===== Cross-Validation Results =====")
cv_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
cv_precision = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='precision')
cv_recall = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='recall')
cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
cv_roc_auc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

print(f"Cross-validated Accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
print(f"Cross-validated Precision: {cv_precision.mean():.4f} (±{cv_precision.std():.4f})")
print(f"Cross-validated Recall: {cv_recall.mean():.4f} (±{cv_recall.std():.4f})")
print(f"Cross-validated F1 Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
print(f"Cross-validated ROC AUC: {cv_roc_auc.mean():.4f} (±{cv_roc_auc.std():.4f})")

# Define parameter grid to search
param_grid = {
    'C': [0.01, 0.1, 1.0, 5.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # liblinear supports both l1 and l2
}

# Set up grid search with cross-validation
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=cv,
    scoring='roc_auc',  # You can change this to 'accuracy', 'f1', etc.
    n_jobs=-1  # Use all available cores
)

# Fit grid search
print("\n===== Performing Hyperparameter Tuning with Cross-Validation =====")
grid_search.fit(X_train_scaled, y_train)

# Get best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use the best model for final evaluation
model = grid_search.best_estimator_

# Train the final model on the full training data
print("\n===== Training Final Model on Full Training Dataset =====")
model.fit(X_train_scaled, y_train)

# Evaluate overall model performance on the test set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

print("\nFinal Model Performance on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Function to evaluate model performance on a specific subgroup
def evaluate_subgroup_performance(subgroup_name, subgroup_condition):
    """Evaluate model performance on a specific demographic subgroup"""
    # Get indices of test set samples in this subgroup
    subgroup_indices = X_test[subgroup_condition].index
    
    if len(subgroup_indices) < 10:
        print(f"Skipping {subgroup_name} due to insufficient test samples")
        return None
    
    # Get predictions for this subgroup
    X_sub_test = X_test.loc[subgroup_indices]
    y_sub_test = y_test.loc[subgroup_indices]
    X_sub_test_scaled = scaler.transform(X_sub_test)
    
    y_sub_pred = model.predict(X_sub_test_scaled)
    y_sub_prob = model.predict_proba(X_sub_test_scaled)[:, 1]
    
    # Calculate performance metrics
    metrics = {
        'subgroup': subgroup_name,
        'size': len(subgroup_indices),
        'accuracy': accuracy_score(y_sub_test, y_sub_pred),
        'precision': precision_score(y_sub_test, y_sub_pred, zero_division=0),
        'recall': recall_score(y_sub_test, y_sub_pred, zero_division=0),
        'f1': f1_score(y_sub_test, y_sub_pred, zero_division=0)
    }
    
    # Add ROC AUC if both classes are present
    if len(np.unique(y_sub_test)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_sub_test, y_sub_prob)
    
    print(f"\nPerformance for {subgroup_name} (n={metrics['size']}):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

# Evaluate performance across gender groups
print("\n===== Model Performance by Gender =====")
male_metrics = evaluate_subgroup_performance("Male", male_condition)
female_metrics = evaluate_subgroup_performance("Female", female_condition)

# Evaluate performance across age groups
print("\n===== Model Performance by Age Group =====")
age_metrics = {}
for age_group in ["30s-40s", "50s", "60+"]:
    age_condition = df['Age Group'] == age_group
    age_metrics[age_group] = evaluate_subgroup_performance(f"Age {age_group}", age_condition)

# Evaluate performance across intersectional groups
print("\n===== Model Performance by Intersectional Group =====")
intersect_metrics = {}
for group_name, condition in intersectional_groups:
    intersect_metrics[group_name] = evaluate_subgroup_performance(group_name, condition)

# Visualize performance metrics across groups
def plot_performance_comparison(metrics_list, metric_name="accuracy", title=None):
    """Plot comparison of a specific performance metric across groups"""
    # Filter out None values
    metrics_list = [m for m in metrics_list if m is not None]
    
    if not metrics_list:
        print(f"No metrics to visualize for {metric_name}")
        return
    
    # Create dataframe for plotting
    plot_data = pd.DataFrame([
        {'Group': m['subgroup'], metric_name: m[metric_name]}
        for m in metrics_list if metric_name in m
    ])
    
    if len(plot_data) < 2:
        print(f"Not enough groups to compare for {metric_name}")
        return
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='Group', y=metric_name, data=plot_data)
    
    # Add value labels
    for bar in bars.patches:
        bars.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.01, 
                f'{bar.get_height():.3f}', 
                ha='center')
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f'{metric_name.capitalize()} Comparison Across Groups', fontsize=16)
    
    plt.ylim(0, max(plot_data[metric_name]) * 1.2)  # Add some space for the labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'performance_{metric_name}.png', dpi=300)
    # plt.show()

# Combine all metrics for visualization
all_metrics = [male_metrics, female_metrics]
all_metrics.extend([m for m in age_metrics.values() if m is not None])
all_metrics.extend([m for m in intersect_metrics.values() if m is not None])

# Plot comparisons of different metrics
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    plot_performance_comparison(all_metrics, metric)

# Plot ROC AUC separately (since not all groups might have it)
roc_metrics = [m for m in all_metrics if m is not None and 'roc_auc' in m]
if roc_metrics:
    plot_performance_comparison(roc_metrics, 'roc_auc')