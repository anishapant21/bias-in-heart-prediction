def main():
    """
    Main execution function
    """
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    # Step 2: Create demographic groups
    df = create_demographic_groups(df)
    
    # Step 3: Analyze class balance in different demographic groups
    analyze_class_balance(df, 'Sex')
    analyze_class_balance(df, 'Age Group')
    analyze_class_balance(df, 'Gender_Age_Group')
    
    # Step 4: Prepare data for modeling
    X, y, X_train, X_test, y_train, y_test, X_train_with_demo, X_test_with_demo = prepare_data_with_demographics(df)
    
    # Step 5: Train and evaluate a baseline model
    print("\n===== Training Baseline Model (No SMOTE) =====")
    baseline_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    baseline_model.fit(X_train_scaled, y_train)
    
    # Evaluate baseline model performance
    y_pred = baseline_model.predict(X_test_scaled)
    y_prob = baseline_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nBaseline model performance on test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Step 6: Evaluate baseline model by demographic groups
    baseline_gender_metrics = evaluate_by_demographic_group(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Sex')
    baseline_age_metrics = evaluate_by_demographic_group(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Age Group')
    baseline_intersect_metrics = evaluate_by_demographic_group(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Gender_Age_Group')
    
    # Step 7: Calculate fairness metrics for baseline model
    baseline_fairness = calculate_fairness_metrics(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Sex')
    
    # Step 8: Apply SMOTE for gender-based balancing
    X_gender_resampled, y_gender_resampled, demo_gender_resampled = apply_smote_to_demographic_groups(
        X_train, y_train, X_train_with_demo, 'Sex'
    )
    
    # Step 9: Check demo_gender_resampled has 'Diagnosis' column
    print("\nChecking SMOTE-resampled data columns:")
    print(f"Original demo columns: {list(X_train_with_demo.columns)}")
    print(f"Resampled demo columns: {list(demo_gender_resampled.columns)}")
    print(f"Diagnosis column present: {'Diagnosis' in demo_gender_resampled.columns}")
    
    # Step 10: Add diagnosis column if missing
    if 'Diagnosis' not in demo_gender_resampled.columns:
        print("Adding Diagnosis column to resampled demographic data")
        demo_gender_resampled['Diagnosis'] = y_gender_resampled.values
    
    # Step 11: Train and evaluate model with gender-based SMOTE
    gender_smote_model, gender_smote_scaler = train_evaluate_smote_model(
        X_gender_resampled, y_gender_resampled, X_test, y_test, X_test_with_demo
    )
    
    # Step 12: Get metrics for gender-based SMOTE model
    gender_smote_metrics = evaluate_by_demographic_group(gender_smote_model, X_test, y_test, X_test_with_demo, gender_smote_scaler, 'Sex')
    
    # Step 13: Calculate fairness metrics for gender-based SMOTE model
    gender_smote_fairness = calculate_fairness_metrics(gender_smote_model, X_test, y_test, X_test_with_demo, gender_smote_scaler, 'Sex')
    
    # Step 14: Visualize effect of SMOTE on gender class distribution
    try:
        visualize_smote_effect(X_train_with_demo, demo_gender_resampled, 'Sex', 'Diagnosis')
    except Exception as e:
        print(f"Error visualizing SMOTE effect: {e}")
    
    # Step 15: Apply SMOTE for intersectional balancing
    X_intersect_resampled, y_intersect_resampled, demo_intersect_resampled = apply_smote_to_demographic_groups(
        X_train, y_train, X_train_with_demo, 'Gender_Age_Group'
    )
    
    # Step 16: Add diagnosis column if missing to intersection demographic data
    if 'Diagnosis' not in demo_intersect_resampled.columns:
        print("Adding Diagnosis column to intersectional resampled demographic data")
        demo_intersect_resampled['Diagnosis'] = y_intersect_resampled.values
    
    # Step 17: Visualize effect of SMOTE on intersectional class distribution
    try:
        visualize_smote_effect(X_train_with_demo, demo_intersect_resampled, 'Gender_Age_Group', 'Diagnosis')
    except Exception as e:
        print(f"Error visualizing intersectional SMOTE effect: {e}")
    
    # Step 18: Train and evaluate model with intersectional SMOTE
    intersect_smote_model, intersect_smote_scaler = train_evaluate_smote_model(
        X_intersect_resampled, y_intersect_resampled, X_test, y_test, X_test_with_demo
    )
    
    # Step 19: Get metrics for intersectional SMOTE model
    intersect_smote_metrics = evaluate_by_demographic_group(
        intersect_smote_model, X_test, y_test, X_test_with_demo, 
        intersect_smote_scaler, 'Gender_Age_Group'
    )
    
    # Step 20: Calculate fairness metrics for intersectional SMOTE model
    intersect_smote_fairness = calculate_fairness_metrics(
        intersect_smote_model, X_test, y_test, X_test_with_demo, 
        intersect_smote_scaler, 'Sex'
    )
    
    # Step 21: Compare model performance before and after SMOTE
    gender_comparison = compare_models(baseline_gender_metrics, gender_smote_metrics, 'Sex')
    intersect_comparison = compare_models(baseline_intersect_metrics, intersect_smote_metrics, 'Gender_Age_Group')
    
    # Step 22: Compare fairness metrics
    print("\n===== Fairness Metrics Comparison (Before vs. After SMOTE) =====")
    
    if baseline_fairness and gender_smote_fairness and 'overall' in baseline_fairness and 'overall' in gender_smote_fairness:
        print("\nGender-based fairness metrics:")
        print(f"  Disparate Impact: {baseline_fairness['overall']['disparate_impact']:.4f} → {gender_smote_fairness['overall']['disparate_impact']:.4f}")
        print(f"  Equal Opportunity Diff: {baseline_fairness['overall']['equal_opportunity_diff']:.4f} → {gender_smote_fairness['overall']['equal_opportunity_diff']:.4f}")
        print(f"  Equalized Odds: {baseline_fairness['overall']['equalized_odds']:.4f} → {gender_smote_fairness['overall']['equalized_odds']:.4f}")
        
        # Create fairness comparison plot
        metrics = ['disparate_impact', 'equal_opportunity_diff', 'equalized_odds']
        metric_names = ['Disparate Impact', 'Equal Opportunity Diff', 'Equalized Odds']
        baseline_values = [baseline_fairness['overall'][m] for m in metrics]
        smote_values = [gender_smote_fairness['overall'][m] for m in metrics]
        
        improvement = []
        for i, metric in enumerate(metrics):
            # For disparate impact, closer to 1 is better
            if metric == 'disparate_impact':
                imp = abs(smote_values[i] - 1) < abs(baseline_values[i] - 1)
            # For others, closer to 0 is better
            else:
                imp = abs(smote_values[i]) < abs(baseline_values[i])
            improvement.append('Improved' if imp else 'Worse')
        
        fairness_df = pd.DataFrame({
            'Metric': metric_names,
            'Baseline': baseline_values,
            'SMOTE': smote_values,
            'Status': improvement
        })
        
        plt.figure(figsize=(10, 6))
        fairness_melted = fairness_df.melt(id_vars=['Metric', 'Status'], 
                                        value_vars=['Baseline', 'SMOTE'],
                                        var_name='Model', value_name='Value')
        
        # For plotting purposes
        ideal_values = {'Disparate Impact': 1.0, 'Equal Opportunity Diff': 0.0, 'Equalized Odds': 0.0}
        
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metric_names):
            metric_data = fairness_melted[fairness_melted['Metric'] == metric]
            sns.barplot(x='Model', y='Value', data=metric_data, ax=axes[i])
            
            # Add ideal value line
            ideal = ideal_values[metric]
            axes[i].axhline(y=ideal, color='green', linestyle='--', label='Ideal')
            
            axes[i].set_title(metric)
            axes[i].set_ylim(0 if metric != 'Disparate Impact' else 0.5, 
                           1.5 if metric == 'Disparate Impact' else max(0.5, metric_data['Value'].max() * 1.2))
        
        plt.tight_layout()
        plt.savefig('fairness_metrics_comparison.png', dpi=300)
    
    # Return results
    return {
        'baseline_model': {
            'model': baseline_model,
            'scaler': scaler,
            'metrics': {
                'gender': baseline_gender_metrics,
                'age': baseline_age_metrics,
                'intersectional': baseline_intersect_metrics
            },
            'fairness': baseline_fairness
        },
        'gender_smote_model': {
            'model': gender_smote_model,
            'scaler': gender_smote_scaler,
            'metrics': gender_smote_metrics,
            'fairness': gender_smote_fairness
        },
        'intersect_smote_model': {
            'model': intersect_smote_model,
            'scaler': intersect_smote_scaler,
            'metrics': intersect_smote_metrics,
            'fairness': intersect_smote_fairness
        },
        'comparisons': {
            'gender': gender_comparison,
            'intersect': intersect_comparison
        }
    }

"""Heart Disease Analysis with SMOTE for Underrepresented Groups
This script extends the heart disease analysis by implementing SMOTE to balance underrepresented groups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ----------------- HELPER FUNCTIONS FROM THE PREVIOUS CODE -----------------

def load_and_preprocess_data(file_path='./dataset/heart_disease_uci.csv'):
    """
    Load and preprocess the heart disease dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df = df.drop(['id', 'dataset'], axis=1)
    
    # Rename columns for better readability
    column_mapping = {
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
    df = df.rename(columns=column_mapping)
    
    # Convert diagnosis to binary
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 0 if x == 0 else 1)
    
    # Handle missing values
    print("Missing values before removal:")
    print(df.isnull().sum())
    print(f"Original dataset shape: {df.shape}")
    
    df = df.dropna()
    print("\nDataset shape after removing missing values:")
    print(df.shape)
    
    return df

def create_demographic_groups(df):
    """
    Create demographic groups based on age and gender
    """
    # Set up age groups
    df['Age Group'] = pd.cut(df['Age'], bins=[29, 50, 60, 100], 
                            labels=["30s-40s", "50s", "60+"])
    
    # Create gender-age intersectional groups
    df['Gender_Age_Group'] = df['Sex'].astype(str) + "_" + df['Age Group'].astype(str)
    
    # Print demographic distributions
    print("\nGender distribution:")
    print(df['Sex'].value_counts())
    
    print("\nAge group distribution:")
    print(df['Age Group'].value_counts())
    
    print("\nIntersectional group distribution:")
    print(df['Gender_Age_Group'].value_counts())
    
    return df

# ----------------- SMOTE IMPLEMENTATION -----------------

def analyze_class_balance(df, group_column, diagnosis_column='Diagnosis'):
    """
    Analyze class balance within different demographic groups
    """
    balance_stats = {}
    
    print(f"\n===== Class Balance Analysis by {group_column} =====")
    
    # Overall class balance
    overall_balance = df[diagnosis_column].value_counts()
    overall_ratio = overall_balance[1] / overall_balance[0] if overall_balance[0] > 0 else float('inf')
    
    print(f"Overall class distribution:")
    print(f"  Negative (0): {overall_balance[0]} ({overall_balance[0]/len(df):.2%})")
    print(f"  Positive (1): {overall_balance[1]} ({overall_balance[1]/len(df):.2%})")
    print(f"  Positive/Negative ratio: {overall_ratio:.2f}")
    
    # Class balance by group
    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]
        
        # Skip empty groups
        if len(group_df) == 0:
            print(f"\n{group_column} = {group} (n={len(group_df)}): [No samples]")
            continue
            
        group_balance = group_df[diagnosis_column].value_counts()
        
        # Handle potential missing classes
        neg_count = group_balance.get(0, 0)
        pos_count = group_balance.get(1, 0)
        
        # Calculate ratio (handle division by zero)
        ratio = pos_count / neg_count if neg_count > 0 else float('inf')
        
        balance_stats[group] = {
            'total': len(group_df),
            'negative': neg_count,
            'positive': pos_count,
            'pos_ratio': ratio
        }
        
        print(f"\n{group_column} = {group} (n={len(group_df)}):")
        print(f"  Negative (0): {neg_count} ({neg_count/len(group_df):.2%})")
        print(f"  Positive (1): {pos_count} ({pos_count/len(group_df):.2%})")
        print(f"  Positive/Negative ratio: {ratio:.2f}")
    
    return balance_stats

def prepare_data_with_demographics(df):
    """
    Prepare data for modeling while preserving demographic information
    """
    # Keep demographic columns for subgroup analysis with diagnosis
    X_with_demographics = df.copy()
    
    # Create a version without demographic columns for actual modeling
    X = df.drop(['Diagnosis', 'Age Group', 'Gender_Age_Group'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
    y = df['Diagnosis']
    
    # Split data into train and test sets (using stratified sampling)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(X)), y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Get the features for training/testing (without demographic columns)
    X_train = X.iloc[X_train_idx]
    X_test = X.iloc[X_test_idx]
    
    # Also keep demographic data for analysis (including diagnosis column)
    X_train_with_demo = X_with_demographics.iloc[X_train_idx]
    X_test_with_demo = X_with_demographics.iloc[X_test_idx]
    
    return X, y, X_train, X_test, y_train, y_test, X_train_with_demo, X_test_with_demo

def apply_smote_to_demographic_groups(X_train, y_train, X_train_with_demo, group_column, random_state=42):
    """
    Apply SMOTE separately to each demographic group to balance classes
    """
    print(f"\n===== Applying SMOTE to balance classes within {group_column} groups =====")
    
    # Store the resampled data
    X_resampled_parts = []
    y_resampled_parts = []
    demo_resampled_parts = []
    
    # Process each demographic group separately
    for group in X_train_with_demo[group_column].unique():
        # Get indices for this group
        group_indices = np.where(X_train_with_demo[group_column] == group)[0]
        
        if len(group_indices) < 10:
            print(f"Skipping {group_column}={group} - insufficient samples ({len(group_indices)})")
            # Add the original samples without resampling
            X_resampled_parts.append(X_train.iloc[group_indices])
            y_resampled_parts.append(y_train.iloc[group_indices])
            demo_resampled_parts.append(X_train_with_demo.iloc[group_indices])
            continue
        
        # Get data for this group
        X_group = X_train.iloc[group_indices]
        y_group = y_train.iloc[group_indices]
        demo_group = X_train_with_demo.iloc[group_indices]
        
        # Check class balance
        class_counts = y_group.value_counts()
        print(f"\n{group_column}={group} before SMOTE:")
        print(f"  Class 0: {class_counts.get(0, 0)}")
        print(f"  Class 1: {class_counts.get(1, 0)}")
        
        # Only apply SMOTE if both classes are present
        if len(class_counts) > 1 and min(class_counts) >= 3:  # Need at least 3 samples for SMOTE with k=2
            # Apply SMOTE
            k_neighbors = min(5, min(class_counts)-1)  # Adjust k based on class size
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_group_resampled, y_group_resampled = smote.fit_resample(X_group, y_group)
            
            # We need to reconstruct the demographic information for the synthetic samples
            demo_columns = demo_group.columns
            demo_group_resampled = pd.DataFrame(index=range(len(y_group_resampled)), columns=demo_columns)
            
            # Original samples keep their demographic info
            demo_group_resampled.iloc[:len(demo_group)] = demo_group.values
            
            # Synthetic samples get the demographics of this group
            for i in range(len(demo_group), len(demo_group_resampled)):
                for col in demo_columns:
                    if col == group_column or col == 'Gender_Age_Group' or col == 'Age Group':
                        # For the grouping column and related columns, use the current group value
                        demo_group_resampled.iloc[i][col] = group
                    elif col == 'Diagnosis':
                        # Set diagnosis to match the resampled y value
                        demo_group_resampled.iloc[i][col] = y_group_resampled[i]
                    else:
                        # For other demographic columns, use the mode (most common) value
                        demo_group_resampled.iloc[i][col] = demo_group[col].mode()[0]
            
            # Print resampled class distribution
            resampled_counts = pd.Series(y_group_resampled).value_counts()
            print(f"{group_column}={group} after SMOTE:")
            print(f"  Class 0: {resampled_counts.get(0, 0)}")
            print(f"  Class 1: {resampled_counts.get(1, 0)}")
            
            # Convert resampled data to DataFrames with proper indices
            X_group_resampled = pd.DataFrame(X_group_resampled, columns=X_group.columns)
            y_group_resampled = pd.Series(y_group_resampled)
            
            X_resampled_parts.append(X_group_resampled)
            y_resampled_parts.append(y_group_resampled)
            demo_resampled_parts.append(demo_group_resampled)
        else:
            print(f"Skipping SMOTE for {group_column}={group} - only one class present or too few samples")
            # Add the original samples without resampling
            X_resampled_parts.append(X_group)
            y_resampled_parts.append(y_group)
            demo_resampled_parts.append(demo_group)
    
    # Combine all parts
    X_resampled = pd.concat(X_resampled_parts, ignore_index=True)
    y_resampled = pd.concat(y_resampled_parts, ignore_index=True)
    demo_resampled = pd.concat(demo_resampled_parts, ignore_index=True)
    
    # Print overall results
    print("\nOverall resampled dataset:")
    print(f"  Original size: {len(X_train)} samples")
    print(f"  Resampled size: {len(X_resampled)} samples")
    print(f"  Class 0: {sum(y_resampled == 0)} ({sum(y_resampled == 0)/len(y_resampled):.2%})")
    print(f"  Class 1: {sum(y_resampled == 1)} ({sum(y_resampled == 1)/len(y_resampled):.2%})")
    
    return X_resampled, y_resampled, demo_resampled

def train_evaluate_smote_model(X_train, y_train, X_test, y_test, X_test_with_demo):
    """
    Train and evaluate a model using the SMOTE-resampled data
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n===== Training Model on SMOTE-Balanced Data =====")
    model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate overall performance
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nOverall model performance on test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Evaluate performance by demographic groups
    evaluate_by_demographic_group(model, X_test, y_test, X_test_with_demo, scaler, 'Sex')
    evaluate_by_demographic_group(model, X_test, y_test, X_test_with_demo, scaler, 'Age Group')
    evaluate_by_demographic_group(model, X_test, y_test, X_test_with_demo, scaler, 'Gender_Age_Group')
    
    return model, scaler

def evaluate_by_demographic_group(model, X_test, y_test, X_test_with_demo, scaler, group_column):
    """
    Evaluate model performance for each demographic group
    """
    print(f"\n===== Model Performance by {group_column} =====")
    group_metrics = {}
    
    for group in X_test_with_demo[group_column].unique():
        # Get indices for this group
        group_mask = X_test_with_demo[group_column] == group
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < 5:
            print(f"Skipping {group_column}={group} - insufficient test samples ({len(group_indices)})")
            continue
        
        # Get test data for this group
        X_group_test = X_test.iloc[group_indices]
        y_group_test = y_test.iloc[group_indices]
        
        # Make predictions
        X_group_scaled = scaler.transform(X_group_test)
        y_group_pred = model.predict(X_group_scaled)
        y_group_prob = model.predict_proba(X_group_scaled)[:, 1]
        
        # Calculate metrics (handling edge cases)
        metrics = {
            'size': len(group_indices),
            'accuracy': accuracy_score(y_group_test, y_group_pred),
            'precision': precision_score(y_group_test, y_group_pred, zero_division=0),
            'recall': recall_score(y_group_test, y_group_pred, zero_division=0),
            'f1': f1_score(y_group_test, y_group_pred, zero_division=0)
        }
        
        # Only calculate ROC AUC if both classes are present
        if len(np.unique(y_group_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_group_test, y_group_prob)
        
        group_metrics[group] = metrics
        
        # Print results
        print(f"\nPerformance for {group_column}={group} (n={metrics['size']}):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    return group_metrics

def compare_models(original_metrics, smote_metrics, group_column):
    """
    Compare model performance before and after SMOTE
    """
    print(f"\n===== Performance Comparison Before vs. After SMOTE ({group_column}) =====")
    
    # Ensure we have metrics to compare
    if not original_metrics or not smote_metrics:
        print("Insufficient metrics for comparison")
        return
    
    # Prepare data for plotting
    groups = []
    original_values = []
    smote_values = []
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Extract metrics for each group and model
    for metric in metrics:
        for group in original_metrics:
            if group in smote_metrics and metric in original_metrics[group] and metric in smote_metrics[group]:
                groups.append(f"{group} ({metric})")
                original_values.append(original_metrics[group][metric])
                smote_values.append(smote_metrics[group][metric])
    
    # Create DataFrame for plotting
    comparison_df = pd.DataFrame({
        'Group': groups,
        'Original Model': original_values,
        'SMOTE Model': smote_values,
        'Improvement': np.array(smote_values) - np.array(original_values)
    })
    
    # Print comparison table
    print("\nMetric differences (SMOTE - Original):")
    print(comparison_df[['Group', 'Original Model', 'SMOTE Model', 'Improvement']].to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(14, 10))
    comparison_df_melted = comparison_df.melt(id_vars='Group', value_vars=['Original Model', 'SMOTE Model'], 
                                            var_name='Model', value_name='Score')
    
    ax = sns.barplot(x='Group', y='Score', hue='Model', data=comparison_df_melted)
    
    plt.title(f'Performance Comparison Before vs. After SMOTE ({group_column})', fontsize=16)
    plt.xlabel('Group and Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'smote_comparison_{group_column}.png', dpi=300)
    
    # Plot improvement
    plt.figure(figsize=(12, 8))
    
    improvement_df = comparison_df[['Group', 'Improvement']].copy()
    improvement_df['Color'] = improvement_df['Improvement'].apply(lambda x: 'green' if x > 0 else 'red')
    
    ax = sns.barplot(x='Group', y='Improvement', data=improvement_df, palette=improvement_df['Color'])
    
    plt.title(f'Performance Improvement After SMOTE ({group_column})', fontsize=16)
    plt.xlabel('Group and Metric', fontsize=12)
    plt.ylabel('Improvement (SMOTE - Original)', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'smote_improvement_{group_column}.png', dpi=300)
    
    return comparison_df

def calculate_fairness_metrics(model, X_test, y_test, X_test_with_demo, scaler, protected_attribute='Sex'):
    """
    Calculate fairness metrics for the model
    """
    print(f"\n===== Fairness Metrics for {protected_attribute} =====")
    
    # Get unique groups
    groups = X_test_with_demo[protected_attribute].unique()
    
    if len(groups) < 2:
        print(f"Need at least 2 groups for fairness analysis, found: {groups}")
        return None
    
    # Make predictions on the test set
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Dictionary to store metrics by group
    fairness_metrics = {}
    
    # Calculate overall selection rate (proportion of positive predictions)
    overall_selection_rate = sum(y_pred == 1) / len(y_pred)
    print(f"Overall selection rate: {overall_selection_rate:.4f}")
    
    # Calculate metrics for each group
    for group in groups:
        # Get indices for this group
        group_mask = X_test_with_demo[protected_attribute] == group
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < 5:
            print(f"Skipping {protected_attribute}={group} - insufficient test samples")
            continue
        
        # Get predictions for this group
        y_group_true = y_test.iloc[group_indices]
        y_group_pred = y_pred[group_indices]
        
        # Calculate selection rate for this group
        selection_rate = sum(y_group_pred == 1) / len(y_group_pred)
        
        # Calculate false positive and false negative rates
        if sum(y_group_true == 0) > 0:
            fpr = sum((y_group_pred == 1) & (y_group_true == 0)) / sum(y_group_true == 0)
        else:
            fpr = float('nan')
            
        if sum(y_group_true == 1) > 0:
            fnr = sum((y_group_pred == 0) & (y_group_true == 1)) / sum(y_group_true == 1)
            tpr = 1 - fnr  # True positive rate = 1 - false negative rate
        else:
            fnr = float('nan')
            tpr = float('nan')
        
        # Store metrics
        fairness_metrics[group] = {
            'selection_rate': selection_rate,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positive_rate': tpr
        }
        
        print(f"\nMetrics for {protected_attribute}={group}:")
        print(f"  Selection rate: {selection_rate:.4f}")
        print(f"  False positive rate: {fpr:.4f}")
        print(f"  False negative rate: {fnr:.4f}")
        print(f"  True positive rate: {tpr:.4f}")
    
    # Calculate group fairness metrics (for binary protected attributes)
    if len(groups) == 2:
        group_a, group_b = groups
        
        if group_a in fairness_metrics and group_b in fairness_metrics:
            # Disparate impact (compare selection rates)
            sr_a = fairness_metrics[group_a]['selection_rate']
            sr_b = fairness_metrics[group_b]['selection_rate']
            
            # Calculate disparate impact ratio (put smaller value in numerator for ratio <= 1)
            if sr_a <= sr_b:
                di = sr_a / sr_b if sr_b > 0 else float('nan')
            else:
                di = sr_b / sr_a if sr_a > 0 else float('nan')
            
            # Equal opportunity difference (difference in true positive rates)
            tpr_a = fairness_metrics[group_a]['true_positive_rate']
            tpr_b = fairness_metrics[group_b]['true_positive_rate']
            eod = abs(tpr_a - tpr_b)
            
            # Equalized odds (average difference in FPR and TPR)
            fpr_a = fairness_metrics[group_a]['false_positive_rate']
            fpr_b = fairness_metrics[group_b]['false_positive_rate']
            eo = (abs(fpr_a - fpr_b) + abs(tpr_a - tpr_b)) / 2
            
            print("\nGroup Fairness Metrics:")
            print(f"  Disparate Impact: {di:.4f} (closer to 1 is better, <0.8 may indicate bias)")
            print(f"  Equal Opportunity Difference: {eod:.4f} (closer to 0 is better)")
            print(f"  Equalized Odds: {eo:.4f} (closer to 0 is better)")
            
            # Store overall fairness metrics
            fairness_metrics['overall'] = {
                'disparate_impact': di,
                'equal_opportunity_diff': eod,
                'equalized_odds': eo
            }
    
    return fairness_metrics

# ----------------- MAIN EXECUTION -----------------

def visualize_smote_effect(original_train, smote_train, demographic_col, diagnosis_col='Diagnosis'):
    """
    Visualize the effect of SMOTE on class distribution within demographic groups
    """
    print(f"\n===== Visualizing SMOTE Effect on {demographic_col} Groups =====")
    
    # Check if diagnosis column exists
    if diagnosis_col not in original_train.columns:
        print(f"Error: '{diagnosis_col}' column not found in original data")
        return None
        
    if diagnosis_col not in smote_train.columns:
        print(f"Error: '{diagnosis_col}' column not found in SMOTE data")
        return None
    
    # Get unique demographic groups
    demographics = sorted(original_train[demographic_col].unique())
    
    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors for visualization
    colors = {'0': 'skyblue', '1': 'salmon'}
    
    # Original data distribution
    original_data = []
    for demo in demographics:
        demo_df = original_train[original_train[demographic_col] == demo]
        class_counts = demo_df[diagnosis_col].value_counts().to_dict()
        # Ensure both classes are represented
        for cls in [0, 1]:
            original_data.append({
                'Demographic': demo,
                'Class': str(cls),
                'Count': class_counts.get(cls, 0),
                'Type': 'Original'
            })
    
    # SMOTE data distribution
    smote_data = []
    for demo in demographics:
        demo_df = smote_train[smote_train[demographic_col] == demo]
        class_counts = demo_df[diagnosis_col].value_counts().to_dict()
        # Ensure both classes are represented
        for cls in [0, 1]:
            smote_data.append({
                'Demographic': demo,
                'Class': str(cls),
                'Count': class_counts.get(cls, 0),
                'Type': 'SMOTE'
            })
    
    # Convert to DataFrames
    orig_df = pd.DataFrame(original_data)
    smote_df = pd.DataFrame(smote_data)
    
    # Plot original data
    sns.barplot(x='Demographic', y='Count', hue='Class', data=orig_df, ax=axes[0], palette=colors)
    axes[0].set_title(f'Original Class Distribution by {demographic_col}', fontsize=14)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xlabel(demographic_col, fontsize=12)
    
    # Plot SMOTE data
    sns.barplot(x='Demographic', y='Count', hue='Class', data=smote_df, ax=axes[1], palette=colors)
    axes[1].set_title(f'After SMOTE Class Distribution by {demographic_col}', fontsize=14)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xlabel(demographic_col, fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(f'smote_effect_{demographic_col}.png', dpi=300)
    
    # Calculate and visualize the class ratios
    ratio_data = []
    
    for demo in demographics:
        # Original data ratio
        orig_demo = orig_df[orig_df['Demographic'] == demo]
        if len(orig_demo) >= 2:
            pos_count = orig_demo[orig_demo['Class'] == '1']['Count'].values[0]
            neg_count = orig_demo[orig_demo['Class'] == '0']['Count'].values[0]
            orig_ratio = pos_count / max(1, neg_count)  # Avoid division by zero
            ratio_data.append({
                'Demographic': demo,
                'Ratio': orig_ratio,
                'Type': 'Original'
            })
        
        # SMOTE data ratio
        smote_demo = smote_df[smote_df['Demographic'] == demo]
        if len(smote_demo) >= 2:
            pos_count = smote_demo[smote_demo['Class'] == '1']['Count'].values[0]
            neg_count = smote_demo[smote_demo['Class'] == '0']['Count'].values[0]
            smote_ratio = pos_count / max(1, neg_count)  # Avoid division by zero
            ratio_data.append({
                'Demographic': demo,
                'Ratio': smote_ratio,
                'Type': 'SMOTE'
            })
    
    # Create ratio plot
    ratio_df = pd.DataFrame(ratio_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Demographic', y='Ratio', hue='Type', data=ratio_df)
    plt.title(f'Positive/Negative Class Ratio by {demographic_col}', fontsize=14)
    plt.ylabel('Ratio (Positive/Negative)', fontsize=12)
    plt.xlabel(demographic_col, fontsize=12)
    plt.axhline(y=1.0, color='green', linestyle='--', label='Balanced (1:1)')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f'smote_ratio_{demographic_col}.png', dpi=300)
    
    return ratio_df

def main():
    """
    Main execution function
    """
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    # Step 2: Create demographic groups
    df = create_demographic_groups(df)
    
    # Step 3: Analyze class balance in different demographic groups
    analyze_class_balance(df, 'Sex')
    analyze_class_balance(df, 'Age Group')
    analyze_class_balance(df, 'Gender_Age_Group')
    
    # Step 4: Prepare data for modeling
    X, y, X_train, X_test, y_train, y_test, X_train_with_demo, X_test_with_demo = prepare_data_with_demographics(df)
    
    # Step 5: Train and evaluate a baseline model
    print("\n===== Training Baseline Model (No SMOTE) =====")
    baseline_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    baseline_model.fit(X_train_scaled, y_train)
    
    # Evaluate baseline model performance
    y_pred = baseline_model.predict(X_test_scaled)
    y_prob = baseline_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nBaseline model performance on test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Step 6: Evaluate baseline model by demographic groups
    baseline_gender_metrics = evaluate_by_demographic_group(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Sex')
    baseline_age_metrics = evaluate_by_demographic_group(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Age Group')
    baseline_intersect_metrics = evaluate_by_demographic_group(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Gender_Age_Group')
    
    # Step 7: Calculate fairness metrics for baseline model
    baseline_fairness = calculate_fairness_metrics(baseline_model, X_test, y_test, X_test_with_demo, scaler, 'Sex')
    
    # Step 8: Apply SMOTE for gender-based balancing
    X_gender_resampled, y_gender_resampled, demo_gender_resampled = apply_smote_to_demographic_groups(
        X_train, y_train, X_train_with_demo, 'Sex'
    )
    
    # Step 9: Train and evaluate model with gender-based SMOTE
    gender_smote_model, gender_smote_scaler = train_evaluate_smote_model(
        X_gender_resampled, y_gender_resampled, X_test, y_test, X_test_with_demo
    )
    
    # Step 10: Get metrics for gender-based SMOTE model
    gender_smote_metrics = evaluate_by_demographic_group(gender_smote_model, X_test, y_test, X_test_with_demo, gender_smote_scaler, 'Sex')
    
    # Step 11: Calculate fairness metrics for gender-based SMOTE model
    gender_smote_fairness = calculate_fairness_metrics(gender_smote_model, X_test, y_test, X_test_with_demo, gender_smote_scaler, 'Sex')
    
    # Step 12: Visualize effect of SMOTE on gender class distribution
    visualize_smote_effect(X_train_with_demo, demo_gender_resampled, 'Sex', 'Diagnosis')
    
    # Step 13: Apply SMOTE for intersectional balancing
    X_intersect_resampled, y_intersect_resampled, demo_intersect_resampled = apply_smote_to_demographic_groups(
        X_train, y_train, X_train_with_demo, 'Gender_Age_Group'
    )
    
    # Step 14: Visualize effect of SMOTE on intersectional class distribution
    visualize_smote_effect(X_train_with_demo, demo_intersect_resampled, 'Gender_Age_Group', 'Diagnosis')
    
    # Step 15: Train and evaluate model with intersectional SMOTE
    intersect_smote_model, intersect_smote_scaler = train_evaluate_smote_model(
        X_intersect_resampled, y_intersect_resampled, X_test, y_test, X_test_with_demo
    )
    
    # Step 16: Get metrics for intersectional SMOTE model
    intersect_smote_metrics = evaluate_by_demographic_group(intersect_smote_model, X_test, y_test, X_test_with_demo, intersect_smote_scaler, 'Gender_Age_Group')
    
    # Step 17: Calculate fairness metrics for intersectional SMOTE model
    intersect_smote_fairness = calculate_fairness_metrics(intersect_smote_model, X_test, y_test, X_test_with_demo, intersect_smote_scaler, 'Sex')
    
    # Step 18: Compare model performance before and after SMOTE
    gender_comparison = compare_models(baseline_gender_metrics, gender_smote_metrics, 'Sex')
    intersect_comparison = compare_models(baseline_intersect_metrics, intersect_smote_metrics, 'Gender_Age_Group')
    
    # Step 16: Compare fairness metrics
    print("\n===== Fairness Metrics Comparison (Before vs. After SMOTE) =====")
    
    if baseline_fairness and gender_smote_fairness and 'overall' in baseline_fairness and 'overall' in gender_smote_fairness:
        print("\nGender-based fairness metrics:")
        print(f"  Disparate Impact: {baseline_fairness['overall']['disparate_impact']:.4f} → {gender_smote_fairness['overall']['disparate_impact']:.4f}")
        print(f"  Equal Opportunity Diff: {baseline_fairness['overall']['equal_opportunity_diff']:.4f} → {gender_smote_fairness['overall']['equal_opportunity_diff']:.4f}")
        print(f"  Equalized Odds: {baseline_fairness['overall']['equalized_odds']:.4f} → {gender_smote_fairness['overall']['equalized_odds']:.4f}")
        
        # Create fairness comparison plot
        metrics = ['disparate_impact', 'equal_opportunity_diff', 'equalized_odds']
        metric_names = ['Disparate Impact', 'Equal Opportunity Diff', 'Equalized Odds']
        baseline_values = [baseline_fairness['overall'][m] for m in metrics]
        smote_values = [gender_smote_fairness['overall'][m] for m in metrics]
        
        improvement = []
        for i, metric in enumerate(metrics):
            # For disparate impact, closer to 1 is better
            if metric == 'disparate_impact':
                imp = abs(smote_values[i] - 1) < abs(baseline_values[i] - 1)
            # For others, closer to 0 is better
            else:
                imp = abs(smote_values[i]) < abs(baseline_values[i])
            improvement.append('Improved' if imp else 'Worse')
        
        fairness_df = pd.DataFrame({
            'Metric': metric_names,
            'Baseline': baseline_values,
            'SMOTE': smote_values,
            'Status': improvement
        })
        
        plt.figure(figsize=(10, 6))
        fairness_melted = fairness_df.melt(id_vars=['Metric', 'Status'], 
                                        value_vars=['Baseline', 'SMOTE'],
                                        var_name='Model', value_name='Value')
        
        # For plotting purposes
        ideal_values = {'Disparate Impact': 1.0, 'Equal Opportunity Diff': 0.0, 'Equalized Odds': 0.0}
        
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metric_names):
            metric_data = fairness_melted[fairness_melted['Metric'] == metric]
            sns.barplot(x='Model', y='Value', data=metric_data, ax=axes[i])
            
            # Add ideal value line
            ideal = ideal_values[metric]
            axes[i].axhline(y=ideal, color='green', linestyle='--', label='Ideal')
            
            axes[i].set_title(metric)
            axes[i].set_ylim(0 if metric != 'Disparate Impact' else 0.5, 
                           1.5 if metric == 'Disparate Impact' else max(0.5, metric_data['Value'].max() * 1.2))
        
        plt.tight_layout()
        plt.savefig('fairness_metrics_comparison.png', dpi=300)
    
    # Return results
    return {
        'baseline_model': {
            'model': baseline_model,
            'scaler': scaler,
            'metrics': {
                'gender': baseline_gender_metrics,
                'age': baseline_age_metrics,
                'intersectional': baseline_intersect_metrics
            },
            'fairness': baseline_fairness
        },
        'gender_smote_model': {
            'model': gender_smote_model,
            'scaler': gender_smote_scaler,
            'metrics': gender_smote_metrics,
            'fairness': gender_smote_fairness
        },
        'intersect_smote_model': {
            'model': intersect_smote_model,
            'scaler': intersect_smote_scaler,
            'metrics': intersect_smote_metrics
        },
        'comparisons': {
            'gender': gender_comparison,
            'intersect': intersect_comparison
        }
    }

if __name__ == "__main__":
    results = main()