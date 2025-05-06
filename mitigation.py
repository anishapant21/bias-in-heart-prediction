# mitigation_approach1.py - Demographic-Specific Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and prepare the data
def load_and_prepare_data():
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
    
    # Convert diagnosis to binary (0 = no disease, 1 = disease)
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 0 if x == 0 else 1)
    
    # Handle missing values
    print("Missing values before removal:")
    print(df.isnull().sum())
    print(f"Original dataset shape: {df.shape}")
    
    df = df.dropna()
    print("\nDataset shape after removing missing values:")
    print(df.shape)
    
    # Create age groups
    df['Age Group'] = pd.cut(df['Age'], bins=[29, 40, 50, 60, 70, 100], 
                            labels=["30s", "40s", "50s", "60s", "70+"])
    
    # Create gender-age intersectional groups
    df['Gender_Age_Group'] = df['Sex'].astype(str) + "_" + df['Age Group'].astype(str)
    
    # Print demographics
    print("\nGender distribution:")
    print(df['Sex'].value_counts())
    
    print("\nAge group distribution:")
    print(df['Age Group'].value_counts())
    
    print("\nIntersectional group distribution:")
    print(df['Gender_Age_Group'].value_counts())
    
    return df

# Define demographic groups and prepare data for modeling
def prepare_data_for_modeling(df):
    # Define the gender conditions
    male_condition = df['Sex'] == 'Male'
    female_condition = df['Sex'] == 'Female'
    
    # Print demographics for verification
    print("\nMales selected:", male_condition.sum())
    print("Females selected:", female_condition.sum())
    
    # Define intersectional groups
    intersectional_groups = [
        ("Male 50s", (male_condition) & (df['Age Group'] == "50s")),
        ("Male 40s", (male_condition) & (df['Age Group'] == "40s")),
        ("Female 50s", (female_condition) & (df['Age Group'] == "50s"))
    ]
    
    # Verify group sizes
    for name, condition in intersectional_groups:
        print(f"{name} size: {condition.sum()}")
    
    # Prepare data for modeling
    X = df.drop(['Diagnosis', 'Age Group', 'Gender_Age_Group'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
    y = df['Diagnosis']
    
    return X, y, male_condition, female_condition, intersectional_groups

# Step 1: Create Demographic-Specific Models
def create_demographic_models(df, X, y, male_condition, female_condition):
    # Define the demographic groups we want to create models for
    demographic_groups = [
        ('All', None),  # Baseline model for everyone
        ('Male', male_condition),
        ('Female', female_condition),
        ('Age_40s', df['Age Group'] == '40s'),
        ('Age_50s', df['Age Group'] == '50s'),
        ('Age_60s', df['Age Group'] == '60s'),
        ('Male_50s', (male_condition) & (df['Age Group'] == '50s')),
        ('Female_50s', (female_condition) & (df['Age Group'] == '50s')),
    ]
    
    # Dictionary to store our trained models
    demographic_models = {}
    
    # Train a specific model for each demographic group
    for group_name, group_mask in demographic_groups:
        print(f"\n===== Training model for {group_name} group =====")
        
        if group_mask is None:
            # This is our baseline model (all data)
            X_group = X
            y_group = y
        else:
            # Filter data for this specific demographic group
            X_group = X[group_mask.reindex(X.index, fill_value=False)]
            y_group = y[group_mask.reindex(y.index, fill_value=False)]
        
        # Check if we have enough samples
        if len(X_group) < 30:  # Minimum threshold for reliable model training
            print(f"Skipping {group_name} due to insufficient samples ({len(X_group)} < 30)")
            continue
        
        print(f"Training with {len(X_group)} samples")
        
        # Split into train/test sets (maintaining demographic-specific data)
        X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(
            X_group, y_group, test_size=0.3, random_state=42, stratify=y_group
        )
        
        # Standardize features
        scaler_group = StandardScaler()
        X_train_group_scaled = scaler_group.fit_transform(X_train_group)
        X_test_group_scaled = scaler_group.transform(X_test_group)
        
        # Train the model
        model_group = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
        model_group.fit(X_train_group_scaled, y_train_group)
        
        # Evaluate on test set
        y_pred_group = model_group.predict(X_test_group_scaled)
        
        print(f"Performance for {group_name} model:")
        print(f"Accuracy: {accuracy_score(y_test_group, y_pred_group):.4f}")
        print(f"Precision: {precision_score(y_test_group, y_pred_group):.4f}")
        print(f"Recall: {recall_score(y_test_group, y_pred_group):.4f}")
        print(f"F1 Score: {f1_score(y_test_group, y_pred_group):.4f}")
        
        # Store the model and scaler for this demographic group
        demographic_models[group_name] = {
            'model': model_group,
            'scaler': scaler_group,
            'train_performance': {
                'accuracy': accuracy_score(y_test_group, y_pred_group),
                'precision': precision_score(y_test_group, y_pred_group),
                'recall': recall_score(y_test_group, y_pred_group),
                'f1': f1_score(y_test_group, y_pred_group)
            }
        }
    
    return demographic_models

# Step 2: Create Model Selection Function
def predict_with_demographic_models(X_data, demographics, demographic_models):
    """
    Make predictions using the appropriate demographic-specific model.
    
    Args:
        X_data: Features to predict on
        demographics: DataFrame with demographic info (Sex, Age Group)
        demographic_models: Dictionary of trained models
    
    Returns:
        Array of predictions
    """
    predictions = []
    
    for i in range(len(X_data)):
        # Get demographic information for this individual
        sex = demographics.iloc[i]['Sex'] if 'Sex' in demographics.columns else None
        age_group = demographics.iloc[i]['Age Group'] if 'Age Group' in demographics.columns else None
        
        # Determine which model to use (from most specific to least specific)
        if sex == 'Male' and age_group == '50s' and 'Male_50s' in demographic_models:
            group_name = 'Male_50s'
        elif sex == 'Female' and age_group == '50s' and 'Female_50s' in demographic_models:
            group_name = 'Female_50s'
        elif sex == 'Male' and age_group == '40s' and 'Male_40s' in demographic_models:
            group_name = 'Male_40s'
        elif age_group == '40s' and 'Age_40s' in demographic_models:
            group_name = 'Age_40s'
        elif age_group == '50s' and 'Age_50s' in demographic_models:
            group_name = 'Age_50s'
        elif age_group == '60s' and 'Age_60s' in demographic_models:
            group_name = 'Age_60s'
        elif sex == 'Male' and 'Male' in demographic_models:
            group_name = 'Male'
        elif sex == 'Female' and 'Female' in demographic_models:
            group_name = 'Female'
        else:
            group_name = 'All'  # Fallback to the baseline model
        
        # Get the appropriate model and scaler
        model = demographic_models[group_name]['model']
        scaler = demographic_models[group_name]['scaler']
        
        # Preprocess the features
        X_scaled = scaler.transform([X_data.iloc[i]])
        
        # Make prediction
        pred = model.predict(X_scaled)[0]
        predictions.append(pred)
    
    return np.array(predictions)

# Step 3: Evaluate the Demographic-Specific Approach
def evaluate_demographic_approach(df, X, y, demographic_models):
    # Create a common test set to compare both approaches
    X_final, X_test, y_final, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Extract demographic information for the test set
    test_demographics = pd.DataFrame({
        'Sex': df.loc[X_test.index, 'Sex'],
        'Age Group': df.loc[X_test.index, 'Age Group']
    })
    
    # Get predictions from the baseline model
    baseline_model = demographic_models['All']['model']
    baseline_scaler = demographic_models['All']['scaler']
    X_test_scaled = baseline_scaler.transform(X_test)
    baseline_predictions = baseline_model.predict(X_test_scaled)
    
    # Get predictions from the demographic-specific approach
    demographic_predictions = predict_with_demographic_models(X_test, test_demographics, demographic_models)
    
    results = evaluate_approach_by_demographic(
        y_test, baseline_predictions, demographic_predictions, test_demographics
    )
    
    return results, X_test, y_test, test_demographics, baseline_predictions, demographic_predictions

# Function to evaluate by demographic
def evaluate_approach_by_demographic(y_true, y_pred_baseline, y_pred_demographic, demographics):
    results = []
    
    # Define demographic slices to evaluate
    demographic_slices = [
        ('Overall', slice(None)),
        ('Male', demographics['Sex'] == 'Male'),
        ('Female', demographics['Sex'] == 'Female'),
        ('Age 40s', demographics['Age Group'] == '40s'),
        ('Age 50s', demographics['Age Group'] == '50s'),
        ('Age 60s', demographics['Age Group'] == '60s'),
        ('Male 50s', (demographics['Sex'] == 'Male') & (demographics['Age Group'] == '50s')),
        ('Female 50s', (demographics['Sex'] == 'Female') & (demographics['Age Group'] == '50s'))
    ]
    
    for name, mask in demographic_slices:
        # Skip if too few samples
        if isinstance(mask, pd.Series) and sum(mask) < 10:
            continue
        
        # Apply the mask to get subgroup data
        if isinstance(mask, slice):
            y_true_group = y_true
            y_pred_baseline_group = y_pred_baseline
            y_pred_demographic_group = y_pred_demographic
            group_size = len(y_true)
        else:
            y_true_group = y_true[mask]
            y_pred_baseline_group = y_pred_baseline[mask]
            y_pred_demographic_group = y_pred_demographic[mask]
            group_size = sum(mask)
        
        # Calculate metrics for baseline model
        baseline_metrics = {
            'accuracy': accuracy_score(y_true_group, y_pred_baseline_group),
            'precision': precision_score(y_true_group, y_pred_baseline_group, zero_division=0),
            'recall': recall_score(y_true_group, y_pred_baseline_group, zero_division=0),
            'f1': f1_score(y_true_group, y_pred_baseline_group, zero_division=0)
        }
        
        # Calculate metrics for demographic-specific approach
        demographic_metrics = {
            'accuracy': accuracy_score(y_true_group, y_pred_demographic_group),
            'precision': precision_score(y_true_group, y_pred_demographic_group, zero_division=0),
            'recall': recall_score(y_true_group, y_pred_demographic_group, zero_division=0),
            'f1': f1_score(y_true_group, y_pred_demographic_group, zero_division=0)
        }
        
        # Calculate improvements
        improvements = {
            metric: demographic_metrics[metric] - baseline_metrics[metric]
            for metric in baseline_metrics
        }
        
        results.append({
            'group': name,
            'size': group_size,
            'baseline': baseline_metrics,
            'demographic_specific': demographic_metrics,
            'improvements': improvements
        })
    
    return results

# Step 4: Visualize the Results
def visualize_demographic_improvements(results):
    # Prepare data for plotting
    plot_data = []
    for result in results:
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plot_data.append({
                'Group': result['group'],
                'Metric': metric.capitalize(),
                'Baseline': result['baseline'][metric],
                'Demographic-Specific': result['demographic_specific'][metric],
                'Improvement': result['improvements'][metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 1. Performance comparison chart
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
        plt.subplot(2, 2, i+1)
        
        # Filter data for this metric
        metric_data = plot_df[plot_df['Metric'] == metric]
        
        # Convert to long format for easier plotting
        long_data = pd.melt(
            metric_data, 
            id_vars=['Group'], 
            value_vars=['Baseline', 'Demographic-Specific'],
            var_name='Model', value_name=metric
        )
        
        # Create plot
        sns.barplot(x='Group', y=metric, hue='Model', data=long_data)
        plt.title(f'{metric} by Demographic Group')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend(title='Approach')
    
    plt.tight_layout()
    plt.savefig('demographic_models_comparison.png', dpi=300)
    plt.show()
    
    # 2. Improvement chart
    plt.figure(figsize=(12, 8))
    
    # Filter to just show F1 improvement
    f1_improvements = plot_df[plot_df['Metric'] == 'F1']
    
    # Sort groups by improvement magnitude
    f1_improvements = f1_improvements.sort_values('Improvement')
    
    # Create horizontal bar chart
    plt.barh(f1_improvements['Group'], f1_improvements['Improvement'])
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('F1 Score Improvement')
    plt.ylabel('Demographic Group')
    plt.title('F1 Score Improvement with Demographic-Specific Models')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(f1_improvements['Improvement']):
        plt.text(v + (0.01 if v >= 0 else -0.04), i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig('f1_improvements.png', dpi=300)
    plt.show()

# Step 5: Analyze Fairness Metrics
def calculate_fairness_metrics(results):
    """Calculate fairness metrics before and after applying demographic-specific models."""
    # Extract results by group
    by_group = {result['group']: result for result in results}
    overall = by_group['Overall']
    
    # Initialize metrics
    fairness_metrics = {
        'performance_gaps': {
            'baseline': {},
            'demographic_specific': {}
        },
        'equity_ratio': {
            'baseline': {},
            'demographic_specific': {}
        }
    }
    
    # Calculate performance gaps and equity ratios for each protected group
    for group_name, result in by_group.items():
        if group_name == 'Overall':
            continue
        
        # Performance gaps (difference from overall performance)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            # For baseline model
            baseline_gap = result['baseline'][metric] - overall['baseline'][metric]
            fairness_metrics['performance_gaps']['baseline'][f"{group_name}_{metric}"] = baseline_gap
            
            # For demographic-specific model
            demo_gap = result['demographic_specific'][metric] - overall['demographic_specific'][metric]
            fairness_metrics['performance_gaps']['demographic_specific'][f"{group_name}_{metric}"] = demo_gap
            
            # Equity ratio (group performance / overall performance)
            baseline_ratio = result['baseline'][metric] / overall['baseline'][metric]
            fairness_metrics['equity_ratio']['baseline'][f"{group_name}_{metric}"] = baseline_ratio
            
            demo_ratio = result['demographic_specific'][metric] / overall['demographic_specific'][metric]
            fairness_metrics['equity_ratio']['demographic_specific'][f"{group_name}_{metric}"] = demo_ratio
    
    # Calculate aggregate fairness metrics
    fairness_metrics['avg_abs_gap'] = {
        'baseline': np.mean([abs(gap) for gap in fairness_metrics['performance_gaps']['baseline'].values()]),
        'demographic_specific': np.mean([abs(gap) for gap in fairness_metrics['performance_gaps']['demographic_specific'].values()])
    }
    
    fairness_metrics['max_abs_gap'] = {
        'baseline': max([abs(gap) for gap in fairness_metrics['performance_gaps']['baseline'].values()]),
        'demographic_specific': max([abs(gap) for gap in fairness_metrics['performance_gaps']['demographic_specific'].values()])
    }
    
    # Calculate improvement in fairness
    fairness_metrics['gap_reduction'] = {
        'avg': fairness_metrics['avg_abs_gap']['baseline'] - fairness_metrics['avg_abs_gap']['demographic_specific'],
        'max': fairness_metrics['max_abs_gap']['baseline'] - fairness_metrics['max_abs_gap']['demographic_specific']
    }
    
    return fairness_metrics

# Visualize fairness metrics
def visualize_fairness_metrics(fairness_metrics):
    # Prepare data for plotting
    gap_data = []
    for model_type in ['baseline', 'demographic_specific']:
        for metric_key, gap_value in fairness_metrics['performance_gaps'][model_type].items():
            group_name, metric = metric_key.rsplit('_', 1)
            gap_data.append({
                'Group': group_name,
                'Metric': metric.capitalize(),
                'Model': 'Baseline' if model_type == 'baseline' else 'Demographic-Specific',
                'Absolute Gap': abs(gap_value)
            })
    
    gap_df = pd.DataFrame(gap_data)
    
    # Plot fairness gaps by group and metric
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
        plt.subplot(2, 2, i+1)
        
        # Filter data for this metric
        metric_data = gap_df[gap_df['Metric'] == metric]
        
        # Create grouped bar chart
        sns.barplot(x='Group', y='Absolute Gap', hue='Model', data=metric_data)
        
        plt.title(f'Absolute {metric} Gap from Overall Performance')
        plt.ylabel(f'Absolute Gap in {metric}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Approach')
    
    plt.tight_layout()
    plt.savefig('fairness_gaps.png', dpi=300)
    plt.show()
    
    # Overall fairness improvement
    plt.figure(figsize=(10, 6))
    
    metrics = ['Average Absolute Gap', 'Maximum Absolute Gap']
    baseline_vals = [fairness_metrics['avg_abs_gap']['baseline'], 
                    fairness_metrics['max_abs_gap']['baseline']]
    demo_vals = [fairness_metrics['avg_abs_gap']['demographic_specific'], 
                fairness_metrics['max_abs_gap']['demographic_specific']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, baseline_vals, width, label='Baseline')
    ax.bar(x + width/2, demo_vals, width, label='Demographic-Specific')
    
    # Add value labels
    for i, v in enumerate(baseline_vals):
        ax.text(i - width/2, v + 0.01, f"{v:.4f}", ha='center')
    
    for i, v in enumerate(demo_vals):
        ax.text(i + width/2, v + 0.01, f"{v:.4f}", ha='center')
    
    ax.set_ylabel('Gap Magnitude')
    ax.set_title('Fairness Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('overall_fairness_comparison.png', dpi=300)
    plt.show()

# Main execution flow
def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare data for modeling
    X, y, male_condition, female_condition, intersectional_groups = prepare_data_for_modeling(df)
    
    # Create demographic-specific models
    demographic_models = create_demographic_models(df, X, y, male_condition, female_condition)
    
    # Evaluate the approach
    results, X_test, y_test, test_demographics, baseline_predictions, demographic_predictions = (
        evaluate_demographic_approach(df, X, y, demographic_models)
    )
    
    # Display results
    print("\n===== Performance Comparison: Baseline vs. Demographic-Specific =====")
    for result in results:
        print(f"\n{result['group']} (n={result['size']}):")
        print(f"  Baseline Model - Accuracy: {result['baseline']['accuracy']:.4f}, F1: {result['baseline']['f1']:.4f}")
        print(f"  Demo-Specific  - Accuracy: {result['demographic_specific']['accuracy']:.4f}, F1: {result['demographic_specific']['f1']:.4f}")
        print(f"  Improvement    - Accuracy: {result['improvements']['accuracy']:.4f}, F1: {result['improvements']['f1']:.4f}")
    
    # Visualize performance improvements
    visualize_demographic_improvements(results)
    
    # Calculate and visualize fairness metrics
    fairness_metrics = calculate_fairness_metrics(results)
    visualize_fairness_metrics(fairness_metrics)
    
    # Display fairness results
    print("\n===== Fairness Metrics =====")
    print(f"Average Absolute Performance Gap:")
    print(f"  Baseline: {fairness_metrics['avg_abs_gap']['baseline']:.4f}")
    print(f"  Demographic-Specific: {fairness_metrics['avg_abs_gap']['demographic_specific']:.4f}")
    print(f"  Improvement: {fairness_metrics['gap_reduction']['avg']:.4f}")
    
    print(f"\nMaximum Absolute Performance Gap:")
    print(f"  Baseline: {fairness_metrics['max_abs_gap']['baseline']:.4f}")
    print(f"  Demographic-Specific: {fairness_metrics['max_abs_gap']['demographic_specific']:.4f}")
    print(f"  Improvement: {fairness_metrics['gap_reduction']['max']:.4f}")
    
    return {
        'demographic_models': demographic_models,
        'results': results,
        'fairness_metrics': fairness_metrics,
        'test_data': {
            'X_test': X_test,
            'y_test': y_test,
            'demographics': test_demographics,
            'baseline_predictions': baseline_predictions,
            'demographic_predictions': demographic_predictions
        }
    }

if __name__ == "__main__":
    output = main()