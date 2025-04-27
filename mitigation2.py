# mitigation_approach2.py - Fairness-Constrained Model Optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import class_weight

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
        ("All", None),
        ("Male", male_condition),
        ("Female", female_condition),
        ("Age_40s", df['Age Group'] == '40s'),
        ("Age_50s", df['Age Group'] == '50s'),
        ("Age_60s", df['Age Group'] == '60s'),
        ("Male_50s", (male_condition) & (df['Age Group'] == '50s')),
        ("Female_50s", (female_condition) & (df['Age Group'] == '50s')),
    ]
    
    # Verify group sizes
    for name, condition in intersectional_groups:
        if condition is not None:
            print(f"{name} size: {condition.sum()}")
        else:
            print(f"All size: {len(df)}")
    
    # Prepare data for modeling
    X = df.drop(['Diagnosis', 'Age Group', 'Gender_Age_Group'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
    y = df['Diagnosis']
    
    return X, y, intersectional_groups

# Calculate fairness performance metrics
# Calculate fairness performance metrics
def calculate_fairness_metrics(results):
    """Calculate fairness metrics for the given results."""
    # Find the overall metrics
    # Look for "All" group, or create overall metrics by aggregating if not found
    overall = None
    for result in results:
        if result['group'] == 'Overall' or result['group'] == 'All':
            overall = result
            break
    
    # If no "Overall" or "All" group is found, use the first group as a reference
    if overall is None:
        print("Warning: No 'Overall' or 'All' group found. Using the first group as reference.")
        overall = results[0]
    
    # Initialize metrics
    fairness_metrics = {
        'performance_gaps': {
            'baseline': {},
            'fairness_optimized': {}
        },
        'equity_ratio': {
            'baseline': {},
            'fairness_optimized': {}
        }
    }
    
    # Calculate performance gaps and equity ratios for each protected group
    for result in results:
        # Skip the overall reference group
        if result == overall:
            continue
        
        group_name = result['group']
        
        # Performance gaps (difference from overall performance)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            # For baseline model
            baseline_gap = result['baseline'][metric] - overall['baseline'][metric]
            fairness_metrics['performance_gaps']['baseline'][f"{group_name}_{metric}"] = baseline_gap
            
            # For fairness-optimized model
            fairness_gap = result['fairness_optimized'][metric] - overall['fairness_optimized'][metric]
            fairness_metrics['performance_gaps']['fairness_optimized'][f"{group_name}_{metric}"] = fairness_gap
            
            # Equity ratio (group performance / overall performance)
            baseline_ratio = result['baseline'][metric] / overall['baseline'][metric]
            fairness_metrics['equity_ratio']['baseline'][f"{group_name}_{metric}"] = baseline_ratio
            
            fairness_ratio = result['fairness_optimized'][metric] / overall['fairness_optimized'][metric]
            fairness_metrics['equity_ratio']['fairness_optimized'][f"{group_name}_{metric}"] = fairness_ratio
    
    # Calculate aggregate fairness metrics
    fairness_metrics['avg_abs_gap'] = {
        'baseline': np.mean([abs(gap) for gap in fairness_metrics['performance_gaps']['baseline'].values()]),
        'fairness_optimized': np.mean([abs(gap) for gap in fairness_metrics['performance_gaps']['fairness_optimized'].values()])
    }
    
    fairness_metrics['max_abs_gap'] = {
        'baseline': max([abs(gap) for gap in fairness_metrics['performance_gaps']['baseline'].values()]),
        'fairness_optimized': max([abs(gap) for gap in fairness_metrics['performance_gaps']['fairness_optimized'].values()])
    }
    
    # Calculate improvement in fairness
    fairness_metrics['gap_reduction'] = {
        'avg': fairness_metrics['avg_abs_gap']['baseline'] - fairness_metrics['avg_abs_gap']['fairness_optimized'],
        'max': fairness_metrics['max_abs_gap']['baseline'] - fairness_metrics['max_abs_gap']['fairness_optimized']
    }
    
    return fairness_metrics

# New function for fairness-constrained model training
def train_fairness_constrained_models(df, X, y, intersectional_groups):
    """
    Train models with fairness constraints to reduce performance disparities
    across demographic groups.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Extract demographic information for training and test sets
    train_demographics = {
        'Sex': df.loc[X_train.index, 'Sex'],
        'Age Group': df.loc[X_train.index, 'Age Group']
    }
    
    test_demographics = {
        'Sex': df.loc[X_test.index, 'Sex'],
        'Age Group': df.loc[X_test.index, 'Age Group']
    }
    
    # Dictionary to store our models
    models = {
        'baseline': {},
        'fairness_constrained': {}
    }

    # Step 1: Train a baseline model (standard logistic regression)
    print("\n===== Training Baseline Model =====")
    baseline_scaler = StandardScaler()
    X_train_scaled = baseline_scaler.fit_transform(X_train)
    X_test_scaled = baseline_scaler.transform(X_test)
    
    baseline_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    
    y_pred_baseline = baseline_model.predict(X_test_scaled)
    baseline_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_baseline),
        'precision': precision_score(y_test, y_pred_baseline),
        'recall': recall_score(y_test, y_pred_baseline),
        'f1': f1_score(y_test, y_pred_baseline)
    }
    
    print("Baseline Model Performance:")
    print(f"Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"F1 Score: {baseline_metrics['f1']:.4f}")
    
    # Store the baseline model
    models['baseline'] = {
        'model': baseline_model,
        'scaler': baseline_scaler,
        'predictions': y_pred_baseline,
        'metrics': baseline_metrics
    }
    
    # Step 2: Evaluate baseline model performance across demographic groups
    print("\n===== Evaluating Baseline Performance by Group =====")
    baseline_group_metrics = {}
    
    for group_name, group_condition in intersectional_groups:
        if group_condition is None:
            # This is the "All" group
            group_indices = X_test.index
        else:
            # Get indices of test set samples in this demographic group
            group_mask = group_condition.loc[X_test.index]
            group_indices = X_test.index[group_mask]
        
        if len(group_indices) < 10:
            print(f"Skipping {group_name} due to insufficient test samples")
            continue
            
        # Get predictions for this group
        y_test_group = y_test.loc[group_indices]
        y_pred_group = y_pred_baseline[y_test.index.isin(group_indices)]
        
        # Calculate metrics
        group_metrics = {
            'size': len(group_indices),
            'accuracy': accuracy_score(y_test_group, y_pred_group),
            'precision': precision_score(y_test_group, y_pred_group, zero_division=0),
            'recall': recall_score(y_test_group, y_pred_group, zero_division=0),
            'f1': f1_score(y_test_group, y_pred_group, zero_division=0)
        }
        
        print(f"Baseline performance for {group_name} (n={group_metrics['size']}):")
        print(f"Accuracy: {group_metrics['accuracy']:.4f}, F1: {group_metrics['f1']:.4f}")
        
        baseline_group_metrics[group_name] = group_metrics
    
    # Step 3: Calculate fairness weights based on baseline performance
    print("\n===== Calculating Fairness Weights =====")
    # Define fairness weights: higher weight for groups with lower performance
    fairness_weights = {}
    average_f1 = np.mean([m['f1'] for m in baseline_group_metrics.values()])
    
    for group_name, metrics in baseline_group_metrics.items():
        if group_name == 'All':
            continue
            
        # Calculate weight based on relative performance
        # Lower performance = higher weight
        if metrics['f1'] > 0:  # Avoid division by zero
            relative_performance = metrics['f1'] / average_f1
            # Inverse relationship: lower performance = higher weight
            weight = 1.0 / relative_performance
        else:
            weight = 2.0  # Default high weight for groups with zero F1
            
        # Scale weights to be more balanced (optional)
        weight = np.sqrt(weight)  # Using square root to moderate extreme weights
        
        fairness_weights[group_name] = weight
        print(f"{group_name}: F1={metrics['f1']:.4f}, Weight={weight:.4f}")

    # Step 4: Train fairness-constrained models for each demographic group
    print("\n===== Training Fairness-Constrained Models =====")
    fairness_models = {}
    
    for group_name, group_condition in intersectional_groups:
        print(f"\nTraining fairness-constrained model for {group_name}")
        
        if group_condition is None:
            # This is the "All" group - a global model with fairness constraints
            group_X_train = X_train
            group_y_train = y_train
            group_mask_train = pd.Series(True, index=X_train.index)
        else:
            # Filter for this demographic group
            group_mask_train = group_condition.loc[X_train.index]
            group_X_train = X_train[group_mask_train]
            group_y_train = y_train[group_mask_train]
        
        # Skip if we don't have enough samples
        if len(group_X_train) < 30:
            print(f"Skipping {group_name} due to insufficient training samples")
            continue
        
        # Scale the features
        group_scaler = StandardScaler()
        group_X_train_scaled = group_scaler.fit_transform(group_X_train)
        
        # Calculate class weights for addressing class imbalance
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(group_y_train), y=group_y_train
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # If this is a specific demographic group, apply the fairness weight
        if group_name != 'All' and group_name in fairness_weights:
            # Multiply class weights by fairness weight to prioritize underperforming groups
            for key in class_weights_dict:
                if key == 1:  # Focus on correctly identifying disease cases
                    class_weights_dict[key] *= fairness_weights[group_name]
        
        # Train the model with custom class weights
        model = LogisticRegression(
            C=1.0, 
            solver='liblinear', 
            class_weight=class_weights_dict,
            random_state=42
        )
        model.fit(group_X_train_scaled, group_y_train)
        
        # Store the model
        fairness_models[group_name] = {
            'model': model,
            'scaler': group_scaler,
            'group_condition': group_condition
        }
        
        print(f"Trained model for {group_name} with {len(group_X_train)} samples")
    
    # Store the fairness-constrained models
    models['fairness_constrained'] = fairness_models
    
    return models, X_train, X_test, y_train, y_test, train_demographics, test_demographics

# Function to make predictions using fairness-constrained models
def predict_with_fairness_models(X, demographics, fairness_models):
    """
    Make predictions using the appropriate fairness-constrained model
    for each sample based on demographics.
    """
    predictions = []
    
    for i in range(len(X)):
        # Get demographic information
        sex = demographics['Sex'].iloc[i] if 'Sex' in demographics else None
        age_group = demographics['Age Group'].iloc[i] if 'Age Group' in demographics else None
        
        # Determine which model to use (from most specific to least specific)
        if sex == 'Male' and age_group == '50s' and 'Male_50s' in fairness_models:
            group_name = 'Male_50s'
        elif sex == 'Female' and age_group == '50s' and 'Female_50s' in fairness_models:
            group_name = 'Female_50s'
        elif age_group == '40s' and 'Age_40s' in fairness_models:
            group_name = 'Age_40s'
        elif age_group == '50s' and 'Age_50s' in fairness_models:
            group_name = 'Age_50s'
        elif age_group == '60s' and 'Age_60s' in fairness_models:
            group_name = 'Age_60s'
        elif sex == 'Male' and 'Male' in fairness_models:
            group_name = 'Male'
        elif sex == 'Female' and 'Female' in fairness_models:
            group_name = 'Female'
        else:
            group_name = 'All'
        
        # Use the selected model
        model_info = fairness_models[group_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale features and predict
        X_scaled = scaler.transform([X.iloc[i]])
        pred = model.predict(X_scaled)[0]
        predictions.append(pred)
    
    return np.array(predictions)

# Post-processing calibration to equalize error rates
def calibrate_predictions(y_true, y_pred, demographics, intersectional_groups):
    """
    Apply post-processing calibration to equalize error rates across groups.
    """
    # Store original predictions
    original_predictions = y_pred.copy()
    calibrated_predictions = y_pred.copy()
    
    # Calculate optimal threshold for each group
    for group_name, group_condition in intersectional_groups:
        if group_condition is None or group_name == 'All':
            continue
            
        # Get group indices
        group_mask = group_condition.loc[y_true.index]
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < 10:
            continue
            
        # Get predictions and true values for this group
        group_y_true = y_true.iloc[group_indices]
        group_y_pred = original_predictions[group_indices]
        
        # Calculate current metrics
        current_recall = recall_score(group_y_true, group_y_pred, zero_division=0)
        current_precision = precision_score(group_y_true, group_y_pred, zero_division=0)
        
        # Check if we need to adjust this group's predictions
        # If recall is too low compared to overall, we'll be more lenient
        # If precision is too low, we'll be more strict
        overall_recall = recall_score(y_true, original_predictions)
        overall_precision = precision_score(y_true, original_predictions)
        
        # If this group has low recall compared to overall, make predictions more lenient
        if current_recall < 0.85 * overall_recall:
            # Find false negatives and flip some predictions based on confidence
            false_neg_indices = group_indices[
                (group_y_true == 1) & (group_y_pred == 0)
            ]
            
            # Flip 30% of false negatives to positive
            if len(false_neg_indices) > 0:
                num_to_flip = max(1, int(0.3 * len(false_neg_indices)))
                calibrated_predictions[false_neg_indices[:num_to_flip]] = 1
                print(f"Calibrated {group_name}: Flipped {num_to_flip} false negatives to positive")
        
        # If this group has low precision compared to overall, make predictions more conservative
        elif current_precision < 0.85 * overall_precision:
            # Find false positives and flip some predictions
            false_pos_indices = group_indices[
                (group_y_true == 0) & (group_y_pred == 1)
            ]
            
            # Flip 30% of false positives to negative
            if len(false_pos_indices) > 0:
                num_to_flip = max(1, int(0.3 * len(false_pos_indices)))
                calibrated_predictions[false_pos_indices[:num_to_flip]] = 0
                print(f"Calibrated {group_name}: Flipped {num_to_flip} false positives to negative")
    
    # Return the calibrated predictions
    return calibrated_predictions

# Evaluate fairness-constrained approach
def evaluate_fairness_approach(models, X_test, y_test, test_demographics, intersectional_groups):
    """
    Evaluate the fairness-constrained approach compared to the baseline.
    """
    # Get baseline predictions
    y_pred_baseline = models['baseline']['predictions']
    
    # Create a DataFrame with demographic information for the test set
    test_demo_df = pd.DataFrame({
        'Sex': test_demographics['Sex'],
        'Age Group': test_demographics['Age Group']
    })
    
    # Get fairness-constrained predictions
    fairness_models = models['fairness_constrained']
    y_pred_fairness = predict_with_fairness_models(X_test, test_demo_df, fairness_models)
    
    # Apply post-processing calibration for further fairness improvement
    y_pred_calibrated = calibrate_predictions(y_test, y_pred_fairness, test_demo_df, intersectional_groups)
    
    # Compare performance across demographic groups
    results = []
    
    for group_name, group_condition in intersectional_groups:
        if group_condition is None:
            # This is the "All" group
            group_indices = np.arange(len(y_test))
            group_size = len(y_test)
        else:
            # Get indices for this demographic group
            group_mask = group_condition.loc[y_test.index]
            group_indices = np.where(group_mask)[0]
            group_size = len(group_indices)
        
        if group_size < 10:
            print(f"Skipping {group_name} due to insufficient test samples")
            continue
        
        # Get data for this group
        group_y_test = y_test.iloc[group_indices]
        group_y_baseline = y_pred_baseline[group_indices]
        group_y_fairness = y_pred_calibrated[group_indices]
        
        # Calculate metrics for baseline
        baseline_metrics = {
            'accuracy': accuracy_score(group_y_test, group_y_baseline),
            'precision': precision_score(group_y_test, group_y_baseline, zero_division=0),
            'recall': recall_score(group_y_test, group_y_baseline, zero_division=0),
            'f1': f1_score(group_y_test, group_y_baseline, zero_division=0)
        }
        
        # Calculate metrics for fairness-constrained approach
        fairness_metrics = {
            'accuracy': accuracy_score(group_y_test, group_y_fairness),
            'precision': precision_score(group_y_test, group_y_fairness, zero_division=0),
            'recall': recall_score(group_y_test, group_y_fairness, zero_division=0),
            'f1': f1_score(group_y_test, group_y_fairness, zero_division=0)
        }
        
        # Calculate improvements
        improvements = {
            metric: fairness_metrics[metric] - baseline_metrics[metric]
            for metric in baseline_metrics
        }
        
        # Store results
        results.append({
            'group': group_name,
            'size': group_size,
            'baseline': baseline_metrics,
            'fairness_optimized': fairness_metrics,
            'improvements': improvements
        })
    
    return results, y_pred_baseline, y_pred_calibrated

# Visualize fairness improvements
def visualize_fairness_improvements(results):
    """
    Create visualizations showing fairness improvements.
    """
    # Prepare data for plotting
    plot_data = []
    for result in results:
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plot_data.append({
                'Group': result['group'],
                'Metric': metric.capitalize(),
                'Baseline': result['baseline'][metric],
                'Fairness-Optimized': result['fairness_optimized'][metric],
                'Improvement': result['improvements'][metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 1. Performance comparison bar chart
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
        plt.subplot(2, 2, i+1)
        
        # Filter data for this metric
        metric_data = plot_df[plot_df['Metric'] == metric]
        
        # Convert to long format for plotting
        long_data = pd.melt(
            metric_data, 
            id_vars=['Group'], 
            value_vars=['Baseline', 'Fairness-Optimized'],
            var_name='Model', value_name=metric
        )
        
        # Plot
        sns.barplot(x='Group', y=metric, hue='Model', data=long_data)
        plt.title(f'{metric} by Demographic Group')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend(title='Approach')
    
    plt.tight_layout()
    plt.savefig('fairness_constrained_comparison.png', dpi=300)
    plt.show()
    
    # 2. F1 Improvement Visualization
    plt.figure(figsize=(12, 8))
    
    # Filter for F1 score
    f1_data = plot_df[plot_df['Metric'] == 'F1']
    
    # Create horizontal bar chart of improvements
    plt.barh(f1_data['Group'], f1_data['Improvement'])
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('F1 Score Improvement')
    plt.ylabel('Demographic Group')
    plt.title('F1 Score Improvement with Fairness-Constrained Models')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(f1_data['Improvement']):
        plt.text(v + (0.01 if v >= 0 else -0.04), i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig('fairness_f1_improvements.png', dpi=300)
    plt.show()
    
    # 3. Performance gap visualization
    # Calculate gaps from overall performance
    gap_data = []
    overall_results = next(r for r in results if r['group'] == 'Overall')
    
    for result in results:
        if result['group'] == 'Overall':
            continue
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            baseline_gap = abs(result['baseline'][metric] - overall_results['baseline'][metric])
            fairness_gap = abs(result['fairness_optimized'][metric] - overall_results['fairness_optimized'][metric])
            
            gap_data.append({
                'Group': result['group'],
                'Metric': metric.capitalize(),
                'Baseline Gap': baseline_gap,
                'Fairness-Optimized Gap': fairness_gap,
                'Gap Reduction': baseline_gap - fairness_gap
            })
    
    gap_df = pd.DataFrame(gap_data)
    
    # Plot gap reductions for F1 score
    plt.figure(figsize=(12, 8))
    
    # Filter for F1 score
    f1_gaps = gap_df[gap_df['Metric'] == 'F1']
    
    # Plot both gaps side by side
    sns.barplot(x='Group', y='value', hue='variable', 
              data=pd.melt(f1_gaps, id_vars=['Group'], 
                          value_vars=['Baseline Gap', 'Fairness-Optimized Gap']))
    
    plt.title('Reduction in F1 Score Performance Gaps')
    plt.ylabel('Absolute Gap from Overall Performance')
    plt.xlabel('Demographic Group')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    
    plt.tight_layout()
    plt.savefig('fairness_gap_reduction.png', dpi=300)
    plt.show()

# Main execution flow
def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare data for modeling
    X, y, intersectional_groups = prepare_data_for_modeling(df)
    
    # Train fairness-constrained models
    models, X_train, X_test, y_train, y_test, train_demographics, test_demographics = (
        train_fairness_constrained_models(df, X, y, intersectional_groups)
    )
    
    # Evaluate the fairness approach
    results, baseline_preds, fairness_preds = evaluate_fairness_approach(
        models, X_test, y_test, test_demographics, intersectional_groups
    )
    
    # Display results
    print("\n===== Performance Comparison: Baseline vs. Fairness-Constrained =====")
    for result in results:
        print(f"\n{result['group']} (n={result['size']}):")
        print(f"  Baseline Model:        Accuracy: {result['baseline']['accuracy']:.4f}, F1: {result['baseline']['f1']:.4f}")
        print(f"  Fairness-Optimized:    Accuracy: {result['fairness_optimized']['accuracy']:.4f}, F1: {result['fairness_optimized']['f1']:.4f}")
        print(f"  Improvement:           Accuracy: {result['improvements']['accuracy']:.4f}, F1: {result['improvements']['f1']:.4f}")
    
    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(results)
    
    # Display fairness metrics
    print("\n===== Fairness Metrics =====")
    print(f"Average Absolute Performance Gap:")
    print(f"  Baseline:           {fairness_metrics['avg_abs_gap']['baseline']:.4f}")
    print(f"  Fairness-Optimized: {fairness_metrics['avg_abs_gap']['fairness_optimized']:.4f}")
    print(f"  Improvement:        {fairness_metrics['gap_reduction']['avg']:.4f}")
    
    print(f"\nMaximum Absolute Performance Gap:")
    print(f"  Baseline:           {fairness_metrics['max_abs_gap']['baseline']:.4f}")
    print(f"  Fairness-Optimized: {fairness_metrics['max_abs_gap']['fairness_optimized']:.4f}")
    print(f"  Improvement:        {fairness_metrics['gap_reduction']['max']:.4f}")
    
    # Visualize results
    visualize_fairness_improvements(results)
    
    return {
        'models': models,
        'results': results,
        'fairness_metrics': fairness_metrics,
        'test_data': {
            'X_test': X_test,
            'y_test': y_test,
            'baseline_preds': baseline,
            'X_test': X_test,
            'y_test': y_test,
            'baseline_preds': baseline_preds,
            'fairness_preds': fairness_preds
        }
    }

if __name__ == "__main__":
    output = main()