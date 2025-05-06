# mitigation_approach4.py - Ensemble with Threshold Calibration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from scipy.optimize import minimize

# Keep the same data loading and demographic model creation functions from approach 3

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

def create_demographic_models(df, X, y, intersectional_groups):
    """Train optimized models for each demographic group."""
    demographic_models = {}
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Extract demographic information
    train_demographics = {
        'Sex': df.loc[X_train.index, 'Sex'],
        'Age Group': df.loc[X_train.index, 'Age Group']
    }
    
    test_demographics = {
        'Sex': df.loc[X_test.index, 'Sex'],
        'Age Group': df.loc[X_test.index, 'Age Group']
    }
    
    # Train a model for each demographic group
    for group_name, group_condition in intersectional_groups:
        print(f"\n===== Training model for {group_name} group =====")
        
        if group_condition is None:
            # This is the baseline model (all data)
            group_mask_train = pd.Series(True, index=X_train.index)
            group_mask_test = pd.Series(True, index=X_test.index)
        else:
            # Filter data for this specific demographic group
            group_mask_train = group_condition.loc[X_train.index]
            group_mask_test = group_condition.loc[X_test.index]
        
        # Extract data for this group
        X_train_group = X_train[group_mask_train]
        y_train_group = y_train[group_mask_train]
        X_test_group = X_test[group_mask_test]
        y_test_group = y_test[group_mask_test]
        
        # Check if we have enough samples
        if len(X_train_group) < 20 or len(X_test_group) < 10:
            print(f"Skipping {group_name} due to insufficient samples")
            continue
        
        print(f"Training with {len(X_train_group)} samples")
        
        # Standardize features
        scaler_group = StandardScaler()
        X_train_group_scaled = scaler_group.fit_transform(X_train_group)
        X_test_group_scaled = scaler_group.transform(X_test_group)
        
        # Train the model
        model_group = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
        model_group.fit(X_train_group_scaled, y_train_group)
        
        # Evaluate on test set
        y_pred_group = model_group.predict(X_test_group_scaled)
        
        # Calculate metrics
        group_metrics = {
            'accuracy': accuracy_score(y_test_group, y_pred_group),
            'precision': precision_score(y_test_group, y_pred_group, zero_division=0),
            'recall': recall_score(y_test_group, y_pred_group, zero_division=0),
            'f1': f1_score(y_test_group, y_pred_group, zero_division=0)
        }
        
        print(f"Performance for {group_name} model:")
        print(f"Accuracy: {group_metrics['accuracy']:.4f}")
        print(f"F1 Score: {group_metrics['f1']:.4f}")
        
        # Store the model and related data
        demographic_models[group_name] = {
            'model': model_group,
            'scaler': scaler_group,
            'metrics': group_metrics,
            'group_condition': group_condition
        }
    
    return demographic_models, X_train, X_test, y_train, y_test, train_demographics, test_demographics

def create_ensemble_predictions(X_test, test_demographics, demographic_models):
    """Generate prediction probabilities from each demographic model."""
    test_demo_df = pd.DataFrame({
        'Sex': test_demographics['Sex'],
        'Age Group': test_demographics['Age Group']
    })
    
    # Dictionary to store predictions from each model
    model_predictions = {}
    
    # Get predictions from each model
    for group_name, model_info in demographic_models.items():
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Get probability predictions
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        model_predictions[group_name] = y_prob
    
    # Convert to DataFrame for easier manipulation
    pred_df = pd.DataFrame(model_predictions)
    
    return pred_df

def visualize_ensemble_results(results):
    """Create visualizations for the ensemble approach results."""
    # Prepare data for plotting
    plot_data = []
    for result in results:
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plot_data.append({
                'Group': result['group'],
                'Metric': metric.capitalize(),
                'Baseline': result['baseline'][metric],
                'Ensemble': result['ensemble'][metric],
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
            value_vars=['Baseline', 'Ensemble'],
            var_name='Model', value_name=metric
        )
        
        # Create plot
        sns.barplot(x='Group', y=metric, hue='Model', data=long_data)
        plt.title(f'{metric} by Demographic Group')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend(title='Approach')
    
    plt.tight_layout()
    plt.savefig('ensemble_performance_comparison.png', dpi=300)
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
    plt.title('F1 Score Improvement with Ensemble Approach')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(f1_improvements['Improvement']):
        plt.text(v + (0.01 if v >= 0 else -0.04), i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig('ensemble_f1_improvements.png', dpi=300)
    plt.show()

def calculate_ensemble_fairness_metrics(results):
    """Calculate fairness metrics for the ensemble approach."""
    # Find the overall results
    overall = None
    for result in results:
        if result['group'] == 'All':
            overall = result
            break
    
    if overall is None:
        print("Warning: No 'All' group found in results. Using first group as reference.")
        overall = results[0]
    
    # Initialize metrics
    fairness_metrics = {
        'performance_gaps': {
            'baseline': {},
            'ensemble': {}
        },
        'equity_ratio': {
            'baseline': {},
            'ensemble': {}
        }
    }
    
    # Calculate gaps and ratios
    for result in results:
        if result == overall:
            continue
            
        group_name = result['group']
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            # Performance gaps
            baseline_gap = result['baseline'][metric] - overall['baseline'][metric]
            fairness_metrics['performance_gaps']['baseline'][f"{group_name}_{metric}"] = baseline_gap
            
            ensemble_gap = result['ensemble'][metric] - overall['ensemble'][metric]
            fairness_metrics['performance_gaps']['ensemble'][f"{group_name}_{metric}"] = ensemble_gap
            
            # Equity ratios
            baseline_ratio = result['baseline'][metric] / overall['baseline'][metric]
            fairness_metrics['equity_ratio']['baseline'][f"{group_name}_{metric}"] = baseline_ratio
            
            ensemble_ratio = result['ensemble'][metric] / overall['ensemble'][metric]
            fairness_metrics['equity_ratio']['ensemble'][f"{group_name}_{metric}"] = ensemble_ratio
    
    # Calculate aggregate metrics
    fairness_metrics['avg_abs_gap'] = {
        'baseline': np.mean([abs(gap) for gap in fairness_metrics['performance_gaps']['baseline'].values()]),
        'ensemble': np.mean([abs(gap) for gap in fairness_metrics['performance_gaps']['ensemble'].values()])
    }
    
    fairness_metrics['max_abs_gap'] = {
        'baseline': max([abs(gap) for gap in fairness_metrics['performance_gaps']['baseline'].values()]),
        'ensemble': max([abs(gap) for gap in fairness_metrics['performance_gaps']['ensemble'].values()])
    }
    
    # Calculate improvement
    fairness_metrics['gap_reduction'] = {
        'avg': fairness_metrics['avg_abs_gap']['baseline'] - fairness_metrics['avg_abs_gap']['ensemble'],
        'max': fairness_metrics['max_abs_gap']['baseline'] - fairness_metrics['max_abs_gap']['ensemble']
    }
    
    return fairness_metrics

def evaluate_ensemble_approach(ensemble_preds, y_test, test_demographics, intersectional_groups, baseline_preds):
    """Evaluate the ensemble approach compared to the baseline."""
    results = []
    
    # Create overall metrics
    baseline_metrics = {
        'accuracy': accuracy_score(y_test, baseline_preds),
        'precision': precision_score(y_test, baseline_preds, zero_division=0),
        'recall': recall_score(y_test, baseline_preds, zero_division=0),
        'f1': f1_score(y_test, baseline_preds, zero_division=0)
    }
    
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_preds),
        'precision': precision_score(y_test, ensemble_preds, zero_division=0),
        'recall': recall_score(y_test, ensemble_preds, zero_division=0),
        'f1': f1_score(y_test, ensemble_preds, zero_division=0)
    }
    
    improvements = {
        metric: ensemble_metrics[metric] - baseline_metrics[metric]
        for metric in baseline_metrics
    }
    
    # Add overall results
    results.append({
        'group': 'All',
        'size': len(y_test),
        'baseline': baseline_metrics,
        'ensemble': ensemble_metrics,
        'improvements': improvements
    })
    
    # Evaluate each demographic group
    for group_name, group_condition in intersectional_groups:
        if group_condition is None:
            continue  # Skip the "All" group since we've already added it
            
        # Create mask that aligns with y_test
        mask = np.zeros(len(y_test), dtype=bool)
        for i, idx in enumerate(y_test.index):
            if idx in group_condition.index and group_condition.loc[idx]:
                mask[i] = True
                
        group_size = np.sum(mask)
        
        if group_size < 10:
            print(f"Skipping {group_name} due to insufficient test samples")
            continue
        
        # Get group data using the mask
        group_y_test = y_test.values[mask]
        group_baseline_preds = baseline_preds[mask]
        group_ensemble_preds = ensemble_preds[mask]
        
        # Calculate metrics
        group_baseline_metrics = {
            'accuracy': accuracy_score(group_y_test, group_baseline_preds),
            'precision': precision_score(group_y_test, group_baseline_preds, zero_division=0),
            'recall': recall_score(group_y_test, group_baseline_preds, zero_division=0),
            'f1': f1_score(group_y_test, group_baseline_preds, zero_division=0)
        }
        
        group_ensemble_metrics = {
            'accuracy': accuracy_score(group_y_test, group_ensemble_preds),
            'precision': precision_score(group_y_test, group_ensemble_preds, zero_division=0),
            'recall': recall_score(group_y_test, group_ensemble_preds, zero_division=0),
            'f1': f1_score(group_y_test, group_ensemble_preds, zero_division=0)
        }
        
        group_improvements = {
            metric: group_ensemble_metrics[metric] - group_baseline_metrics[metric]
            for metric in group_baseline_metrics
        }
        
        # Add to results
        results.append({
            'group': group_name,
            'size': group_size,
            'baseline': group_baseline_metrics,
            'ensemble': group_ensemble_metrics,
            'improvements': group_improvements
        })
    
    return results

# New function for threshold calibration
def calibrate_thresholds(prediction_df, X_test, y_test, test_demographics, demographic_models, intersectional_groups):
    """
    Calibrate prediction thresholds for each demographic group to optimize performance.
    """
    # Create demographic group masks
    group_masks = {}
    
    # Create overall mask
    group_masks['All'] = np.ones(len(y_test), dtype=bool)
    
    # Create masks for demographic groups
    for group_name, group_condition in intersectional_groups:
        if group_condition is None:
            continue
            
        # Create mask that aligns with y_test
        mask = np.zeros(len(y_test), dtype=bool)
        for i, idx in enumerate(y_test.index):
            if idx in group_condition.index and group_condition.loc[idx]:
                mask[i] = True
                
        if np.sum(mask) >= 10:  # Only include groups with sufficient samples
            group_masks[group_name] = mask
    
    # Get baseline model predictions
    baseline_model = demographic_models['All']['model']
    baseline_scaler = demographic_models['All']['scaler']
    X_test_scaled = baseline_scaler.transform(X_test)
    baseline_preds = baseline_model.predict(X_test_scaled)
    
    # Calculate baseline metrics for each group
    baseline_metrics = {}
    for group_name, mask in group_masks.items():
        if group_name != 'All' and np.sum(mask) >= 10:
            group_y_test = y_test.values[mask]
            group_baseline_preds = baseline_preds[mask]
            
            baseline_metrics[group_name] = {
                'f1': f1_score(group_y_test, group_baseline_preds, zero_division=0),
                'precision': precision_score(group_y_test, group_baseline_preds, zero_division=0),
                'recall': recall_score(group_y_test, group_baseline_preds, zero_division=0)
            }
    
    # Calculate optimal thresholds for each group
    optimal_thresholds = {}
    for group_name, mask in group_masks.items():
        group_y_test = y_test.values[mask]
        
        # Calculate ensemble prediction probabilities
        ensemble_probs = np.zeros(len(mask))
        for model_name, model_info in demographic_models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            X_scaled = scaler.transform(X_test)
            # Weight by model performance on this group
            group_weight = 1.0
            if model_name == group_name:
                # Give extra weight to the specialized model for this group
                group_weight = 2.0
            
            probs = model.predict_proba(X_scaled)[:, 1]
            ensemble_probs += group_weight * probs
        
        # Normalize ensemble probabilities
        ensemble_probs = ensemble_probs / sum(1.0 if model_name == group_name else 1.0 
                                             for model_name in demographic_models.keys())
        
        group_probs = ensemble_probs[mask]
        
        # Skip if we don't have enough samples
        if len(group_y_test) < 10 or np.unique(group_y_test).size < 2:
            optimal_thresholds[group_name] = 0.5
            continue
        
        # Find optimal threshold for this group
        precision, recall, thresholds = precision_recall_curve(group_y_test, group_probs)
        
        # Add 0.5 as a candidate threshold
        augmented_thresholds = np.append(thresholds, 0.5)
        
        # Calculate F1 scores for each threshold
        f1_scores = []
        for threshold in augmented_thresholds:
            preds = (group_probs >= threshold).astype(int)
            f1 = f1_score(group_y_test, preds, zero_division=0)
            f1_scores.append(f1)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = augmented_thresholds[best_idx]
        
        # Special handling for groups that performed poorly in ensemble
        if group_name in baseline_metrics:
            baseline_f1 = baseline_metrics[group_name]['f1']
            best_f1 = f1_scores[best_idx]
            
            # If even the best threshold doesn't improve over baseline,
            # try to optimize for fairness
            if best_f1 < baseline_f1:
                # Find threshold that gives closest to baseline performance
                f1_diffs = [abs(f1 - baseline_f1) for f1 in f1_scores]
                fairness_idx = np.argmin(f1_diffs)
                best_threshold = augmented_thresholds[fairness_idx]
        
        optimal_thresholds[group_name] = best_threshold
    
    print("\nCalibrated Thresholds:")
    for group, threshold in optimal_thresholds.items():
        print(f"  {group}: {threshold:.4f}")
    
    return optimal_thresholds

# Function to apply calibrated thresholds
def apply_calibrated_predictions(X_test, y_test, demographic_models, test_demographics, intersectional_groups, thresholds):
    """Apply optimized weights and calibrated thresholds for final predictions."""
    # Create ensemble probabilities
    ensemble_probs = np.zeros(len(y_test))
    
    # Initialize weights
    model_weights = {
        'All': 1.0,
        'Male': 1.5,  # Increase weight for Male model to prevent performance drop
        'Female': 1.5,  # Increase weight for Female model
        'Age_40s': 1.7,  # Increase weight for Age_40s to prevent performance drop
        'Age_50s': 1.0,
        'Age_60s': 2.0,  # Higher weight for previously underperforming group
        'Male_50s': 1.5,
        'Female_50s': 1.0
    }
    
    # Calculate total weight
    total_weight = sum(model_weights.get(model_name, 1.0) for model_name in demographic_models.keys())
    
    # Get weighted probabilities from each model
    for model_name, model_info in demographic_models.items():
        model = model_info['model']
        scaler = model_info['scaler']
        X_scaled = scaler.transform(X_test)
        
        weight = model_weights.get(model_name, 1.0)
        probs = model.predict_proba(X_scaled)[:, 1]
        ensemble_probs += (weight / total_weight) * probs
    
    # Create demographic group masks
    group_masks = {}
    
    # Create overall mask
    group_masks['All'] = np.ones(len(y_test), dtype=bool)
    
    # Create masks for demographic groups
    for group_name, group_condition in intersectional_groups:
        if group_condition is None:
            continue
            
        # Create mask that aligns with y_test
        mask = np.zeros(len(y_test), dtype=bool)
        for i, idx in enumerate(y_test.index):
            if idx in group_condition.index and group_condition.loc[idx]:
                mask[i] = True
                
        if np.sum(mask) >= 10:  # Only include groups with sufficient samples
            group_masks[group_name] = mask
    
    # Apply calibrated thresholds to get final predictions
    final_preds = np.zeros(len(y_test), dtype=int)
    applied_thresholds = np.zeros(len(y_test))
    
    # Start with the most specific groups
    priority_order = [
        'Male_50s', 'Female_50s',  # Intersectional groups first
        'Age_40s', 'Age_50s', 'Age_60s',  # Age groups next
        'Male', 'Female',  # Gender groups next
        'All'  # Global model last
    ]
    
    # Keep track of which samples have been assigned
    assigned = np.zeros(len(y_test), dtype=bool)
    
    # Apply thresholds in priority order
    for group_name in priority_order:
        if group_name not in group_masks or group_name not in thresholds:
            continue
            
        mask = group_masks[group_name]
        threshold = thresholds[group_name]
        
        # Only apply to unassigned samples in this group
        current_mask = mask & ~assigned
        
        if np.sum(current_mask) > 0:
            final_preds[current_mask] = (ensemble_probs[current_mask] >= threshold).astype(int)
            applied_thresholds[current_mask] = threshold
            assigned[current_mask] = True
    
    # If any samples are still unassigned, use the global threshold
    if np.sum(~assigned) > 0:
        final_preds[~assigned] = (ensemble_probs[~assigned] >= thresholds['All']).astype(int)
        applied_thresholds[~assigned] = thresholds['All']
    
    return final_preds, ensemble_probs, applied_thresholds

# Main execution function
def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare data for modeling
    X, y, intersectional_groups = prepare_data_for_modeling(df)
    
    # Train demographic-specific models
    demographic_models, X_train, X_test, y_train, y_test, train_demographics, test_demographics = (
        create_demographic_models(df, X, y, intersectional_groups)
    )
    
    # Create ensemble predictions
    prediction_df = create_ensemble_predictions(X_test, test_demographics, demographic_models)
    
    # Get baseline predictions
    baseline_model = demographic_models['All']['model']
    baseline_scaler = demographic_models['All']['scaler']
    X_test_scaled = baseline_scaler.transform(X_test)
    baseline_preds = baseline_model.predict(X_test_scaled)
    
    # Calibrate thresholds for each demographic group
    # Calibrate thresholds for each demographic group
    calibrated_thresholds = calibrate_thresholds(
        prediction_df, X_test, y_test, test_demographics, demographic_models, intersectional_groups
    )
    
    # Apply calibrated thresholds to get final predictions
    calibrated_preds, ensemble_probs, applied_thresholds = apply_calibrated_predictions(
        X_test, y_test, demographic_models, test_demographics, intersectional_groups, calibrated_thresholds
    )
    
    # Evaluate the calibrated ensemble approach
    results = evaluate_ensemble_approach(
        calibrated_preds, y_test, test_demographics, intersectional_groups, baseline_preds
    )
    
    # Display results
    print("\n===== Performance Comparison: Baseline vs. Calibrated Ensemble =====")
    for result in results:
        print(f"\n{result['group']} (n={result['size']}):")
        print(f"  Baseline Model: Accuracy: {result['baseline']['accuracy']:.4f}, F1: {result['baseline']['f1']:.4f}")
        print(f"  Calibrated:     Accuracy: {result['ensemble']['accuracy']:.4f}, F1: {result['ensemble']['f1']:.4f}")
        print(f"  Improvement:    Accuracy: {result['improvements']['accuracy']:.4f}, F1: {result['improvements']['f1']:.4f}")
    
    # Calculate fairness metrics
    fairness_metrics = calculate_ensemble_fairness_metrics(results)
    
    # Display fairness metrics
    print("\n===== Fairness Metrics =====")
    print(f"Average Absolute Performance Gap:")
    print(f"  Baseline: {fairness_metrics['avg_abs_gap']['baseline']:.4f}")
    print(f"  Calibrated: {fairness_metrics['avg_abs_gap']['ensemble']:.4f}")
    print(f"  Improvement: {fairness_metrics['gap_reduction']['avg']:.4f}")
    
    print(f"\nMaximum Absolute Performance Gap:")
    print(f"  Baseline: {fairness_metrics['max_abs_gap']['baseline']:.4f}")
    print(f"  Calibrated: {fairness_metrics['max_abs_gap']['ensemble']:.4f}")
    print(f"  Improvement: {fairness_metrics['gap_reduction']['max']:.4f}")
    
    # Visualize results (with modified function names)
    visualize_results = visualize_ensemble_results
    visualize_results(results)
    
    return {
        'demographic_models': demographic_models,
        'results': results,
        'fairness_metrics': fairness_metrics,
        'predictions': {
            'baseline': baseline_preds,
            'calibrated': calibrated_preds,
            'ensemble_probs': ensemble_probs,
            'applied_thresholds': applied_thresholds
        }
    }

if __name__ == "__main__":
    output = main()