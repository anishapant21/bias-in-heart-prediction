from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_specialized_model(X_train, y_train, demographic_condition, cv=5, random_state=42):
    """
    Train a specialized model for a specific demographic group using grid search
    with cross-validation for hyperparameter tuning
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    demographic_condition : Series of bool
        Boolean mask for selecting the demographic group
    cv : int
        Number of cross-validation folds (default: 5)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    best_model : LogisticRegression
        The best model for the demographic group
    cv_results : dict
        Cross-validation results including best parameters
    """
    # Filter training data for the demographic group
    X_train_demo = X_train[demographic_condition.loc[X_train.index]]
    y_train_demo = y_train[demographic_condition.loc[y_train.index]]
    
    # Check if we have enough samples
    if len(X_train_demo) < cv * 2:
        print(f"Insufficient samples for this demographic group: {len(X_train_demo)}")
        return None, None
    
    # Check if we have samples from both classes
    if len(y_train_demo.unique()) < 2:
        print(f"Only one class present in this demographic group. Cannot train a classifier.")
        return None, None
    
    print(f"Training specialized model with {len(X_train_demo)} samples")
    print(f"Class distribution: {y_train_demo.value_counts().to_dict()}")
    
    # Set up cross-validation
    min_samples_per_fold = 2  # Need at least 2 samples per fold
    n_folds = min(cv, len(X_train_demo) // min_samples_per_fold)
    
    if n_folds < 2:
        print(f"Not enough samples for cross-validation. Using a single train/test split instead.")
        # Fall back to a simple model without cross-validation
        from sklearn.model_selection import train_test_split
        X_demo_train, X_demo_val, y_demo_train, y_demo_val = train_test_split(
            X_train_demo, y_train_demo, test_size=0.3, random_state=random_state, stratify=y_train_demo
        )
        
        # Use a simple LogisticRegression with default parameters
        model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=random_state)
        model.fit(X_demo_train, y_demo_train)
        
        print(f"Simple model trained without cross-validation.")
        return model, {'best_params': {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}}
    
    cv_folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Define parameter grid to search - simplified for small sample sizes
    param_grid = {
        'C': [0.1, 1.0, 10.0],  # Reduced parameter options
        'penalty': ['l2'],      # Simpler regularization
        'solver': ['liblinear'] # Reliable solver
    }
    
    # Set up grid search with cross-validation
    grid_search = GridSearchCV(
        LogisticRegression(random_state=random_state),
        param_grid,
        cv=cv_folds,
        scoring='accuracy',  # Changed to accuracy which works better with small datasets
        n_jobs=-1           # Use all available cores
    )
    
    try:
        # Fit grid search with error handling
        grid_search.fit(X_train_demo, y_train_demo)
        
        # Get best parameters and score
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Get and train the best model
        best_model = grid_search.best_estimator_
        
        # Return the best model and cross-validation results
        return best_model, {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    except Exception as e:
        print(f"Error during model training: {e}")
        print(f"Falling back to simple model...")
        
        # Fall back to a simple model with fixed parameters
        model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=random_state)
        
        try:
            model.fit(X_train_demo, y_train_demo)
            print(f"Simple model trained successfully.")
            return model, {'best_params': {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}}
        except Exception as e2:
            print(f"Could not train even a simple model: {e2}")
            return None, None

def evaluate_model(model, X_test, y_test, demographic_condition=None, group_name=None):
    """
    Evaluate a model on test data, optionally for a specific demographic group
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : DataFrame
        Test features
    y_test : Series
        Test target
    demographic_condition : Series of bool, optional
        Boolean mask for selecting the demographic group
    group_name : str, optional
        Name of the demographic group for reporting
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # If model is None, return None
    if model is None:
        print(f"No model to evaluate for {group_name if group_name else 'this group'}")
        return None
        
    # If demographic condition is provided, filter test data
    if demographic_condition is not None:
        test_indices = [idx for idx in X_test.index if idx in demographic_condition.index]
        valid_condition = demographic_condition.loc[test_indices]
        X_test_demo = X_test[valid_condition]
        y_test_demo = y_test[valid_condition.index]
        group_prefix = f"{group_name} " if group_name else "Group "
    else:
        X_test_demo = X_test
        y_test_demo = y_test
        group_prefix = ""
    
    # Skip if we don't have enough samples
    if len(X_test_demo) < 5:
        print(f"Skipping evaluation - insufficient test samples: {len(X_test_demo)}")
        return None
        
    # Check if we have samples from both classes
    if len(y_test_demo.unique()) < 2:
        print(f"Only one class present in test set. Some metrics will not be calculated.")
    
    # Make predictions
    y_pred = model.predict(X_test_demo)
    y_prob = model.predict_proba(X_test_demo)[:, 1]  # Probability of positive class
    
    # Calculate performance metrics
    metrics = {
        'size': len(X_test_demo),
        'accuracy': accuracy_score(y_test_demo, y_pred)
    }
    
    # Calculate precision, recall, F1 only if both classes are present in predictions
    if len(np.unique(y_pred)) > 1 and len(np.unique(y_test_demo)) > 1:
        metrics['precision'] = precision_score(y_test_demo, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_test_demo, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_test_demo, y_pred, zero_division=0)
    else:
        print(f"Warning: Not all classes present in predictions. Some metrics unavailable.")
        # Add placeholder values
        metrics['precision'] = np.nan
        metrics['recall'] = np.nan
        metrics['f1'] = np.nan
    
    # Add ROC AUC if both classes are present in true labels
    if len(np.unique(y_test_demo)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_test_demo, y_prob)
    else:
        metrics['roc_auc'] = np.nan
    
    # Print results
    print(f"\n{group_prefix}Performance (n={metrics['size']}):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    if not np.isnan(metrics['precision']):
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    if not np.isnan(metrics['roc_auc']):
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

def train_and_evaluate_specialized_models(X_train, X_test, y_train, y_test, 
                                          demographic_conditions, scaler):
    """
    Train and evaluate specialized models for different demographic groups
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target
    y_test : Series
        Test target
    demographic_conditions : dict
        Dictionary of demographic conditions
    scaler : StandardScaler
        Fitted scaler for feature standardization
        
    Returns:
    --------
    specialized_models : dict
        Dictionary of specialized models for each demographic group
    specialized_metrics : dict
        Dictionary of performance metrics for specialized models
    baseline_metrics : dict
        Dictionary of performance metrics for the baseline model on each group
    """
    # Train a baseline model on all data
    print("\n===== Training Baseline Model =====")
    baseline_model, _ = train_specialized_model(
        X_train, y_train, 
        demographic_condition=pd.Series(True, index=X_train.index),
        cv=5
    )
    
    # Initialize dictionaries to store models and metrics
    specialized_models = {}
    specialized_metrics = {}
    baseline_metrics = {}
    
    # Evaluate baseline model on test set
    print("\n===== Evaluating Baseline Model on Full Test Set =====")
    baseline_metrics['Overall'] = evaluate_model(
        baseline_model, X_test, y_test
    )
    
    # Train and evaluate specialized models for each demographic group
    for group_name, condition in demographic_conditions.items():
        # Skip very small groups
        group_size = condition.sum()
        if group_size < 10:
            print(f"\n===== Skipping {group_name} - too few samples ({group_size}) =====")
            specialized_models[group_name] = None
            specialized_metrics[group_name] = None
            baseline_metrics[group_name] = None
            continue
            
        print(f"\n===== Training Specialized Model for {group_name} =====")
        
        # Use a try-except block to handle potential errors
        try:
            # Train specialized model
            specialized_model, cv_results = train_specialized_model(
                X_train, y_train, 
                demographic_condition=condition,
                cv=5
            )
            
            # Store the model
            specialized_models[group_name] = specialized_model
            
            # Evaluate specialized model on this demographic group's test data
            if specialized_model is not None:
                print(f"\n===== Evaluating Specialized Model for {group_name} =====")
                specialized_metrics[group_name] = evaluate_model(
                    specialized_model, X_test, y_test, 
                    demographic_condition=condition, 
                    group_name=group_name
                )
            else:
                specialized_metrics[group_name] = None
            
            # Also evaluate the baseline model on this demographic group
            print(f"\n===== Evaluating Baseline Model on {group_name} =====")
            baseline_metrics[group_name] = evaluate_model(
                baseline_model, X_test, y_test, 
                demographic_condition=condition, 
                group_name=group_name
            )
        
        except Exception as e:
            print(f"Error processing {group_name}: {e}")
            specialized_models[group_name] = None
            specialized_metrics[group_name] = None
            baseline_metrics[group_name] = None
    
    # Return models and metrics
    return specialized_models, specialized_metrics, baseline_metrics

def compare_specialized_vs_baseline(specialized_metrics, baseline_metrics, metric='accuracy'):
    """
    Compare performance of specialized models vs. baseline model across demographic groups
    
    Parameters:
    -----------
    specialized_metrics : dict
        Dictionary of performance metrics for specialized models
    baseline_metrics : dict
        Dictionary of performance metrics for baseline model
    metric : str
        Performance metric to compare (default: 'accuracy')
        
    Returns:
    --------
    comparison_df : DataFrame
        DataFrame comparing specialized and baseline model performance
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Initialize lists to store data
    groups = []
    baseline_values = []
    specialized_values = []
    improvements = []
    
    # Collect data for each group
    for group in specialized_metrics.keys():
        if group in baseline_metrics and specialized_metrics[group] is not None and baseline_metrics[group] is not None:
            # Skip if metric is not available
            if metric not in specialized_metrics[group] or metric not in baseline_metrics[group]:
                continue
                
            groups.append(group)
            baseline_val = baseline_metrics[group][metric]
            specialized_val = specialized_metrics[group][metric]
            
            baseline_values.append(baseline_val)
            specialized_values.append(specialized_val)
            improvements.append(specialized_val - baseline_val)
    
    # Create DataFrame
    comparison_df = pd.DataFrame({
        'Group': groups,
        'Baseline': baseline_values,
        'Specialized': specialized_values,
        'Improvement': improvements
    })
    
    # Sort by improvement
    comparison_df = comparison_df.sort_values('Improvement', ascending=False)
    
    # Print comparison
    print(f"\n===== Specialized vs. Baseline Model Comparison ({metric}) =====")
    print(comparison_df)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Reshape data for grouped bar chart
    plot_data = comparison_df.melt(
        id_vars='Group', 
        value_vars=['Baseline', 'Specialized'],
        var_name='Model Type', 
        value_name=metric.capitalize()
    )
    
    # Create grouped bar chart
    sns.barplot(x='Group', y=metric.capitalize(), hue='Model Type', data=plot_data)
    
    plt.title(f'Specialized vs. Baseline Model Performance ({metric.capitalize()})', fontsize=16)
    plt.xlabel('Demographic Group', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model Type')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'specialized_vs_baseline_{metric}.png', dpi=300)
    plt.show()
    
    return comparison_df
