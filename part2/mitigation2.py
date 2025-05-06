import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_and_prepare_data(file_path='./dataset/heart_disease_uci.csv'):
    """
    Load and preprocess the heart disease dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the heart disease dataset CSV file
        
    Returns:
    --------
    df : DataFrame
        Preprocessed dataframe with demographic groups
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df = df.drop(['id', 'dataset'], axis=1)
    
    # Rename columns for clarity
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
    
    # Convert diagnosis to binary (0 = no heart disease, 1 = heart disease)
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 0 if x == 0 else 1)
    
    # Print missing values
    print("Missing values before removal:")
    print(df.isnull().sum())
    print(f"Original dataset shape: {df.shape}")
    
    # Drop rows with missing values
    df = df.dropna()
    print("\nDataset shape after removing missing values:")
    print(df.shape)
    
    # Identify feature types
    numerical_features, categorical_features = identify_feature_types(df)
    print("\nNumerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    
    # Create demographic groups
    df = create_demographic_groups(df)
    
    # Print demographic distributions
    print_demographic_distributions(df)
    
    return df

def identify_feature_types(df):
    """
    Identify numerical and categorical features in the dataset
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe
        
    Returns:
    --------
    numerical_features : list
        List of numerical feature names
    categorical_features : list
        List of categorical feature names
    """
    numerical_features = []
    categorical_features = []
    
    for column in df.columns:
        if column == 'Diagnosis':
            continue
            
        if df[column].dtype == 'object' or df[column].nunique() < 10:
            categorical_features.append(column)
        else:
            numerical_features.append(column)
            
    return numerical_features, categorical_features

def create_demographic_groups(df):
    """
    Create age groups and intersectional groups
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe
        
    Returns:
    --------
    df : DataFrame
        DataFrame with added demographic group columns
    """
    # Set up the age groups
    df['Age Group'] = pd.cut(df['Age'], bins=[29, 50, 60, 100], 
                            labels=["30s-40s", "50s", "60+"])
    
    # Create gender-age intersectional groups
    df['Gender_Age_Group'] = df['Sex'].astype(str) + "_" + df['Age Group'].astype(str)
    
    return df

def print_demographic_distributions(df):
    """
    Print the distribution of demographic groups
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe with demographic groups
    """
    print("\nGender distribution:")
    print(df['Sex'].value_counts())
    
    print("\nAge group distribution:")
    print(df['Age Group'].value_counts())
    
    print("\nIntersectional group distribution:")
    print(df['Gender_Age_Group'].value_counts())

def prepare_data_for_modeling(df, test_size=0.3, random_state=42):
    """
    Prepare data for modeling by creating feature matrix and target variable,
    splitting into train and test sets, and standardizing features
    
    Parameters:
    -----------
    df : DataFrame
        The preprocessed dataframe
    test_size : float
        Proportion of data to use for testing (default: 0.3)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Testing features
    y_train : Series
        Training target
    y_test : Series
        Testing target
    X_train_scaled : ndarray
        Standardized training features
    X_test_scaled : ndarray
        Standardized testing features
    scaler : StandardScaler
        Fitted scaler for future transformations
    """
    # Prepare feature matrix and target variable
    X = df.drop(['Diagnosis', 'Age Group', 'Gender_Age_Group'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
    y = df['Diagnosis']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def get_demographic_conditions(df):
    """
    Create conditions for selecting different demographic groups
    
    Parameters:
    -----------
    df : DataFrame
        The preprocessed dataframe
        
    Returns:
    --------
    conditions : dict
        Dictionary of demographic conditions
    intersectional_groups : list
        List of tuples with intersectional group names and conditions
    """
    # Gender conditions
    male_condition = df['Sex'] == 'Male'
    female_condition = df['Sex'] == 'Female'
    
    # Print counts
    print("\nMales selected:", male_condition.sum())
    print("Females selected:", female_condition.sum())
    
    # Create dictionary of conditions
    conditions = {
        'Male': male_condition,
        'Female': female_condition
    }
    
    # Age conditions
    for age_group in ["30s-40s", "50s", "60+"]:
        conditions[f'Age {age_group}'] = df['Age Group'] == age_group
    
    # Define intersectional groups
    intersectional_groups = [
        ("Male 30s-40s", (male_condition) & (df['Age Group'] == "30s-40s")),
        ("Male 50s", (male_condition) & (df['Age Group'] == "50s")),
        ("Male 60+", (male_condition) & (df['Age Group'] == "60+")),
        ("Female 30s-40s", (female_condition) & (df['Age Group'] == "30s-40s")),
        ("Female 50s", (female_condition) & (df['Age Group'] == "50s")),
        ("Female 60+", (female_condition) & (df['Age Group'] == "60+"))
    ]
    
    # Print intersectional group sizes
    for name, condition in intersectional_groups:
        print(f"{name} size: {condition.sum()}")
        conditions[name] = condition
    
    return conditions, intersectional_groups

    from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
    
    print(f"Training specialized model with {len(X_train_demo)} samples")
    
    # Set up cross-validation
    cv_folds = StratifiedKFold(n_splits=min(cv, len(X_train_demo)//2), 
                              shuffle=True, 
                              random_state=random_state)
    
    # Define parameter grid to search
    param_grid = {
        'C': [0.01, 0.1, 1.0, 5.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # liblinear supports both l1 and l2
    }
    
    # Set up grid search with cross-validation
    grid_search = GridSearchCV(
        LogisticRegression(random_state=random_state),
        param_grid,
        cv=cv_folds,
        scoring='roc_auc',  # You can change this to 'accuracy', 'f1', etc.
        n_jobs=-1,
        class_weight='balanced'       # Use all available cores
    )
    
    # Fit grid search
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
    # If demographic condition is provided, filter test data
    if demographic_condition is not None:
        X_test_demo = X_test[demographic_condition.loc[X_test.index]]
        y_test_demo = y_test[demographic_condition.loc[y_test.index]]
        group_prefix = f"{group_name} " if group_name else "Group "
    else:
        X_test_demo = X_test
        y_test_demo = y_test
        group_prefix = ""
    
    # Skip if we don't have enough samples
    if len(X_test_demo) < 10:
        print(f"Skipping evaluation - insufficient test samples: {len(X_test_demo)}")
        return None
    
    # Make predictions
    y_pred = model.predict(X_test_demo)
    y_prob = model.predict_proba(X_test_demo)[:, 1]  # Probability of positive class
    
    # Calculate performance metrics
    metrics = {
        'size': len(X_test_demo),
        'accuracy': accuracy_score(y_test_demo, y_pred),
        'precision': precision_score(y_test_demo, y_pred, zero_division=0),
        'recall': recall_score(y_test_demo, y_pred, zero_division=0),
        'f1': f1_score(y_test_demo, y_pred, zero_division=0)
    }
    
    # Add ROC AUC if both classes are present
    if len(set(y_test_demo)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_test_demo, y_prob)
    
    # Print results
    print(f"\n{group_prefix}Performance (n={metrics['size']}):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
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

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def create_weighted_ensemble(baseline_model, specialized_models, X_val, y_val, demographic_conditions):
    """
    Create a weighted ensemble combining baseline and specialized models
    
    Parameters:
    -----------
    baseline_model : estimator
        Trained baseline model
    specialized_models : dict
        Dictionary of specialized models for each demographic group
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation target
    demographic_conditions : dict
        Dictionary of demographic conditions
        
    Returns:
    --------
    ensemble_model : VotingClassifier
        Weighted ensemble model
    model_weights : dict
        Dictionary of model weights for each demographic group
    """
    # Dictionary to store weights for each model and demographic group
    model_weights = {}
    
    # Calculate weights based on validation accuracy for each demographic group
    for group_name, condition in demographic_conditions.items():
        # Get validation samples for this demographic group
        valid_indices = [idx for idx in X_val.index if idx in condition.index]
        valid_condition = condition.loc[valid_indices]
        X_val_demo = X_val[valid_condition]
        y_val_demo = y_val[valid_condition.index]
        
        if len(X_val_demo) < 5 or group_name not in specialized_models or specialized_models[group_name] is None:
            # Use baseline model if specialized model is unavailable
            model_weights[group_name] = {'baseline': 1.0, 'specialized': 0.0}
            continue
            
        # Calculate accuracy for baseline model on this group
        baseline_pred = baseline_model.predict(X_val_demo)
        baseline_acc = accuracy_score(y_val_demo, baseline_pred)
        
        # Calculate accuracy for specialized model on this group
        specialized_pred = specialized_models[group_name].predict(X_val_demo)
        specialized_acc = accuracy_score(y_val_demo, specialized_pred)
        
        # Normalize to sum to 1
        total_acc = baseline_acc + specialized_acc
        if total_acc > 0:
            model_weights[group_name] = {
                'baseline': baseline_acc / total_acc,
                'specialized': specialized_acc / total_acc
            }
        else:
            model_weights[group_name] = {'baseline': 0.5, 'specialized': 0.5}
    
    # Create a class that implements the weighted ensemble
    class DemographicEnsemble:
        def __init__(self, baseline_model, specialized_models, model_weights, demographic_conditions):
            self.baseline_model = baseline_model
            self.specialized_models = specialized_models
            self.model_weights = model_weights
            self.demographic_conditions = demographic_conditions
            self.classes_ = baseline_model.classes_
            
        def predict(self, X):
            predictions = []
            for idx, sample in X.iterrows():
                # Find which demographic groups this sample belongs to
                group_weights = {}
                for group_name, condition in self.demographic_conditions.items():
                    if idx in condition.index and condition.loc[idx]:
                        if group_name in self.model_weights:
                            group_weights[group_name] = self.model_weights[group_name]
                
                if not group_weights:  # No matching group
                    # Use baseline model
                    predictions.append(self.baseline_model.predict(sample.values.reshape(1, -1))[0])
                else:
                    # Weighted vote from all applicable demographic models
                    baseline_weight = sum(w['baseline'] for w in group_weights.values()) / len(group_weights)
                    specialized_votes = []
                    specialized_weights = []
                    
                    for group, weights in group_weights.items():
                        if group in self.specialized_models and self.specialized_models[group] is not None:
                            specialized_votes.append(
                                self.specialized_models[group].predict(sample.values.reshape(1, -1))[0]
                            )
                            specialized_weights.append(weights['specialized'])
                    
                    # If no specialized models available, use baseline
                    if not specialized_votes:
                        predictions.append(self.baseline_model.predict(sample.values.reshape(1, -1))[0])
                    else:
                        # Weighted voting
                        baseline_vote = self.baseline_model.predict(sample.values.reshape(1, -1))[0]
                        if baseline_weight > sum(specialized_weights) / len(specialized_weights):
                            predictions.append(baseline_vote)
                        else:
                            # Take the most common specialized prediction
                            from collections import Counter
                            specialized_counter = Counter(specialized_votes)
                            predictions.append(specialized_counter.most_common(1)[0][0])
            
            return np.array(predictions)
            
        def predict_proba(self, X):
            probas = []
            for idx, sample in X.iterrows():
                # Similar logic to predict method but for probabilities
                # This is a simplified version
                group_weights = {}
                for group_name, condition in self.demographic_conditions.items():
                    if idx in condition.index and condition.loc[idx]:
                        if group_name in self.model_weights:
                            group_weights[group_name] = self.model_weights[group_name]
                
                if not group_weights:  # No matching group
                    probas.append(self.baseline_model.predict_proba(sample.values.reshape(1, -1))[0])
                else:
                    baseline_weight = sum(w['baseline'] for w in group_weights.values()) / len(group_weights)
                    baseline_proba = self.baseline_model.predict_proba(sample.values.reshape(1, -1))[0] * baseline_weight
                    
                    specialized_probas = np.zeros_like(baseline_proba)
                    total_specialized_weight = 0
                    
                    for group, weights in group_weights.items():
                        if group in self.specialized_models and self.specialized_models[group] is not None:
                            specialized_proba = self.specialized_models[group].predict_proba(
                                sample.values.reshape(1, -1)
                            )[0]
                            specialized_probas += specialized_proba * weights['specialized']
                            total_specialized_weight += weights['specialized']
                    
                    if total_specialized_weight > 0:
                        specialized_probas /= total_specialized_weight
                        # Weighted average of baseline and specialized probabilities
                        final_proba = baseline_proba + specialized_probas
                        final_proba /= 2  # Normalize
                    else:
                        final_proba = baseline_proba
                    
                    probas.append(final_proba)
            
            return np.array(probas)
    
    # Create the ensemble model
    ensemble_model = DemographicEnsemble(
        baseline_model, 
        specialized_models, 
        model_weights, 
        demographic_conditions
    )
    
    return ensemble_model, model_weights


def create_comparison_table(baseline_metrics, specialized_metrics, ensemble_metrics):
    """
    Create a comprehensive comparison table for baseline, specialized, and ensemble models
    """
    # Initialize lists for table rows
    rows = []
    
    # Add overall performance
    if 'Overall' in baseline_metrics and baseline_metrics['Overall'] is not None:
        rows.append({
            'Category': 'Overall',
            'Group': 'All Data',
            'Model Type': 'Baseline',
            'Sample Size': baseline_metrics['Overall']['size'],
            'Accuracy': baseline_metrics['Overall'].get('accuracy', np.nan),
            'Precision': baseline_metrics['Overall'].get('precision', np.nan),
            'Recall': baseline_metrics['Overall'].get('recall', np.nan),
            'F1 Score': baseline_metrics['Overall'].get('f1', np.nan),
            'ROC AUC': baseline_metrics['Overall'].get('roc_auc', np.nan)
        })
    
    if 'Overall' in ensemble_metrics and ensemble_metrics['Overall'] is not None:
        rows.append({
            'Category': 'Overall',
            'Group': 'All Data',
            'Model Type': 'Ensemble',
            'Sample Size': ensemble_metrics['Overall']['size'],
            'Accuracy': ensemble_metrics['Overall'].get('accuracy', np.nan),
            'Precision': ensemble_metrics['Overall'].get('precision', np.nan),
            'Recall': ensemble_metrics['Overall'].get('recall', np.nan),
            'F1 Score': ensemble_metrics['Overall'].get('f1', np.nan),
            'ROC AUC': ensemble_metrics['Overall'].get('roc_auc', np.nan)
        })
    
    # Add performance for each demographic group
    for group_name in baseline_metrics:
        if group_name == 'Overall' or baseline_metrics[group_name] is None:
            continue
            
        category = 'By Gender' if group_name in ['Male', 'Female'] else \
                  'By Age Group' if group_name in ['Age 30s-40s', 'Age 50s', 'Age 60+'] else \
                  'By Intersectional Group'
                  
        display_group = group_name.replace('Age ', '') if category == 'By Age Group' else group_name
        
        # Add baseline model performance
        rows.append({
            'Category': category,
            'Group': display_group,
            'Model Type': 'Baseline',
            'Sample Size': baseline_metrics[group_name]['size'],
            'Accuracy': baseline_metrics[group_name].get('accuracy', np.nan),
            'Precision': baseline_metrics[group_name].get('precision', np.nan),
            'Recall': baseline_metrics[group_name].get('recall', np.nan),
            'F1 Score': baseline_metrics[group_name].get('f1', np.nan),
            'ROC AUC': baseline_metrics[group_name].get('roc_auc', np.nan)
        })
        
        # Add specialized model performance if available
        if group_name in specialized_metrics and specialized_metrics[group_name] is not None:
            rows.append({
                'Category': category,
                'Group': display_group,
                'Model Type': 'Specialized',
                'Sample Size': specialized_metrics[group_name]['size'],
                'Accuracy': specialized_metrics[group_name].get('accuracy', np.nan),
                'Precision': specialized_metrics[group_name].get('precision', np.nan),
                'Recall': specialized_metrics[group_name].get('recall', np.nan),
                'F1 Score': specialized_metrics[group_name].get('f1', np.nan),
                'ROC AUC': specialized_metrics[group_name].get('roc_auc', np.nan)
            })
        
        # Add ensemble model performance if available
        if group_name in ensemble_metrics and ensemble_metrics[group_name] is not None:
            rows.append({
                'Category': category,
                'Group': display_group,
                'Model Type': 'Ensemble',
                'Sample Size': ensemble_metrics[group_name]['size'],
                'Accuracy': ensemble_metrics[group_name].get('accuracy', np.nan),
                'Precision': ensemble_metrics[group_name].get('precision', np.nan),
                'Recall': ensemble_metrics[group_name].get('recall', np.nan),
                'F1 Score': ensemble_metrics[group_name].get('f1', np.nan),
                'ROC AUC': ensemble_metrics[group_name].get('roc_auc', np.nan)
            })
    
    # Create DataFrame and sort
    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(['Category', 'Group', 'Model Type'])
    
    # Save to CSV
    comparison_df.to_csv('results/model_comparison_with_ensemble.csv', index=False)
    
    # Print summary
    print("\nModel Comparison Summary Table:")
    print(comparison_df.to_string())
    
    # Create visualizations
    plot_model_comparison(comparison_df)
    
    return comparison_df

def plot_model_comparison(comparison_df):
    """Create visualizations comparing model performance across demographic groups"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Reshape data for grouped bar chart
        groups = comparison_df['Group'].unique()
        model_types = comparison_df['Model Type'].unique()
        
        for i, group in enumerate(groups):
            group_data = comparison_df[comparison_df['Group'] == group]
            x = np.arange(len(model_types))
            width = 0.8 / len(groups)
            offset = width * i - width * len(groups) / 2 + width / 2
            
            values = []
            for model_type in model_types:
                model_data = group_data[group_data['Model Type'] == model_type]
                if not model_data.empty and not pd.isna(model_data[metric].values[0]):
                    values.append(model_data[metric].values[0])
                else:
                    values.append(0)
            
            plt.bar(x + offset, values, width, label=group)
        
        plt.xlabel('Model Type')
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison Across Models and Demographics')
        plt.xticks(np.arange(len(model_types)), model_types)
        plt.legend(title='Demographic Group')
        plt.tight_layout()
        plt.savefig(f'results/comparison_{metric.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Set up matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

def main():
    """
    Main function to run the heart disease prediction pipeline with ensemble modeling
    """
    print("======= Heart Disease Prediction with Ensemble Modeling =======\n")
    
    # 1. Load and prepare data
    print("Step 1: Loading and preparing data...\n")
    df = load_and_prepare_data('./dataset/heart_disease_uci.csv')
    
    # 2. Get demographic conditions
    print("\nStep 2: Defining demographic groups...\n")
    conditions, _ = get_demographic_conditions(df)
    
    # 3. Prepare data for modeling with train/validation/test split
    print("\nStep 3: Preparing data for modeling...\n")
    X = df.drop(['Diagnosis', 'Age Group', 'Gender_Age_Group'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
    y = df['Diagnosis']
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train baseline and specialized models
    print("\nStep 4: Training baseline and specialized models...\n")
    
    # Train baseline model
    print("\n===== Training Baseline Model =====")
    baseline_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42, class_weight='balanced')
    baseline_model.fit(X_train_scaled, y_train)
    
    # Evaluate baseline model
    print("\n===== Evaluating Baseline Model on Test Set =====")
    baseline_metrics = {}
    baseline_metrics['Overall'] = evaluate_model(baseline_model, X_test, y_test)
    
    # Evaluate baseline model on demographic groups
    for group_name, condition in conditions.items():
        print(f"\n===== Evaluating Baseline Model on {group_name} =====")
        baseline_metrics[group_name] = evaluate_model(
            baseline_model, X_test, y_test, 
            demographic_condition=condition, 
            group_name=group_name
        )
    
    # Train specialized models
    print("\n===== Training Specialized Models =====")
    specialized_models = {}
    specialized_metrics = {}
    
    for group_name, condition in conditions.items():
        # Skip very small groups
        group_size = condition.sum()
        if group_size < 10:
            print(f"\n===== Skipping {group_name} - too few samples ({group_size}) =====")
            specialized_models[group_name] = None
            specialized_metrics[group_name] = None
            continue
            
        print(f"\n===== Training Specialized Model for {group_name} =====")
        
        try:
            # Get training data for this group
            train_indices = [idx for idx in X_train.index if idx in condition.index]
            valid_condition = condition.loc[train_indices]
            X_train_demo = X_train[valid_condition]
            y_train_demo = y_train[valid_condition.index]
            
            # Check if enough samples and classes
            if len(X_train_demo) < 10 or len(np.unique(y_train_demo)) < 2:
                print(f"Insufficient samples or classes for {group_name}")
                specialized_models[group_name] = None
                specialized_metrics[group_name] = None
                continue
                
            # Train model
            specialized_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42, class_weight='balanced')
            specialized_model.fit(scaler.transform(X_train_demo), y_train_demo)
            specialized_models[group_name] = specialized_model
            
            # Evaluate specialized model
            print(f"\n===== Evaluating Specialized Model for {group_name} =====")
            specialized_metrics[group_name] = evaluate_model(
                specialized_model, X_test, y_test, 
                demographic_condition=condition, 
                group_name=group_name
            )
            
        except Exception as e:
            print(f"Error processing {group_name}: {e}")
            specialized_models[group_name] = None
            specialized_metrics[group_name] = None
    
    # 5. Create and evaluate ensemble model
    print("\nStep 5: Creating and evaluating ensemble model...\n")
    
    # Create ensemble model using validation set
    ensemble_model, model_weights = create_weighted_ensemble(
        baseline_model, 
        specialized_models, 
        X_val, y_val, 
        conditions
    )
    
    # Print model weights
    print("\nEnsemble Model Weights:")
    for group, weights in model_weights.items():
        print(f"{group}: Baseline={weights['baseline']:.2f}, Specialized={weights['specialized']:.2f}")
    
    # Evaluate ensemble model
    print("\n===== Evaluating Ensemble Model on Test Set =====")
    ensemble_metrics = {}
    ensemble_metrics['Overall'] = evaluate_model(ensemble_model, X_test, y_test)
    
    # Evaluate ensemble model on demographic groups
    for group_name, condition in conditions.items():
        if group_name in baseline_metrics and baseline_metrics[group_name] is not None:
            print(f"\n===== Evaluating Ensemble Model on {group_name} =====")
            ensemble_metrics[group_name] = evaluate_model(
                ensemble_model, X_test, y_test, 
                demographic_condition=condition, 
                group_name=group_name
            )
    
    # 6. Create comparison table and visualizations
    print("\nStep 6: Creating comparison table and visualizations...\n")
    comparison_table = create_comparison_table(
        baseline_metrics, specialized_metrics, ensemble_metrics
    )
    
    # 7. Save models
    print("\nStep 7: Saving models...\n")
    import joblib
    
    # Save models
    joblib.dump(baseline_model, 'results/baseline_model.pkl')
    joblib.dump(specialized_models, 'results/specialized_models.pkl')
    joblib.dump(ensemble_model, 'results/ensemble_model.pkl')
    joblib.dump(scaler, 'results/scaler.pkl')
    
    print("\n======= Analysis Complete =======")
    print("\nCheck the 'results' directory for saved models, visualizations, and data.")
    
    return baseline_model, specialized_models, ensemble_model, comparison_table

if __name__ == "__main__":
    main()

def create_performance_summary(specialized_metrics, baseline_metrics):
    """
    Create and save a comprehensive table summarizing model performance across all groups
    """
    # Initialize lists for table rows
    rows = []
    
    # Add overall baseline performance
    if 'Overall' in baseline_metrics and baseline_metrics['Overall'] is not None:
        rows.append({
            'Category': 'Baseline',
            'Group': 'Overall',
            'Model Type': 'Baseline',
            'Sample Size': baseline_metrics['Overall']['size'],
            'Accuracy': baseline_metrics['Overall'].get('accuracy', np.nan),
            'Precision': baseline_metrics['Overall'].get('precision', np.nan),
            'Recall': baseline_metrics['Overall'].get('recall', np.nan),
            'F1 Score': baseline_metrics['Overall'].get('f1', np.nan),
            'ROC AUC': baseline_metrics['Overall'].get('roc_auc', np.nan)
        })
    
    # Add gender groups
    for group in ['Male', 'Female']:
        if group in baseline_metrics and baseline_metrics[group] is not None:
            rows.append({
                'Category': 'By Gender',
                'Group': group,
                'Model Type': 'Baseline',
                'Sample Size': baseline_metrics[group]['size'],
                'Accuracy': baseline_metrics[group].get('accuracy', np.nan),
                'Precision': baseline_metrics[group].get('precision', np.nan),
                'Recall': baseline_metrics[group].get('recall', np.nan),
                'F1 Score': baseline_metrics[group].get('f1', np.nan),
                'ROC AUC': baseline_metrics[group].get('roc_auc', np.nan)
            })
            
        if group in specialized_metrics and specialized_metrics[group] is not None:
            rows.append({
                'Category': 'By Gender',
                'Group': group,
                'Model Type': 'Specialized',
                'Sample Size': specialized_metrics[group]['size'],
                'Accuracy': specialized_metrics[group].get('accuracy', np.nan),
                'Precision': specialized_metrics[group].get('precision', np.nan),
                'Recall': specialized_metrics[group].get('recall', np.nan),
                'F1 Score': specialized_metrics[group].get('f1', np.nan),
                'ROC AUC': specialized_metrics[group].get('roc_auc', np.nan)
            })
    
    # Add age groups
    for group in ['Age 30s-40s', 'Age 50s', 'Age 60+']:
        if group in baseline_metrics and baseline_metrics[group] is not None:
            rows.append({
                'Category': 'By Age Group',
                'Group': group.replace('Age ', ''),
                'Model Type': 'Baseline',
                'Sample Size': baseline_metrics[group]['size'],
                'Accuracy': baseline_metrics[group].get('accuracy', np.nan),
                'Precision': baseline_metrics[group].get('precision', np.nan),
                'Recall': baseline_metrics[group].get('recall', np.nan),
                'F1 Score': baseline_metrics[group].get('f1', np.nan),
                'ROC AUC': baseline_metrics[group].get('roc_auc', np.nan)
            })
            
        if group in specialized_metrics and specialized_metrics[group] is not None:
            rows.append({
                'Category': 'By Age Group',
                'Group': group.replace('Age ', ''),
                'Model Type': 'Specialized',
                'Sample Size': specialized_metrics[group]['size'],
                'Accuracy': specialized_metrics[group].get('accuracy', np.nan),
                'Precision': specialized_metrics[group].get('precision', np.nan),
                'Recall': specialized_metrics[group].get('recall', np.nan),
                'F1 Score': specialized_metrics[group].get('f1', np.nan),
                'ROC AUC': specialized_metrics[group].get('roc_auc', np.nan)
            })
    
    # Add intersectional groups
    for group in ['Male 30s-40s', 'Male 50s', 'Male 60+', 'Female 30s-40s', 'Female 50s', 'Female 60+']:
        if group in baseline_metrics and baseline_metrics[group] is not None:
            rows.append({
                'Category': 'By Intersectional Group',
                'Group': group,
                'Model Type': 'Baseline',
                'Sample Size': baseline_metrics[group]['size'],
                'Accuracy': baseline_metrics[group].get('accuracy', np.nan),
                'Precision': baseline_metrics[group].get('precision', np.nan),
                'Recall': baseline_metrics[group].get('recall', np.nan),
                'F1 Score': baseline_metrics[group].get('f1', np.nan),
                'ROC AUC': baseline_metrics[group].get('roc_auc', np.nan)
            })
            
        if group in specialized_metrics and specialized_metrics[group] is not None:
            rows.append({
                'Category': 'By Intersectional Group',
                'Group': group,
                'Model Type': 'Specialized',
                'Sample Size': specialized_metrics[group]['size'],
                'Accuracy': specialized_metrics[group].get('accuracy', np.nan),
                'Precision': specialized_metrics[group].get('precision', np.nan),
                'Recall': specialized_metrics[group].get('recall', np.nan),
                'F1 Score': specialized_metrics[group].get('f1', np.nan),
                'ROC AUC': specialized_metrics[group].get('roc_auc', np.nan)
            })
    
    # Create DataFrame
    summary_df = pd.DataFrame(rows)
    
    # Save to CSV
    summary_df.to_csv('results/performance_summary.csv', index=False)
    
    # Print summary table
    print("\nPerformance Summary Table:")
    print(summary_df.to_string())
    
    return summary_df


# def plot_model_comparison(comparison_df):
#     """Create visualizations comparing model performance across demographic groups"""
#     metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
#     for metric in metrics:
#         plt.figure(figsize=(12, 8))
        
#         # Reshape data for grouped bar chart
#         groups = comparison_df['Group'].unique()
#         model_types = comparison_df['Model Type'].unique()
        
#         for i, group in enumerate(groups):
#             group_data = comparison_df[comparison_df['Group'] == group]
#             x = np.arange(len(model_types))
#             width = 0.8 / len(groups)
#             offset = width * i - width * len(groups) / 2 + width / 2
            
#             values = []
#             for model_type in model_types:
#                 model_data = group_data[group_data['Model Type'] == model_type]
#                 if not model_data.empty and not pd.isna(model_data[metric].values[0]):
#                     values.append(model_data[metric].values[0])
#                 else:
#                     values.append(0)
            
#             plt.bar(x + offset, values, width, label=group)
        
#         plt.xlabel('Model Type')
#         plt.ylabel(metric)
#         plt.title(f'{metric} Comparison Across Models and Demographics')
#         plt.xticks(np.arange(len(model_types)), model_types)
#         plt.legend(title='Demographic Group')
#         plt.tight_layout()
#         plt.savefig(f'results/comparison_{metric.lower().replace(" ", "_")}.png', dpi=300)
#         plt.close()

if __name__ == "__main__":
    main()