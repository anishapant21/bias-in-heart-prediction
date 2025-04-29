import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ----------------- DATA LOADING AND PREPROCESSING -----------------

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

def identify_feature_types(df):
    """
    Identify numerical and categorical features in the dataset
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
    
    print("\nNumerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    
    return numerical_features, categorical_features

def create_demographic_groups(df):
    """
    Create demographic groups based on age and gender
    """
    # Convert Sex to string representation for clarity
    df['Sex'] = df['Sex'].replace({0: 'Female', 1: 'Male'})
    
    # Set up age groups
    df['Age Group'] = pd.cut(df['Age'], bins=[29, 50, 60, 100], 
                            labels=["30s-40s", "50s", "60+"])
    
    # Create gender-age intersectional groups
    df['Gender_Age_Group'] = df['Sex'] + "_" + df['Age Group'].astype(str)
    
    # Print demographic distributions
    print("\nGender distribution:")
    print(df['Sex'].value_counts())
    
    print("\nAge group distribution:")
    print(df['Age Group'].value_counts())
    
    print("\nIntersectional group distribution:")
    print(df['Gender_Age_Group'].value_counts())
    
    # Drop rows with NaN in Age Group to avoid issues
    df = df.dropna(subset=['Age Group'])
    
    return df

# ----------------- PREPROCESSING PIPELINE -----------------

def get_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline for numerical and categorical features
    """
    # Create transformers for different column types
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified in the transformers
    )
    
    return preprocessor

# ----------------- FAIRNESS-CONSTRAINED OPTIMIZATION -----------------

def train_fairness_constrained_models(df, numerical_features, categorical_features):
    """
    Train separate models for each demographic subgroup with fairness constraints
    """
    # Prepare features and target
    X = df[numerical_features + categorical_features]
    y = df['Diagnosis']
    
    # Get all demographic groups
    demographic_groups = df['Gender_Age_Group'].unique()
    
    # Dictionary to store models and results
    group_models = {}
    group_results = {}
    
    # First, measure baseline performance for each group
    print("\n--- Baseline Group Performance ---")
    
    # Create preprocessing pipeline
    preprocessor = get_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train baseline model
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    baseline_pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_pred = baseline_pipeline.predict(X_test)
    
    # Calculate group-specific performance
    group_performance = {}
    for group in demographic_groups:
        group_mask = df.loc[X_test.index, 'Gender_Age_Group'] == group
        if sum(group_mask) > 0:  # Ensure we have test samples for this group
            group_y_test = y_test[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Calculate metrics
            group_accuracy = accuracy_score(group_y_test, group_y_pred)
            # Use zero_division=0 to avoid warnings
            group_f1 = f1_score(group_y_test, group_y_pred, zero_division=0)
            
            group_performance[group] = {
                'accuracy': group_accuracy,
                'f1_score': group_f1,
                'sample_count': sum(group_mask)
            }
            
            print(f"Group {group}: Accuracy = {group_accuracy:.4f}, F1 = {group_f1:.4f}, Samples = {sum(group_mask)}")
    
    # Identify underperforming groups (groups with below-average F1 score)
    avg_f1 = np.mean([perf['f1_score'] for perf in group_performance.values()])
    print(f"\nAverage F1 Score across groups: {avg_f1:.4f}")
    
    underperforming_groups = {group: perf for group, perf in group_performance.items() 
                             if perf['f1_score'] < avg_f1}
    
    print("\nUnderperforming groups:")
    for group, perf in underperforming_groups.items():
        print(f"Group {group}: F1 = {perf['f1_score']:.4f}")
    
    # Now train separate models for each demographic group with appropriate weighting
    print("\n--- Training Group-Specific Models with Fairness Constraints ---")
    
    for group in demographic_groups:
        print(f"\nTraining model for group: {group}")
        
        # Get group-specific data
        group_mask = df['Gender_Age_Group'] == group
        group_df = df[group_mask]
        
        if len(group_df) < 20:  # Skip if too few samples
            print(f"  Skipping group {group} due to insufficient samples ({len(group_df)})")
            continue
            
        # Prepare features and target for this group
        group_X = group_df[numerical_features + categorical_features]
        group_y = group_df['Diagnosis']
        
        # Split the data
        group_X_train, group_X_test, group_y_train, group_y_test = train_test_split(
            group_X, group_y, test_size=0.3, random_state=42
        )
        
        # Create preprocessing pipeline for this group
        group_preprocessor = get_preprocessing_pipeline(numerical_features, categorical_features)
        
        # Determine appropriate class weights
        # If this group is underperforming, apply stronger weights to balance
        if group in underperforming_groups:
            # Calculate class distribution
            class_counts = group_y_train.value_counts()
            # Stronger weights for minority class in underperforming groups
            weight_factor = 1.5  # Increase this for stronger fairness constraints
            
            # Handle the case where a class might be missing
            if len(class_counts) < 2:
                class_weight = 'balanced'
            else:
                ratio = (class_counts.max() / class_counts.min()) * weight_factor
                # Create class weight dictionary - stronger weighting for underperforming groups
                class_weight = {0: 1, 1: ratio if class_counts.idxmin() == 1 else 1/ratio}
            
            print(f"  Applied stronger class weights for underperforming group: {class_weight}")
        else:
            # Standard balanced weighting for well-performing groups
            class_weight = 'balanced'
        
        # Create and train model pipeline with appropriate constraints
        group_pipeline = Pipeline([
            ('preprocessor', group_preprocessor),
            ('classifier', LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=42
            ))
        ])
        
        # Fit the model
        group_pipeline.fit(group_X_train, group_y_train)
        
        # Evaluate on test set
        group_y_pred = group_pipeline.predict(group_X_test)
        
        # Calculate and store metrics (use zero_division=0 to avoid warnings)
        group_accuracy = accuracy_score(group_y_test, group_y_pred)
        group_precision = precision_score(group_y_test, group_y_pred, zero_division=0)
        group_recall = recall_score(group_y_test, group_y_pred, zero_division=0)
        group_f1 = f1_score(group_y_test, group_y_pred, zero_division=0)
        
        # Store results
        group_results[group] = {
            'accuracy': group_accuracy,
            'precision': group_precision,
            'recall': group_recall,
            'f1_score': group_f1,
            'sample_count': len(group_X_test)
        }
        
        # Store model and pipeline
        group_models[group] = {
            'pipeline': group_pipeline
        }
        
        print(f"  Results for group {group}:")
        print(f"    Accuracy: {group_accuracy:.4f}")
        print(f"    Precision: {group_precision:.4f}")
        print(f"    Recall: {group_recall:.4f}")
        print(f"    F1 Score: {group_f1:.4f}")
        print(f"    Test samples: {len(group_X_test)}")
    
    # Compare results before and after fairness constraints
    print("\n--- Performance Comparison (Before vs. After Fairness Constraints) ---")
    for group in demographic_groups:
        if group in group_results:
            baseline_f1 = group_performance.get(group, {}).get('f1_score', 0)
            constrained_f1 = group_results[group]['f1_score']
            improvement = constrained_f1 - baseline_f1
            
            # Handle division by zero
            if baseline_f1 == 0:
                if constrained_f1 > 0:
                    percent_change = "∞ (infinite improvement from zero)"
                else:
                    percent_change = "0.0% (no change from zero)"
            else:
                percent_change = f"{improvement/baseline_f1*100:.1f}%"
            
            print(f"Group {group}:")
            print(f"  Baseline F1: {baseline_f1:.4f}")
            print(f"  Fairness-Constrained F1: {constrained_f1:.4f}")
            print(f"  Improvement: {improvement:.4f} ({percent_change} change)")
    
    return group_models, group_results

def evaluate_fairness_metrics(group_results):
    """
    Calculate fairness metrics across all demographic groups
    """
    print("\n--- Fairness Metrics ---")
    
    # Extract F1 scores for all groups
    f1_scores = [results['f1_score'] for group, results in group_results.items()]
    
    if not f1_scores:  # Handle the case when no groups have results
        print("No group results available for fairness metrics calculation.")
        return {}
    
    # Calculate fairness metrics
    min_f1 = min(f1_scores)
    max_f1 = max(f1_scores)
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    # Calculate disparate impact (ratio of minimum to maximum performance)
    # Handle the case where max_f1 is zero
    if max_f1 == 0:
        disparate_impact = 1.0 if min_f1 == 0 else 0.0
    else:
        disparate_impact = min_f1 / max_f1
    
    # Calculate coefficient of variation (measure of dispersion)
    # Handle the case where avg_f1 is zero
    if avg_f1 == 0:
        cv = 0.0
    else:
        cv = std_f1 / avg_f1
    
    print(f"Minimum F1 Score: {min_f1:.4f}")
    print(f"Maximum F1 Score: {max_f1:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Standard Deviation of F1 Scores: {std_f1:.4f}")
    print(f"Disparate Impact Ratio (min/max): {disparate_impact:.4f}")
    print(f"Coefficient of Variation: {cv:.4f}")
    
    # A higher disparate impact ratio (closer to 1.0) indicates more equitable performance
    # A lower coefficient of variation indicates more consistent performance across groups
    
    return {
        'min_f1': min_f1,
        'max_f1': max_f1,
        'avg_f1': avg_f1,
        'std_f1': std_f1,
        'disparate_impact': disparate_impact,
        'coefficient_of_variation': cv
    }

def visualize_fairness_results(group_results):
    """
    Visualize the performance of each demographic group after fairness constraints
    """
    if not group_results:  # Handle the case when no groups have results
        print("No group results available for visualization.")
        return
    
    # Set up figure
    plt.figure(figsize=(14, 8))
    
    # Extract data for plotting
    groups = list(group_results.keys())
    accuracies = [results['accuracy'] for results in group_results.values()]
    precisions = [results['precision'] for results in group_results.values()]
    recalls = [results['recall'] for results in group_results.values()]
    f1_scores = [results['f1_score'] for results in group_results.values()]
    
    # Setup bar chart
    x = np.arange(len(groups))
    width = 0.2
    
    # Create grouped bar chart
    plt.bar(x - width*1.5, accuracies, width, label='Accuracy')
    plt.bar(x - width/2, precisions, width, label='Precision')
    plt.bar(x + width/2, recalls, width, label='Recall')
    plt.bar(x + width*1.5, f1_scores, width, label='F1 Score')
    
    # Add labels and legend
    plt.xlabel('Demographic Groups')
    plt.ylabel('Score')
    plt.title('Model Performance Across Demographic Groups')
    plt.xticks(x, groups, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save or show the plot
    plt.savefig('fairness_results.png')
    plt.show()

# ----------------- MAIN EXECUTION -----------------
def main():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Identify feature types
        numerical_features, categorical_features = identify_feature_types(df)
        
        # Create demographic groups
        df = create_demographic_groups(df)
        
        # Train fairness-constrained models
        group_models, group_results = train_fairness_constrained_models(df, numerical_features, categorical_features)
        
        # Evaluate fairness metrics
        fairness_metrics = evaluate_fairness_metrics(group_results)
        
        # Visualize results
        visualize_fairness_results(group_results)
        
        print("\nFairness-Constrained Optimization completed successfully.")
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()