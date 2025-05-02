"""
Heart Disease Analysis - Demographic Subgroup Analysis
This script analyzes heart disease prediction models across different demographic subgroups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

# ----------------- DEMOGRAPHIC SUBGROUP ANALYSIS -----------------

def analyze_coefficients_for_subgroup(df, subgroup_name, subgroup_condition):
    """
    Train a logistic regression model on a specific subgroup and analyze coefficients
    """
    # Filter the dataframe for the subgroup
    subgroup_df = df[subgroup_condition]
    
    print(f"\n{subgroup_name} sample size: {len(subgroup_df)}")
    
    # Skip if we don't have enough samples
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

def analyze_demographic_groups(df):
    """
    Analyze coefficient patterns across different demographic groups
    """
    # Define gender conditions
    print("\nUnique values in Sex column:", df['Sex'].unique())
    male_condition = df['Sex'] == 'Male'
    female_condition = df['Sex'] == 'Female'
    
    # Verify selection counts
    print("\nMales selected:", male_condition.sum())
    print("Females selected:", female_condition.sum())
    
    # Analyze by gender
    print("\n===== Gender-based Coefficient Analysis =====")
    male_coeffs = analyze_coefficients_for_subgroup(df, "Male", male_condition)
    female_coeffs = analyze_coefficients_for_subgroup(df, "Female", female_condition)
    
    # Analyze by age groups
    print("\n===== Age-based Coefficient Analysis =====")
    age_coeffs = {}
    for age_group in ["30s-40s", "50s", "60+"]:
        age_coeffs[age_group] = analyze_coefficients_for_subgroup(
            df, f"Age {age_group}", df['Age Group'] == age_group
        )
    
    # Define and analyze intersectional groups
    intersectional_groups = [
        ("Male 30s-40s", (male_condition) & (df['Age Group'] == "30s-40s")),
        ("Male 50s", (male_condition) & (df['Age Group'] == "50s")),
        ("Male 60+", (male_condition) & (df['Age Group'] == "60+")),
        ("Female 30s-40s", (female_condition) & (df['Age Group'] == "30s-40s")),
        ("Female 50s", (female_condition) & (df['Age Group'] == "50s")),
        ("Female 60+", (female_condition) & (df['Age Group'] == "60+"))
    ]
    
    # Verify group sizes
    print("\n===== Intersectional Group Sizes =====")
    for name, condition in intersectional_groups:
        print(f"{name} size: {condition.sum()}")
    
    # Analyze intersectional groups
    print("\n===== Intersectional Group Coefficient Analysis =====")
    intersect_coeffs = {}
    for group_name, condition in intersectional_groups:
        intersect_coeffs[group_name] = analyze_coefficients_for_subgroup(
            df, group_name, condition
        )
    
    return {
        'male_condition': male_condition,
        'female_condition': female_condition,
        'male_coeffs': male_coeffs,
        'female_coeffs': female_coeffs,
        'age_coeffs': age_coeffs,
        'intersectional_groups': intersectional_groups,
        'intersect_coeffs': intersect_coeffs
    }

# ----------------- VISUALIZATION FUNCTIONS -----------------
def plot_gender_disease_distribution(df):
    """
    Display bar plot of gender vs disease presence
    """
    # Map 'Sex' column if it's numeric (ensure compatibility)
    if df['Sex'].dtype in [int, float]:
        df['Sex'] = df['Sex'].map({0: 'Female', 1: 'Male'})

    # Create a grouped bar plot
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Sex', hue='Diagnosis', palette='Set2')
    plt.title('Heart Disease Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Heart Disease', labels=['No Disease', 'Disease'])
    plt.tight_layout()
    plt.show()

def visualize_coefficient_comparison(group1_name, group1_coeffs, group2_name, group2_coeffs, top_n=5):
    """
    Create a visualization comparing coefficients between two demographic groups
    """
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
    
    return comparison_df

def create_coefficient_visualizations(analysis_results):
    """
    Create various coefficient comparison visualizations
    """
    # Gender comparison
    visualize_coefficient_comparison(
        "Male", analysis_results['male_coeffs'], 
        "Female", analysis_results['female_coeffs']
    )
    
    # Age comparison (40s vs 60s)
    visualize_coefficient_comparison(
        "Age 40s", analysis_results['age_coeffs']["30s-40s"], 
        "Age 60s", analysis_results['age_coeffs']["60+"]
    )
    
    # Intersectional comparison (Male 50s vs Female 50s)
    visualize_coefficient_comparison(
        "Male 50s", analysis_results['intersect_coeffs']["Male 50s"], 
        "Female 50s", analysis_results['intersect_coeffs']["Female 50s"]
    )

# ----------------- MODEL TRAINING AND EVALUATION -----------------

def prepare_data_for_modeling(df):
    """
    Prepare data for model training and evaluation
    """
    # Keep demographic columns for subgroup analysis
    X_with_demographics = df.drop(['Diagnosis'], axis=1)
    
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
    
    # Also keep demographic data for analysis
    X_train_with_demo = X_with_demographics.iloc[X_train_idx]
    X_test_with_demo = X_with_demographics.iloc[X_test_idx]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X_train_with_demo, X_test_with_demo

def cross_validate_model(X_train_scaled, y_train):
    """
    Perform cross-validation to evaluate model performance
    """
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define model
    model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
    
    # Perform cross-validation and get scores
    print("\n===== Cross-Validation Results =====")
    cv_metrics = {
        'accuracy': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy'),
        'precision': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='precision'),
        'recall': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='recall'),
        'f1': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1'),
        'roc_auc': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    }
    
    # Print cross-validation results
    for metric_name, scores in cv_metrics.items():
        print(f"Cross-validated {metric_name.capitalize()}: {scores.mean():.4f} (±{scores.std():.4f})")
    
    return cv_metrics

def tune_hyperparameters(X_train_scaled, y_train):
    """
    Perform hyperparameter tuning via grid search
    """
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
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
        scoring='roc_auc',
        n_jobs=-1  # Use all available cores
    )
    
    # Fit grid search
    print("\n===== Performing Hyperparameter Tuning with Cross-Validation =====")
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluate overall model performance on the test set
    """
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print results
    print("\nFinal Model Performance on Test Set:")
    for metric_name, score in metrics.items():
        print(f"{metric_name.capitalize()}: {score:.4f}")
    
    return metrics

def evaluate_subgroup_performance(model, X_test, y_test, scaler, subgroup_name, subgroup_condition):
    """
    Evaluate model performance on a specific demographic subgroup
    """
    # Get indices of test set samples in this subgroup
    subgroup_indices = np.where(subgroup_condition)[0]
    
    if len(subgroup_indices) < 10:
        print(f"Skipping {subgroup_name} due to insufficient test samples")
        return None
    
    # Get predictions for this subgroup
    X_sub_test = X_test.iloc[subgroup_indices]
    y_sub_test = y_test.iloc[subgroup_indices]
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

def evaluate_demographic_subgroups(model, X_test, y_test, scaler, analysis_results, X_test_with_demo):
    """
    Evaluate model performance across different demographic subgroups
    """
    # Evaluate performance across gender groups
    print("\n===== Model Performance by Gender =====")
    male_condition = X_test_with_demo['Sex'] == 'Male'
    female_condition = X_test_with_demo['Sex'] == 'Female'
    
    male_metrics = evaluate_subgroup_performance(
        model, X_test, y_test, scaler, 
        "Male", male_condition
    )
    female_metrics = evaluate_subgroup_performance(
        model, X_test, y_test, scaler, 
        "Female", female_condition
    )
    
    # Evaluate performance across age groups
    print("\n===== Model Performance by Age Group =====")
    age_metrics = {}
    for age_group in ["30s-40s", "50s", "60+"]:
        age_condition = X_test_with_demo['Age Group'] == age_group
        age_metrics[age_group] = evaluate_subgroup_performance(
            model, X_test, y_test, scaler,
            f"Age {age_group}", age_condition
        )
    
    # Evaluate performance across intersectional groups
    print("\n===== Model Performance by Intersectional Group =====")
    intersect_metrics = {}
    intersectional_groups = [
        ("Male 30s-40s", (male_condition) & (X_test_with_demo['Age Group'] == "30s-40s")),
        ("Male 50s", (male_condition) & (X_test_with_demo['Age Group'] == "50s")),
        ("Male 60+", (male_condition) & (X_test_with_demo['Age Group'] == "60+")),
        ("Female 30s-40s", (female_condition) & (X_test_with_demo['Age Group'] == "30s-40s")),
        ("Female 50s", (female_condition) & (X_test_with_demo['Age Group'] == "50s")),
        ("Female 60+", (female_condition) & (X_test_with_demo['Age Group'] == "60+"))
    ]
    
    for group_name, condition in intersectional_groups:
        intersect_metrics[group_name] = evaluate_subgroup_performance(
            model, X_test, y_test, scaler,
            group_name, condition
        )
    
    return {
        'male_metrics': male_metrics,
        'female_metrics': female_metrics,
        'age_metrics': age_metrics,
        'intersect_metrics': intersect_metrics
    }

def plot_performance_comparison(metrics_list, metric_name="accuracy", title=None):
    """
    Plot comparison of a specific performance metric across groups
    """
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
    
    plt.ylim(0, max(plot_data[metric_name]) * 1.2)  # Add space for labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'performance_{metric_name}.png', dpi=300)

def visualize_performance_metrics(evaluation_results):
    """
    Create visualizations of performance metrics across demographic subgroups
    """
    # Combine all metrics for visualization
    all_metrics = [
        evaluation_results['male_metrics'], 
        evaluation_results['female_metrics']
    ]
    all_metrics.extend([m for m in evaluation_results['age_metrics'].values() if m is not None])
    all_metrics.extend([m for m in evaluation_results['intersect_metrics'].values() if m is not None])
    
    # Plot comparisons of different metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_performance_comparison(all_metrics, metric)
    
    # Plot ROC AUC separately (since not all groups might have it)
    roc_metrics = [m for m in all_metrics if m is not None and 'roc_auc' in m]
    if roc_metrics:
        plot_performance_comparison(roc_metrics, 'roc_auc')

def plot_combined_performance_metrics(evaluation_results, save_path='combined_performance_metrics.png'):
    """
    Create a single visualization showing accuracy, precision, recall, and F1 score
    across different demographic subgroups.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Combine all metrics for visualization
    all_metrics = [
        evaluation_results['male_metrics'], 
        evaluation_results['female_metrics']
    ]
    all_metrics.extend([m for m in evaluation_results['age_metrics'].values() if m is not None])
    all_metrics.extend([m for m in evaluation_results['intersect_metrics'].values() if m is not None])
    
    # Filter out None values
    all_metrics = [m for m in all_metrics if m is not None]
    
    if not all_metrics:
        print("No metrics to visualize")
        return
    
    # Create dataframe for plotting with all metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    plot_data = []
    
    for metric_dict in all_metrics:
        group_name = metric_dict['subgroup']
        for metric in metrics_to_plot:
            if metric in metric_dict:
                plot_data.append({
                    'Group': group_name,
                    'Metric': metric.capitalize(),
                    'Value': metric_dict[metric]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Create the grouped bar chart
    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="Group", y="Value", hue="Metric",
        palette="Set2", alpha=0.9, height=8, aspect=1.5,
        legend_out=False
    )
    
    # Customize the plot
    g.set_xticklabels(rotation=45, ha='right')
    g.set(ylim=(0, 1.0))
    g.set_axis_labels("Demographic Group", "Score Value")
    g.legend.set_title("Performance Metrics")
    
    plt.title('Heart Disease Model Performance Across Demographics', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Add value labels on bars
    # Get the current axis from the FacetGrid
    ax = g.axes[0, 0]
    
    # Iterate through the bars
    for i, bar in enumerate(ax.patches):
        # Get the height of the bar
        height = bar.get_height()
        # Add text label
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01, 
            f'{height:.2f}', 
            ha='center', va='bottom',
            fontsize=8
        )
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined metrics visualization saved to {save_path}")
    
    return g

def plot_advanced_combined_metrics(evaluation_results, save_path='advanced_performance_metrics.png'):
    """
    Create an advanced visualization showing all metrics in a single figure
    with a more sophisticated layout using subplots.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Combine all metrics for visualization
    all_metrics = [
        evaluation_results['male_metrics'], 
        evaluation_results['female_metrics']
    ]
    all_metrics.extend([m for m in evaluation_results['age_metrics'].values() if m is not None])
    all_metrics.extend([m for m in evaluation_results['intersect_metrics'].values() if m is not None])
    
    # Filter out None values
    all_metrics = [m for m in all_metrics if m is not None]
    
    if not all_metrics:
        print("No metrics to visualize")
        return
    
    # Create dataframe for plotting with all metrics
    plot_data = pd.DataFrame([
        {
            'Group': m['subgroup'],
            'Accuracy': m.get('accuracy', np.nan),
            'Precision': m.get('precision', np.nan),
            'Recall': m.get('recall', np.nan),
            'F1 Score': m.get('f1', np.nan),
            'Sample Size': m['size']
        }
        for m in all_metrics
    ])
    
    # Set up the figure with GridSpec for more control
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = sns.color_palette("Set2", len(plot_data))
    
    # Create a bar plot for each metric
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[i//2, i%2])
        
        # Sort data by this metric value
        sorted_data = plot_data.sort_values(by=metric, ascending=False)
        
        # Create the bar plot
        bars = sns.barplot(x='Group', y=metric, data=sorted_data, ax=ax, palette=colors)
        
        # Add value labels
        for j, bar in enumerate(bars.patches):
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom',
                fontsize=9
            )
        
        # Customize subplot
        ax.set_title(f'{metric} by Demographic Group', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Add sample size as text below x-labels
        for tick, group in zip(ax.get_xticklabels(), sorted_data['Group']):
            sample = sorted_data[sorted_data['Group'] == group]['Sample Size'].values[0]
            ax.text(
                tick.get_position()[0],
                -0.07,
                f'n={sample}',
                ha='center',
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                alpha=0.7
            )
    
    # Add overall title
    plt.suptitle('Heart Disease Model Performance Metrics by Demographic Group', 
                fontsize=16, y=0.98)
    
    # Add a text note about sample sizes
    fig.text(0.5, 0.01, "Note: 'n' values indicate sample size for each group", 
             ha='center', fontsize=10, style='italic')
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Advanced combined metrics visualization saved to {save_path}")
    
    return fig

# ----------------- MAIN EXECUTION -----------------

def main():
    """
    Main execution function
    """
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    # Step 2: Identify feature types
    numerical_features, categorical_features = identify_feature_types(df)
    
    # Step 3: Create demographic groups
    df = create_demographic_groups(df)
    
    # Step 4: Analyze demographic groups and their feature coefficients
    analysis_results = analyze_demographic_groups(df)

    plot_gender_disease_distribution(df)
    
    # Step 5: Create coefficient visualizations
    create_coefficient_visualizations(analysis_results)
    
    # Step 6: Prepare data for modeling
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X_train_with_demo, X_test_with_demo = prepare_data_for_modeling(df)
    
    # Step 7: Cross-validate model
    cv_metrics = cross_validate_model(X_train_scaled, y_train)
    
    # Step 8: Perform hyperparameter tuning
    best_model = tune_hyperparameters(X_train_scaled, y_train)
    
    # Step 9: Train and evaluate final model
    best_model.fit(X_train_scaled, y_train)
    overall_metrics = evaluate_model(best_model, X_test_scaled, y_test)
    
    # Step 10: Evaluate model performance on demographic subgroups
    evaluation_results = evaluate_demographic_subgroups(best_model, X_test, y_test, scaler, analysis_results, X_test_with_demo)
    
    # Step 11: Visualize performance metrics across groups
    visualize_performance_metrics(evaluation_results)
    
    # Step 12: Create combined visualization of all metrics
    plot_combined_performance_metrics(evaluation_results)

    plot_advanced_combined_metrics(evaluation_results)
    
    return {
        'df': df,
        'analysis_results': analysis_results,
        'model': best_model,
        'overall_metrics': overall_metrics,
        'evaluation_results': evaluation_results
    }

if __name__ == "__main__":
    results = main()