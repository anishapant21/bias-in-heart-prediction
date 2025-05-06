def _calculate_instance_weights(self, y_true, y_pred, sensitive_attr):
        """
        Calculate instance weights to improve fairness
        """
        # Start with uniform weights
        weights = np.ones(len(y_true))
        
        # Calculate current unfairness
        if self.fairness_metric == 'demographic_parity':
            # Upweight examples that would improve demographic parity
            # For disadvantaged group, upweight positive predictions
            # For advantaged group, upweight negative predictions
            group_0_selection = np.mean(y_pred[sensitive_attr == 0])
            group_1_selection = np.mean(y_pred[sensitive_attr == 1])
            
            if group_0_selection < group_1_selection:
                # Group 0 is disadvantaged (lower selection rate)
                weights[(y_true == 1) & (sensitive_attr == 0)] *= (1 + self.fairness_weight)
                weights[(y_true == 0) & (sensitive_attr == 1)] *= (1 + self.fairness_weight)
            else:
                # Group 1 is disadvantaged
                weights[(y_true == 1) & (sensitive_attr == 1)] *= (1 + self.fairness_weight)
                weights[(y_true == 0) & (sensitive_attr == 0)] *= (1 + self.fairness_weight)
        
        elif self.fairness_metric == 'equal_opportunity':
            # Upweight examples that would improve equal opportunity
            # For disadvantaged group, upweight true positives
            # For advantaged group, upweight false negatives
            group_0_tpr = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_attr == 0)) / max(1, np.sum((y_true == 1) & (sensitive_attr == 0)))
            group_1_tpr = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_attr == 1)) / max(1, np.sum((y_true == 1) & (sensitive_attr == 1)))
            
            if group_0_tpr < group_1_tpr:
                # Group 0 is disadvantaged (lower true positive rate)
                weights[(y_true == 1) & (y_pred == 0) & (sensitive_attr == 0)] *= (1 + self.fairness_weight)
                weights[(y_true == 1) & (y_pred == 1) & (sensitive_attr == 1)] *= (1 - min(self.fairness_weight, 0.5))
            else:
                # Group 1 is disadvantaged
                weights[(y_true == 1) & (y_pred == 0) & (sensitive_attr == 1)] *= (1 + self.fairness_weight)
                weights[(y_true == 1) & (y_pred == 1) & (sensitive_attr == 0)] *= (1 - min(self.fairness_weight, 0.5))
        
        # Normalize weights
        weights = weights / np.mean(weights) * len(weights)
        
        return weightsy_pred == 1) & (sensitive_attr == 1)] *= (1 - min(self.fairness_weight, 0.5))
            else:
                # Group 1 is disadvantaged
                weights[(y_true == 1) & (y_pred == 0) & (sensitive_attr == 1)] *= (1 + self.fairness_weight)
                weights[(y_true == 1) & (y_pred == 1) & (sensitive_attr == 0)] *= (1 - min(self.fairness_weight, 0.5))
        
        # Normalize weights
        weights = weights / np.mean(weights) * len(weights)
        
        return weightsy_pred == 1) & (sensitive_attr == 1)] *= (1 - min(self.fairness_weight, 0.5))
            else:
                # Group 1 is disadvantaged
                weights[(y_true == 1) & (y_pred == 0) & (sensitive_attr == 1)] *= (1 + self.fairness_weight)
                weights[(y_true == 1) & (y_pred == 1) & (sensitive_attr == 0)] *= (1 - min(self.fairness_weight, 0.5))
        
        # Normalize weights
        weights = weights / np.mean(weights) * len(weights)
        
        return weights

def train_fairness_constrained_model(X_train, y_train, X_train_with_demo, sensitive_attribute='Sex', fairness_metric='demographic_parity'):
    """
    Train a model with fairness constraints
    """
    print(f"\n===== Training Fairness-Constrained Model for {sensitive_attribute} =====")
    print(f"Fairness metric: {fairness_metric}")
    
    # Try different fairness weights
    fairness_weights = [0.1, 0.5, 1.0, 2.0, 5.0]
    fairness_models = {}
    fairness_metrics = {}
    
    for weight in fairness_weights:
        print(f"\nTraining with fairness weight: {weight}")
        
        # Train model with fairness constraint
        fair_model = FairnessLogisticRegression(
            fairness_weight=weight,
            sensitive_attribute=sensitive_attribute,
            fairness_metric=fairness_metric,
            max_iter=50
        )
        
        fair_model.fit(X_train, y_train, X_train_with_demo)
        
        # Store model
        fairness_models[weight] = fair_model
        
        # Get predictions
        y_pred = fair_model.predict(X_train)
        y_prob = fair_model.predict_proba(X_train)[:, 1]
        
        # Get sensitive attribute values
        sensitive_values = X_train_with_demo[sensitive_attribute].values
        
        # Map sensitive attribute to binary (0/1)
        sensitive_groups = np.unique(sensitive_values)
        sensitive_binary = (sensitive_values == sensitive_groups[1]).astype(int)
        
        # Calculate fairness metrics
        demo_parity = fair_model._demographic_parity_violation(y_pred, sensitive_binary)
        equal_opp = fair_model._equal_opportunity_violation(y_pred, y_train.values, sensitive_binary)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_train, y_pred)
        
        # Store metrics
        fairness_metrics[weight] = {
            'demographic_parity_violation': demo_parity,
            'equal_opportunity_violation': equal_opp,
            'accuracy': accuracy
        }
        
        print(f"Demographic Parity Violation: {demo_parity:.4f}")
        print(f"Equal Opportunity Violation: {equal_opp:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
    
    # Choose best model based on fairness-accuracy trade-off
    best_weight = None
    best_score = float('-inf')
    
    for weight, metrics in fairness_metrics.items():
        # Score = accuracy - fairness violations
        if fairness_metric == 'demographic_parity':
            fairness_violation = metrics['demographic_parity_violation']
        else:
            fairness_violation = metrics['equal_opportunity_violation']
        
        score = metrics['accuracy'] - fairness_violation
        
        if score > best_score:
            best_score = score
            best_weight = weight
    
    print(f"\nBest fairness weight: {best_weight}")
    print(f"Metrics: {fairness_metrics[best_weight]}")
    
    best_model = fairness_models[best_weight]
    
    return best_model, fairness_metrics

def evaluate_fairness_model(fairness_model, X_test, y_test, X_test_with_demo, sensitive_attribute='Sex'):
    """
    Evaluate the fairness-constrained model
    """
    print(f"\n===== Evaluating Fairness-Constrained Model for {sensitive_attribute} =====")
    
    # Make predictions
    y_pred = fairness_model.predict(X_test)
    y_prob = fairness_model.predict_proba(X_test)[:, 1]
    
    # Calculate overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print("\nOverall performance:")
    for metric, value in overall_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Calculate metrics by demographic group
    demographic_metrics = {}
    
    for group in X_test_with_demo[sensitive_attribute].unique():
        # Skip null/NaN groups
        if pd.isna(group):
            continue
            
        # Get indices for this group
        group_mask = X_test_with_demo[sensitive_attribute] == group
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < 5:
            print(f"Skipping {sensitive_attribute}={group} - insufficient test samples ({len(group_indices)})")
            continue
        
        # Get test data for this group
        X_group_test = X_test.iloc[group_indices]
        y_group_test = y_test.iloc[group_indices]
        
        # Make predictions
        y_group_pred = fairness_model.predict(X_group_test)
        y_group_prob = fairness_model.predict_proba(X_group_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'size': len(group_indices),
            'accuracy': accuracy_score(y_group_test, y_group_pred),
            'precision': precision_score(y_group_test, y_group_pred, zero_division=0),
            'recall': recall_score(y_group_test, y_group_pred, zero_division=0),
            'f1': f1_score(y_group_test, y_group_pred, zero_division=0)
        }
        
        # Add ROC AUC if both classes are present
        if len(np.unique(y_group_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_group_test, y_group_prob)
        
        demographic_metrics[group] = metrics
        
        # Print results
        print(f"\nPerformance for {sensitive_attribute}={group} (n={metrics['size']}):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Calculate fairness metrics
    sensitive_values = X_test_with_demo[sensitive_attribute].values
    sensitive_groups = np.unique(sensitive_values[~pd.isna(sensitive_values)])
    
    if len(sensitive_groups) == 2:
        # Map sensitive attribute to binary (0/1)
        sensitive_binary = (sensitive_values == sensitive_groups[1]).astype(int)
        
        # Calculate demographic parity (difference in selection rates)
        group_0_selection = np.mean(y_pred[sensitive_values == sensitive_groups[0]])
        group_1_selection = np.mean(y_pred[sensitive_values == sensitive_groups[1]])
        demographic_parity_diff = abs(group_0_selection - group_1_selection)
        
        # Calculate equal opportunity (difference in true positive rates)
        group_0_tpr = np.sum((y_pred == 1) & (y_test == 1) & (sensitive_values == sensitive_groups[0])) / max(1, np.sum((y_test == 1) & (sensitive_values == sensitive_groups[0])))
        group_1_tpr = np.sum((y_pred == 1) & (y_test == 1) & (sensitive_values == sensitive_groups[1])) / max(1, np.sum((y_test == 1) & (sensitive_values == sensitive_groups[1])))
        equal_opportunity_diff = abs(group_0_tpr - group_1_tpr)
        
        # Calculate equalized odds (average difference in FPR and TPR)
        group_0_fpr = np.sum((y_pred == 1) & (y_test == 0) & (sensitive_values == sensitive_groups[0])) / max(1, np.sum((y_test == 0) & (sensitive_values == sensitive_groups[0])))
        group_1_fpr = np.sum((y_pred == 1) & (y_test == 0) & (sensitive_values == sensitive_groups[1])) / max(1, np.sum((y_test == 0) & (sensitive_values == sensitive_groups[1])))
        equalized_odds = (abs(group_0_fpr - group_1_fpr) + abs(group_0_tpr - group_1_tpr)) / 2
        
        fairness_metrics = {
            'demographic_parity_diff': demographic_parity_diff,
            'equal_opportunity_diff': equal_opportunity_diff,
            'equalized_odds': equalized_odds
        }
        
        print("\nFairness metrics:")
        print(f"  Demographic Parity Difference: {demographic_parity_diff:.4f}")
        print(f"  Equal Opportunity Difference: {equal_opportunity_diff:.4f}")
        print(f"  Equalized Odds: {equalized_odds:.4f}")
        
        # Selection rates by group
        print(f"\nSelection rates by {sensitive_attribute}:")
        print(f"  {sensitive_groups[0]}: {group_0_selection:.4f}")
        print(f"  {sensitive_groups[1]}: {group_1_selection:.4f}")
        
        # True positive rates by group
        print(f"\nTrue Positive Rates by {sensitive_attribute}:")
        print(f"  {sensitive_groups[0]}: {group_0_tpr:.4f}")
        print(f"  {sensitive_groups[1]}: {group_1_tpr:.4f}")
    else:
        print(f"Cannot calculate fairness metrics: expected 2 {sensitive_attribute} groups, got {len(sensitive_groups)}")
        fairness_metrics = {}
    
    return overall_metrics, demographic_metrics, fairness_metrics

# ----------------- COMPARISON AND VISUALIZATION -----------------

def compare_all_models(baseline_metrics, demographic_model_metrics, fairness_model_metrics, sensitive_attribute='Sex'):
    """
    Compare all three approaches: baseline, demographic-specific, and fairness-constrained
    """
    print(f"\n===== Comparing All Models for {sensitive_attribute} =====")
    
    # Check if we have metrics for all three approaches
    if not baseline_metrics or not demographic_model_metrics or not fairness_model_metrics:
        print("Missing metrics for one or more approaches")
        return None
    
    # Prepare comparison data
    comparison_data = []
    
    # Common metrics to compare
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Overall performance comparison
    if 'overall' in baseline_metrics and 'overall' in fairness_model_metrics:
        for metric in metrics_to_compare:
            if metric in baseline_metrics['overall'] and metric in fairness_model_metrics['overall']:
                comparison_data.append({
                    'Group': 'Overall',
                    'Metric': metric.capitalize(),
                    'Baseline': baseline_metrics['overall'][metric],
                    'Demographic-Specific': None,  # May not have overall for demographic models
                    'Fairness-Constrained': fairness_model_metrics['overall'][metric]
                })
    
    # Group-specific performance comparison
    for group in baseline_metrics.keys():
        if group == 'overall':
            continue
            
        # Skip if we don't have metrics for this group in all approaches
        if group not in fairness_model_metrics:
            continue
        
        for metric in metrics_to_compare:
            # Convert metric name to match keys in different dictionaries
            baseline_key = metric.lower()
            demographic_key = metric.capitalize()
            fairness_key = metric.lower()
            
            # Check if metric exists for all approaches
            has_baseline = baseline_key in baseline_metrics[group]
            has_fairness = fairness_key in fairness_model_metrics[group]
            
            # Find this group in demographic model results
            has_demographic = False
            demographic_value = None
            
            if demographic_model_metrics is not None:
                for idx, row in demographic_model_metrics.iterrows():
                    if row['Group'] == group and demographic_key in row:
                        has_demographic = True
                        demographic_value = row[demographic_key]
                        break
            
            # Only add comparison if we have at least two approaches to compare
            if has_baseline or has_demographic or has_fairness:
                comparison_data.append({
                    'Group': group,
                    'Metric': metric.capitalize(),
                    'Baseline': baseline_metrics[group][baseline_key] if has_baseline else None,
                    'Demographic-Specific': demographic_value,
                    'Fairness-Constrained': fairness_model_metrics[group][fairness_key] if has_fairness else None
                })
    
    # Convert to DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison table
        print("\nPerformance comparison table:")
        for group in comparison_df['Group'].unique():
            group_data = comparison_df[comparison_df['Group'] == group]
            print(f"\n{sensitive_attribute}={group}:")
            for _, row in group_data.iterrows():
                print(f"  {row['Metric']}:", end=" ")
                if pd.notna(row['Baseline']):
                    print(f"Baseline: {row['Baseline']:.4f}", end=" | ")
                if pd.notna(row['Demographic-Specific']):
                    print(f"Demo-Specific: {row['Demographic-Specific']:.4f}", end=" | ")
                if pd.notna(row['Fairness-Constrained']):
                    print(f"Fairness: {row['Fairness-Constrained']:.4f}", end="")
                print()
        
        # Visualize comparison
        # Melt the DataFrame for easier plotting
        plot_df = comparison_df.melt(
            id_vars=['Group', 'Metric'],
            value_vars=['Baseline', 'Demographic-Specific', 'Fairness-Constrained'],
            var_name='Model',
            value_name='Performance'
        )
        
        # Remove rows with missing values
        plot_df = plot_df.dropna()
        
        # Create a figure for each metric
        for metric in plot_df['Metric'].unique():
            metric_df = plot_df[plot_df['Metric'] == metric]
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Group', y='Performance', hue='Model', data=metric_df)
            
            plt.title(f'{metric} Comparison Across Models', fontsize=14)
            plt.xlabel(sensitive_attribute, fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'model_comparison_{sensitive_attribute}_{metric}.png', dpi=300)
        
        return comparison_df
    else:
        print("No comparison data available")
        return None

def compare_fairness_metrics(baseline_fairness, fairness_constrained_fairness, fairness_metric='demographic_parity'):
    """
    Compare fairness metrics between baseline and fairness-constrained models
    """
    print("\n===== Fairness Metrics Comparison =====")
    
    # Check if we have fairness metrics for both models
    if not baseline_fairness or not fairness_constrained_fairness:
        print("Missing fairness metrics for one or more models")
        return None
    
    # Map fairness metric names between different formats
    metric_mapping = {
        'demographic_parity': 'demographic_parity_diff',
        'equal_opportunity': 'equal_opportunity_diff',
        'equalized_odds': 'equalized_odds'
    }
    
    # Prepare comparison data
    comparison_data = []
    
    for fair_metric, metric_key in metric_mapping.items():
        if metric_key in baseline_fairness and metric_key in fairness_constrained_fairness:
            baseline_value = baseline_fairness[metric_key]
            fairness_value = fairness_constrained_fairness[metric_key]
            improvement = baseline_value - fairness_value
            
            comparison_data.append({
                'Metric': fair_metric.replace('_', ' ').title(),
                'Baseline': baseline_value,
                'Fairness-Constrained': fairness_value,
                'Improvement': improvement,
                'Percent_Improvement': improvement / max(0.0001, baseline_value) * 100
            })
    
    # Convert to DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison table
        print("\nFairness metrics comparison:")
        for _, row in comparison_df.iterrows():
            print(f"  {row['Metric']}:")
            print(f"    Baseline: {row['Baseline']:.4f}")
            print(f"    Fairness-Constrained: {row['Fairness-Constrained']:.4f}")
            print(f"    Improvement: {row['Improvement']:.4f} ({row['Percent_Improvement']:.1f}%)")
            print()
        
        # Visualize comparison
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        plot_df = comparison_df.melt(
            id_vars=['Metric'],
            value_vars=['Baseline', 'Fairness-Constrained'],
            var_name='Model',
            value_name='Value'
        )
        
        sns.barplot(x='Metric', y='Value', hue='Model', data=plot_df)
        
        plt.title('Fairness Metrics Comparison', fontsize=14)
        plt.xlabel('Fairness Metric', fontsize=12)
        plt.ylabel('Violation (lower is better)', fontsize=12)
        plt.tight_layout()
        plt.savefig('fairness_metrics_comparison.png', dpi=300)
        
        # Visualize improvement
        plt.figure(figsize=(10, 6))
        
        bars = sns.barplot(x='Metric', y='Improvement', data=comparison_df, 
                  palette=['green' if x > 0 else 'red' for x in comparison_df['Improvement']])
        
        # Add percentage labels
        for i, bar in enumerate(bars.patches):
            bars.text(bar.get_x() + bar.get_width()/2., 
                     bar.get_height() + 0.001, 
                     f"{comparison_df.iloc[i]['Percent_Improvement']:.1f}%", 
                     ha='center', fontsize=10)
        
        plt.title('Fairness Improvement (Baseline - Fairness-Constrained)', fontsize=14)
        plt.xlabel('Fairness Metric', fontsize=12)
        plt.ylabel('Improvement (higher is better)', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.tight_layout()
        plt.savefig('fairness_improvement.png', dpi=300)
        
        return comparison_df
    else:
        print("No fairness metrics available for comparison")
        return None

# ----------------- MAIN EXECUTION -----------------

def main():
    """
    Main execution function
    """
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    # Step 2: Create demographic groups
    df = create_demographic_groups(df)
    
    # Step 3: Prepare data for modeling
    X, y, X_train, X_test, y_train, y_test, X_train_with_demo, X_test_with_demo, feature_names = prepare_data_for_modeling(df)
    
    # Step 4: Train and evaluate baseline model
    print("\n===== Training Baseline Model =====")
    baseline_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    baseline_model.fit(X_train_scaled, y_train)
    
    # Evaluate baseline model
    baseline_preds = baseline_model.predict(X_test_scaled)
    baseline_probs = baseline_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nBaseline model performance:")
    baseline_overall_metrics = {
        'accuracy': accuracy_score(y_test, baseline_preds),
        'precision': precision_score(y_test, baseline_preds),
        'recall': recall_score(y_test, baseline_preds),
        'f1': f1_score(y_test, baseline_preds),
        'roc_auc': roc_auc_score(y_test, baseline_probs)
    }
    
    for metric, value in baseline_overall_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Step 5: Calculate baseline metrics by demographic group
    baseline_metrics = {'overall': baseline_overall_metrics}
    
    # Gender metrics
    gender_groups = X_test_with_demo['Sex'].unique()
    for gender in gender_groups:
        if pd.isna(gender):
            continue
            
        gender_mask = X_test_with_demo['Sex'] == gender
        gender_indices = np.where(gender_mask)[0]
        
        if len(gender_indices) < 5:
            continue
            
        gender_preds = baseline_preds[gender_indices]
        gender_probs = baseline_probs[gender_indices]
        gender_true = y_test.iloc[gender_indices]
        
        gender_metrics = {
            'accuracy': accuracy_score(gender_true, gender_preds),
            'precision': precision_score(gender_true, gender_preds, zero_division=0),
            'recall': recall_score(gender_true, gender_preds, zero_division=0),
            'f1': f1_score(gender_true, gender_preds, zero_division=0)
        }
        
        if len(np.unique(gender_true)) > 1:
            gender_metrics['roc_auc'] = roc_auc_score(gender_true, gender_probs)
        
        baseline_metrics[gender] = gender_metrics
        
        print(f"\nBaseline performance for Sex={gender}:")
        for metric, value in gender_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Step 6: Calculate baseline fairness metrics
    sensitive_attr = 'Sex'
    sensitive_values = X_test_with_demo[sensitive_attr].values
    sensitive_groups = np.unique(sensitive_values[~pd.isna(sensitive_values)])
    
    if len(sensitive_groups) == 2:
        # Calculate demographic parity
        group_0_selection = np.mean(baseline_preds[sensitive_values == sensitive_groups[0]])
        group_1_selection = np.mean(baseline_preds[sensitive_values == sensitive_groups[1]])
        demographic_parity_diff = abs(group_0_selection - group_1_selection)
        
        # Calculate equal opportunity
        group_0_tpr = np.sum((baseline_preds == 1) & (y_test == 1) & (sensitive_values == sensitive_groups[0])) / max(1, np.sum((y_test == 1) & (sensitive_values == sensitive_groups[0])))
        group_1_tpr = np.sum((baseline_preds == 1) & (y_test == 1) & (sensitive_values == sensitive_groups[1])) / max(1, np.sum((y_test == 1) & (sensitive_values == sensitive_groups[1])))
        equal_opportunity_diff = abs(group_0_tpr - group_1_tpr)
        
        # Calculate equalized odds
        group_0_fpr = np.sum((baseline_preds == 1) & (y_test == 0) & (sensitive_values == sensitive_groups[0])) / max(1, np.sum((y_test == 0) & (sensitive_values == sensitive_groups[0])))
        group_1_fpr = np.sum((baseline_preds == 1) & (y_test == 0) & (sensitive_values == sensitive_groups[1])) / max(1, np.sum((y_test == 0) & (sensitive_values == sensitive_groups[1])))
        equalized_odds = (abs(group_0_fpr - group_1_fpr) + abs(group_0_tpr - group_1_tpr)) / 2
        
        baseline_fairness_metrics = {
            'demographic_parity_diff': demographic_parity_diff,
            'equal_opportunity_diff': equal_opportunity_diff,
            'equalized_odds': equalized_odds
        }
        
        print("\nBaseline fairness metrics:")
        print(f"  Demographic Parity Difference: {demographic_parity_diff:.4f}")
        print(f"  Equal Opportunity Difference: {equal_opportunity_diff:.4f}")
        print(f"  Equalized Odds: {equalized_odds:.4f}")
    else:
        baseline_fairness_metrics = {}
    
    # Step 7: Train demographic-specific models
    demographic_models, demographic_scalers, feature_importances = train_demographic_specific_models(
        X_train, y_train, X_train_with_demo, demographic_col='Sex', use_smote=True
    )
    
    # Step 8: Evaluate demographic-specific models
    demographic_results, demographic_preds, demographic_probs, evaluated_mask = evaluate_demographic_models(
        demographic_models, demographic_scalers, X_test, y_test, X_test_with_demo, demographic_col='Sex'
    )
    
    # Step 9: Compare baseline with demographic-specific models
    demo_comparison = compare_with_baseline(baseline_metrics, demographic_results, demographic_col='Sex')
    
    # Step 10: Visualize feature importance by demographic group
    feature_importance_comparison = visualize_feature_importance(feature_importances, demographic_col='Sex')
    
    # Step 11: Train fairness-constrained model
    fairness_model, fairness_training_metrics = train_fairness_constrained_model(
        X_train, y_train, X_train_with_demo, sensitive_attribute='Sex', fairness_metric='demographic_parity'
    )
    
    # Step 12: Evaluate fairness-constrained model
    fairness_overall, fairness_demo_metrics, fairness_metrics = evaluate_fairness_model(
        fairness_model, X_test, y_test, X_test_with_demo, sensitive_attribute='Sex'
    )
    
    # Step 13: Compare all models
    all_models_comparison = compare_all_models(
        baseline_metrics, demographic_results, fairness_demo_metrics, sensitive_attribute='Sex'
    )
    
    # Step 14: Compare fairness metrics
    fairness_comparison = compare_fairness_metrics(
        baseline_fairness_metrics, fairness_metrics, fairness_metric='demographic_parity'
    )
    
    return {
        'baseline': {
            'model': baseline_model,
            'scaler': scaler,
            'metrics': baseline_metrics,
            'fairness': {
            'model': fairness_model,
            'metrics': fairness_overall,
            'demographic_metrics': fairness_demo_metrics,
            'fairness_metrics': fairness_metrics
        },
        'comparisons': {
            'demographic_vs_baseline': demo_comparison,
            'all_models': all_models_comparison,
            'fairness_metrics': fairness_comparison
        }
    }

if __name__ == "__main__":
    results = main()

        """
Demographic-Specific Heart Disease Models

This script implements separate machine learning models for different demographic groups
and adds fairness constraints to address bias in heart disease prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

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

def prepare_data_for_modeling(df):
    """
    Prepare data for model training and evaluation
    """
    # Keep demographic information
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
    
    return X, y, X_train, X_test, y_train, y_test, X_train_with_demo, X_test_with_demo, X.columns

# ----------------- DEMOGRAPHIC-SPECIFIC MODELS -----------------

def train_demographic_specific_models(X_train, y_train, X_train_with_demo, demographic_col='Sex', use_smote=True):
    """
    Train separate models for each demographic group
    """
    print(f"\n===== Training Demographic-Specific Models for {demographic_col} =====")
    
    demographic_models = {}
    feature_importances = {}
    demographic_scalers = {}
    
    # Train a model for each demographic group
    for group in X_train_with_demo[demographic_col].unique():
        # Skip null/NaN groups
        if pd.isna(group):
            print(f"Skipping {demographic_col}=NaN")
            continue
            
        # Get indices for this group
        group_mask = X_train_with_demo[demographic_col] == group
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < 20:
            print(f"Warning: Small sample size for {demographic_col}={group} (n={len(group_indices)})")
            
        # Get data for this group
        X_group = X_train.iloc[group_indices]
        y_group = y_train.iloc[group_indices]
        
        print(f"\nTraining model for {demographic_col}={group} (n={len(X_group)})")
        print(f"Class distribution: {y_group.value_counts().to_dict()}")
        
        # Apply SMOTE if requested and if the group has at least 5 samples in each class
        if use_smote:
            class_counts = y_group.value_counts()
            if len(class_counts) > 1 and min(class_counts) >= 5:
                print(f"Applying SMOTE to balance classes for {demographic_col}={group}")
                k_neighbors = min(5, min(class_counts) - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_group, y_group = smote.fit_resample(X_group, y_group)
                print(f"After SMOTE: {pd.Series(y_group).value_counts().to_dict()}")
        
        # Standardize features
        scaler = StandardScaler()
        X_group_scaled = scaler.fit_transform(X_group)
        demographic_scalers[group] = scaler
        
        # Choose the best model type for this group
        models_to_try = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_model_name = None
        
        # Find the best model type for this demographic
        for model_name, model in models_to_try.items():
            # Use cross-validation to evaluate
            cv_scores = cross_val_score(model, X_group_scaled, y_group, cv=5, scoring='roc_auc')
            avg_score = np.mean(cv_scores)
            
            print(f"{model_name} CV Score: {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_model_name = model_name
        
        print(f"Best model for {demographic_col}={group}: {best_model_name} (CV Score: {best_score:.4f})")
        
        # If Logistic Regression is best, tune hyperparameters
        if best_model_name == 'LogisticRegression':
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid,
                cv=5,
                scoring='roc_auc'
            )
            
            grid_search.fit(X_group_scaled, y_group)
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Train the final model
        best_model.fit(X_group_scaled, y_group)
        
        # Store model and feature importance
        demographic_models[group] = best_model
        
        # Get feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            # For tree-based models
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X_group.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
        elif hasattr(best_model, 'coef_'):
            # For linear models
            importances = np.abs(best_model.coef_[0])
            feature_importance = pd.DataFrame({
                'Feature': X_group.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            feature_importance = None
        
        feature_importances[group] = feature_importance
        
        if feature_importance is not None:
            print(f"\nTop 5 important features for {demographic_col}={group}:")
            print(feature_importance.head(5))
    
    return demographic_models, demographic_scalers, feature_importances

def evaluate_demographic_models(demographic_models, demographic_scalers, X_test, y_test, X_test_with_demo, demographic_col):
    """
    Evaluate the demographic-specific models
    """
    print(f"\n===== Evaluating Demographic-Specific Models for {demographic_col} =====")
    
    # Create a unified DataFrame for results
    all_results = []
    demographic_specific_predictions = np.zeros(len(X_test))
    demographic_specific_probas = np.zeros(len(X_test))
    
    # Track which samples were evaluated by their specific model
    evaluated_mask = np.zeros(len(X_test), dtype=bool)
    
    # Evaluate each demographic model on its specific group
    for group, model in demographic_models.items():
        # Get indices for this group in the test set
        group_mask = X_test_with_demo[demographic_col] == group
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < 5:
            print(f"Skipping evaluation for {demographic_col}={group} - insufficient test samples ({len(group_indices)})")
            continue
        
        # Get test data for this group
        X_group_test = X_test.iloc[group_indices]
        y_group_test = y_test.iloc[group_indices]
        
        # Scale the test data
        scaler = demographic_scalers[group]
        X_group_scaled = scaler.transform(X_group_test)
        
        # Make predictions
        y_group_pred = model.predict(X_group_scaled)
        
        # Get probabilities for ROC AUC (if available)
        if hasattr(model, 'predict_proba'):
            y_group_proba = model.predict_proba(X_group_scaled)[:, 1]
            has_proba = True
        else:
            y_group_proba = y_group_pred  # Use predictions as a fallback
            has_proba = False
        
        # Store predictions in the combined arrays
        demographic_specific_predictions[group_indices] = y_group_pred
        demographic_specific_probas[group_indices] = y_group_proba
        evaluated_mask[group_indices] = True
        
        # Calculate metrics
        metrics = {
            'Group': group,
            'Model': type(model).__name__,
            'Size': len(group_indices),
            'Accuracy': accuracy_score(y_group_test, y_group_pred),
            'Precision': precision_score(y_group_test, y_group_pred, zero_division=0),
            'Recall': recall_score(y_group_test, y_group_pred, zero_division=0),
            'F1': f1_score(y_group_test, y_group_pred, zero_division=0)
        }
        
        # Add ROC AUC if probabilities are available and both classes are present
        if has_proba and len(np.unique(y_group_test)) > 1:
            metrics['ROC_AUC'] = roc_auc_score(y_group_test, y_group_proba)
        
        all_results.append(metrics)
        
        # Print results
        print(f"\nPerformance for {demographic_col}={group} (n={metrics['Size']}):")
        print(f"  Model type: {metrics['Model']}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1 Score: {metrics['F1']:.4f}")
        if 'ROC_AUC' in metrics:
            print(f"  ROC AUC: {metrics['ROC_AUC']:.4f}")
    
    # Calculate overall performance for samples that had a demographic-specific model
    if np.any(evaluated_mask):
        print("\nOverall performance of demographic-specific models:")
        print(f"  Samples evaluated: {sum(evaluated_mask)} of {len(X_test)}")
        
        overall_metrics = {
            'Accuracy': accuracy_score(y_test[evaluated_mask], demographic_specific_predictions[evaluated_mask]),
            'Precision': precision_score(y_test[evaluated_mask], demographic_specific_predictions[evaluated_mask], zero_division=0),
            'Recall': recall_score(y_test[evaluated_mask], demographic_specific_predictions[evaluated_mask], zero_division=0),
            'F1': f1_score(y_test[evaluated_mask], demographic_specific_predictions[evaluated_mask], zero_division=0)
        }
        
        # Add ROC AUC if both classes are present
        if len(np.unique(y_test[evaluated_mask])) > 1:
            overall_metrics['ROC_AUC'] = roc_auc_score(y_test[evaluated_mask], demographic_specific_probas[evaluated_mask])
        
        for metric, value in overall_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Convert results to DataFrame for further analysis
    results_df = pd.DataFrame(all_results)
    
    return results_df, demographic_specific_predictions, demographic_specific_probas, evaluated_mask

def compare_with_baseline(baseline_metrics, demographic_model_metrics, demographic_col):
    """
    Compare demographic-specific models with baseline model
    """
    print(f"\n===== Comparing Demographic-Specific Models vs. Baseline for {demographic_col} =====")
    
    # Prepare data for comparison
    comparison_data = []
    
    for group in demographic_model_metrics['Group'].unique():
        # Get metrics for this group
        demo_metrics = demographic_model_metrics[demographic_model_metrics['Group'] == group].iloc[0].to_dict()
        
        # Find corresponding baseline metrics
        baseline_group_metrics = baseline_metrics.get(group, None)
        
        if baseline_group_metrics is None:
            print(f"No baseline metrics for {demographic_col}={group}, skipping comparison")
            continue
        
        # Add comparison row
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
            if metric.lower() in baseline_group_metrics and metric in demo_metrics:
                baseline_value = baseline_group_metrics[metric.lower()]
                demo_value = demo_metrics[metric]
                improvement = demo_value - baseline_value
                
                comparison_data.append({
                    'Group': group,
                    'Metric': metric,
                    'Baseline': baseline_value,
                    'Demographic-Specific': demo_value,
                    'Improvement': improvement
                })
        
        # Add ROC AUC comparison if available
        if 'roc_auc' in baseline_group_metrics and 'ROC_AUC' in demo_metrics:
            improvement = demo_metrics['ROC_AUC'] - baseline_group_metrics['roc_auc']
            
            comparison_data.append({
                'Group': group,
                'Metric': 'ROC_AUC',
                'Baseline': baseline_group_metrics['roc_auc'],
                'Demographic-Specific': demo_metrics['ROC_AUC'],
                'Improvement': improvement
            })
    
    # Convert to DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison
        print("\nPerformance difference (Demographic-Specific - Baseline):")
        for group in comparison_df['Group'].unique():
            group_data = comparison_df[comparison_df['Group'] == group]
            print(f"\n{demographic_col}={group}:")
            for _, row in group_data.iterrows():
                print(f"  {row['Metric']}: {row['Baseline']:.4f} → {row['Demographic-Specific']:.4f} ({row['Improvement']:+.4f})")
        
        # Visualize improvement
        plt.figure(figsize=(12, 8))
        
        # Plot improvement by group and metric
        plot_data = comparison_df.copy()
        plot_data['Color'] = plot_data['Improvement'].apply(lambda x: 'green' if x > 0 else 'red')
        
        ax = sns.barplot(x='Group', y='Improvement', hue='Metric', data=plot_data)
        
        plt.title(f'Performance Improvement: Demographic-Specific vs. Baseline ({demographic_col})', fontsize=14)
        plt.xlabel(demographic_col, fontsize=12)
        plt.ylabel('Improvement (Demographic - Baseline)', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'demographic_model_improvement_{demographic_col}.png', dpi=300)
        
        return comparison_df
    else:
        print("No comparison data available")
        return None

def visualize_feature_importance(feature_importances, demographic_col):
    """
    Visualize and compare feature importance across demographic groups
    """
    print(f"\n===== Feature Importance Comparison for {demographic_col} =====")
    
    # Check if we have feature importances to visualize
    if not feature_importances:
        print("No feature importance data available")
        return
    
    # Get all groups
    groups = list(feature_importances.keys())
    
    # Get top N features from each group
    top_n = 5
    all_top_features = set()
    
    for group, importance_df in feature_importances.items():
        if importance_df is not None:
            top_features = importance_df.head(top_n)['Feature'].tolist()
            all_top_features.update(top_features)
    
    # Create a DataFrame for visualization
    comparison_data = []
    
    for feature in all_top_features:
        for group in groups:
            if feature_importances[group] is not None and feature in feature_importances[group]['Feature'].values:
                importance = feature_importances[group][feature_importances[group]['Feature'] == feature]['Importance'].values[0]
                
                comparison_data.append({
                    'Feature': feature,
                    'Group': group,
                    'Importance': importance
                })
    
    if not comparison_data:
        print("No common features found for importance comparison")
        return
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Pivot data for heatmap
    pivot_data = comparison_df.pivot(index='Feature', columns='Group', values='Importance')
    
    # Plot heatmap
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".3f")
    plt.title(f'Feature Importance by {demographic_col}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{demographic_col}.png', dpi=300)
    
    # Create bar plot for each group
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 6), squeeze=False)
    
    for i, group in enumerate(groups):
        group_data = comparison_df[comparison_df['Group'] == group]
        if len(group_data) > 0:
            group_data = group_data.sort_values('Importance', ascending=False).head(top_n)
            
            sns.barplot(x='Importance', y='Feature', data=group_data, ax=axes[0, i])
            axes[0, i].set_title(f'{demographic_col}={group}')
            
            # Only show y-label for first subplot
            if i > 0:
                axes[0, i].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_by_group_{demographic_col}.png', dpi=300)
    
    return comparison_df

# ----------------- FAIRNESS-CONSTRAINED MODEL -----------------

class FairnessLogisticRegression:
    """
    Logistic Regression with fairness constraints
    """
    def __init__(self, fairness_weight=1.0, sensitive_attribute='Sex', 
                 base_model=LogisticRegression(C=1.0, random_state=42),
                 fairness_metric='demographic_parity', max_iter=100):
        """
        Initialize fairness-constrained model
        
        Parameters:
        -----------
        fairness_weight : float
            Weight of the fairness constraint in the loss function
        sensitive_attribute : str
            Name of the sensitive attribute column
        base_model : classifier
            Base model to use (must have fit, predict, and predict_proba methods)
        fairness_metric : str
            Fairness metric to use ('demographic_parity', 'equal_opportunity')
        max_iter : int
            Maximum number of iterations for the optimization
        """
        self.fairness_weight = fairness_weight
        self.sensitive_attribute = sensitive_attribute
        self.base_model = base_model
        self.fairness_metric = fairness_metric
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y, X_with_demo):
        """
        Fit the model with fairness constraints
        """
        # Get sensitive attribute values
        sensitive_values = X_with_demo[self.sensitive_attribute].values
        
        # Store unique values and map to binary (for now, assume binary sensitive attribute)
        self.sensitive_groups = np.unique(sensitive_values)
        if len(self.sensitive_groups) != 2:
            raise ValueError(f"Expected 2 unique values for {self.sensitive_attribute}, got {len(self.sensitive_groups)}")
        
        # Map sensitive attribute to binary (0/1)
        sensitive_binary = (sensitive_values == self.sensitive_groups[1]).astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initial fit with base model
        self.base_model.fit(X_scaled, y)
        self.model = self.base_model
        
        # Iteratively improve fairness
        best_fairness = float('inf')
        best_model = None
        
        for iteration in range(self.max_iter):
            # Get predictions
            y_pred = self.model.predict(X_scaled)
            y_prob = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate fairness metric
            if self.fairness_metric == 'demographic_parity':
                fairness_violation = self._demographic_parity_violation(y_pred, sensitive_binary)
            elif self.fairness_metric == 'equal_opportunity':
                fairness_violation = self._equal_opportunity_violation(y_pred, y, sensitive_binary)
            else:
                raise ValueError(f"Unknown fairness metric: {self.fairness_metric}")
            
            # Check if we've improved
            if fairness_violation < best_fairness:
                best_fairness = fairness_violation
                best_model = self.model
            
            # Early stopping if fairness is good enough
            if fairness_violation < 0.01:
                print(f"Reached fairness threshold at iteration {iteration}")
                break
                
            # Calculate and apply instance weights to improve fairness
            instance_weights = self._calculate_instance_weights(y, y_pred, sensitive_binary)
            
            # Refit model with weights
            if hasattr(self.model, 'sample_weight'):
                self.model.fit(X_scaled, y, sample_weight=instance_weights)
            else:
                # For models that don't support sample_weight, we can apply them
                # by duplicating samples according to their weight
                # (Only works for integer weights)
                weighted_indices = []
                for i, weight in enumerate(instance_weights):
                    weighted_indices.extend([i] * int(np.round(weight)))
                
                X_weighted = X_scaled[weighted_indices]
                y_weighted = y.iloc[weighted_indices]
                self.model.fit(X_weighted, y_weighted)
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration+1}/{self.max_iter}, fairness violation: {fairness_violation:.4f}")
        
        # Use the best model we found
        if best_model is not None:
            self.model = best_model
            print(f"Final fairness violation: {best_fairness:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the fairness-constrained model
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get probability estimates
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def _demographic_parity_violation(self, y_pred, sensitive_attr):
        """
        Calculate demographic parity violation
        (difference in selection rates between groups)
        """
        group_0_selection = np.mean(y_pred[sensitive_attr == 0])
        group_1_selection = np.mean(y_pred[sensitive_attr == 1])
        return abs(group_0_selection - group_1_selection)
    
    def _equal_opportunity_violation(self, y_pred, y_true, sensitive_attr):
        """
        Calculate equal opportunity violation
        (difference in true positive rates between groups)
        """
        group_0_tpr = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_attr == 0)) / max(1, np.sum((y_true == 1) & (sensitive_attr == 0)))
        group_1_tpr = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_attr == 1)) / max(1, np.sum((y_true == 1) & (sensitive_attr == 1)))
        return abs(group_0_tpr - group_1_tpr)
    
