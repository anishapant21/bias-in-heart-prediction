import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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