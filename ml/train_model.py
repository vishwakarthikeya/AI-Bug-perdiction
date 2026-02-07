"""
Script to train and save the bug prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pickle
import os
import sys

def load_and_prepare_data(data_path: str) -> tuple:
    """Load and prepare the dataset"""
    print(f"Loading dataset from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        print("Creating synthetic dataset for demonstration...")
        df = create_synthetic_dataset()
    
    # Check if target column exists
    if 'has_bug' not in df.columns:
        print("Warning: 'has_bug' column not found. Creating synthetic target...")
        df['has_bug'] = create_synthetic_target(df)
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Separate features and target
    X = df.drop('has_bug', axis=1)
    y = df['has_bug']
    
    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Bug rate: {y.mean():.2%}")
    
    return X, y, df

def create_synthetic_dataset() -> pd.DataFrame:
    """Create synthetic dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'loc': np.random.exponential(50, n_samples).astype(int) + 10,
        'cyclomatic_complexity': np.random.gamma(2, 2, n_samples),
        'halstead_volume': np.random.lognormal(5, 1, n_samples),
        'num_functions': np.random.poisson(3, n_samples),
        'num_loops': np.random.poisson(2, n_samples),
        'num_conditionals': np.random.poisson(5, n_samples),
        'num_try_except': np.random.poisson(1, n_samples),
        'num_null_checks': np.random.poisson(2, n_samples),
        'nested_depth': np.random.poisson(2, n_samples),
        'avg_line_length': np.random.uniform(20, 80, n_samples),
        'comment_density': np.random.uniform(5, 30, n_samples),
        'num_imports': np.random.poisson(2, n_samples),
        'num_div_operations': np.random.poisson(1, n_samples),
        'num_array_access': np.random.poisson(3, n_samples),
        'parameter_count': np.random.uniform(0, 5, n_samples)
    }
    
    return pd.DataFrame(data)

def create_synthetic_target(df: pd.DataFrame) -> pd.Series:
    """Create synthetic target based on features"""
    # Simple rule: higher complexity = higher bug probability
    complexity_score = (
        df['cyclomatic_complexity'] / 10 +
        df['nested_depth'] / 5 +
        df['num_loops'] / 3 +
        df['num_conditionals'] / 8
    )
    
    # Normalize and add randomness
    bug_prob = complexity_score / complexity_score.max()
    bug_prob = bug_prob * 0.7 + np.random.normal(0, 0.1, len(df))
    
    # Convert to binary
    y = (bug_prob > 0.5).astype(int)
    
    print(f"Synthetic target created. Bug rate: {y.mean():.2%}")
    return y

def train_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train the logistic regression model"""
    print("\nTraining model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        C=1.0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\\nModel trained successfully!")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Feature coefficients
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    print("\nTop 5 features increasing bug probability:")
    for _, row in coefficients.head(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    print("\nTop 5 features decreasing bug probability:")
    for _, row in coefficients.tail(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'coefficients': coefficients
    }

def save_model(model_data: dict, output_path: str):
    """Save the trained model"""
    print(f"\nSaving model to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved successfully!")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")

def main():
    """Main training function"""
    print("=" * 60)
    print("AI BUG PREDICTOR - MODEL TRAINING")
    print("=" * 60)
    
    # Configuration
    data_path = "../dataset/bug_dataset_50k.csv"
    model_path = "model.pkl"
    
    # Load and prepare data
    X, y, df = load_and_prepare_data(data_path)
    
    # Train model
    model_data = train_model(X, y)
    
    # Save model
    save_model(model_data, model_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Features used: {len(model_data['feature_names'])}")
    print(f"Model accuracy: {model_data['accuracy']:.4f}")
    print(f"ROC AUC: {model_data['roc_auc']:.4f}")
    print(f"Training samples: {model_data['training_samples']}")
    print(f"Test samples: {model_data['test_samples']}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    
    # Test the saved model
    print("\nTesting saved model...")
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # Create test prediction
    test_features = {feature: 1.0 for feature in loaded_data['feature_names']}
    
    # Prepare feature array
    features = [test_features.get(name, 0) for name in loaded_data['feature_names']]
    features_array = np.array(features).reshape(1, -1)
    
    # Scale and predict
    features_scaled = loaded_data['scaler'].transform(features_array)
    probability = loaded_data['model'].predict_proba(features_scaled)[0, 1]
    
    print(f"Test prediction with all features=1.0:")
    print(f"  Bug probability: {probability:.4f}")
    print(f"  Prediction: {'BUG' if probability > 0.5 else 'NO BUG'}")
    
    return model_data

if __name__ == "__main__":
    main()