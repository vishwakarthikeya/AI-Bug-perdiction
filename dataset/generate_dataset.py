"""
Dataset Generator for AI Bug Predictor
Creates a synthetic dataset of 50,000 code samples with bug labels
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def generate_bug_dataset(n_samples=50000, output_file="bug_dataset_50k.csv"):
    """
    Generate a synthetic dataset for bug prediction
    
    Args:
        n_samples: Number of samples to generate
        output_file: Output CSV file path
    
    Returns:
        pandas.DataFrame: Generated dataset
    """
    print(f"Generating synthetic bug dataset with {n_samples:,} samples...")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate base features
    print("1. Generating base features...")
    
    data = {
        # Code size metrics
        'loc': np.random.exponential(50, n_samples).astype(int) + 10,
        'avg_line_length': np.random.uniform(20, 80, n_samples),
        'comment_density': np.random.uniform(5, 30, n_samples),
        
        # Complexity metrics
        'cyclomatic_complexity': np.random.gamma(2, 2, n_samples),
        'halstead_volume': np.random.lognormal(5, 1, n_samples),
        
        # Structural features
        'num_functions': np.random.poisson(3, n_samples),
        'num_classes': np.random.poisson(1, n_samples),
        'num_loops': np.random.poisson(2, n_samples),
        'num_conditionals': np.random.poisson(5, n_samples),
        'num_try_except': np.random.poisson(1, n_samples),
        'num_imports': np.random.poisson(2, n_samples),
        
        # Bug pattern indicators
        'num_null_checks': np.random.poisson(2, n_samples),
        'num_div_operations': np.random.poisson(1, n_samples),
        'num_array_access': np.random.poisson(3, n_samples),
        
        # Code quality metrics
        'nested_depth': np.random.poisson(2, n_samples),
        'parameter_count': np.random.uniform(0, 5, n_samples),
        
        # Python-specific features
        'num_list_comprehensions': np.random.poisson(0.5, n_samples),
        'num_dict_comprehensions': np.random.poisson(0.3, n_samples),
        'num_lambda_expressions': np.random.poisson(0.5, n_samples),
        
        # Binary indicators
        'has_assert': np.random.binomial(1, 0.3, n_samples),
        'has_raise': np.random.binomial(1, 0.2, n_samples),
        'has_yield': np.random.binomial(1, 0.1, n_samples),
        'has_async': np.random.binomial(1, 0.05, n_samples),
        'has_decorator': np.random.binomial(1, 0.2, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # 2. Generate realistic correlations between features
    print("2. Adding realistic correlations...")
    
    # Larger code tends to have more functions
    df['num_functions'] = df['num_functions'] + (df['loc'] // 50).astype(int)
    
    # More complex code has more conditionals
    df['num_conditionals'] = df['num_conditionals'] + (df['cyclomatic_complexity'] // 2).astype(int)
    
    # Code with many functions tends to have more parameters
    df['parameter_count'] = df['parameter_count'] + df['num_functions'] * 0.3
    
    # Code with many loops tends to have higher nested depth
    df['nested_depth'] = np.maximum(df['nested_depth'], df['num_loops'] // 2)
    
    # 3. Generate target variable (has_bug)
    print("3. Generating target variable (has_bug)...")
    
    # Create a bug probability based on features
    bug_probability = np.zeros(n_samples)
    
    # Factors that increase bug probability
    bug_probability += df['cyclomatic_complexity'] * 0.05
    bug_probability += df['nested_depth'] * 0.1
    bug_probability += df['num_loops'] * 0.08
    bug_probability += df['num_conditionals'] * 0.04
    bug_probability += (df['loc'] > 100) * 0.1
    bug_probability += (df['num_functions'] > 5) * 0.05
    bug_probability += (df['num_div_operations'] > 0) * 0.15  # Division operations risky
    bug_probability += (df['num_array_access'] > 3) * 0.08
    
    # Factors that decrease bug probability
    bug_probability -= df['num_null_checks'] * 0.05
    bug_probability -= df['num_try_except'] * 0.08
    bug_probability -= df['comment_density'] * 0.01
    bug_probability -= (df['has_assert'] == 1) * 0.05
    
    # Add some noise
    bug_probability += np.random.normal(0, 0.1, n_samples)
    
    # Normalize to 0-1 range
    bug_probability = (bug_probability - bug_probability.min()) / (bug_probability.max() - bug_probability.min())
    
    # Adjust to get desired bug rate (~30%)
    threshold = np.percentile(bug_probability, 70)  # Top 30% have bugs
    df['has_bug'] = (bug_probability >= threshold).astype(int)
    
    # 4. Add some noise to features based on bug status
    print("4. Adding noise patterns...")
    
    # Buggy code tends to have slightly different distributions
    bug_mask = df['has_bug'] == 1
    
    # Buggy code tends to be larger
    df.loc[bug_mask, 'loc'] = df.loc[bug_mask, 'loc'] * np.random.uniform(1.1, 1.3, bug_mask.sum())
    
    # Buggy code tends to be more complex
    df.loc[bug_mask, 'cyclomatic_complexity'] = df.loc[bug_mask, 'cyclomatic_complexity'] * np.random.uniform(1.1, 1.4, bug_mask.sum())
    
    # Buggy code has fewer null checks
    df.loc[bug_mask, 'num_null_checks'] = df.loc[bug_mask, 'num_null_checks'] * np.random.uniform(0.7, 0.9, bug_mask.sum())
    
    # 5. Ensure all values are valid
    print("5. Cleaning and validating data...")
    
    # Convert to appropriate types
    int_columns = [
        'loc', 'num_functions', 'num_classes', 'num_loops', 
        'num_conditionals', 'num_try_except', 'num_imports',
        'num_null_checks', 'num_div_operations', 'num_array_access',
        'nested_depth', 'num_list_comprehensions', 'num_dict_comprehensions',
        'num_lambda_expressions', 'has_assert', 'has_raise', 
        'has_yield', 'has_async', 'has_decorator', 'has_bug'
    ]
    
    for col in int_columns:
        df[col] = df[col].astype(int)
        # Ensure non-negative
        df[col] = df[col].clip(lower=0)
    
    # Round float columns
    float_columns = ['avg_line_length', 'comment_density', 'cyclomatic_complexity',
                    'halstead_volume', 'parameter_count']
    
    for col in float_columns:
        df[col] = df[col].round(3)
    
    # 6. Statistics and validation
    print("6. Generating statistics...")
    
    bug_rate = df['has_bug'].mean()
    print(f"   Bug rate: {bug_rate:.2%} ({df['has_bug'].sum():,} buggy samples)")
    
    print(f"   Average LOC: {df['loc'].mean():.1f}")
    print(f"   Average complexity: {df['cyclomatic_complexity'].mean():.2f}")
    print(f"   Average functions: {df['num_functions'].mean():.2f}")
    
    # Feature correlation with target
    correlations = {}
    for col in df.columns:
        if col != 'has_bug':
            corr = df[col].corr(df['has_bug'])
            correlations[col] = corr
    
    top_positive = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negative = sorted(correlations.items(), key=lambda x: x[1])[:5]
    
    print("\n   Top 5 features positively correlated with bugs:")
    for feature, corr in top_positive:
        print(f"     {feature}: {corr:.3f}")
    
    print("\n   Top 5 features negatively correlated with bugs:")
    for feature, corr in top_negative:
        print(f"     {feature}: {corr:.3f}")
    
    # 7. Save to file
    print("\n7. Saving dataset...")
    
    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   Dataset saved to: {output_file}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # 8. Create a sample preview
    print("\n8. Sample preview:")
    print("=" * 60)
    print(df.head(10).to_string())
    print("=" * 60)
    
    return df

def create_sample_code_snippets(df, n_samples=10):
    """
    Create sample code snippets that match the feature patterns
    Useful for testing and demonstration
    """
    print("\n" + "=" * 60)
    print("GENERATING SAMPLE CODE SNIPPETS")
    print("=" * 60)
    
    samples = []
    
    for i in range(min(n_samples, len(df))):
        features = df.iloc[i]
        
        # Create a code snippet based on features
        loc = features['loc']
        num_funcs = features['num_functions']
        num_loops = features['num_loops']
        has_bug = features['has_bug']
        
        # Generate simple Python-like code
        code = []
        code.append(f"# Sample code {i+1} (Bug: {'YES' if has_bug else 'NO'})")
        code.append(f"# LOC: {loc}, Functions: {num_funcs}, Loops: {num_loops}")
        code.append("")
        
        # Add imports
        if features['num_imports'] > 0:
            code.append("import random")
            code.append("import math")
            code.append("")
        
        # Add functions
        for f in range(num_funcs):
            code.append(f"def function_{f+1}(data):")
            code.append(f"    \"\"\"Function {f+1}\"\"\"")
            
            # Add potential bug if this is a buggy sample
            if has_bug and f == 0:
                code.append("    # Potential bug: no null check")
                code.append("    result = data.process()  # Might fail if data is None")
            else:
                code.append("    if data is not None:")
                code.append("        result = data.process()")
                code.append("    else:")
                code.append("        result = None")
            
            code.append("    return result")
            code.append("")
        
        # Add main logic
        code.append("def main():")
        
        # Add loops
        for l in range(num_loops):
            if has_bug and l == 0:
                code.append(f"    # Loop {l+1} - potential off-by-one")
                code.append(f"    for i in range(len(data) + 1):  # Might cause index error")
                code.append(f"        value = data[i]")
            else:
                code.append(f"    # Loop {l+1}")
                code.append(f"    for item in data:")
                code.append(f"        process_item(item)")
        
        # Add conditionals
        num_conds = min(features['num_conditionals'], 3)
        for c in range(num_conds):
            code.append(f"    if condition_{c+1}:")
            code.append(f"        do_something()")
        
        code.append("")
        code.append("if __name__ == \"__main__\":")
        code.append("    main()")
        
        samples.append("\n".join(code))
    
    # Save samples to file
    samples_file = "sample_code_snippets.txt"
    with open(samples_file, "w") as f:
        f.write("\n" + "="*60 + "\n")
        f.write("SAMPLE CODE SNIPPETS FOR TESTING\n")
        f.write("="*60 + "\n\n")
        
        for i, sample in enumerate(samples):
            f.write(f"SAMPLE {i+1}\n")
            f.write("="*40 + "\n")
            f.write(sample)
            f.write("\n\n" + "="*60 + "\n\n")
    
    print(f"Sample code snippets saved to: {samples_file}")
    
    return samples

def validate_dataset(df):
    """
    Validate the generated dataset
    """
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    
    validation_passed = True
    issues = []
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        validation_passed = False
        issues.append(f"Found {nan_count} NaN values in dataset")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        inf_count += np.isinf(df[col]).sum()
    
    if inf_count > 0:
        validation_passed = False
        issues.append(f"Found {inf_count} infinite values in dataset")
    
    # Check target distribution
    bug_rate = df['has_bug'].mean()
    if bug_rate < 0.2 or bug_rate > 0.4:
        issues.append(f"Bug rate ({bug_rate:.2%}) outside typical range (20-40%)")
    
    # Check feature ranges
    for col in df.columns:
        if col != 'has_bug':
            min_val = df[col].min()
            max_val = df[col].max()
            
            if col in ['loc', 'cyclomatic_complexity', 'halstead_volume']:
                if min_val < 0:
                    validation_passed = False
                    issues.append(f"{col} has negative values (min: {min_val})")
            
            if col in ['has_assert', 'has_raise', 'has_yield', 'has_async', 'has_decorator']:
                if min_val < 0 or max_val > 1:
                    validation_passed = False
                    issues.append(f"{col} has values outside [0,1] range")
    
    # Print validation results
    if validation_passed:
        print("✅ Dataset validation PASSED")
    else:
        print("❌ Dataset validation FAILED")
        for issue in issues:
            print(f"   - {issue}")
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Bug rate: {bug_rate:.2%}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    return validation_passed

def main():
    """Main function to generate the dataset"""
    print("=" * 60)
    print("AI BUG PREDICTOR - DATASET GENERATOR")
    print("=" * 60)
    
    # Configuration
    n_samples = 50000
    output_file = "bug_dataset_50k.csv"
    
    try:
        # Generate the dataset
        df = generate_bug_dataset(n_samples, output_file)
        
        # Validate the dataset
        validation_passed = validate_dataset(df)
        
        # Create sample code snippets for testing
        samples = create_sample_code_snippets(df, n_samples=5)
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE!")
        print("=" * 60)
        
        if validation_passed:
            print(f"\n✅ Dataset ready for use:")
            print(f"   File: {output_file}")
            print(f"   Samples: {len(df):,}")
            print(f"   Features: {len(df.columns)}")
            print(f"   Bug rate: {df['has_bug'].mean():.2%}")
            
            print("\n📊 Feature Overview:")
            print(df.describe().round(2).to_string())
            
            print("\n🚀 Next steps:")
            print("1. Use this dataset to train the ML model:")
            print("   cd ../ml && python train_model.py")
            print("\n2. Test the trained model:")
            print("   python -c \"import pickle; import pandas as pd;")
            print("   data = pd.read_csv('bug_dataset_50k.csv');")
            print("   print(f'Dataset loaded: {len(data)} samples')\"")
            
        else:
            print("\n⚠️  Dataset generated but validation failed.")
            print("   Check the issues above before using for training.")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error generating dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df = main()
    
    # Quick test to verify the dataset
    if df is not None:
        print("\n" + "=" * 60)
        print("QUICK TEST: Loading and verifying the dataset...")
        
        try:
            # Try to load the generated file
            test_df = pd.read_csv("bug_dataset_50k.csv")
            print(f"✅ File loaded successfully")
            print(f"   Shape: {test_df.shape}")
            print(f"   Columns: {list(test_df.columns)}")
            print(f"   Bug samples: {test_df['has_bug'].sum():,}")
            
            # Check a few sample predictions
            print("\n🔍 Sample predictions preview:")
            sample = test_df.sample(5)
            for i, (_, row) in enumerate(sample.iterrows()):
                bug_prob = (
                    row['cyclomatic_complexity'] * 0.05 +
                    row['nested_depth'] * 0.1 +
                    row['num_div_operations'] * 0.15 -
                    row['num_null_checks'] * 0.05
                )
                bug_prob = min(1.0, max(0.0, bug_prob))
                
                print(f"   Sample {i+1}: LOC={row['loc']}, "
                      f"Complexity={row['cyclomatic_complexity']:.1f}, "
                      f"Actual Bug={'Yes' if row['has_bug'] else 'No'}, "
                      f"Estimated Prob={bug_prob:.2f}")
            
        except Exception as e:
            print(f"❌ Error testing dataset: {str(e)}")