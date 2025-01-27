import pandas as pd
from sklearn.datasets import make_regression

# Add 'src' to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the function
from data_analysis import categorize_correlation, analyze_r_squared, random_forest_analysis

def test_categorize_correlation_positive():
    """Test with a valid correlation matrix."""
    data = {
        "A": [1.0, 0.5, 0.2],
        "B": [0.5, 1.0, 0.8],
        "C": [0.2, 0.8, 1.0]
    }
    correlation_matrix = pd.DataFrame(data, index=["A", "B", "C"])

    try:
        categorize_correlation(correlation_matrix)
        print("Positive test passed.")
    except Exception as e:
        print(f"Positive test failed: {e}")

def test_categorize_correlation_correctness():
    # Create a sample correlation matrix
    example_data = pd.DataFrame({
        'A': [1, 0.1, 0.4],
        'B': [0.1, 1, 0.8],
        'C': [0.4, 0.8, 1]
    }, index=['A', 'B', 'C'])

    # Run the function
    result = categorize_correlation(example_data)

    # Print results for manual verification
    print("Strong correlations:", result['strong'])
    print("Moderate correlations:", result['moderate'])
    print("Weak correlations:", result['weak'])

def test_no_variance_in_data():
    """Test that the function handles lack of variance ok."""
    data = {
        "alpha": [1, 1, 1, 1, 1],  # No variance in 'alpha'
        "attention": [1, 4, 9, 16, 25],
        "meditation": [2, 4, 6, 8, 10]  
    }
    dataset = pd.DataFrame(data)

    try:
        results = analyze_r_squared(dataset)
        
        # Check if results DataFrame is empty
        if results.empty:
            print("No valid R² values were calculated. Test passed for no variance.")
        else:
            # Assert that 'alpha' is not in the results if the DataFrame is not empty
            assert "alpha" not in results['Brainwave'].values, "Expected 'alpha' to be excluded due to lack of variance."
            print("No variance test passed.")
    except Exception as e:
        print(f"No variance test failed: {e}")

def test_random_forest_analysis():
    # Create a synthetic dataset
    brainwave_columns = ['delta', 'theta', 'lowAlpha', 'highAlpha', 
                         'lowBeta', 'highBeta', 'lowGamma', 'highGamma']
    data, target = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
    df = pd.DataFrame(data, columns=brainwave_columns)
    df['attention'] = target
    df['meditation'] = target * 0.5  # Add a second target column for testing

    # Run the Random Forest analysis
    results = random_forest_analysis(df)

    # Test R² score range
    for target in ['attention', 'meditation']:
        r2_score = results[target]['r2']
        assert 0 <= r2_score <= 1, f"R² score for {target} is out of range: {r2_score}"

        # Test feature importance output
        feature_importance = results[target]['feature_importance']
        assert set(feature_importance['Feature']) == set(brainwave_columns), \
            f"Feature importance does not include all expected columns for {target}"

    print("All tests passed!")

if __name__ == "__main__":
    test_categorize_correlation_positive()
    test_categorize_correlation_correctness()
    test_no_variance_in_data()
    test_random_forest_analysis()
