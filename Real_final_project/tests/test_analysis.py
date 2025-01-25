import pandas as pd
# Add 'src' to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the function
from data_analysis import categorize_correlation

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

def test_categorize_correlation_invalid_input():
    """Test with invalid inputs to ensure proper exception handling."""
    # Empty DataFrame
    empty_df = pd.DataFrame()
    try:
        categorize_correlation(empty_df)
        print("Empty DataFrame test failed: No exception was raised.")
    except Exception:
        print("Empty DataFrame test passed: Exception was raised as expected.")

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
    data = {"alpha": [1, 1, 1, 1, 1], "attention": [1, 4, 9, 16, 25]}  # No variance in 'alpha'
    dataset = pd.DataFrame(data)

    try:
        results = analyze_r_squared(dataset)
        assert "alpha" not in results['Brainwave'].values, "Expected 'alpha' to be excluded due to lack of variance."
        print("No variance test passed.")
    except Exception as e:
        print(f"No variance test failed: {e}")




if __name__ == "__main__":
    test_categorize_correlation_positive()
    test_categorize_correlation_invalid_input()
    test_categorize_correlation_correctness()
    test_no_variance_in_data()
