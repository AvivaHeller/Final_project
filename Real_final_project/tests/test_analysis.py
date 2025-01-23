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



if __name__ == "__main__":
    test_categorize_correlation_positive()
    test_categorize_correlation_invalid_input()
    
