import pandas as pd
# Add 'src' to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the function
from data_cleaning import split_dataset_by_classification

def test_split_dataset_positive():
    """Test with a valid dataset and classification column."""
    data = {
        "Feature1": [10, 20, 30, 40],
        "Feature2": [5, 15, 25, 35],
        "Classification": [0, 1, 0, 1]
    }
    dataset = pd.DataFrame(data)

    try:
        split_dataset_by_classification(dataset, "Classification")
        print("Dataset 0:")
        print(dataset_0)
        print("Dataset 1:")
        print(dataset_1)
        print("Positive test passed.")
    except Exception as e:
        print(f"Positive test failed: {e}")

def test_split_dataset_invalid_column():
    """Test when the classification column does not exist, and our ValueError works"""
    data = {
        "Feature1": [10, 20, 30, 40],
        "Feature2": [5, 15, 25, 35]
    }
    dataset = pd.DataFrame(data)

    try:
        split_dataset_by_classification(dataset, "Classification")
        print("Invalid column test failed: No exception was raised.")
    except ValueError as e:
        print(f"Invalid column test passed: {e}")

def test_split_dataset_invalid_values():
    """Test when the classification column contains invalid values, and our ValueError works."""
    data = {
        "Feature1": [10, 20, 30, 40],
        "Feature2": [5, 15, 25, 35],
        "Classification": [0, 1, 2, 1]  # Invalid value (2)
    }
    dataset = pd.DataFrame(data)

    try:
        split_dataset_by_classification(dataset, "Classification")
        print("Invalid values test failed: No exception was raised.")
    except ValueError as e:
        print(f"Invalid values test passed: {e}")

def test_split_dataset_NaNs_float():
    # Test data for NaN and non-float values
    data = {
        "classification": [0, 1, 0, None, 1],  # NaN value included
        "alpha": [1.0, 2.0, "3.0", 4.0, 5.0],  # "3.0" is a string (non-float)
        "beta": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    dataset = pd.DataFrame(data)

    # Test for NaN in the dataset
    try:
        split_dataset_by_classification(dataset, "classification")
        print("Test failed: No exception raised for NaN values.")
    except ValueError as e:
        print(f"Test passed for NaN values: {e}")

    # Test for non-float values in the dataset
    try:
        split_dataset_by_classification(dataset, "classification")
        print("Test failed: No exception raised for non-float values.")
    except ValueError as e:
        print(f"Test passed for non-float values: {e}")


if __name__ == "__main__":
    test_split_dataset_positive()
    test_split_dataset_invalid_column()
    test_split_dataset_invalid_values()
    test_split_dataset_NaNs_float()

