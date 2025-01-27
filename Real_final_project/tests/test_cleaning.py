import pandas as pd
# Add 'src' to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the function
from data_cleaning import split_dataset_by_classification, dataset, dataset_0, dataset_1

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

def test_split_dataset_NaNs_numeric():
    # Test data for NaN and non-numeric values
    data = {
        "classification": [0, 1, 0, None, 1],  # NaN value included
        "alpha": [1.0, 2.0, "3.0", 4.0, 5.0],  # "3.0" is a string (non-numeric)
        "beta": [0.1, 0.2, 0.3, 0.4, 0.5]  # Valid numeric column
    }
    dataset = pd.DataFrame(data)

    # Test for NaN in the dataset
    print("Testing for NaN values...")
    try:
        dataset_0, dataset_1 = split_dataset_by_classification(dataset, "classification")
        assert not dataset_0.isnull().values.any() and not dataset_1.isnull().values.any(), \
            "NaN values were not handled correctly."
        print("Test passed for NaN handling: Invalid rows skipped.")
    except Exception as e:
        print(f"Test failed for NaN handling: {e}")

    # Test for non-numeric (non-float/int) values in the dataset
    print("\nTesting for non-numeric values...")
    try:
        # Convert "alpha" column to numeric, coercing invalid values to NaN
        dataset["alpha"] = pd.to_numeric(dataset["alpha"], errors="coerce")
        
        # Change the column type to allow for non-numeric insertion
        dataset["alpha"] = dataset["alpha"].astype(object)
        dataset.iloc[1, dataset.columns.get_loc("alpha")] = "non-numeric"  # Insert a non-numeric value

        dataset_0, dataset_1 = split_dataset_by_classification(dataset, "classification")
        
        # Validate that invalid rows were skipped
        assert all(isinstance(x, (float, int)) for x in dataset_0["alpha"] if x is not None) and \
               all(isinstance(x, (float, int)) for x in dataset_1["alpha"] if x is not None), \
            "Non-numeric values were not skipped correctly."
        print("Test passed for non-numeric handling: Invalid rows skipped.")
    except Exception as e:
        print(f"Test failed for non-numeric handling: {e}")

if __name__ == "__main__":
    test_split_dataset_invalid_column()
    test_split_dataset_invalid_values()
    test_split_dataset_NaNs_numeric()

