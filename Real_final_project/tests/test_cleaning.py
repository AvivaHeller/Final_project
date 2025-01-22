'''# Debugging: Check if the dataset is loaded correctly
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The input data must be a pandas DataFrame.")

    print(f"Dataset contains {len(data)} rows and the following columns: {data.columns.tolist()}")

    # Ensure the wavelength column exists
    if wavelength not in data.columns:
        raise ValueError(f"Wavelength '{wavelength}' not found in the dataset.")

    # Ensure the column contains numeric data
    try:
        data[wavelength] = pd.to_numeric(data[wavelength], errors='coerce')
    except Exception as e:
        raise ValueError(f"Error converting column '{wavelength}' to numeric: {e}")'''

import pandas as pd

# Import the function
from data_cleaning.py import split_dataset_by_classification

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
    """Test when the classification column does not exist."""
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
    """Test when the classification column contains invalid values."""
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

def test_split_dataset_empty_classification():
    """Test when the classification column is empty."""
    data = {
        "Feature1": [10, 20, 30, 40],
        "Feature2": [5, 15, 25, 35],
        "Classification": []  # Empty column
    }
    dataset = pd.DataFrame(data)

    try:
        split_dataset_by_classification(dataset, "Classification")
        print("Empty classification test failed: No exception was raised.")
    except ValueError as e:
        print(f"Empty classification test passed: {e}")

if __name__ == "__main__":
    test_split_dataset_positive()
    test_split_dataset_invalid_column()
    test_split_dataset_invalid_values()
    test_split_dataset_empty_classification()
