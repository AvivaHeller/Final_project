import pandas as pd

pd.set_option('display.max_columns', None)

dataset = pd.read_csv("acquiredDataset.csv")
print(dataset.head())

print(f'Data size: {dataset.shape[0]} rows and {dataset.shape[1]} columns')

print("Data set Information")
dataset.info()

print('Duplicate rows:')
print(dataset.duplicated().sum())

#Main full dataset is clean - No NaN and no duplicates.  dataset clean.

#split dataset to awake vs sleep readings
# Declare variables at the module level
dataset_0 = None
dataset_1 = None

def split_dataset_by_classification(dataset, classification):
    """
    Splits a dataset into two based on the values in a classification column.

    Parameters:
        dataset (pd.DataFrame): The input dataset.
        classification (str): The name of the column used for classification (e.g., 0 and 1).

    Returns:
        None: Assigns the split datasets to module-level variables `dataset_0` and `dataset_1`.
    """
    global dataset_0, dataset_1  # Make variables accessible across the module

    # Ensure the classification column exists and values are 0 ror 1
    if classification not in dataset.columns:
        raise ValueError(f"Column '{classification}' does not exist in the dataset.")
    # Check for invalid values
    if not all(dataset[classification].isin([0, 1])):
        raise ValueError("The classification column contains values other than 0 and 1.")

    # Filter out invalid classification values (keep only 0 and 1)
    valid_values = [0, 1]
    dataset = dataset[dataset[classification].isin(valid_values)]

    # Split the dataset
    dataset_0 = dataset[dataset[classification] == 0]
    dataset_1 = dataset[dataset[classification] == 1]

split_dataset_by_classification(dataset, "classification")
print(dataset_0.head())
print(dataset_1.head())

