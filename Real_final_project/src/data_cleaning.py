import pandas as pd
from scipy.stats import zscore

pd.set_option('display.max_columns', None)

dataset = pd.read_csv("acquiredDataset.csv")
print(dataset.head())

print(f'Data size: {dataset.shape[0]} rows and {dataset.shape[1]} columns')

print("Data set Information")
dataset.info()

print('Duplicate rows:')
print(dataset.duplicated().sum())

print("Checking original dataset for NaN...")
print(dataset.isnull().sum())

# Function to remove outliers using z-score
def remove_outliers(df):
    '''this function will clean outliers with Z scores test
    parameter:    
    dataframe

    returns: 
    clean dataframe
    '''
    z_scores = df.apply(zscore, nan_policy='omit')  # Calculate z-scores for all columns
    filtered_entries = (z_scores < 3.5).all(axis=1)  # Keep only rows with z-scores < 3.5 (more lenient)
    return df[filtered_entries]

# Remove outliers
print("Removing outliers...")
dataset = remove_outliers(dataset)
print(f"Data size after outlier removal: {dataset.shape[0]} rows and {dataset.shape[1]} columns")

#Main full dataset is clean - No NaN and no duplicates and no outliers.  dataset clean.

# Split dataset into awake vs sleep readings
def split_dataset_by_classification(dataset, classification):
    """
    Splits a dataset into two based on the values in a classification column.

    Parameters:
        dataset (pd.DataFrame)- The input dataset.
        classification (str)- The name of the column used for classification (0 and 1).

    Returns:
        dataset_0 (pd.DataFrame)- Subset of the dataset where classification == 0.
        dataset_1 (pd.DataFrame)- Subset of the dataset where classification == 1.
    """
    # Remove rows with NaN values or non-numeric data
    dataset = dataset.apply(pd.to_numeric, errors="coerce")  # Convert non-numeric to NaN
    dataset = dataset.dropna()  # Drop rows with NaN

    # Ensure the classification column exists
    if classification not in dataset.columns:
        raise ValueError(f"Column '{classification}' does not exist in the dataset.")
    
    # Check for invalid values in the classification column
    if not all(dataset[classification].isin([0, 1])):
        raise ValueError("The classification column contains values other than 0 and 1.")

    # Filter valid classification values and split the dataset
    dataset_0 = dataset[dataset[classification] == 0]
    dataset_1 = dataset[dataset[classification] == 1]

    return dataset_0, dataset_1

# Call the function and capture the returned datasets
dataset_0, dataset_1 = split_dataset_by_classification(dataset, "classification")

# Remove classification column from all datasets
dataset = dataset.drop(columns=["classification"])
dataset_0 = dataset_0.drop(columns=["classification"])
dataset_1 = dataset_1.drop(columns=["classification"])

print(dataset.head())
print(dataset_0.head())
print(dataset_1.head())

