import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_cleaning import dataset, dataset_0, dataset_1


# Correlation matrix
correlation_matrix_full = dataset.corr()
correlation_matrix_0 = dataset_0.corr()
correlation_matrix_1 = dataset_1.corr()

#catagorize correlations
def categorize_correlation(correlation_matrix):
    """
    Categorizes correlation matrix values into weak, moderate, and strong.
    Prints each category with a title.
    """
    # Thresholds for correlation levels
    weak_threshold = (0.0, 0.3)
    moderate_threshold = (0.3, 0.7)
    strong_threshold = (0.7, 1.0)

    # Initialize lists to store correlation pairs in each category
    weak = []
    moderate = []
    strong = []

    # Iterate through the correlation matrix
    for row in correlation_matrix.index:
        for col in correlation_matrix.columns:
            if row != col:  # Exclude self-correlation (always 1)
                corr_value = abs(correlation_matrix.loc[row, col])
                pair = f"{row} - {col}: {corr_value:.2f}"

                # Categorize based on thresholds
                if weak_threshold[0] <= corr_value < weak_threshold[1]:
                    weak.append(pair)
                elif moderate_threshold[0] <= corr_value < moderate_threshold[1]:
                    moderate.append(pair)
                elif strong_threshold[0] <= corr_value <= strong_threshold[1]:
                    strong.append(pair)

    # Print the categorized correlations
    print("High Correlation (Strong, 0.7 - 1.0):")
    print("\n".join(strong) if strong else "None")

    print("\nModerate Correlation (0.3 - 0.7):")
    print("\n".join(moderate) if moderate else "None")

    print("\nWeak Correlation (0.0 - 0.3):")
    print("\n".join(weak) if weak else "None")


