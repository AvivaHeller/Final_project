import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

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

#test for non-linear relationship between brainwaves and attention, meditation

def analyze_r_squared(data):
   
    # Identify columns
    attention = data['attention']
    meditation = data['meditation']
    brainwaves = data.drop(columns=['attention', 'meditation'])

    # Simplest way: Using curve_fit to calculate quadratic R^2
    results = []

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    for column in brainwaves.columns:
        for target, target_name in zip([attention, meditation], ["Attention", "Meditation"]):
            x = brainwaves[column]
            y = target

            try:
                # Fit quadratic model using curve_fit
                params, _ = curve_fit(quadratic, x, y, maxfev=10000)
                predicted_y = quadratic(x, *params)

                # Calculate R^2
                ss_res = np.sum((y - predicted_y) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Append results
                results.append({"Brainwave": column, "Target": target_name, "R^2": r_squared})

            except Exception as e:
                print(f"Error fitting {column} vs {target_name}: {e}")

    # Convert results to a DataFrame for reuse
    results_df = pd.DataFrame(results)

    # Categorize and group results
    low = results_df[results_df['R^2'] < 0.3]
    medium = results_df[(results_df['R^2'] >= 0.3) & (results_df['R^2'] < 0.7)]
    high = results_df[results_df['R^2'] >= 0.7]

    # Print categorized results
    print("Low:")
    for _, row in low.iterrows():
        print(f"Brainwave: {row['Brainwave']}, Target: {row['Target']}, R^2: {row['R^2']:.4f}")

    print("\nMedium:")
    for _, row in medium.iterrows():
        print(f"Brainwave: {row['Brainwave']}, Target: {row['Target']}, R^2: {row['R^2']:.4f}")

    print("\nHigh:")
    for _, row in high.iterrows():
        print(f"Brainwave: {row['Brainwave']}, Target: {row['Target']}, R^2: {row['R^2']:.4f}")

    return results_df
