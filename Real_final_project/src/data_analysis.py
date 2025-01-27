import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from data_cleaning import dataset, dataset_0, dataset_1

# Correlation matrix
correlation_matrix_full = dataset.corr()
correlation_matrix_0 = dataset_0.corr()
correlation_matrix_1 = dataset_1.corr()

def categorize_correlation(correlation_matrix):
    """
    Groups values from a correlation matrix into three categories:
    - Strong correlations (0.7 - 1.0)
    - Moderate correlations (0.3 - 0.7)
    - Weak correlations (0.0 - 0.3)

    Excludes self-correlations (diagonal values). Returns categorized pairs and prints them in a readable format.
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

    # Print the categorized correlations in the desired format
    print("High Correlation (Strong, 0.7 - 1.0):")
    print("\n".join(strong) if strong else "None")

    print("\nModerate Correlation (0.3 - 0.7):")
    print("\n".join(moderate) if moderate else "None")

    print("\nWeak Correlation (0.0 - 0.3):")
    print("\n".join(weak) if weak else "None")

    # Return the categorized correlations as a dictionary
    return {
        "weak": weak,
        "moderate": moderate,
        "strong": strong,
    }

# looking for non-linear relationship between brainwaves and attention, meditation
def analyze_r_squared(data):
    
    """
    Analyze the non-linear relationship between brainwave features and attention/meditation levels
    by calculating the R² values.

    Parameters:
    data: pandas.DataFrame
        Includes 'attention', 'meditation', and brainwave feature columns.

    Returns:
    results_df: pandas.DataFrame
        Contains 'Brainwave', 'Target' ('Attention' or 'Meditation'), and 'R^2'.

    Prints:
    R² categories-
        - Low: R² < 0.3
        - Medium: 0.3 ≤ R² < 0.7
        - High: R² ≥ 0.7
    """
    # Identify columns
    attention = data['attention']
    meditation = data['meditation']
    brainwaves = data.drop(columns=['attention', 'meditation'])

    # Using curve_fit to calculate quadratic R^2
    results = []

    def quadratic(x, a, b, c):
        """Defines a quadratic function."""
        return a * x**2 + b * x + c

    for column in brainwaves.columns:
        # Check if the column has variance
        if brainwaves[column].nunique() <= 1:
            print(f"Skipping {column} due to lack of variance.")
            continue

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

    if results_df.empty:
        print("No valid R² values were calculated.")
        return results_df

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

def random_forest_analysis(data, test_size=0.2, random_state=42, n_estimators=100):
    """
    Perform Random Forest analysis on a dataset for both attention and meditation with predefined brainwave columns.

    Parameters:
    data : DataFrame - Input dataset.

    Returns:
    dict - R² scores and DataFrames of feature importances for attention and meditation.
    """
    # Predefined brainwave columns
    brainwave_columns = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']

    results = {}

    for target in ['attention', 'meditation']:
        X = data[brainwave_columns]
        y = data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Initialize and fit the Random Forest Regressor
        model = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        r2 = r2_score(y_test, y_pred)

        # Print R² score
        print(f"R² score for {target.capitalize()}: {r2:.4f}")

        # Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': brainwave_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        results[target] = {'r2': r2, 'feature_importance': feature_importance}

    return results

