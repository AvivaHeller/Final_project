from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

# Add 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_cleaning import dataset, dataset_0, dataset_1

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

def visualize_feature_importance(results):
    """
    Visualize the feature importance as bar charts for both attention and meditation.

    Parameters:
    results : dict - Results containing feature importances for attention and meditation.
    """
    for target, data in results.items():
        color = 'skyblue' if target == 'attention' else 'lightcoral'
        title = f"Feature Importance for {target.capitalize()}"
        feature_importance = data['feature_importance']

        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['Feature'], feature_importance['Importance'], color=color)
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage:
# Load the dataset
data = dataset_0

# Run analysis for both attention and meditation
results = random_forest_analysis(data)

# Visualize the results
visualize_feature_importance(results)

# Load the dataset
data = dataset_1

# Run analysis for both attention and meditation
results = random_forest_analysis(data)

# Visualize the results
visualize_feature_importance(results)
