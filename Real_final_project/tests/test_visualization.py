import pandas as pd

# Add 'src' to the Python path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the functions
from data_visualization import plot_correlation_heatmap, plot_brainwave_relationships

def test_plot_correlation_heatmap_positive():
    """Test with a valid correlation matrix."""
    data = {
        "A": [1.0, 0.5, 0.2],
        "B": [0.5, 1.0, 0.8],
        "C": [0.2, 0.8, 1.0]
    }
    correlation_matrix = pd.DataFrame(data, index=["A", "B", "C"])

    try:
        plot_correlation_heatmap(correlation_matrix)
        print("Positive test passed.")
    except Exception as e:
        print(f"Positive test failed: {e}")

def test_plot_correlation_heatmap_empty():
    """Test with an empty DataFrame."""
    correlation_matrix = pd.DataFrame()

    try:
        plot_correlation_heatmap(correlation_matrix)
        print("Empty DataFrame test failed: No exception was raised.")
    except ValueError as e:
        print(f"Empty DataFrame test passed: {e}")
    except Exception as e:
        print(f"Empty DataFrame test encountered unexpected error: {e}")

def test_plot_correlation_heatmap_invalid_values():
    """Test with a correlation matrix containing invalid values."""
    data = {
        "A": [1.0, 0.5, 1.5],  # Invalid value (1.5)
        "B": [0.5, 1.0, 0.8],
        "C": [0.2, 0.8, 1.0]
    }
    correlation_matrix = pd.DataFrame(data, index=["A", "B", "C"])

    try:
        plot_correlation_heatmap(correlation_matrix)
        print("Invalid values test failed: No exception was raised.")
    except ValueError as e:
        print(f"Invalid values test passed: {e}")
    except Exception as e:
        print(f"Invalid values test encountered unexpected error: {e}")

def test_plot_brainwave_relationships_positive():
    """Test with a valid dataset."""
    data = {
        "alpha": [0.1, 0.2, 0.3, 0.4],
        "beta": [0.5, 0.6, 0.7, 0.8],
        "attention": [10, 20, 30, 40]
    }
    dataset = pd.DataFrame(data)

    try:
        plot_brainwave_relationships(dataset, brainwave_columns=["alpha", "beta"], target_column="attention",
                                     color="blue", title_prefix="Attention vs")
        print("Positive test passed.")
    except Exception as e:
        print(f"Positive test failed: {e}")

def test_plot_brainwave_relationships_missing_columns():
    """Test with missing brainwave or target columns."""
    data = {
        "alpha": [0.1, 0.2, 0.3, 0.4],
        "attention": [10, 20, 30, 40]
    }  # Missing 'beta'
    dataset = pd.DataFrame(data)

    try:
        plot_brainwave_relationships(dataset, brainwave_columns=["alpha", "beta"], target_column="attention",
                                     color="blue", title_prefix="Attention vs")
        print("Missing columns test failed: No exception was raised.")
    except KeyError as e:
        print(f"Missing columns test passed: {e}")

def test_plot_brainwave_relationships_empty_dataframe():
    """Test with an empty DataFrame."""
    dataset = pd.DataFrame()

    try:
        plot_brainwave_relationships(dataset, brainwave_columns=["alpha", "beta"], target_column="attention",
                                     color="blue", title_prefix="Attention vs")
        print("Empty DataFrame test failed: No exception was raised.")
    except ValueError as e:
        print(f"Empty DataFrame test passed: {e}")
    except Exception as e:
        print(f"Empty DataFrame test encountered unexpected error: {e}")

def test_plot_brainwave_relationships_invalid_values():
    """Test with invalid values in the dataset."""
    data = {
        "alpha": [0.1, "invalid", 0.3, 0.4],  # Invalid value in 'alpha'
        "beta": [0.5, 0.6, 0.7, 0.8],
        "attention": [10, 20, 30, 40]
    }
    dataset = pd.DataFrame(data)

    try:
        plot_brainwave_relationships(dataset, brainwave_columns=["alpha", "beta"], target_column="attention",
                                     color="blue", title_prefix="Attention vs")
        print("Invalid values test failed: No exception was raised.")
    except ValueError as e:
        print(f"Invalid values test passed: {e}")
    except Exception as e:
        print(f"Invalid values test encountered unexpected error: {e}")

if __name__ == "__main__":
    test_plot_correlation_heatmap_positive()
    test_plot_correlation_heatmap_empty()
    test_plot_correlation_heatmap_invalid_values()
    test_plot_brainwave_relationships_positive()
    test_plot_brainwave_relationships_missing_columns()
    test_plot_brainwave_relationships_empty_dataframe()
    test_plot_brainwave_relationships_invalid_values()
