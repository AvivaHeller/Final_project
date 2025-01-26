import pandas as pd

# Add 'src' to the Python path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the functions
from data_visualization import plot_correlation_heatmap, plot_brainwave_relationships

def test_plot_correlation_heatmap_invalid_values():
    """Test that the function runs with a correlation matrix containing invalid values."""
    data = {
        "A": [1.0, 0.5, 1.5],  # Invalid value (1.5)
        "B": [0.5, 1.0, 0.8],
        "C": [0.2, 0.8, 1.0]
    }
    correlation_matrix = pd.DataFrame(data, index=["A", "B", "C"])

    try:
        # Run the function to ensure it doesn't raise unexpected errors
        plot_correlation_heatmap(correlation_matrix)
        print("Invalid values test passed: Function executed without errors.")
    except Exception as e:
        print(f"Invalid values test failed: {e}")


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


if __name__ == "__main__":
    test_plot_correlation_heatmap_invalid_values()
    test_plot_brainwave_relationships_positive()
    
