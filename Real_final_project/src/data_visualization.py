import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(correlation_matrix, title="Correlation Matrix", dataset_name=""):
    """
    Plots a heatmap of the correlation matrix.

    Parameters:
        correlation_matrix (pd.DataFrame): The correlation matrix to be visualized.
        title (str): The title of the heatmap. Default is "Correlation Matrix".
        dataset_name (str): Name of the dataset to append to the plot titles.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    if dataset_name:
        full_title = f"{title} of {dataset_name}"
    else:
        full_title = title
    plt.title(full_title)
    plt.show()


#function to compare attention and meditation to each brainwave length
def plot_brainwave_relationships(dataset, brainwave_columns, target_column, color, title_prefix, dataset_name=""):
    """
    Plots scatter plots with trendlines for brainwave metrics vs. a target column (e.g., attention or meditation).

    Args:
        dataset (DataFrame): The dataset containing brainwave metrics and target column.
        brainwave_columns (list): List of brainwave column names.
        target_column (str): The column to compare against (e.g., 'attention' or 'meditation').
        color (str): The color for scatter points.
        title_prefix (str): Prefix for the plot titles (e.g., 'Attention vs' or 'Meditation vs').
        dataset_name (str): Name of the dataset to append to the plot titles.
    """
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(brainwave_columns, 1):
        plt.subplot(4, 2, i)
        sns.scatterplot(x=dataset[column], y=dataset[target_column], alpha=0.6, color=color)
        plt.title(f'{title_prefix} {column}')
        plt.xlabel(column)
        plt.ylabel(target_column.capitalize())
        plt.xscale('log')
        plt.yticks([0, 25, 50, 75, 100])
    
    plt.subplots_adjust(hspace=0.7)
    plt.suptitle(f"{dataset_name} - {title_prefix} Brainwave Relationships", fontsize=16, y=0.98)
    
    plt.show()


def visualize_r_squared(results_df, suffix=""):
    """
    Visualizes the R² values from a quadratic analysis of brainwave features 
    against attention and meditation levels using a heatmap.

    Parameters:
    ----------
    results_df : pandas.DataFrame
        A DataFrame containing the results of R² analysis with the following columns:
        - 'Brainwave': Names of the brainwave features.
        - 'Target': The target variable ('Attention' or 'Meditation').
        - 'R^2': The R² value representing the fit of a quadratic relationship.
    
    suffix : str, optional
        An optional suffix to append to the title of the heatmap (default is "").

    Returns:
    -------
    None
        The function displays a heatmap of R² values for visual analysis.
    """
    # Create a pivot table for visualization
    heatmap_data = results_df.pivot(index="Brainwave", columns="Target", values="R^2")

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", vmin=0, vmax=1)
    plt.title(f"Quadratic R^2 Heatmap for Brainwaves vs Attention/Meditation {suffix}")
    plt.show()
