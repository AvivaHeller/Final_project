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
    Plots scatter plots with trendlines for brainwave features vs. a target column.

    Parameters:
        dataset (DataFrame): Data containing brainwave features and the target column.
        brainwave_columns (list): Brainwave column names to plot.
        target_column (str): Target variable (e.g., 'attention' or 'meditation').
        color (str): Color of scatter points.
        title_prefix (str): Title prefix for each plot.
        dataset_name (str): Optional dataset name for the plot title.

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
    Displays a heatmap of R² values for brainwave features vs. attention/meditation.

    Parameters:
        results_df (DataFrame): Contains 'Brainwave', 'Target', and 'R^2' columns.
        suffix (str, optional): Text to add to the heatmap title (default: "").
    
    """
    # Create a pivot table for visualization
    heatmap_data = results_df.pivot(index="Brainwave", columns="Target", values="R^2")

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", vmin=0, vmax=1)
    plt.title(f"Quadratic R^2 Heatmap for Brainwaves vs Attention/Meditation {suffix}")
    plt.show()
