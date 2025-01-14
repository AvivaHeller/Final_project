import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_cleaning import dataset, dataset_0, dataset_1
from data_analysis import correlation_matrix_0, correlation_matrix_1, correlation_matrix_full

def plot_attention_and_meditation(data, title_suffix=""):
    """
    Plots the distributions of attention and meditation levels for a given dataset.
    
    Parameters:
        data (pd.DataFrame): The dataset containing 'attention' and 'meditation' columns.
        title_suffix (str): Optional suffix to add to plot titles for context.
    """
    plt.figure(figsize=(12, 5))

    # Set y-axis ticks
    y_ticks = range(0, 501, 50)

    # Distribution of Attention
    plt.subplot(1, 2, 1)
    sns.histplot(data['attention'], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of Attention Levels {title_suffix}')
    plt.xlabel('Attention')
    plt.yticks(y_ticks)

    # Distribution of Meditation
    plt.subplot(1, 2, 2)
    sns.histplot(data['meditation'], kde=True, bins=20, color='orange')
    plt.title(f'Distribution of Meditation Levels {title_suffix}')
    plt.xlabel('Meditation')
    plt.yticks(y_ticks)

    plt.tight_layout()
    plt.show()


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
    plt.title(title)
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
    plt.figure(figsize=(16, 12))
    for i, column in enumerate(brainwave_columns, 1):
        plt.subplot(4, 2, i)
        sns.scatterplot(x=dataset[column], y=dataset[target_column], alpha=0.6, color=color)
        sns.lineplot(x=dataset[column], y=dataset[target_column], color='red', label='Trendline', ci=None)
        plt.title(f'{title_prefix} {column}')
        plt.xlabel(column)
        plt.ylabel(target_column.capitalize())
        plt.xscale('log')

    plt.tight_layout()
    plt.show()

