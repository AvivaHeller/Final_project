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