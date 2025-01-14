import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.data_cleaning import dataset, dataset_0, dataset_1
from src.data_analysis import correlation_matrix_full, correlation_matrix_0, correlation_matrix_1, categorize_correlation
from src.data_visualization import plot_attention_and_meditation, plot_correlation_heatmap, plot_brainwave_relationships

#implementation of plotting
#all 3

#implemenation of correlation heatmaps and catagorizing
#all 3

#impletmentation of comparison of attentiona dn meditation to each brainwave type
# Define brainwave columns
brainwave_columns = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']

# Plot Attention vs. Brainwaves
plot_brainwave_relationships(dataset, brainwave_columns, target_column='attention', color='blue', title_prefix='Attention vs')
plot_brainwave_relationships(dataset_0, brainwave_columns, target_column='attention', color='blue', title_prefix='Attention vs')
plot_brainwave_relationships(dataset_1, brainwave_columns, target_column='attention', color='blue', title_prefix='Attention vs')

# Plot Meditation vs. Brainwaves
plot_brainwave_relationships(dataset, brainwave_columns, target_column='meditation', color='green', title_prefix='Meditation vs')
plot_brainwave_relationships(dataset_0, brainwave_columns, target_column='meditation', color='green', title_prefix='Meditation vs')
plot_brainwave_relationships(dataset_1, brainwave_columns, target_column='meditation', color='green', title_prefix='Meditation vs')