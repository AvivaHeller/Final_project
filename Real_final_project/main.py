import sys
import os

# Add 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_cleaning import dataset, dataset_0, dataset_1
from data_analysis import correlation_matrix_full, correlation_matrix_0, correlation_matrix_1, categorize_correlation, analyze_r_squared, random_forest_analysis
from data_visualization import plot_correlation_heatmap, plot_brainwave_relationships, visualize_r_squared, visualize_feature_importance

def main():
    #implemenation of correlation heatmaps and catagorizing
    plot_correlation_heatmap(correlation_matrix_full, dataset_name="Full Dataset")
    categorize_correlation(correlation_matrix_full)

    plot_correlation_heatmap(correlation_matrix_0, dataset_name="Sleep Dataset")
    categorize_correlation(correlation_matrix_0)

    plot_correlation_heatmap(correlation_matrix_1, dataset_name="Awake Dataset")
    categorize_correlation(correlation_matrix_1)

    #impletmentation of comparison of attentiona and meditation to each brainwave type
    # Define brainwave columns
    brainwave_columns = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']

    # Plot Attention vs. Brainwaves
    plot_brainwave_relationships(dataset_0, brainwave_columns, target_column='attention', color='blue', title_prefix='Attention vs', dataset_name="Sleep Dataset")
    plot_brainwave_relationships(dataset_1, brainwave_columns, target_column='attention', color='blue', title_prefix='Attention vs', dataset_name="Awake Dataset")

    # Plot Meditation vs. Brainwaves
    plot_brainwave_relationships(dataset_0, brainwave_columns, target_column='meditation', color='green', title_prefix='Meditation vs', dataset_name="Sleep Dataset")
    plot_brainwave_relationships(dataset_1, brainwave_columns, target_column='meditation', color='green', title_prefix='Meditation vs', dataset_name="Awake Dataset")

    #implementation of non- linear regression test
    results_df_dataset = analyze_r_squared(dataset)
    visualize_r_squared(results_df_dataset, suffix="Full dataset" )
    results_df_datatset_0 = analyze_r_squared(dataset_0)
    visualize_r_squared(results_df_datatset_0, suffix= "Sleep dataset")
    results_df_datatset_1 = analyze_r_squared(dataset_1)
    visualize_r_squared(results_df_datatset_1, suffix= "Awake dataset")

    #implememtation of RF Non-linear relationship and relative importance test
    results_rf_0= random_forest_analysis(dataset_0)
    visualize_feature_importance(results_rf_0, dataset_name="Sleep dataset")
    results_rf_1= random_forest_analysis(dataset_1)
    visualize_feature_importance(results_rf_1, dataset_name="Awake dataset")

if __name__=="__main__":
    main()