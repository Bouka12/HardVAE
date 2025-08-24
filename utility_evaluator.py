"""This script evaluates and compares classification performance (Utility Gain/Gain Index) between baseline and HardVAE configurations.
    -> Calculates utility gains/gaine index and generates comprehensive plots.
    -> Outputs results to CSV and LaTeX tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp
import os
from itertools import combinations

class UtilityEvaluator:
    """
    Synthetic Data Utility Evaluator for classification results analysis.
    Analyzes utility gains and generates comprehensive plots.
    """
    
    def __init__(self, csv_file_path, save_path=None):
        """
        Initialize the UtilityEvaluator.
        
        Parameters:
        -----------
        csv_file_path : str
            Path to the 'all_classification_results.csv' file
        save_path : str, optional
            Path to save plots and results
        """
        self.csv_file_path = csv_file_path
        self.save_path = save_path
        self.df = None
        self.baseline_df = None
        self.augmented_df = None
        self.utility_gains = None
        self.gain_indices = None
        self.statistical_results = None
        
        # Load and prepare data
        self._load_data()
        self._prepare_baseline_and_augmented()
    
    def _load_data(self):
        """Load the CSV file and validate columns."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Loaded data with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Validate required columns
            required_cols = ['Classifier', 'cv_k_folds', 'Random_State',
                           'Precision', 'Recall', 'F1 Score', 'Specificity', 
                           'Balanced Accuracy', 'dataset', 'hardness_metric', 
                           'seed', 'weighting_strategy']

            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.df = self.df[required_cols]
            self.df = self.df[~self.df['hardness_metric'].isin(['relative_entropy', 'pca_contribution'])]
                
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    def _prepare_baseline_and_augmented(self):
        """Separate baseline and augmented configurations."""
        # Baseline: weighting_strategy='static' and hardness_metric=None
        self.baseline_df = self.df[
            (self.df['weighting_strategy'] == 'static') & 
            (self.df['hardness_metric'].isna() | (self.df['hardness_metric'] == 'None'))
        ].copy()
        
        # Augmented: all other configurations
        self.augmented_df = self.df[
            ~((self.df['weighting_strategy'] == 'static') & 
              (self.df['hardness_metric'].isin(['relative_entropy', 'pca_contribution'])) &
              (self.df['hardness_metric'].isna() | (self.df['hardness_metric'] == 'None')))
        ].copy()
        
        print(f"Baseline configurations: {len(self.baseline_df)} rows")
        print(f"Augmented configurations: {len(self.augmented_df)} rows")
    
    def calculate_utility_gains(self):
        """Step 1: Calculate utility gains for each configuration and run."""
        print("Calculating utility gains...")
        
        metrics = [ 'Precision', 'Recall', 'F1 Score', 'Specificity', 'Balanced Accuracy']
        utility_gains = []
        
        # Group augmented data by configuration
        config_cols = ['dataset', 'Classifier', 'weighting_strategy', 'hardness_metric']
        
        for config, aug_group in self.augmented_df.groupby(config_cols):
            dataset, classifier, weighting_strategy, hardness_metric = config
            
            # Find corresponding baseline
            baseline_group = self.baseline_df[
                (self.baseline_df['dataset'] == dataset) & 
                (self.baseline_df['Classifier'] == classifier)
            ]
            
            if baseline_group.empty:
                print(f"Warning: No baseline found for {config}")
                continue
            
            # Calculate gains for each run (seed)
            for _, aug_row in aug_group.iterrows():
                seed = aug_row['seed']
                
                # Find baseline with same seed
                baseline_row = baseline_group[baseline_group['seed'] == seed]
                
                if baseline_row.empty:
                    print(f"Warning: No baseline with seed {seed} for {config}")
                    continue
                
                baseline_row = baseline_row.iloc[0]
                
                # Calculate utility gains for each metric
                for metric in metrics:
                    gain = aug_row[metric] - baseline_row[metric]
                    
                    utility_gains.append({
                        'dataset': dataset,
                        'Classifier': classifier,
                        'weighting_strategy': weighting_strategy,
                        'hardness_metric': hardness_metric,
                        'seed': seed,
                        'metric': metric,
                        'baseline_value': baseline_row[metric],
                        'augmented_value': aug_row[metric],
                        'utility_gain': gain
                    })
        
        self.utility_gains = pd.DataFrame(utility_gains)
        print(f"Calculated {len(self.utility_gains)} utility gain records")
        return self.utility_gains
    
    def calculate_gain_indices(self):
        """Step 2: Calculate gain indices and disparity."""
        print("Calculating gain indices...")
        
        if self.utility_gains is None:
            self.calculate_utility_gains()
        
        # Group by configuration and metric
        config_cols = ['dataset', 'Classifier', 'weighting_strategy', 'hardness_metric', 'metric']
        
        gain_indices = []
        for config, group in self.utility_gains.groupby(config_cols):
            dataset, classifier, weighting_strategy, hardness_metric, metric = config
            
            gain_index = group['utility_gain'].mean()
            gain_disparity = group['utility_gain'].std()
            n_runs = len(group)
            
            # Count wins (positive gains)
            wins = (group['utility_gain'] > 0).sum()
            win_rate = wins / n_runs if n_runs > 0 else 0
            
            gain_indices.append({
                'dataset': dataset,
                'Classifier': classifier,
                'weighting_strategy': weighting_strategy,
                'hardness_metric': hardness_metric,
                'metric': metric,
                'gain_index': gain_index,
                'gain_disparity': gain_disparity,
                'n_runs': n_runs,
                'wins': wins,
                'win_rate': win_rate
            })
        
        self.gain_indices = pd.DataFrame(gain_indices)
        print(f"Calculated gain indices for {len(self.gain_indices)} configurations")
        return self.gain_indices


    def create_comprehensive_plots(self, dataset_name="utility_analysis"):
        """Create comprehensive visualization plots."""
        if self.save_path is None:
            print("No save path specified, skipping plots")
            return
        
        try:
            os.makedirs(self.save_path, exist_ok=True)

            if self.gain_indices is None or self.gain_indices.empty:
                print("No gain index data available.")
                return
            
            df = self.gain_indices.copy()
            print(f"df head: {df.head()}")
         
            # Create figure with n*k subplots: n is the number of classifiers and k is the number of metrics
            # Then create plot for each classifier with subplots as (classifier,Metric) pair

            classifiers = df['Classifier'].unique()
            print(f"the number of classifiers: {classifiers}")
            metrics = df['metric'].unique()
            print(f"the number of the metrics: {metrics}")

            for clf in classifiers:
                clf_df = df[df['Classifier'] == clf]

                # Plot settings
                n_metrics = len(metrics)
                ncols= 1 # Weighting strategies: curriculum, self-paced, static for each hardness_measures
                nrows = (n_metrics + ncols -1) // ncols # ceil division

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
                axes = axes.flatten() # indexing


                for i, metric in enumerate(metrics):
                    ax = axes[i]
                    subset = clf_df[clf_df['metric'] == metric]

                    # Create pivot : rows = hardness_metric, cols = weighting_strategy

                    pivot = subset.pivot_table(
                        index = 'hardness_metric',
                        columns = 'weighting_strategy',
                        values = 'gain_index',
                        aggfunc = 'mean'
                    )

                    sns.heatmap(
                        pivot,
                        annot=True,
                        fmt=".2f",
                        cmap="RdYlGn",
                        center = 0,
                        ax =ax,
                        cbar=False
                    )
                    ax.set_title(f"{metric}")
                    ax.set_xlabel("Weighting Strategy")
                    ax.set_ylabel("Hardness Measure")

                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])

                # plt.suptitle(f"{clf}", fontsize=18, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                # fig.supxlabel("Weighting Strategy", fontsize=14)
                # fig.supylabel("Hardness Metric", fontsize=14)
                # save plot
                plot_path = os.path.join(self.save_path, f"utility_heatmap_{clf}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved heatmap plot for {clf} to : {plot_path}")

  
        except Exception as e:
            print(f"Error generating comprehensive plots: {e}")

   
    def create_comprehensive_plots_(self, dataset_name="utility_analysis"):
        """Create comprehensive visualization plots."""
        if self.save_path is None:
            print("No save path specified, skipping plots")
            return
        
        try:
            os.makedirs(self.save_path, exist_ok=True)
            
            # Create figure with 6 subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 12))
            
            
            # Plot 4: Gain Distribution
            if self.utility_gains is not None:
                sns.boxplot(data=self.utility_gains, x='metric', y='utility_gain', ax=axes[0])
                axes[0].set_title('Utility Gain Distribution')
                axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[0].tick_params(axis='x', rotation=45)
            
            # Plot 5: Strategy Performance Comparison
            if self.gain_indices is not None:
                strategy_perf = self.gain_indices.groupby('weighting_strategy')['gain_index'].agg(['mean', 'std']).reset_index()
                
                axes[1].bar(strategy_perf['weighting_strategy'], strategy_perf['mean'], 
                              yerr=strategy_perf['std'], capsize=5)
                axes[1].set_title('Strategy Performance (Mean ± Std)')
                axes[1].set_ylabel('Mean Gain Index')
                axes[1].set_xlabel('Weighting Strategy')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            
            # plt.suptitle(f'Synthetic Data Utility Analysis - {dataset_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.save_path, f'utility_analysis_{dataset_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Utility analysis plots saved to: {plot_path}")
            
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")


    def create_dataset_summary_heatmaps(self):
        """
        For each dataset, average over classifiers and weighting strategies.
        Creates a heatmap: rows = hardness_metric, columns = metric.
        """
        if self.save_path is None:
            print("No save path specified, skipping plots")
            return

        try:
            os.makedirs(self.save_path, exist_ok=True)

            if self.gain_indices is None or self.gain_indices.empty:
                print("No gain index data available.")
                return

            df = self.gain_indices.copy()

            if 'dataset' not in df.columns:
                print("Dataset column not found in gain_indices.")
                return

            datasets = df['dataset'].unique()
            print(f"Found {len(datasets)} datasets: {datasets}")

            for dataset_name in datasets:
                dataset_df = df[df['dataset'] == dataset_name]

                # Group over classifier and weighting strategy
                grouped = dataset_df.groupby(['hardness_metric', 'metric'])['gain_index'].mean().reset_index()

                pivot = grouped.pivot(index='hardness_metric', columns='metric', values='gain_index')

                # Sort: put Baseline first
                if 'Baseline' in pivot.index:
                    ordered_rows = ['Baseline'] + sorted([r for r in pivot.index if r != 'Baseline'])
                    pivot = pivot.reindex(ordered_rows)

                fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*1.5), max(8, len(pivot)*0.6)))

                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn",
                    center=0,
                    ax=ax,
                    cbar_kws={'label': 'Avg Gain Index'}
                )

                ax.set_title(f"Average Gain Index Heatmap\n{dataset_name}")
                ax.set_xlabel("Metric")
                ax.set_ylabel("Hardness Metric")

                plt.tight_layout()

                # Save
                plot_path = os.path.join(self.save_path, f"summary_heatmap_{dataset_name}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved summary heatmap for {dataset_name} to: {plot_path}")

        except Exception as e:
            print(f"Error generating summary heatmaps: {e}")

    def export_gain_index_tables(self):
        """
        Export gain index summary tables (mean ± std) per dataset as CSV and LaTeX.
        Rows = hardness_metric, Columns = metric.
        """
        if self.save_path is None:
            print("No save path specified, skipping export.")
            return

        try:
            os.makedirs(self.save_path, exist_ok=True)

            if self.gain_indices is None or self.gain_indices.empty:
                print("No gain index data available.")
                return

            df = self.gain_indices.copy()

            if 'dataset' not in df.columns:
                print("Missing 'dataset' column in gain_indices.")
                return

            datasets = df['dataset'].unique()

            for dataset in datasets:
                dataset_df = df[df['dataset'] == dataset]

                # Compute mean and std
                grouped = dataset_df.groupby(['hardness_metric', 'metric'])['gain_index'].agg(['mean', 'std']).reset_index()

                # Format as "mean ± std"
                grouped['mean_std'] = grouped.apply(
                    lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}", axis=1
                )

                # Pivot to wide format
                pivot_table = grouped.pivot(index='hardness_metric', columns='metric', values='mean_std')

                # Sort: Baseline first
                if 'Baseline' in pivot_table.index:
                    ordered_rows = ['Baseline'] + sorted([r for r in pivot_table.index if r != 'Baseline'])
                    pivot_table = pivot_table.reindex(ordered_rows)

                # Save CSV
                csv_path = os.path.join(self.save_path, f"gain_index_table_{dataset}.csv")
                pivot_table.to_csv(csv_path)
                print(f"Saved CSV table to: {csv_path}")

                # Save LaTeX
                latex_path = os.path.join(self.save_path, f"gain_index_table_{dataset}.tex")
                with open(latex_path, 'w', encoding='utf-8') as f:
                    f.write(pivot_table.to_latex(na_rep='', escape=False, column_format='l' + 'c'*len(pivot_table.columns)))
                print(f"Saved LaTeX table to: {latex_path}")

        except Exception as e:
            print(f"Error exporting gain index tables: {e}")


    def run_complete_analysis(self):
        """Run the complete utility analysis pipeline."""
        print("Starting complete utility analysis...")
        
        # Step 1: Calculate utility gains
        self.calculate_utility_gains()
        
        # Step 2: Calculate gain indices
        self.calculate_gain_indices()

        # Step 3: Create the comprehensive plots
        self.create_comprehensive_plots()

        # Step 3: Create the comprehensive plots
        self.create_comprehensive_plots_()

        self.create_dataset_summary_heatmaps()

        self.export_gain_index_tables()
        
        
        # Save results to CSV if save_path is provided
        if self.save_path:
            try:
                os.makedirs(self.save_path, exist_ok=True)
                
                self.utility_gains.to_csv(os.path.join(self.save_path, 'utility_gains.csv'), index=False)
                self.gain_indices.to_csv(os.path.join(self.save_path, 'gain_indices.csv'), index=False) 
                print(f"Results saved to {self.save_path}")

            except Exception as e:
                print(f"Warning: Could not save results: {e}")
        
        print("Complete utility analysis finished!")
        
        return {
            'utility_gains': self.utility_gains,
            'gain_indices': self.gain_indices
        }


evaluator = UtilityEvaluator(r'C:\Users\BOUKA\Downloads\PART2-Hardness-CSVAE\Hard-CSVAE\results_medical\all_classification_results_medical_complete.csv', save_path='./utility_results_complete')
results = evaluator.run_complete_analysis()

