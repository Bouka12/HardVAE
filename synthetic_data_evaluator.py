""" 
Enhanced Synthetic Minority Data Evaluation Module

This module contains comprehensive evaluation classes for assessing the quality of synthetic minority data
used for handling class imbalance. It implements 6 key evaluation aspects:

1. Statistical evaluation: correlation, distribution similarity, range coverage
2. Clustering-based evaluation: cluster quality metrics -> Check the implementation here
3. Topological Data Analysis (TDA): shape and structure analysis -> Modify to return a distance-based score!
4. Instance hardness analysis: complexity similarity measurement -> Check the logic 
5. Complexity-based analysis: data complexity metrics -> Check the logic
------------- TO-DO-HOY:

- Fix the storage of the results of the evaluation:
    1. each aspect results will be stored in a separate csv, for that, the output should be formatted as a row of csv file



- In a separate class, in this script, we evaluate the utility of synthetic data :
6. Model-based utility evaluation: landmarking and classification performance -> Check the implementation



"""
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from pymfe.mfe import MFE
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ks_2samp
import problexity as px
import persim
from ripser import ripser
from persim import plot_diagrams, bottleneck, wasserstein
import random
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# Try to import optional dependencies
try:
    import pyhard
    PYHARD_AVAILABLE = True
except ImportError:
    PYHARD_AVAILABLE = False
    print("Warning: pyhard not available. Instance hardness analysis will be limited.")

try:
    from ripser import ripser
    from persim import plot_diagrams
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("Warning: TDA libraries not available. Topological analysis will be limited.")


class SyntheticDataEvaluator:
    """
    Comprehensive evaluator for synthetic minority data quality assessment.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        """
        self.random_state = random_state
        self.results = {}

        
    def evaluate_all(self, X_real, y_real, X_synth, y_synth, save_path=None, dataset_name="dataset"):
        """
        Run comprehensive evaluation across all aspects.
        
        Parameters:
        -----------
        X_real : array-like
            Original training data features of minority and majority
        y_real : array-like
            Original training data labels of minority and majority
        X_synth : array-like
            Synthetic minority data features
        y_synth : array-like
            Synthetic minority data labels
        save_path : str, optional
            Path to save results and plots
        dataset_name : str, default="dataset"
            Name of the dataset for labeling
            
        Returns:
        --------
        dict : Comprehensive evaluation results
        """
        print(f"Starting comprehensive evaluation for {dataset_name}...")
        
        # 1. Statistical Evaluation
        print("1. Running statistical evaluation...")
        stat_results = self.statistical_evaluation(X_real, y_real, X_synth, y_synth, save_path, dataset_name)
        self.results['statistical'] = stat_results['summary'] 
        
        # 2. Clustering-based Evaluation
        print("2. Running clustering-based evaluation...")
        cluster_results = self.clustering_evaluation(X_real, y_real, X_synth, y_synth, save_path, dataset_name)
        self.results['clustering'] = cluster_results['summary']
        
        # 3. Complexity Analysis
        print("3. Running complexity analysis...")
        complexity_results = self.complexity_evaluation(X_real, y_real, X_synth, y_synth, 
                                                       save_path, dataset_name)
        self.results['complexity'] = complexity_results['summary']
             
        # 4. Instance Hardness Analysis
        print("4. Running instance hardness analysis...")
        hardness_results = self.hardness_evaluation(X_real,y_real, X_synth, y_synth, save_path, dataset_name)
        self.results['hardness'] = hardness_results['summary']
        
        # 5. Topological Data Analysis
        if TDA_AVAILABLE:
            print("5. Running topological data analysis...")
            tda_results = self.topological_evaluation(X_real, y_real, X_synth, y_synth, save_path, dataset_name)
            self.results['topological'] = tda_results['summary']
        else:
            print("5. Skipping topological analysis (libraries not available)")
            self.results['topological'] = None
        
        
        return self.results
    
    def statistical_evaluation(self, X_real, y_real, X_synth, y_synth, save_path=None, dataset_name="dataset"):
        """
        Evaluate statistical similarity between real and synthetic data with integrated plotting.
        """
        # Sample data if too large
        X_real_sampled, y_real_sampled = self._cluster_sampling(X_real, y_real)
        X_synth_sampled, y_synth_sampled = self._cluster_sampling(X_synth, y_synth)
        
        # Statistical meta-features
        stat_features = ['cor', 'cov', 'eigenvalues', 'gravity', 'iq_range', 'kurtosis', 
                        'mad', 'max', 'mean', 'median', 'min', 'nr_cor_attr', 'nr_norm', 
                        'nr_outliers', 'range', 'sd', 'sd_ratio', 'skewness', 'sparsity', 
                        't_mean', 'var']
        
        mfe = MFE(groups='statistical', features=stat_features, random_state=self.random_state)
        
        # Extract features for real data
        mfe.fit(X_real_sampled, y_real_sampled)
        ft_real = mfe.extract()
        
        # Extract features for synthetic data
        mfe.fit(X_synth_sampled, y_synth_sampled)
        ft_synth = mfe.extract()
        
        # Calculate differences and similarities
        feature_names = ft_real[0]
        real_values = np.array(ft_real[1])
        synth_values = np.array(ft_synth[1])
        
        # Handle NaN values 
        valid_indices = ~(np.isnan(real_values) | np.isnan(synth_values))
        real_values_clean = real_values[valid_indices]
        synth_values_clean = synth_values[valid_indices]
        feature_names_clean = [feature_names[i] for i in range(len(feature_names)) if valid_indices[i]]
        
        # Calculate similarity metrics
        differences = real_values_clean - synth_values_clean
        relative_differences = np.abs(differences) / (np.abs(real_values_clean) + 1e-8)
        similarity_scores = 1 / (1 + relative_differences)  # Similarity score [0,1]
        
        # Statistical tests
        ks_statistics = []
        ks_pvalues = []
        
        for i in range(X_real_sampled.shape[1]):
            ks_stat, ks_pval = ks_2samp(X_real_sampled[:, i], X_synth_sampled[:, i])
            ks_statistics.append(ks_stat)
            ks_pvalues.append(ks_pval)
        
        # INTEGRATED PLOTTING FUNCTIONALITY
        if save_path:
            try:
                import matplotlib.pyplot as plt
                import os
                
                print(f"Saving statistical evaluation plots...")
                os.makedirs(save_path, exist_ok=True)
                
                # Create comprehensive statistical plots (6 subplots)
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Plot 1: Feature Values Comparison (Real vs Synthetic)
                x = np.arange(len(feature_names_clean))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, real_values_clean, width, label='Real', alpha=0.7, color='blue')
                axes[0, 0].bar(x + width/2, synth_values_clean, width, label='Synthetic', alpha=0.7, color='orange')
                axes[0, 0].set_xlabel('Statistical Features')
                axes[0, 0].set_ylabel('Feature Values')
                axes[0, 0].set_title('Statistical Features Comparison')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(feature_names_clean, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Similarity Scores
                colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in similarity_scores]
                axes[0, 1].bar(range(len(feature_names_clean)), similarity_scores, alpha=0.7, color=colors)
                axes[0, 1].set_xlabel('Statistical Features')
                axes[0, 1].set_ylabel('Similarity Score')
                axes[0, 1].set_title('Feature Similarity Scores')
                axes[0, 1].set_xticks(range(len(feature_names_clean)))
                axes[0, 1].set_xticklabels(feature_names_clean, rotation=45, ha='right')
                axes[0, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent')
                axes[0, 1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: KS Statistics per Feature
                axes[0, 2].bar(range(len(ks_statistics)), ks_statistics, alpha=0.7, color='lightcoral')
                axes[0, 2].set_xlabel('Features')
                axes[0, 2].set_ylabel('KS Statistic')
                axes[0, 2].set_title('KS Statistics per Feature')
                axes[0, 2].grid(True, alpha=0.3)
                
                # Plot 4: P-values Distribution
                axes[1, 0].bar(range(len(ks_pvalues)), ks_pvalues, alpha=0.7, color='lightblue')
                axes[1, 0].set_xlabel('Features')
                axes[1, 0].set_ylabel('P-value')
                axes[1, 0].set_title('KS Test P-values')
                axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 5: Scatter Plot (Real vs Synthetic Values)
                axes[1, 1].scatter(real_values_clean, synth_values_clean, alpha=0.6, color='purple')
                min_val = min(min(real_values_clean), min(synth_values_clean))
                max_val = max(max(real_values_clean), max(synth_values_clean))
                axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Match')
                axes[1, 1].set_xlabel('Real Data Values')
                axes[1, 1].set_ylabel('Synthetic Data Values')
                axes[1, 1].set_title('Real vs Synthetic Values Correlation')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # Plot 6: Summary Statistics
                summary_data = {
                    'Mean\nSimilarity': np.mean(similarity_scores),
                    'Mean KS\nStatistic': np.mean(ks_statistics),
                    'Std KS\nStatistic': np.std(ks_statistics),
                    'Valid\nFeatures': len(feature_names_clean)
                }
                
                colors = ['lightgreen', 'lightyellow', 'lightpink', 'lightblue']
                bars = axes[1, 2].bar(summary_data.keys(), summary_data.values(), alpha=0.7, color=colors)
                axes[1, 2].set_ylabel('Value')
                axes[1, 2].set_title('Statistical Evaluation Summary')
                axes[1, 2].tick_params(axis='x', rotation=0)
                axes[1, 2].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, summary_data.values()):
                    height = bar.get_height()
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}' if isinstance(value, float) else str(value),
                                ha='center', va='bottom', fontsize=10)
                
                plt.suptitle(f'Statistical Evaluation Results - {dataset_name}', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save the plot
                plot_path = os.path.join(save_path, f'statistical_evaluation_{dataset_name}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Statistical evaluation plots saved to: {plot_path}")
                
            except Exception as e:
                print(f"Warning: Could not save statistical plots: {e}")
        
        final_results = {
            'detailed':{
                'feature_names': feature_names_clean,
                'real_values': real_values_clean,
                'synth_values': synth_values_clean,
                'differences': differences,
                'similarity_scores': similarity_scores,
                'ks_statistics': ks_statistics,
                'ks_pvalues': ks_pvalues
            },
            'summary':{
                'mean_similarity': np.mean(similarity_scores),
                'mean_ks_statistic': np.mean(ks_statistics),
                'std_ks_statistic': np.std(ks_statistics)
            }
        }
        return final_results

    def clustering_evaluation(self, X_real, y_real, X_synth, y_synth, save_path=None, dataset_name="dataset", cluster_features = None):
        """
        Evaluate clustering-based metrics for data quality assessment.
        Clustering features from `pymfe`:
        - 'ch': Compute the Calinski and Harabasz index.(T. Calinski, J. Harabasz, A dendrite method for cluster analysis, Commun. Stat. Theory Methods 3 (1) (1974) 1–27.)
        - 'int': Compute the INT index. (SOUZA, Bruno Feres de. Meta-aprendizagem aplicada à classificação de dados de expressão gênica. 2010. Tese (Doutorado em Ciências de Computação e Matemática Computacional), Instituto de Ciências Matemáticas e de Computação, Universidade de São Paulo, São Carlos, 2010. doi:10.11606/T.55.2010.tde-04012011-142551. [2] Bezdek, J. C.; Pal, N. R. (1998a). Some new indexes of cluster validity. IEEE Transactions on Systems, Man, and Cybernetics, Part B, v.28, n.3, p.301–315.)
        - 'nre': Compute the normalized relative entropy. (Bruno Almeida Pimentel, André C.P.L.F. de Carvalho. A new data characterization for selecting clustering algorithms using meta-learning. Information Sciences, Volume 477, 2019, Pages 203-219)
        - 'pb': Compute the pearson correlation between class matching and instance distances. (J. Lev, “The Point Biserial Coefficient of Correlation”, Ann. Math. Statist., Vol. 20, no.1, pp. 125-126, 1949.)
        - 'sc': Compute the number of clusters with size smaller than a given size.(Bruno Almeida Pimentel, André C.P.L.F. de Carvalho. A new data characterization for selecting clustering algorithms using meta-learning. Information Sciences, Volume 477, 2019, Pages 203-219.)
        - 'sil': Compute the mean silhouette value.(P.J. Rousseeuw, Silhouettes: a graphical aid to the interpretation and validation of cluster analysis, J. Comput. Appl. Math. 20 (1987) 53–65.)
        - 'vdb': Compute the Davies and Bouldin Index. (D.L. Davies, D.W. Bouldin, A cluster separation measure, IEEE Trans. Pattern Anal. Mach. Intell. 1 (2) (1979) 224–227.)
        - 'vdu': Compute the Dunn Index.(J.C. Dunn, Well-separated clusters and optimal fuzzy partitions, J. Cybern. 4 (1) (1974) 95–104.)
        """
        # Sample data if too large
        # X_real_sampled, y_real_sampled = self._cluster_sampling(X_real, y_real)
        # X_synth_sampled, y_synth_sampled = self._cluster_sampling(X_synth, y_synth)
        
        # Clustering meta-features
        if cluster_features is None:
            cluster_features = ['ch', 'int', 'nre', 'pb', 'sc', 'sil', 'vdb', 'vdu']
        
        try:
            mfe = MFE(groups='clustering', features=cluster_features, random_state=self.random_state)
            
            # Extract clustering features for real data
            mfe.fit(X_real, y_real)
            ft_real = mfe.extract()
            
            # Extract clustering features for synthetic data
            mfe.fit(X_synth, y_synth)
            ft_synth = mfe.extract()
            
            # Calculate similarities
            feature_names = ft_real[0]
            real_values = np.array(ft_real[1])
            synth_values = np.array(ft_synth[1])
            
            # Handle NaN values
            valid_indices = ~(np.isnan(real_values) | np.isnan(synth_values))
            real_values_clean = real_values[valid_indices]
            synth_values_clean = synth_values[valid_indices]
            feature_names_clean = [feature_names[i] for i in range(len(feature_names)) if valid_indices[i]]
            
            # Calculate similarity
            differences = real_values_clean - synth_values_clean
            relative_differences = np.abs(differences) / (np.abs(real_values_clean) + 1e-8)
            similarity_scores = 1 / (1 + relative_differences)
            
        except Exception as e:
            print(f"Warning: Clustering evaluation failed: {e}")
            return {
                'feature_names': [],
                'real_values': [],
                'synth_values': [],
                'similarity_scores': [],
                'mean_similarity': 0.0,
                'error': str(e)
            }
            #     # Plotting block: synth_values_clean vs. feature_names_clean & similarity_scores that is will be 2 subplot based figure, with a figure for summary stats
        if save_path:
            try:
                import matplotlib.pyplot as plt
                import os

                print(f"Saving clustering evaluation plots...")
                os.makedirs(save_path, exist_ok=True)

                fig, axes = plt.subplots(1, 2, figsize=(20, 6))

                # Plot 1: Feature Values Comparison (Real vs Synthetic)
                x = np.arange(len(feature_names_clean))
                width = 0.35

                axes[0].bar(x - width/2, real_values_clean, width, label='Real', alpha=0.7, color='blue')
                axes[0].bar(x + width/2, synth_values_clean, width, label='Synthetic', alpha=0.7, color='orange')
                axes[0].set_xlabel('Clustering Features')
                axes[0].set_ylabel('Feature Values')
                axes[0].set_title('Clustering Features Comparison')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(feature_names_clean, rotation=45, ha='right')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # Plot 2: Similarity Scores
                colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in similarity_scores]
                axes[1].bar(range(len(feature_names_clean)), similarity_scores, alpha=0.7, color=colors)
                axes[1].set_xlabel('Clustering Features')
                axes[1].set_ylabel('Similarity Score')
                axes[1].set_title('Feature Similarity Scores')
                axes[1].set_xticks(range(len(feature_names_clean)))
                axes[1].set_xticklabels(feature_names_clean, rotation=45, ha='right')
                axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent')
                axes[1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                # # Plot 3: Summary Statistics
                # summary_data = {
                #     'Mean\nSimilarity': np.mean(similarity_scores),
                #     'Valid\nFeatures': len(feature_names_clean)
                # }

                # colors = ['lightgreen', 'lightblue']
                # bars = axes[2].bar(summary_data.keys(), summary_data.values(), alpha=0.7, color=colors)
                # axes[2].set_ylabel('Value')
                # axes[2].set_title('Clustering Evaluation Summary')
                # axes[2].tick_params(axis='x', rotation=0)
                # axes[2].grid(True, alpha=0.3)

                # for bar, value in zip(bars, summary_data.values()):
                #     height = bar.get_height()
                #     axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                #                 f'{value:.3f}' if isinstance(value, float) else str(value),
                #                 ha='center', va='bottom', fontsize=10)

                plt.suptitle(f'Clustering Evaluation Results - {dataset_name}', fontsize=16, fontweight='bold')
                plt.tight_layout()

                # os.makedirs(save_path, exist=True)
                plot_path = os.path.join(save_path, f'clustering_evaluation_{dataset_name}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Clustering evaluation plots saved to: {plot_path}")

            except Exception as e:
                print(f"Warning: Could not save clustering plots: {e}")

        # Additional clustering analysis
        # cluster_analysis = self._analyze_cluster_structure(X_real, X_synth)
        
        # print results:
        cl_res = { 
            'detailed':{ 
                'feature_names': feature_names_clean,
                'real_values': real_values_clean,
                'synth_values': synth_values_clean,
                'similarity_scores': similarity_scores},

            'summary':{
                'mean_similarity': np.mean(similarity_scores) if len(similarity_scores) > 0 else 0.0}
            # 'cluster_analysis': cluster_analysis
            }
        print(f" results of clustering analysis: {cl_res}")

        return cl_res
    
  
    #-----------------------------------------------------------------------------------------------------#
    def complexity_evaluation(self, X_real, y_real, X_synth, y_synth, save_path=None, dataset_name="dataset", k=3, minority_class=None):
        """
        Evaluate data complexity using problexity package.
        Parameters:
        -----------
            - X_real : array-like
                Complete original training data features (minority + majority classes)
             
            - y_real : array-like
                Complete original training data labels (minority + majority classes)
            - X_synth : array-like
                Synthetic minority class features
            - y_synth : array-like
                Synthetic minority class labels
            - save_path : str, optional
                Path to save complexity plots
            - dataset_name : str, default="dataset"
                Name of the dataset for labeling
            - k : int, default=3
                Number of iteration for sampling and averaging 
            - minority_class : int/str, optional
                Label of the minority class. If None, automatically detected as the least frequent class.

        Returns:
        --------
        - dict : Comprehensive complexity evaluation results

        Algorithm:
        -----------
        1. Automatically separate X_real, y_real into minority and majority classes
        2. Calculate original class imbalance ratio
        3. Dtermine  sampling strategy based on synthetic vs real minority sizes
        4. For k iterations:
            - Sample data according to strategy
            - Combine minority + majority to create balanced dataset
            - Apply clustering sampling if dataset is large
            - Calculate complexity scores using problexity
            - Store iteration results
        5. Avergae complexity scores across iterations
        6. Calculate similarity metrics and statistical summaries
     

        How:
            - check if the minority: synth >= real
                - do for k times the following:
                - sample (# real size) from synthetic data as much as real minority concatenate it with the majority class data in the training
                - if synth and X_real_maj are considerably large we perform clustering check `pymfe_metrics.py` clustering strategy based on the shape of the dataset
                - calculate the complexity scores and store them {comp_1, comp_2, ..., comp_k} using problexity
                - Average across the k dict/df of complexity scores
            - if real >= synth:
                - repeat the procedures of the previous loop but with:
                - sample (# synthetic size) from synthetic and sample from the majority class data to maintain the same class_imbalance_ratio to get a imbalanced dataset like the original
                - if synth and X_real_maj are considerably large we perform clustering check `pymfe_metrics.py` clustering strategy based on the shape of the dataset
                - calculate the complexity scores and store them {comp_1, comp_2, ..., comp_k} using problexity
                - Average across the k dict/df of complexity scores
        """

        print(f"Running enhanced complexity evaluation for {dataset_name}...")
        
        # Convert inputs to numpy arrays
        X_real = np.array(X_real)
        y_real = np.array(y_real)
        X_synth = np.array(X_synth)
        y_synth = np.array(y_synth)
        
        # Let's check the shape of our params:
        print(f" X_real: {X_real.shape}|y_real:{y_real.shape}\nX_synth: {X_synth.shape}|y_synth: {y_synth.shape}")

        # Automatically detect minority and majority classes
        unique_classes, class_counts = np.unique(y_real, return_counts=True)
        
        if minority_class is None:
            # Automatically detect minority class as the least frequent
            minority_class = unique_classes[np.argmin(class_counts)]
            print(f"Auto-detected minority class: {minority_class}")
        
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, but found {len(unique_classes)} classes: {unique_classes}")
        
        # Separate minority and majority classes from original data
        minority_mask = y_real == minority_class
        majority_mask = ~minority_mask
        
        X_real_min = X_real[minority_mask]
        y_real_min = y_real[minority_mask]
        X_real_maj = X_real[majority_mask]
        y_real_maj = y_real[majority_mask]
        
        # Calculate class imbalance ratio
        class_imbalance_ratio = len(X_real_min) / len(X_real_maj)
        
        # print(f"Dataset composition:")
        # print(f"  Minority class ({minority_class}): {len(X_real_min)} samples")
        # print(f"  Majority class: {len(X_real_maj)} samples")
        # print(f"  Class imbalance ratio: {class_imbalance_ratio:.4f}")
        # print(f"  Synthetic samples: {len(X_synth)}")
        
        # Validate synthetic data labels
        unique_synth_classes = np.unique(y_synth)
        if len(unique_synth_classes) != 1 or unique_synth_classes[0] != minority_class:
            print(f"Warning: Synthetic data should contain only minority class ({minority_class}), "
                f"but found: {unique_synth_classes}")
        
        # Determine sizes
        n_real_min = len(X_real_min)
        n_synth = len(X_synth)
        n_real_maj = len(X_real_maj)
        
        # Initialize storage for k iterations
        iteration_results = {
            'real_complexities': [],
            'synth_complexities': [],
            'real_scores': [],
            'synth_scores': [],
            'metrics_names': None,
            'sampling_strategy': None,
            'iteration_details': []
        }
        
        try:
            # Determine sampling strategy
            if n_synth >= n_real_min:
                # Strategy 1: Synthetic >= Real minority
                sampling_strategy = "synth_larger"
                sample_size_min = n_real_min
                print(f"Using Strategy 1: Sampling {sample_size_min} from synthetic data (synth >= real)")
            else:
                # Strategy 2: Real >= Synthetic minority  
                sampling_strategy = "real_larger"
                sample_size_min = n_synth
                print(f"Using Strategy 2: Sampling {sample_size_min} from real minority data (real >= synth)")
            
            iteration_results['sampling_strategy'] = sampling_strategy
            
            # Perform k iterations
            for iteration in range(k):
                print(f"  Iteration {iteration + 1}/{k}")
                
                try:
                    # Sample minority data based on strategy
                    if sampling_strategy == "synth_larger":
                        # Sample from synthetic to match real minority size
                        synth_indices = np.random.choice(n_synth, size=sample_size_min, replace=False)
                        X_min_sampled = X_synth[synth_indices]
                        y_min_sampled = y_synth[synth_indices]
                        
                        # Use all real minority data
                        X_real_min_iter = X_real_min.copy()
                        y_real_min_iter = y_real_min.copy()
                        
                    else:  # real_larger
                        # Sample from real minority to match synthetic size
                        real_indices = np.random.choice(n_real_min, size=sample_size_min, replace=False)
                        X_real_min_iter = X_real_min[real_indices]
                        y_real_min_iter = y_real_min[real_indices]
                        
                        # Use all synthetic data
                        X_min_sampled = X_synth.copy()
                        y_min_sampled = y_synth.copy()
                    
                    # Calculate required majority sample size to maintain class imbalance ratio
                    maj_sample_size = int(sample_size_min / class_imbalance_ratio)
                    maj_sample_size = min(maj_sample_size, n_real_maj)  # Don't exceed available majority samples
                    
                    # Sample majority data
                    maj_indices = np.random.choice(n_real_maj, size=maj_sample_size, replace=False)
                    X_maj_sampled = X_real_maj[maj_indices]
                    y_maj_sampled = y_real_maj[maj_indices]
                    
                    # Create complete datasets for this iteration
                    # Real dataset: real minority + majority
                    X_real_complete = np.vstack([X_real_min_iter, X_maj_sampled])
                    y_real_complete = np.hstack([y_real_min_iter, y_maj_sampled])
                    
                    # Synthetic dataset: synthetic minority + majority  
                    X_synth_complete = np.vstack([X_min_sampled, X_maj_sampled])
                    y_synth_complete = np.hstack([y_min_sampled, y_maj_sampled])
                    
                    # Apply clustering sampling if datasets are large
                    clustering_applied = False
                    if len(X_real_complete) > 10000:
                        print(f"    Applying clustering sampling (dataset size: {len(X_real_complete)})")
                        X_real_complete, y_real_complete = self._cluster_sampling(
                            X_real_complete, y_real_complete, k=4
                        )
                        X_synth_complete, y_synth_complete = self._cluster_sampling(
                            X_synth_complete, y_synth_complete, k=4
                        )
                        clustering_applied = True
                    
                    # Calculate complexity scores for this iteration
                    cc_real = px.ComplexityCalculator().fit(X_real_complete, y_real_complete)
                    cc_synth = px.ComplexityCalculator().fit(X_synth_complete, y_synth_complete)
                    
                    # If Path >> Save Plot

                    # Store results
                    iteration_results['real_complexities'].append(cc_real.complexity)
                    iteration_results['synth_complexities'].append(cc_synth.complexity)
                    iteration_results['real_scores'].append(cc_real.score())
                    iteration_results['synth_scores'].append(cc_synth.score())
                    
                    # Store metrics names (same for all iterations)
                    if iteration_results['metrics_names'] is None:
                        iteration_results['metrics_names'] = cc_real._metrics()
                    
                    # Store iteration details
                    iteration_results['iteration_details'].append({
                        'iteration': iteration + 1,
                        'real_dataset_size': len(X_real_complete),
                        'synth_dataset_size': len(X_synth_complete),
                        'minority_sample_size': sample_size_min,
                        'majority_sample_size': maj_sample_size,
                        'actual_imbalance_ratio': sample_size_min / maj_sample_size,
                        'clustering_applied': clustering_applied,
                        'real_score': cc_real.score(),
                        'synth_score': cc_synth.score()
                    })
                    
                    print(f"    Completed - Real score: {cc_real.score():.3f}, Synth score: {cc_synth.score():.3f}")
                    
                except Exception as e:
                    print(f"    Warning: Iteration {iteration + 1} failed: {e}")
                    # Fill with zeros for failed iteration
                    if iteration_results['metrics_names'] is not None:
                        n_metrics = len(iteration_results['metrics_names'])
                        iteration_results['real_complexities'].append([0.0] * n_metrics)
                        iteration_results['synth_complexities'].append([0.0] * n_metrics)
                    iteration_results['real_scores'].append(0.0)
                    iteration_results['synth_scores'].append(0.0)
                    
                    # Add failed iteration details
                    iteration_results['iteration_details'].append({
                        'iteration': iteration + 1,
                        'failed': True,
                        'error': str(e)
                    })
            
            # Calculate averaged results across iterations
            if iteration_results['real_complexities'] and iteration_results['metrics_names']:
                # Convert to numpy arrays for easier computation
                real_complexities_array = np.array(iteration_results['real_complexities'])
                synth_complexities_array = np.array(iteration_results['synth_complexities'])
                
                # Calculate means and standard deviations
                mean_real_complexity = np.mean(real_complexities_array, axis=0)
                mean_synth_complexity = np.mean(synth_complexities_array, axis=0)
                std_real_complexity = np.std(real_complexities_array, axis=0)
                std_synth_complexity = np.std(synth_complexities_array, axis=0)
                
                # Handle NaN values
                valid_indices = ~(np.isnan(mean_real_complexity) | np.isnan(mean_synth_complexity))
                
                if np.any(valid_indices):
                    mean_real_clean = mean_real_complexity[valid_indices]
                    mean_synth_clean = mean_synth_complexity[valid_indices]
                    std_real_clean = std_real_complexity[valid_indices]
                    std_synth_clean = std_synth_complexity[valid_indices]
                    metrics_names_clean = [iteration_results['metrics_names'][i] 
                                        for i in range(len(iteration_results['metrics_names'])) 
                                        if valid_indices[i]]
                    
                    # Calculate similarity scores
                    differences = mean_real_clean - mean_synth_clean
                    relative_differences = np.abs(differences) / (np.abs(mean_real_clean) + 1e-8)
                    similarity_scores = 1 / (1 + relative_differences)
                    
                    # Calculate overall score statistics
                    real_scores_array = np.array(iteration_results['real_scores'])
                    synth_scores_array = np.array(iteration_results['synth_scores'])
                    
                    # Filter out zero scores (failed iterations)
                    valid_real_scores = real_scores_array[real_scores_array > 0]
                    valid_synth_scores = synth_scores_array[synth_scores_array > 0]
                    
                    if len(valid_real_scores) > 0 and len(valid_synth_scores) > 0:
                        mean_real_score = np.mean(valid_real_scores)
                        mean_synth_score = np.mean(valid_synth_scores)
                        std_real_score = np.std(valid_real_scores)
                        std_synth_score = np.std(valid_synth_scores)
                    else:
                        mean_real_score = 0.0
                        mean_synth_score = 0.0
                        std_real_score = 0.0
                        std_synth_score = 0.0
                    
                else:
                    # All values are NaN
                    mean_real_clean = np.array([])
                    mean_synth_clean = np.array([])
                    std_real_clean = np.array([])
                    std_synth_clean = np.array([])
                    metrics_names_clean = []
                    similarity_scores = np.array([])
                    mean_real_score = 0.0
                    mean_synth_score = 0.0
                    std_real_score = 0.0
                    std_synth_score = 0.0
            
            else:
                # No successful iterations
                mean_real_clean = np.array([])
                mean_synth_clean = np.array([])
                std_real_clean = np.array([])
                std_synth_clean = np.array([])
                metrics_names_clean = []
                similarity_scores = np.array([])
                mean_real_score = 0.0
                mean_synth_score = 0.0
                std_real_score = 0.0
                std_synth_score = 0.0
            
            # Save plots if path provided
            if save_path and len(iteration_results['real_scores']) > 0:
                try:
                    self._save_complexity_plots(iteration_results, save_path, dataset_name, k)
                except Exception as e:
                    print(f"Warning: Could not save complexity plots: {e}")
            
            # Count successful iterations
            successful_iterations = len([detail for detail in iteration_results['iteration_details'] 
                                    if not detail.get('failed', False)])
            
            # Prepare final results
            final_results = { 
                'detailed':
                {
                    # Core complexity results
                    'metrics_names': metrics_names_clean,
                    'mean_real_complexity': mean_real_clean,
                    'mean_synth_complexity': mean_synth_clean,
                    'std_real_complexity': std_real_clean,
                    'std_synth_complexity': std_synth_clean,
                    'similarity_scores': similarity_scores},

                'summary':
                {
                    'mean_similarity': np.mean(similarity_scores) if len(similarity_scores) > 0 else 0.0,
                    
                    # Overall scores
                    'mean_real_score': mean_real_score,
                    'mean_synth_score': mean_synth_score,
                    'std_real_score': std_real_score,
                    'std_synth_score': std_synth_score,
                    'score_difference': abs(mean_real_score - mean_synth_score),
                },    
                # Metadata
                'metadata':
                {
                    'sampling_strategy': sampling_strategy,
                    'k_iterations': k,
                    'successful_iterations': successful_iterations,
                    'minority_class': minority_class,
                    'class_imbalance_ratio': class_imbalance_ratio,
                    'original_data_composition': {
                        'minority_samples': n_real_min,
                        'majority_samples': n_real_maj,
                        'synthetic_samples': n_synth
                    },
                    
                    # Detailed results
                    'iteration_details': iteration_results['iteration_details'],
                    'raw_iteration_results': iteration_results
                }
            }
            
            print(f"Complexity evaluation completed successfully!")
            # print(f"  Successful iterations: {successful_iterations}/{k}")
            # print(f"  Mean real score: {mean_real_score:.3f} ± {std_real_score:.3f}")
            # print(f"  Mean synth score: {mean_synth_score:.3f} ± {std_synth_score:.3f}")
            # print(f"  Mean similarity: {final_results['mean_similarity']:.3f}")
            # print(f"  Sampling strategy: {sampling_strategy}")
            
            return final_results
            
        except Exception as e:
            print(f"Error: Complexity evaluation failed: {e}")
            return {
                'metrics_names': [],
                'mean_real_complexity': np.array([]),
                'mean_synth_complexity': np.array([]),
                'std_real_complexity': np.array([]),
                'std_synth_complexity': np.array([]),
                'similarity_scores': np.array([]),
                'mean_similarity': 0.0,
                'mean_real_score': 0.0,
                'mean_synth_score': 0.0,
                'std_real_score': 0.0,
                'std_synth_score': 0.0,
                'score_difference': 0.0,
                'sampling_strategy': 'unknown',
                'k_iterations': k,
                'successful_iterations': 0,
                'minority_class': minority_class,
                'class_imbalance_ratio': 0.0,
                'original_data_composition': {},
                'iteration_details': [],
                'error': str(e)
            }

        # ----------------------------------#
        # # Sample data if too large
        # X_real_sampled, y_real_sampled = self._cluster_sampling(X_real, y_real)
        # X_synth_sampled, y_synth_sampled = self._cluster_sampling(X_synth, y_synth)
        
        # try:
        #     # Initialize complexity calculators
        #     cc_real = px.ComplexityCalculator().fit(X_real_sampled, y_real_sampled)
        #     cc_synth = px.ComplexityCalculator().fit(X_synth_sampled, y_synth_sampled)
            
        #     # Get complexity metrics
        #     metrics_names = cc_real._metrics()
        #     real_complexity = cc_real.complexity
        #     synth_complexity = cc_synth.complexity
            
        #     # Calculate similarity
        #     real_values = np.array(real_complexity)
        #     synth_values = np.array(synth_complexity)
            
        #     # Handle NaN values
        #     valid_indices = ~(np.isnan(real_values) | np.isnan(synth_values))
        #     real_values_clean = real_values[valid_indices]
        #     synth_values_clean = synth_values[valid_indices]
        #     metrics_names_clean = [metrics_names[i] for i in range(len(metrics_names)) if valid_indices[i]]
            
        #     differences = real_values_clean - synth_values_clean
        #     relative_differences = np.abs(differences) / (np.abs(real_values_clean) + 1e-8)
        #     similarity_scores = 1 / (1 + relative_differences)
            
        #     # Save plots if path provided
        #     if save_path:
        #         try:
        #             fig = plt.figure(figsize=(12, 5))
                    
        #             # Real data complexity plot
        #             plt.subplot(1, 2, 1)
        #             cc_real.plot(fig, (1, 2, 1))
        #             plt.title(f'Real Data Complexity - {dataset_name}')
                    
        #             # Synthetic data complexity plot
        #             plt.subplot(1, 2, 2)
        #             cc_synth.plot(fig, (1, 2, 2))
        #             plt.title(f'Synthetic Data Complexity - {dataset_name}')
                    
        #             plt.tight_layout()
        #             plt.savefig(os.path.join(save_path, f'complexity_comparison_{dataset_name}.png'), 
        #                        dpi=100, bbox_inches='tight')
        #             plt.close()
        #         except Exception as e:
        #             print(f"Warning: Could not save complexity plots: {e}")
            
        #     return {
        #         'metrics_names': metrics_names_clean,
        #         'real_complexity': real_values_clean,
        #         'synth_complexity': synth_values_clean,
        #         'similarity_scores': similarity_scores,
        #         'mean_similarity': np.mean(similarity_scores),
        #         'real_score': cc_real.score(),
        #         'synth_score': cc_synth.score(),
        #         'score_difference': abs(cc_real.score() - cc_synth.score())
        #     }
            
        # except Exception as e:
        #     print(f"Warning: Complexity evaluation failed: {e}")
        #     return {
        #         'metrics_names': [],
        #         'real_complexity': [],
        #         'synth_complexity': [],
        #         'similarity_scores': [],
        #         'mean_similarity': 0.0,
        #         'real_score': 0.0,
        #         'synth_score': 0.0,
        #         'score_difference': 0.0,
        #         'error': str(e)
        #     }

    def _save_complexity_plots(self, iteration_results, save_path, dataset_name, k):
        """
        Save complexity evaluation plots including iteration results and averages.
        """
        import matplotlib.pyplot as plt
        import os
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Score progression across iterations
        real_scores = iteration_results['real_scores']
        synth_scores = iteration_results['synth_scores']
        iterations = range(1, len(real_scores) + 1)
        
        axes[0, 0].plot(iterations, real_scores, 'o-', label='Real Data', color='blue', alpha=0.7, linewidth=2)
        axes[0, 0].plot(iterations, synth_scores, 's-', label='Synthetic Data', color='orange', alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Complexity Score')
        axes[0, 0].set_title(f'Complexity Scores Across {k} Iterations')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add mean lines
        if real_scores:
            valid_real = [s for s in real_scores if s > 0]
            valid_synth = [s for s in synth_scores if s > 0]
            if valid_real:
                axes[0, 0].axhline(y=np.mean(valid_real), color='blue', linestyle='--', alpha=0.5, 
                                label=f'Real Mean: {np.mean(valid_real):.3f}')
            if valid_synth:
                axes[0, 0].axhline(y=np.mean(valid_synth), color='orange', linestyle='--', alpha=0.5,
                                label=f'Synth Mean: {np.mean(valid_synth):.3f}')
        
        # Plot 2: Average complexity metrics comparison
        if iteration_results['metrics_names'] and len(iteration_results['real_complexities']) > 0:
            real_complexities = np.array(iteration_results['real_complexities'])
            synth_complexities = np.array(iteration_results['synth_complexities'])
            
            # Filter out zero rows (failed iterations)
            valid_rows = np.any(real_complexities > 0, axis=1) & np.any(synth_complexities > 0, axis=1)
            
            if np.any(valid_rows):
                mean_real = np.mean(real_complexities[valid_rows], axis=0)
                mean_synth = np.mean(synth_complexities[valid_rows], axis=0)
                metrics_names = iteration_results['metrics_names']
                
                # Only plot metrics that are not NaN
                valid_metrics = ~(np.isnan(mean_real) | np.isnan(mean_synth))
                if np.any(valid_metrics):
                    mean_real_clean = mean_real[valid_metrics]
                    mean_synth_clean = mean_synth[valid_metrics]
                    metrics_clean = [metrics_names[i] for i in range(len(metrics_names)) if valid_metrics[i]]
                    
                    x = np.arange(len(metrics_clean))
                    width = 0.35
                    
                    axes[0, 1].bar(x - width/2, mean_real_clean, width, label='Real', alpha=0.7, color='blue')
                    axes[0, 1].bar(x + width/2, mean_synth_clean, width, label='Synthetic', alpha=0.7, color='orange')
                    axes[0, 1].set_xlabel('Complexity Metrics')
                    axes[0, 1].set_ylabel('Average Score')
                    axes[0, 1].set_title('Average Complexity Metrics Comparison')
                    axes[0, 1].set_xticks(x)
                    axes[0, 1].set_xticklabels(metrics_clean, rotation=45, ha='right')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Score distribution (box plot)
        valid_real_scores = [s for s in real_scores if s > 0]
        valid_synth_scores = [s for s in synth_scores if s > 0]
        
        if valid_real_scores and valid_synth_scores:
            score_data = [valid_real_scores, valid_synth_scores]
            box_plot = axes[1, 0].boxplot(score_data, labels=['Real Data', 'Synthetic Data'], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][1].set_facecolor('lightsalmon')
            axes[1, 0].set_ylabel('Complexity Score')
            axes[1, 0].set_title('Score Distribution Across Iterations')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Iteration details
        if iteration_results['iteration_details']:
            details = iteration_results['iteration_details']
            successful_details = [d for d in details if not d.get('failed', False)]
            
            if successful_details:
                iterations_success = [d['iteration'] for d in successful_details]
                real_sizes = [d['real_dataset_size'] for d in successful_details]
                synth_sizes = [d['synth_dataset_size'] for d in successful_details]
                
                axes[1, 1].plot(iterations_success, real_sizes, 'o-', label='Real Dataset Size', 
                            alpha=0.7, color='blue', linewidth=2)
                axes[1, 1].plot(iterations_success, synth_sizes, 's-', label='Synthetic Dataset Size', 
                            alpha=0.7, color='orange', linewidth=2)
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Dataset Size')
                axes[1, 1].set_title('Dataset Sizes Across Successful Iterations')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Enhanced Complexity Evaluation Results - {dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, f'enhanced_complexity_evaluation_{dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Complexity plots saved to: {plot_path}")

    def hardness_evaluation(self, X_real, y_real, X_synth, y_synth, 
                        save_path=None, dataset_name="dataset", k=3, minority_class=None,
                        hardness_metrics=None):
        """
        Evaluate instance hardness similarity using pyhard package with comprehensive statistical analysis.
        FIXED VERSION - Resolves Tkinter threading errors
        """
        
        print(f"Running comprehensive hardness evaluation for {dataset_name}...")
        
        # Import required libraries
        try:
            import pandas as pd
            from sklearn.preprocessing import MinMaxScaler
            from scipy.stats import ks_2samp
            import os
            # Assuming pyhard imports (adjust based on actual package structure)
            from pyhard.measures import ClassificationMeasures
        except ImportError as e:
            print(f"Warning: Required libraries not available: {e}")
            # return self._fallback_hardness_evaluation(X_real, y_real, X_synth, y_synth, save_path, dataset_name, k, minority_class)
        
        # Convert inputs to numpy arrays
        X_real = np.array(X_real)
        y_real = np.array(y_real)
        X_synth = np.array(X_synth)
        y_synth = np.array(y_synth)
        
        # Default hardness metrics if not provided
        if hardness_metrics is None:
            hardness_metrics = ['feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
                            'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV', 
                            'feature_CB', 'feature_N1', 'feature_N2', 'feature_LSC', 
                            'feature_LSR', 'feature_Harmfulness', 'feature_Usefulness', 
                            'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4']

        print(f"Using hardness metrics: {hardness_metrics}")
        
        # Step 1: Automatically detect minority and majority classes
        unique_classes, class_counts = np.unique(y_real, return_counts=True)
        
        if minority_class is None:
            minority_class = unique_classes[np.argmin(class_counts)]
            print(f"Auto-detected minority class: {minority_class}")
        
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, but found {len(unique_classes)} classes: {unique_classes}")
        
        # Separate minority and majority classes
        minority_mask = y_real == minority_class
        X_real_min = X_real[minority_mask]
        y_real_min = y_real[minority_mask]
        X_real_maj = X_real[~minority_mask]
        y_real_maj = y_real[~minority_mask]
        
        # Step 2: Calculate class imbalance ratio
        class_imbalance_ratio = len(X_real_min) / len(X_real_maj)
        
        # Determine sizes
        n_real_min = len(X_real_min)
        n_synth = len(X_synth)
        n_real_maj = len(X_real_maj)
        
        # Initialize storage for k iterations
        iteration_results = {
            'real_hardness_scores': [],      # List of DataFrames for each iteration
            'synth_hardness_scores': [],     # List of DataFrames for each iteration
            'sampling_strategy': None,
            'iteration_details': [],
            'successful_iterations': 0
        }
        
        try:
            # Step 3: Determine sampling strategy
            if n_synth >= n_real_min:
                sampling_strategy = "synth_larger"
                sample_size_min = n_real_min
                print(f"Using Strategy 1: Sampling {sample_size_min} from synthetic data (synth >= real)")
            else:
                sampling_strategy = "real_larger"
                sample_size_min = n_synth
                print(f"Using Strategy 2: Sampling {sample_size_min} from real minority data (real >= synth)")
            
            iteration_results['sampling_strategy'] = sampling_strategy
            
            # Step 4: Perform k iterations
            for iteration in range(k):
                print(f"  Iteration {iteration + 1}/{k}")
                
                try:
                    # Sample minority data based on strategy
                    if sampling_strategy == "synth_larger":
                        # Sample from synthetic to match real minority size
                        synth_indices = np.random.choice(n_synth, size=sample_size_min, replace=False)
                        X_synth_sampled = X_synth[synth_indices]
                        y_synth_sampled = y_synth[synth_indices]
                        X_real_min_iter = X_real_min.copy()
                        y_real_min_iter = y_real_min.copy()
                    else:
                        # Sample from real minority to match synthetic size
                        real_indices = np.random.choice(n_real_min, size=sample_size_min, replace=False)
                        X_real_min_iter = X_real_min[real_indices]
                        y_real_min_iter = y_real_min[real_indices]
                        X_synth_sampled = X_synth.copy()
                        y_synth_sampled = y_synth.copy()
                    
                    # Calculate required majority sample size to maintain class imbalance ratio
                    maj_sample_size = int(sample_size_min / class_imbalance_ratio)
                    maj_sample_size = min(maj_sample_size, n_real_maj)
                    
                    # Sample majority data
                    maj_indices = np.random.choice(n_real_maj, size=maj_sample_size, replace=False)
                    X_maj_sampled = X_real_maj[maj_indices]
                    y_maj_sampled = y_real_maj[maj_indices]
                    
                    # Create complete datasets for this iteration
                    X_real_complete = np.vstack([X_real_min_iter, X_maj_sampled])
                    y_real_complete = np.hstack([y_real_min_iter, y_maj_sampled])
                    X_synth_complete = np.vstack([X_synth_sampled, X_maj_sampled])
                    y_synth_complete = np.hstack([y_synth_sampled, y_maj_sampled])
                    
                    # Apply clustering sampling if datasets are large
                    clustering_applied = False
                    if len(X_real_complete) > 10000:
                        print(f"    Applying clustering sampling (dataset size: {len(X_real_complete)})")
                        X_real_complete, y_real_complete = self._cluster_sampling(
                            X_real_complete, y_real_complete, k=4
                        )
                        X_synth_complete, y_synth_complete = self._cluster_sampling(
                            X_synth_complete, y_synth_complete, k=4
                        )
                        clustering_applied = True
                    
                    # Calculate hardness scores using pyhard for this iteration
                    print(f"    Calculating hardness scores...")
                    real_hardness_df = self._calculate_hardness_scores(
                        X_real_complete, y_real_complete, hardness_metrics
                    )
                    synth_hardness_df = self._calculate_hardness_scores(
                        X_synth_complete, y_synth_complete, hardness_metrics
                    )
                    
                    # Store results
                    iteration_results['real_hardness_scores'].append(real_hardness_df)
                    iteration_results['synth_hardness_scores'].append(synth_hardness_df)
                    iteration_results['successful_iterations'] += 1
                    
                    # Store iteration details
                    iteration_results['iteration_details'].append({
                        'iteration': iteration + 1,
                        'real_dataset_size': len(X_real_complete),
                        'synth_dataset_size': len(X_synth_complete),
                        'minority_sample_size': sample_size_min,
                        'majority_sample_size': maj_sample_size,
                        'clustering_applied': clustering_applied,
                        'real_hardness_metrics': list(real_hardness_df.columns),
                        'synth_hardness_metrics': list(synth_hardness_df.columns),
                        'real_mean_hardness': real_hardness_df.mean().to_dict(),
                        'synth_mean_hardness': synth_hardness_df.mean().to_dict()
                    })
                    
                    print(f"    Completed - Calculated {len(real_hardness_df.columns)} hardness metrics")
                    
                except Exception as e:
                    print(f"    Warning: Iteration {iteration + 1} failed: {e}")
                    iteration_results['iteration_details'].append({
                        'iteration': iteration + 1,
                        'failed': True,
                        'error': str(e)
                    })
            
            # Step 5: Statistical analysis across iterations for each hardness metric
            if iteration_results['successful_iterations'] > 0:
                print(f"Performing statistical analysis across {iteration_results['successful_iterations']} successful iterations...")
                
                # Get all available metrics (intersection of all successful iterations)
                all_real_dfs = iteration_results['real_hardness_scores']
                all_synth_dfs = iteration_results['synth_hardness_scores']
                
                # Find common metrics across all iterations
                common_metrics = set(all_real_dfs[0].columns)
                for df in all_real_dfs[1:] + all_synth_dfs:
                    common_metrics = common_metrics.intersection(set(df.columns))
                common_metrics = list(common_metrics)
                
                print(f"Common hardness metrics across iterations: {common_metrics}")
                
                # Initialize results storage
                metric_statistics = {}
                ks_statistics = []
                ks_pvalues = []
                
                # Analyze each hardness metric
                for metric in common_metrics:
                    print(f"  Analyzing metric: {metric}")
                    
                    # Collect all values for this metric across iterations
                    real_values_all = []
                    synth_values_all = []
                    
                    for real_df, synth_df in zip(all_real_dfs, all_synth_dfs):
                        real_values_all.extend(real_df[metric].values)
                        synth_values_all.extend(synth_df[metric].values)
                    
                    real_values_all = np.array(real_values_all)
                    synth_values_all = np.array(synth_values_all)
                    
                    # Remove NaN values
                    real_values_clean = real_values_all[~np.isnan(real_values_all)]
                    synth_values_clean = synth_values_all[~np.isnan(synth_values_all)]
                    
                    if len(real_values_clean) > 0 and len(synth_values_clean) > 0:
                        # Calculate statistics
                        real_mean = np.mean(real_values_clean)
                        real_std = np.std(real_values_clean)
                        synth_mean = np.mean(synth_values_clean)
                        synth_std = np.std(synth_values_clean)
                        
                        # Perform KS test
                        ks_stat, ks_pval = ks_2samp(real_values_clean, synth_values_clean)
                        
                        # Store results for this metric
                        metric_statistics[metric] = {
                            'real_mean': real_mean,
                            'real_std': real_std,
                            'synth_mean': synth_mean,
                            'synth_std': synth_std,
                            'ks_statistic': ks_stat,
                            'ks_pvalue': ks_pval,
                            'similarity_score': 1 - ks_stat,  # Convert KS stat to similarity
                            'n_real_samples': len(real_values_clean),
                            'n_synth_samples': len(synth_values_clean)
                        }
                        
                        ks_statistics.append(ks_stat)
                        ks_pvalues.append(ks_pval)
                        
                        print(f"    {metric}: Real={real_mean:.3f}±{real_std:.3f}, "
                            f"Synth={synth_mean:.3f}±{synth_std:.3f}, KS={ks_stat:.3f} (p={ks_pval:.3f})")
                    
                    else:
                        print(f"    Warning: Insufficient data for metric {metric}")
                        metric_statistics[metric] = {
                            'real_mean': 0.0, 'real_std': 0.0,
                            'synth_mean': 0.0, 'synth_std': 0.0,
                            'ks_statistic': 1.0, 'ks_pvalue': 0.0,
                            'similarity_score': 0.0,
                            'n_real_samples': 0, 'n_synth_samples': 0
                        }
                
                # Step 6: Calculate overall statistics
                if ks_statistics:
                    mean_ks_statistic = np.mean(ks_statistics)
                    std_ks_statistic = np.std(ks_statistics)
                    mean_ks_pvalue = np.mean(ks_pvalues)
                    overall_similarity = np.mean([metric_statistics[m]['similarity_score'] 
                                                for m in metric_statistics])
                else:
                    mean_ks_statistic = 1.0
                    std_ks_statistic = 0.0
                    mean_ks_pvalue = 0.0
                    overall_similarity = 0.0
            
            else:
                print("No successful iterations - cannot perform statistical analysis")
                metric_statistics = {}
                mean_ks_statistic = 1.0
                std_ks_statistic = 0.0
                mean_ks_pvalue = 0.0
                overall_similarity = 0.0
            
            # FIXED PLOTTING SECTION - No GUI backend issues
            if save_path and iteration_results['successful_iterations'] > 0:
                try:
                    print(f"Saving hardness evaluation plots...")
                    
                    # Create output directory
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Set matplotlib to use Agg backend explicitly (no GUI)
                    plt.switch_backend('Agg')
                    
                    # Create comprehensive hardness plots (6 subplots)
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    
                    # Plot 1: KS Statistics for each metric
                    if metric_statistics:
                        metrics = list(metric_statistics.keys())
                        ks_stats = [metric_statistics[m]['ks_statistic'] for m in metrics]
                        
                        axes[0, 0].bar(range(len(metrics)), ks_stats, alpha=0.7, color='skyblue')
                        axes[0, 0].set_xlabel('Hardness Metrics')
                        axes[0, 0].set_ylabel('KS Statistic')
                        axes[0, 0].set_title('KS Statistics by Hardness Metric')
                        axes[0, 0].set_xticks(range(len(metrics)))
                        axes[0, 0].set_xticklabels([m.replace('feature_', '') for m in metrics], 
                                                rotation=45, ha='right')
                        axes[0, 0].grid(True, alpha=0.3)
                    
                    # Plot 2: Similarity Scores
                    if metric_statistics:
                        similarity_scores = [metric_statistics[m]['similarity_score'] for m in metrics]
                        colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in similarity_scores]
                        
                        axes[0, 1].bar(range(len(metrics)), similarity_scores, alpha=0.7, color=colors)
                        axes[0, 1].set_xlabel('Hardness Metrics')
                        axes[0, 1].set_ylabel('Similarity Score')
                        axes[0, 1].set_title('Similarity Scores by Hardness Metric')
                        axes[0, 1].set_xticks(range(len(metrics)))
                        axes[0, 1].set_xticklabels([m.replace('feature_', '') for m in metrics], 
                                                rotation=45, ha='right')
                        axes[0, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold')
                        axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
                        axes[0, 1].legend()
                        axes[0, 1].grid(True, alpha=0.3)
                    
                    # Plot 3: Mean comparison (Real vs Synthetic)
                    if metric_statistics:
                        real_means = [metric_statistics[m]['real_mean'] for m in metrics]
                        synth_means = [metric_statistics[m]['synth_mean'] for m in metrics]
                        
                        x = np.arange(len(metrics))
                        width = 0.35
                        
                        axes[0, 2].bar(x - width/2, real_means, width, label='Real', alpha=0.7, color='blue')
                        axes[0, 2].bar(x + width/2, synth_means, width, label='Synthetic', alpha=0.7, color='orange')
                        axes[0, 2].set_xlabel('Hardness Metrics')
                        axes[0, 2].set_ylabel('Mean Hardness Score')
                        axes[0, 2].set_title('Mean Hardness Comparison')
                        axes[0, 2].set_xticks(x)
                        axes[0, 2].set_xticklabels([m.replace('feature_', '') for m in metrics], 
                                                rotation=45, ha='right')
                        axes[0, 2].legend()
                        axes[0, 2].grid(True, alpha=0.3)
                    
                    # Plot 4: P-value distribution
                    if metric_statistics:
                        p_values = [metric_statistics[m]['ks_pvalue'] for m in metrics]
                        
                        axes[1, 0].bar(range(len(metrics)), p_values, alpha=0.7, color='lightcoral')
                        axes[1, 0].set_xlabel('Hardness Metrics')
                        axes[1, 0].set_ylabel('KS Test P-value')
                        axes[1, 0].set_title('Statistical Significance (KS Test P-values)')
                        axes[1, 0].set_xticks(range(len(metrics)))
                        axes[1, 0].set_xticklabels([m.replace('feature_', '') for m in metrics], 
                                                rotation=45, ha='right')
                        axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
                        axes[1, 0].legend()
                        axes[1, 0].grid(True, alpha=0.3)
                    
                    # Plot 5: Standard deviation comparison
                    if metric_statistics:
                        real_stds = [metric_statistics[m]['real_std'] for m in metrics]
                        synth_stds = [metric_statistics[m]['synth_std'] for m in metrics]
                        
                        axes[1, 1].bar(x - width/2, real_stds, width, label='Real', alpha=0.7, color='lightblue')
                        axes[1, 1].bar(x + width/2, synth_stds, width, label='Synthetic', alpha=0.7, color='lightsalmon')
                        axes[1, 1].set_xlabel('Hardness Metrics')
                        axes[1, 1].set_ylabel('Standard Deviation')
                        axes[1, 1].set_title('Hardness Variability Comparison')
                        axes[1, 1].set_xticks(x)
                        axes[1, 1].set_xticklabels([m.replace('feature_', '') for m in metrics], 
                                                rotation=45, ha='right')
                        axes[1, 1].legend()
                        axes[1, 1].grid(True, alpha=0.3)
                    
                    # Plot 6: Overall summary
                    if iteration_results['successful_iterations'] > 0:
                        summary_data = {
                            'Successful\nIterations': iteration_results['successful_iterations'],
                            'Total\nMetrics': len(metric_statistics),
                            'Avg KS\nStatistic': mean_ks_statistic,
                            'Overall\nSimilarity': overall_similarity
                        }
                        
                        colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightpink']
                        bars = axes[1, 2].bar(summary_data.keys(), summary_data.values(), 
                                            alpha=0.7, color=colors)
                        axes[1, 2].set_ylabel('Value')
                        axes[1, 2].set_title('Evaluation Summary')
                        axes[1, 2].tick_params(axis='x', rotation=0)
                        axes[1, 2].grid(True, alpha=0.3)
                        
                        # Add value labels on bars
                        for bar, value in zip(bars, summary_data.values()):
                            height = bar.get_height()
                            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                            f'{value:.2f}' if isinstance(value, float) else str(value),
                                            ha='center', va='bottom', fontsize=10)
                    
                    plt.suptitle(f'Enhanced Hardness Evaluation Results - {dataset_name}', 
                            fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    # Save the comprehensive plot
                    plot_path = os.path.join(save_path, f'enhanced_hardness_evaluation_{dataset_name}.png')
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()  # Important: Close the figure to free memory
                    
                    print(f"Enhanced hardness plots saved to: {plot_path}")
                    
                except Exception as e:
                    print(f"Warning: Could not save hardness plots: {e}")
            
            # Count successful iterations
            successful_iterations = iteration_results['successful_iterations']
            
            final_results = { 
                'detailed':{
                    # Core hardness results
                    'metric_statistics': metric_statistics,
                    'hardness_metrics_used': list(metric_statistics.keys()),
                    'individual_ks_statistics': ks_statistics if 'ks_statistics' in locals() else [],
                    'individual_ks_pvalues': ks_pvalues if 'ks_pvalues' in locals() else []},
                    
                'summary':{
                    'overall_similarity': overall_similarity,
                    # Overall KS statistics
                    'mean_ks_statistic': mean_ks_statistic,
                    'std_ks_statistic': std_ks_statistic,
                    'mean_ks_pvalue': mean_ks_pvalue},

                'metadata':{
                
                    # Metadata
                    'sampling_strategy': sampling_strategy,
                    'k_iterations': k,
                    'successful_iterations': iteration_results['successful_iterations'],
                    'minority_class': minority_class,
                    'class_imbalance_ratio': class_imbalance_ratio,
                    'original_data_composition': {
                        'minority_samples': n_real_min,
                        'majority_samples': n_real_maj,
                        'synthetic_samples': n_synth
                    },
                    
                    # Detailed results
                    'iteration_details': iteration_results['iteration_details']}
            }


            
            print(f"Hardness evaluation completed successfully!")
            print(f"  Successful iterations: {successful_iterations}/{k}")
            print(f"  Metrics analyzed: {len(metric_statistics)}")
            print(f"  Overall similarity: {overall_similarity:.3f}")
            print(f"  Mean KS statistic: {mean_ks_statistic:.3f} ± {std_ks_statistic:.3f}")
            print(f"  Mean KS p-value: {mean_ks_pvalue:.3f}")
            
            return final_results
            
        except Exception as e:
            print(f"Error: Hardness evaluation failed: {e}")
            return {
                'metric_statistics': {},
                'hardness_metrics_used': [],
                'overall_similarity': 0.0,
                'mean_ks_statistic': 1.0,
                'std_ks_statistic': 0.0,
                'mean_ks_pvalue': 0.0,
                'individual_ks_statistics': [],
                'individual_ks_pvalues': [],
                'sampling_strategy': 'unknown',
                'k_iterations': k,
                'successful_iterations': 0,
                'minority_class': minority_class,
                'class_imbalance_ratio': 0.0,
                'original_data_composition': {},
                'iteration_details': [],
                'error': str(e)
            }


    def _calculate_hardness_scores(self, X, y, hardness_metrics):
        """
        Calculate hardness scores using pyhard package (adapted from your helper function).
        """
        try:
            import pandas as pd
            from sklearn.preprocessing import MinMaxScaler
            from pyhard.measures import ClassificationMeasures
            
            # Check if input data is empty
            if X.size == 0 or y.size == 0:
                raise ValueError("Input data X and y should not be empty.")
            
            # Ensure y is a 1D array
            if len(y.shape) > 1 and y.shape[1] > 1:
                raise ValueError("y should be a 1D array of labels.")
            
            # Create a DataFrame from X and y
            data = pd.DataFrame(X)
            data['target'] = y
            column_names = [f"feature_{i}" for i in range(X.shape[1])] + ['target']
            data.columns = column_names
            
            HardnessMetrics = ['feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
                            'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV', 
                            'feature_CB', 'feature_N1', 'feature_N2', 'feature_LSC', 
                            'feature_LSR', 'feature_Harmfulness', 'feature_Usefulness', 
                            'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4']

            # If hardness_metrics is empty, raise an error
            if not hardness_metrics:
                raise ValueError("No hardness metrics specified.")
            
            # Check if all specified hardness metrics are valid
            invalid_metrics = [metric for metric in hardness_metrics 
                            if metric not in HardnessMetrics ]
            if invalid_metrics:
                print(f"Warning: Invalid hardness metrics will be skipped: {invalid_metrics}")
            
            # Filter valid pyhard metrics
            pyhard_hardness_metrics = [metric for metric in hardness_metrics if metric in HardnessMetrics]
            
            if pyhard_hardness_metrics:
                # Initialize ClassificationMeasures with the DataFrame
                HM = ClassificationMeasures(data)
                
                # Calculate the hardness metrics
                data_HM = HM.calculate_all()
                
                # Extract only the specified hardness metrics
                hardness_scores = {metric: data_HM[metric] for metric in pyhard_hardness_metrics 
                                if metric in data_HM}
                
                # Create a DataFrame from the hardness scores
                hardness_df = pd.DataFrame(hardness_scores)
                
                # Check if hardness measures are not null or empty
                if hardness_df.empty:
                    raise ValueError("No hardness metrics were calculated.")
                
                if hardness_df.isnull().values.any():
                    # Which hardness metrics contain null values
                    null_metrics = hardness_df.columns[hardness_df.isnull().any()].tolist()
                    print(f"Warning: The following hardness metrics contain null values: {null_metrics}")
                    
                    # Eliminate any columns that have null values
                    hardness_df = hardness_df.dropna(axis=1, how='any')
                
                # Min-max scale the hardness scores to [0, 1]
                if not hardness_df.empty:
                    scaler = MinMaxScaler()
                    scaled_hardness_df = pd.DataFrame(
                        scaler.fit_transform(hardness_df), 
                        columns=hardness_df.columns
                    )
                    
                    # Avoid 0 values after scaling by replacing them with a small value
                    scaled_hardness_df = scaled_hardness_df.replace(0, 1e-6)
                    
                    return scaled_hardness_df
                else:
                    raise ValueError("All hardness metrics contained null values.")
            
            else:
                raise ValueError("No valid pyhard hardness metrics found.")
        
        except Exception as e:
            print(f"Warning: Hardness calculation failed: {e}")
            # Return empty DataFrame with expected structure
            return pd.DataFrame()


    def topological_evaluation(self, X_real, y_real, X_synth, y_synth, save_path=None, dataset_name="dataset", minority_class=None):
        """
        Evaluate topological similarity using persistent homology.
        """
                # Convert inputs to numpy arrays
        X_real = np.array(X_real)
        y_real = np.array(y_real)
        X_synth = np.array(X_synth)
        y_synth = np.array(y_synth)
        
        # Step 1: Automatically detect minority and majority classes
        unique_classes, class_counts = np.unique(y_real, return_counts=True)
        
        if minority_class is None:
            minority_class = unique_classes[np.argmin(class_counts)]
            print(f"Auto-detected minority class: {minority_class}")
        
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, but found {len(unique_classes)} classes: {unique_classes}")
        
        # Separate minority and majority classes
        minority_mask = y_real == minority_class
        X_real_min = X_real[minority_mask]
        y_real_min = y_real[minority_mask]
        X_real_maj = X_real[~minority_mask]
        y_real_maj = y_real[~minority_mask]

        if not TDA_AVAILABLE:
            return None
        
        results = {'runs':[]}
        try:
            # Sample and reduce dimensionality for TDA
            X_real_min_sampled, _ = self._cluster_sampling(X_real_min, np.zeros(len(X_real_min)))
            X_synth_sampled, _ = self._cluster_sampling(X_synth, np.zeros(len(X_synth)))
            
            # Reduce to 3D for computational efficiency
            if X_real_min_sampled.shape[1] > 3:
                pca = PCA(n_components=3, random_state=self.random_state)
                X_real_pca = pca.fit_transform(X_real_min_sampled)
                X_synth_pca = pca.transform(X_synth_sampled)
            else:
                X_real_pca = X_real_min_sampled
                X_synth_pca = X_synth_sampled
            
            # For k times sample equal sets from X_real_pca and X_synth_pca, where the size is random between in the range((half the size of the smaller set), (the size of the smaller set))
            b = min(X_synth_pca.shape[0] , X_real_pca.shape[0])
            a = max(10, int(0.5*b))
            for k in range(0,3):
                # Compute persistent homology
                r_sample_size = random.randint(a, b)
                
                # Randomly sample without replacement
                idx_real = np.random.choice(X_real_pca.shape[0], r_sample_size, replace=False)
                idx_synth = np.random.choice(X_synth_pca.shape[0], r_sample_size, replace=False)

                # CONTINUE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 17/06/2025
                # sample r_sample_size from X_real_pca and X_synth_pca
                X_real_sample = X_real_pca[idx_real]
                X_synth_sample = X_synth_pca[idx_synth]

                # Compute diagrams
                dgm_real = ripser(X_real_sample)['dgms']  
                dgm_synth = ripser(X_synth_sample)['dgms']

                run_results = {}
                # Compute the Bottleneck distance between persistent homology diagrams
                for homology_dim in [0,1]:
                    if homology_dim < len(dgm_real) and homology_dim < len(dgm_synth):
                        dgm_r = dgm_real[homology_dim]
                        dgm_s = dgm_synth[homology_dim]

                        run_results[f'bottleneck_H{homology_dim}'] = bottleneck(dgm_r, dgm_s)
                        run_results[f'wasserstein_H{homology_dim}'] = wasserstein(dgm_r, dgm_s)

                results['runs'].append(run_results)


            # Save plot for the last run
            if save_path:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                plot_diagrams(dgm_real, ax=axes[0], title=f'Real Data - {dataset_name}')
                plot_diagrams(dgm_synth, ax=axes[1], title=f'Synthetic Data - {dataset_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'persistence_diagrams_{dataset_name}.png'),
                            dpi=100, bbox_inches='tight')
                plt.close()

            # Compute summary statistics
            summary = {}
            for key in ['bottleneck_H0', 'bottleneck_H1', 'wasserstein_H0', 'wasserstein_H1']:
                vals = [run.get(key, np.nan) for run in results['runs']]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    summary[f'{key}_mean'] = np.mean(vals)
                    summary[f'{key}_std'] = np.std(vals)

            results['summary'] = summary

        except Exception as e:
            print(f"Warning: Topological evaluation failed: {e}")
            results['error'] = str(e)
            
        print(f'results of topological analysis:\n {results}')
        return results
                       
    def _cluster_sampling(self, X, y, sampling_ratio=None, k=4):
        """
        Cluster-based sampling for large datasets.
        """
        if sampling_ratio is None:
            sampling_ratio = 0.01 if X.shape[0] >= 100000 else 0.1 if X.shape[0] >= 10000 else 0
        
        if sampling_ratio == 0:
            return X, y
        
        # Shuffle data
        X_shuffled, y_shuffled = shuffle(X, y, random_state=self.random_state)
        
        # Apply clustering
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=self.random_state).fit(X_shuffled)
        cluster_labels = kmeans.predict(X_shuffled)
        
        # Sample from each cluster
        X_sampled = []
        y_sampled = []
        cluster_counts = Counter(cluster_labels)
        
        for i in range(k):
            cluster_count = cluster_counts[i]
            sample_size = int(sampling_ratio * cluster_count)
            if sample_size > 0:
                cluster_mask = cluster_labels == i
                X_sampled.append(X_shuffled[cluster_mask][:sample_size])
                y_sampled.append(y_shuffled[cluster_mask][:sample_size])
        
        return np.array(np.concatenate(X_sampled)), np.array(np.concatenate(y_sampled))
   
    def _analyze_cluster_structure(self, X_real, X_synth):
        """
        Analyze cluster structure differences between real and synthetic data.
        """
        try:
            # Optimal number of clusters using elbow method
            def find_optimal_k(X, max_k=10):
                inertias = []
                K_range = range(2, min(max_k, len(X)//2))
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                    kmeans.fit(X)
                    inertias.append(kmeans.inertia_)
                
                # Simple elbow detection
                if len(inertias) > 2:
                    diffs = np.diff(inertias)
                    optimal_k = K_range[np.argmin(diffs[1:]) + 1]
                else:
                    optimal_k = K_range[0] if K_range else 2
                
                return optimal_k, inertias
            
            # Find optimal clusters for both datasets
            k_real, inertias_real = find_optimal_k(X_real)
            k_synth, inertias_synth = find_optimal_k(X_synth)
            
            # Cluster with optimal k
            kmeans_real = KMeans(n_clusters=k_real, random_state=self.random_state)
            kmeans_synth = KMeans(n_clusters=k_synth, random_state=self.random_state)
            
            labels_real = kmeans_real.fit_predict(X_real)
            labels_synth = kmeans_synth.fit_predict(X_synth)
            
            # Calculate cluster quality metrics
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            sil_real = silhouette_score(X_real, labels_real)
            sil_synth = silhouette_score(X_synth, labels_synth)
            
            ch_real = calinski_harabasz_score(X_real, labels_real)
            ch_synth = calinski_harabasz_score(X_synth, labels_synth)
            
            return {
                'optimal_k_real': k_real,
                'optimal_k_synth': k_synth,
                'silhouette_real': sil_real,
                'silhouette_synth': sil_synth,
                'calinski_harabasz_real': ch_real,
                'calinski_harabasz_synth': ch_synth,
                'silhouette_similarity': 1 - abs(sil_real - sil_synth),
                'ch_similarity': 1 / (1 + abs(ch_real - ch_synth) / (ch_real + 1e-8))
            }
            
        except Exception as e:
            print(f"Warning: Cluster structure analysis failed: {e}")
            return {
                'optimal_k_real': 0,
                'optimal_k_synth': 0,
                'silhouette_real': 0.0,
                'silhouette_synth': 0.0,
                'calinski_harabasz_real': 0.0,
                'calinski_harabasz_synth': 0.0,
                'silhouette_similarity': 0.0,
                'ch_similarity': 0.0,
                'error': str(e)
            }
    

if __name__ == "__main__":
    # Example usage
    print("Synthetic Data Evaluator Module")
    print("Usage:")
    print("evaluator = SyntheticDataEvaluator(random_state=42)")
    print("results = evaluator.evaluate_all(X_real, y_real, X_synth, y_synth)")


