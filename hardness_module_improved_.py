"""
Hardness-Aware Module for CVAE Training

This module provides comprehensive hardness calculation functionality for tabular medical data,
designed to integrate with Conditional Variational Autoencoders (CVAEs) for improved synthetic
data generation in imbalanced datasets.

Features:
- PyHard integration for standard hardness metrics
- Custom hardness metrics (relative entropy, PCA-based, reconstruction error)
- Flexible class-based architecture
- CVAE integration utilities
- Comprehensive error handling and validation
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from scipy.stats import entropy
from imblearn.datasets import fetch_datasets
import torch
import torch.nn as nn

# device related -> GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    from pyhard.measures import ClassificationMeasures
    PYHARD_AVAILABLE = True
except ImportError:
    PYHARD_AVAILABLE = False
    warnings.warn("PyHard package not available. Standard hardness metrics will be disabled.")
    
HardnessMetrics = ['feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
                   'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV', 
                   'feature_CB', 'feature_N1', 'feature_N2', 'feature_LSC', 
                   'feature_LSR', 'feature_Harmfulness', 
                   'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4']

class HardnessCalculator:
    """
    Comprehensive hardness calculator for tabular data with support for multiple hardness metrics.
    """
    
    # Standard PyHard metrics
    PYHARD_METRICS = [
        'feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
        'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV', 
        'feature_CB', 'feature_N1', 'feature_N2', 'feature_LSC', 
        'feature_LSR', 'feature_Harmfulness', 'feature_Usefulness', 
        'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4'
    ]
    
    # Custom proposed metrics
    CUSTOM_METRICS = [
        'relative_entropy', 'pca_contribution' 
    ]
    
    NO_WEIGHT_METRICS = [None]
    # Metric groups for analysis
    METRIC_GROUPS = {
        "linear": ['feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P', 'feature_TD_U'],
        "neighborhood_based": ['feature_CL', 'feature_CLD', 'feature_MV', 'feature_CB'],
        "network_based": ['feature_N1', 'feature_N2'],
        "feature_based": ['feature_LSC', 'feature_LSR'],
        "other": ['feature_Harmfulness', 'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4'],
        "custom": ['relative_entropy', 'pca_contribution'] 
    }
    
    def __init__(self, random_state: int = 42, n_classifiers: int = 5, min_hardness_value: float = 1e-6):
        """
        Initialize the HardnessCalculator.
        
        Args:
            random_state: Random seed for reproducibility
            n_classifiers: Number of classifiers for ensemble-based metrics
            min_hardness_value: Minimum value to replace zeros after scaling
        """
        self.random_state = random_state
        self.n_classifiers = n_classifiers
        self.min_hardness_value = min_hardness_value
        self.scaler = MinMaxScaler()
        self.pca = None
        self.label_encoder = None
        
        # Initialize probabilistic classifiers for relative entropy
        self.classifiers = [
            # LogisticRegression(random_state=random_state),
            RandomForestClassifier(random_state=random_state),
            # GaussianNB(),
            SVC(probability=True, random_state=random_state),
            KNeighborsClassifier(n_neighbors=5)
        ]
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess input data."""
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data X and y should not be empty.")
        
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError("y should be a 1D array of labels.")
        
        # Ensure y is 1D
        y = y.ravel()
        
        # Handle categorical labels
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
            else:
                y = self.label_encoder.transform(y)
        
        return X, y
    
    def _calculate_pyhard_metrics(self, X: np.ndarray, y: np.ndarray, 
                                  metrics: List[str]) -> pd.DataFrame:
        """Calculate PyHard-based hardness metrics."""
        # print("USING _calculate_pyhard_metrics FUNCTION")
        if not PYHARD_AVAILABLE:
            raise ImportError("PyHard package is required for standard hardness metrics.")
        
        # Create DataFrame for PyHard
        data = pd.DataFrame(X)
        data['target'] = y
        column_names = [f"feature_{i}" for i in range(X.shape[1])] + ['target']
        data.columns = column_names
        
        # Calculate hardness metrics
        hm = ClassificationMeasures(data)
        data_hm = hm.calculate_all()
        
        # Extract specified metrics
        hardness_scores = {}
        for metric in metrics:
            if metric in data_hm:
                hardness_scores[metric] = data_hm[metric]
            else:
                warnings.warn(f"Metric {metric} not found in PyHard results.")
        
        if not hardness_scores:
            raise ValueError("No valid PyHard metrics were calculated.")
        
        hardness_df = pd.DataFrame(hardness_scores)
        print(f"Number of NaN values in hardness_df in _calculate_pyhard_metrics: {hardness_df.isnull().sum().sum()}")

        
        # Handle null values
        if hardness_df.isnull().values.any():
            null_metrics = hardness_df.columns[hardness_df.isnull().any()].tolist()
            warnings.warn(f"Dropping metrics with null values: {null_metrics}")
            hardness_df = hardness_df.dropna(axis=1, how='any')
        
        # New Check after dropping NaNs, if the DataFrame is empty--
        # New Check after dropping NaNs, if the DataFrame is empty--
        if hardness_df.empty:
            warnings.warn(
                f"All calculated hardness metrics {metrics} resulted in NaN values or an empty DataFrame after dropping columns with nulls. Returning empty DataFrame."
            )
            return pd.DataFrame() # Return an empty DataFrame
        
        
        return hardness_df
    
    def _calculate_relative_entropy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate relative entropy hardness scores using ensemble of classifiers.
        
        This metric measures the uncertainty/disagreement among multiple classifiers
        for each instance, which indicates instance difficulty.
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # Use subset of classifiers if specified --> MODIFY TO ALLOW FOR RANDOM SELECTION/ SELECTION
        selected_classifiers = self.classifiers[:self.n_classifiers]
        
        # Collect probability predictions from all classifiers
        all_probabilities = []
        
        for clf in selected_classifiers:
            try:
                # Use cross-validation to get unbiased predictions
                probas = cross_val_predict(clf, X, y, cv=3, method='predict_proba')
                all_probabilities.append(probas)
            except Exception as e:
                warnings.warn(f"Classifier {type(clf).__name__} failed: {e}")
                continue
        
        if not all_probabilities:
            raise ValueError("No classifiers could generate probability predictions.")
        
        # Calculate relative entropy for each instance
        relative_entropies = []
        
        for i in range(n_samples):
            instance_probas = [proba[i] for proba in all_probabilities]
            
            # Calculate average probability distribution
            avg_proba = np.mean(instance_probas, axis=0)
            
            # Calculate relative entropy (KL divergence) from uniform distribution
            uniform_dist = np.ones(n_classes) / n_classes
            
            # Add small epsilon to avoid log(0)
            avg_proba = np.clip(avg_proba, 1e-10, 1.0)
            uniform_dist = np.clip(uniform_dist, 1e-10, 1.0)
            
            rel_entropy = entropy(avg_proba, uniform_dist)
            relative_entropies.append(rel_entropy)
        
        return np.array(relative_entropies)
    
    def _calculate_pca_metrics(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate PCA-based hardness metrics.
        
        Returns:
            pca_contribution: Contribution of each instance to principal components
        """
        # Fit PCA if not already fitted
        if self.pca is None:
            # Use 95% variance retention or max 50 components
            n_components = min(X.shape[1], 50)
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            self.pca.fit(X)
        
        # Transform data
        X_transformed = self.pca.transform(X)
        
        # Calculate contribution (sum of squared loadings weighted by explained variance)
        contributions = []
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        selected_components = np.where(cumulative_variance < 0.80)[0]
        contributions = X_transformed[:, selected_components] / self.pca.singular_values_[selected_components]
        total_contributions = np.sum(np.abs(contributions), axis=1)
        total_contributions_tensor = torch.tensor(total_contributions, dtype=torch.float32).to(device)        
        return total_contributions_tensor.cpu().numpy()
    
    def _calculate_custom_metrics(self, X: np.ndarray, y: np.ndarray, 
                                  metrics: List[str]) -> pd.DataFrame:
        """Calculate custom hardness metrics."""
        hardness_scores = {}
        
        for metric in metrics:
            if metric == 'relative_entropy':
                hardness_scores[metric] = self._calculate_relative_entropy(X, y)
            
            elif metric in ['pca_contribution']:
                if 'pca_contribution' not in hardness_scores: # and 'pca_reconstruction_error' not in hardness_scores:
                    # Calculate both PCA metrics at once for efficiency
                    pca_contrib = self._calculate_pca_metrics(X)
                    if 'pca_contribution' in metrics:
                        hardness_scores['pca_contribution'] = pca_contrib
                    # if 'pca_reconstruction_error' in metrics:
                    #     hardness_scores['pca_reconstruction_error'] = pca_recon_error
            
            else:
                warnings.warn(f"Unknown custom metric: {metric}")
        
        return pd.DataFrame(hardness_scores)
    
    def calculate_hardness_scores(self, X: np.ndarray, y: np.ndarray, 
                                  hardness_metrics: List[str]) -> pd.DataFrame:
        """
        Calculate hardness scores for each instance using specified metrics.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            hardness_metrics: List of hardness metrics to calculate
            
        Returns:
            DataFrame with hardness scores (scaled to [0,1])
        """
        # Validate inputs
        X, y = self._validate_inputs(X, y)
        
        if not hardness_metrics:
            raise ValueError("No hardness metrics specified.")
        
        # Separate PyHard and custom metrics
        pyhard_metrics = [m for m in hardness_metrics if m in self.PYHARD_METRICS]
        custom_metrics = [m for m in hardness_metrics if m in self.CUSTOM_METRICS]
        no_metrics = [m for m in hardness_metrics if m in self.NO_WEIGHT_METRICS]
        
        # Check for invalid metrics
        invalid_metrics = [m for m in hardness_metrics 
                          if m not in self.PYHARD_METRICS and m not in self.CUSTOM_METRICS and m not in self.NO_WEIGHT_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid hardness metrics: {invalid_metrics}")
        
        # Calculate metrics
        hardness_dfs = []
        
        if pyhard_metrics:
            pyhard_df = self._calculate_pyhard_metrics(X, y, pyhard_metrics)
            hardness_dfs.append(pyhard_df)

        
        if custom_metrics:
            custom_df = self._calculate_custom_metrics(X, y, custom_metrics)
            hardness_dfs.append(custom_df)
        
        if no_metrics:
            # Create dataframe with ones for no-weight metrics
            no_weight_df = pd.DataFrame(np.ones((X.shape[0], len(no_metrics))),
                                        columns=no_metrics)
            hardness_dfs.append(no_weight_df)
        
        # Combine all hardness scores
        if len(hardness_dfs) == 1:
            hardness_df = hardness_dfs[0]
        else:
            hardness_df = pd.concat(hardness_dfs, axis=1)
        
        if hardness_df.empty:
            warnings.warn("Combined hardness DataFrame is empty after processing. No valid scores to scale.")
            return pd.DataFrame() # Return an empty DataFrame


        # Scale to [0, 1] and avoid zeros
        scaled_hardness = pd.DataFrame(
            self.scaler.fit_transform(hardness_df), 
            columns=hardness_df.columns
        )
        scaled_hardness = scaled_hardness.replace(0, self.min_hardness_value)
        
        return scaled_hardness
    
    def get_metric_info(self, metric: str) -> Dict[str, str]:
        """Get information about a specific hardness metric."""
        metric_definitions = {
            'None': 'No hardness metric - used for uniform weighting',
            'feature_kDN': 'k-Disagreeing Neighbors - measures local class disagreement',
            'feature_DS': 'Decision Surface - complexity of decision boundary',
            'feature_DCP': 'Disjunct Class Percentage - class distribution complexity',
            'feature_TD_P': 'Tree Depth Pruned - decision tree complexity (pruned)',
            'feature_TD_U': 'Tree Depth Unpruned - decision tree complexity (unpruned)',
            'feature_CL': 'Class Likelihood - probability of correct classification',
            'feature_CLD': 'Class Likelihood Difference - margin of classification',
            'feature_MV': 'Minority Value - rarity in feature space',
            'feature_CB': 'Class Balance - local class distribution',
            'feature_N1': 'Borderline Points - fraction of different class neighbors',
            'feature_N2': 'Intra-Extra Ratio - within vs between class distances',
            'feature_LSC': 'Locality Sensitive Complexity - local complexity measure',
            'feature_LSR': 'Locality Sensitive Radius - local neighborhood size',
            'feature_Harmfulness': 'Harmfulness - negative impact on learning',
            'feature_Usefulness': 'Usefulness - positive contribution to learning',
            'feature_F1': 'Feature Overlap F1 - overlapping feature regions',
            'feature_F2': 'Feature Overlap F2 - non-overlapping feature regions',
            'feature_F3': 'Feature F3 - minority class feature characteristics',
            'feature_F4': 'Feature F4 - majority class feature characteristics',
            'relative_entropy': 'Relative Entropy - classifier disagreement measure',
            'pca_contribution': 'PCA Contribution - instance importance in principal components',
            #'pca_reconstruction_error': 'PCA Reconstruction Error - information loss in dimensionality reduction'
        }
        
        group = None
        for group_name, metrics in self.METRIC_GROUPS.items():
            if metric in metrics:
                group = group_name
                break
        
        return {
            'name': metric,
            'definition': metric_definitions.get(metric, 'Definition not available'),
            'group': group or 'unknown'
        }
    
    def get_summary_statistics(self, hardness_df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for calculated hardness scores."""
        return hardness_df.describe()


class CVAEHardnessIntegrator:
    """
    Utility class for integrating hardness scores with CVAE training.
    """
    
    def __init__(self, hardness_strategy: str = 'static', 
                 curriculum_epochs: Optional[Tuple[int, int, int]] = None):
        """
        Initialize the CVAE hardness integrator.
        
        Args:
            hardness_strategy: 'static', 'curriculum', or 'self_paced'
            curriculum_epochs: Tuple of (easy_epochs, mixed_epochs, hard_epochs) for curriculum learning
        """
        self.hardness_strategy = hardness_strategy
        self.curriculum_epochs = curriculum_epochs or (10, 10, 10)
        
    def get_sample_weights(self, hardness_scores: np.ndarray, 
                          epoch: int = 0, total_epochs: int = 100) -> np.ndarray:
        """
        Get sample weights based on hardness strategy.
        
        Args:
            hardness_scores: Array of hardness scores for each sample
            epoch: Current training epoch
            total_epochs: Total number of training epochs
            
        Returns:
            Array of weights for each sample
        """
        if self.hardness_strategy == 'static':
            # Static weighting: higher hardness = higher weight
            return hardness_scores
        
        elif self.hardness_strategy == 'curriculum':
            # Curriculum learning: easy -> mixed -> hard
            easy_epochs, mixed_epochs, hard_epochs = self.curriculum_epochs if self.curriculum_epochs else (int(total_epochs * 0.3), int(total_epochs * 0.3), int(total_epochs * 0.4))
            
            if epoch < easy_epochs:
                # Focus on easy samples (low hardness)
                return 1.0 - hardness_scores
            elif epoch < easy_epochs + mixed_epochs:
                # Uniform weighting
                return np.ones_like(hardness_scores)
            else:
                # Focus on hard samples (high hardness)
                return hardness_scores
        
        elif self.hardness_strategy == 'self_paced':
            # Self-paced: gradually include harder samples
            progress = epoch / total_epochs
            threshold = np.percentile(hardness_scores, progress * 100)
            weights = np.where(hardness_scores <= threshold, 1.0, 0.1)
            return weights
        
        else:
            raise ValueError(f"Unknown hardness strategy: {self.hardness_strategy}")
    
    def weighted_reconstruction_loss(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                                   weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted reconstruction loss for CVAE.
        
        Args:
            x_true: True input data
            x_pred: Reconstructed data
            weights: Sample weights based on hardness
            
        Returns:
            Weighted reconstruction loss
        """
        # Calculate per-sample reconstruction loss (MSE)
        recon_loss = torch.sum((x_true - x_pred) ** 2, dim=1)
        
        # Apply weights
        weighted_loss = recon_loss * weights
        
        return torch.mean(weighted_loss)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = fetch_datasets()['ecoli']
    X, y  = data.data, data.target
    # y values are in -1 and 1, convert to 0 and 1 for binary classification
    y = np.where(y == -1, 0, 1)  # Convert to binary labels
    # check the count of the classes
    print(f"Class distribution: {np.bincount(y)}")
    # X = np.random.randn(200, 10)
    # y = np.random.choice([0, 1], size=200, p=[0.8, 0.2])  # Imbalanced
    
    # Initialize hardness calculator
    calc = HardnessCalculator()
    
    # Calculate custom hardness metrics (since PyHard might not be available)
    metrics = ['feature_TD_P']  #['relative_entropy', 'pca_contribution']
    
    try:
        hardness_df = calc.calculate_hardness_scores(X, y, metrics)
        print("Hardness scores calculated successfully!")
        print(f"Shape: {hardness_df.shape}")
        print("\nSummary statistics:")
        print(calc.get_summary_statistics(hardness_df))
        
        # Test CVAE integration
        integrator = CVAEHardnessIntegrator(hardness_strategy='curriculum')
        
        # Example of getting weights for different epochs
        hardness_values = hardness_df['relative_entropy'].values
        
        for epoch in [5, 15, 25]:
            weights = integrator.get_sample_weights(hardness_values, epoch, 30)
            print(f"\nEpoch {epoch} - Weight statistics:")
            print(f"Mean: {np.mean(weights):.3f}, Std: {np.std(weights):.3f}")
            print(f"Min: {np.min(weights):.3f}, Max: {np.max(weights):.3f}")
        
    except Exception as e:
        print(f"Error: {e}")

