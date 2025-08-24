# Multiview and Multi-objective SD evaluation framework:
| **View**                         | **Goal**                               | **Method**                                                          | **Reference**                                 |
| -------------------------------- | -------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------- |
| **1. Statistical Fidelity**      | Match feature-wise distributions       | Statistical meta-features, KS test                                  | Lorena et al. (2019), Goncalves et al. (2020) |
| **2. Topological Fidelity**      | Preserve shape & structure             | Persistent homology (ripser, bottleneck/wasserstein distances)      | Chazal & Michel (2016)                        |
| **3. Instance-Level Fidelity**   | Preserve instance difficulty           | Instance hardness (KDN, DCP, etc.)                                  | Smith et al. (2014)                           |
| **4. Complexity Fidelity**       | Preserve classification complexity     | Meta-feature complexity measures                                    | Lorena et al. (2019)                          |
| **5. **🆕** Utility Evaluation** | Support downstream predictive modeling | Train-on-synth → test-on-real, CV on real and synth (f1, AUC, etc.) | Jordon et al. (2022), Yale et al. (2020)      |


# Synthetic Data Evaluation Module

A comprehensive Python module for evaluating the quality of synthetic minority data used in class imbalance scenarios. This module implements multiple evaluation aspects including statistical similarity, complexity metrics, utility assessment, and more.

## Features

### 🔍 **Comprehensive Evaluation Framework**
- **Statistical Evaluation**: Distribution similarity, correlation analysis, range coverage
- **Complexity Analysis**: Data complexity patterns using problexity package
- **Instance Hardness**: Nearest neighbor distance analysis
- **Topological Analysis**: Persistent homology (when libraries available)

### 📊 **Advanced Visualization**
- Heatmaps, and comparison matrices
- Individual detailed plots for each evaluation component

### 📈 **Similarity Metrics**
- Multiple similarity calculation methods
- Relative difference, correlation-based, distribution-based metrics
- Advanced data quality metrics (coverage, mutual information)

### 📋 **Comprehensive Reporting**
- CSV exports for detailed analysis
- Summary assessments with recommendations

## Installation

### Required Dependencies
```bash
pip install pymfe problexity scikit-learn scipy pandas matplotlib seaborn plotly
```

### Optional Dependencies (for enhanced features)
```bash
pip install pyhard ripser persim  # For instance hardness and topological analysis
```

## Quick Start

```python
from synthetic_data_evaluator import SyntheticDataEvaluator

# Initialize evaluator
evaluator = SyntheticDataEvaluator(random_state=42)

# Run comprehensive evaluation
results = evaluator.evaluate_all(
    X_real, y_real,           # Original minority data
    X_synth, y_synth,         # Synthetic minority data
    save_path="./results",    # Output directory
    dataset_name="my_dataset" # Dataset identifier
)

```

## Module Structure

### Core Components

1. **`synthetic_data_evaluator.py`** - Main evaluation class
   - `SyntheticDataEvaluator`: Primary evaluation interface
   - Individual evaluation methods for each aspect
   - Legacy function compatibility

2. **`example_usage.py`** - Usage examples and testing

## Evaluation Aspects

### 1. Statistical Evaluation
- **Meta-features**: 21 statistical measures including correlation, covariance, skewness, kurtosis
- **Distribution Tests**: Kolmogorov-Smirnov tests for each feature
- **Similarity Scoring**: Multiple similarity calculation methods

### 3. Complexity Analysis
- **Problexity Integration**: 20+ complexity metrics
- **Categories**: Feature-based, linearity, neighborhood, dimensionality measures
- **Visualization**: Radar plots comparing complexity patterns

### 4. Utility Assessment
- **Model Performance**: RandomForest, LogisticRegression, SVM
- **Cross-validation**: Performance on real vs synthetic data
- **Utility Score**: Train-on-synthetic, test-on-real evaluation

### 5. Instance Hardness
- **Nearest Neighbor Analysis**: Distance-based hardness estimation
- **Distribution Comparison**: Statistical tests on hardness patterns

### 6. Topological Analysis (Optional)
- **Persistent Homology**: Shape and structure analysis
- **Persistence Diagrams**: Topological feature comparison

## Usage Examples

### Basic Evaluation
```python
# Simple evaluation with default settings
evaluator = SyntheticDataEvaluator()
results = evaluator.evaluate_all(X_real, y_real, X_synth, y_synth)
```

### Individual Component Evaluation
```python
# Evaluate specific aspects
stat_results = evaluator.statistical_evaluation(X_real, y_real, X_synth, y_synth)
complexity_results = evaluator.complexity_evaluation(X_real, y_real, X_synth, y_synth)
utility_results = evaluator.utility_evaluation(X_real, y_real, X_synth, y_synth)
```



## Output Files

The evaluation generates several output files:

- **`evaluation_summary_{dataset}.txt`**: Human-readable summary report
- **`detailed_results_{dataset}.csv`**: Detailed metrics in CSV format
- **`dashboard_{dataset}.png`**: Comprehensive visualization dashboard
- **`complexity_comparison_{dataset}.png`**: Complexity analysis plots

## Interpretation Guide


### Component Interpretation
- **Statistical**: How well distributions match
- **Complexity**: Data complexity pattern preservation
- **Utility**: Usefulness for model training
- **Hardness**: Instance difficulty pattern similarity



### Batch Processing
```python
# Process multiple datasets
datasets = ['dataset1', 'dataset2', 'dataset3']
all_results = {}

for dataset in datasets:
    X_real, y_real, X_synth, y_synth = load_data(dataset)
    results = evaluator.evaluate_all(X_real, y_real, X_synth, y_synth)
    all_results[dataset] = results
```



## License

This module is provided as-is for research and educational purposes.

## Citation

If you use this evaluation module in your research, please cite:
```
will be provided later
```

