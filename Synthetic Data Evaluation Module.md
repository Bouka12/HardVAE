# Multiview and Multi-objective SD evaluation framework:
| **View**                         | **Goal**                               | **Method**                                                          | **Reference**                                 |
| -------------------------------- | -------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------- |
| **1. Statistical Fidelity**      | Match feature-wise distributions       | Statistical meta-features, KS test                                  | Lorena et al. (2019), Goncalves et al. (2020) |
| **2. Topological Fidelity**      | Preserve shape & structure             | Persistent homology (ripser, bottleneck/wasserstein distances)      | Chazal & Michel (2016)                        |
| **4. Instance-Level Fidelity**   | Preserve instance difficulty           | Instance hardness (KDN, DCP, etc.)                                  | Smith et al. (2014)                           |
| **5. Complexity Fidelity**       | Preserve classification complexity     | Meta-feature complexity measures                                    | Lorena et al. (2019)                          |
| **6. **🆕** Utility Evaluation** | Support downstream predictive modeling | Train-on-synth → test-on-real, CV on real and synth (f1, AUC, etc.) | Jordon et al. (2022), Yale et al. (2020)      |


# Synthetic Data Evaluation Module

A comprehensive Python module for evaluating the quality of synthetic minority data used in class imbalance scenarios. This module implements multiple evaluation aspects including statistical similarity, clustering analysis, complexity metrics, utility assessment, and more.

## Features

### 🔍 **Comprehensive Evaluation Framework**
- **Statistical Evaluation**: Distribution similarity, correlation analysis, range coverage
- **Complexity Analysis**: Data complexity patterns using problexity package
- **Instance Hardness**: Nearest neighbor distance analysis
- **Topological Analysis**: Persistent homology (when libraries available)

### 📊 **Advanced Visualization**
- Comprehensive dashboards with multiple evaluation aspects
- Interactive Plotly dashboards
- Radar charts, heatmaps, and comparison matrices
- Individual detailed plots for each evaluation component

### 📈 **Similarity Metrics**
- Multiple similarity calculation methods
- Relative difference, correlation-based, distribution-based metrics
- Advanced data quality metrics (coverage, mutual information)

### 📋 **Comprehensive Reporting**
- HTML and Markdown report generation
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

# Access results
print(f"Overall Quality Score: {results['summary']['overall_quality_score']:.3f}")
print(f"Assessment: {results['summary']['assessment']}")
```

## Module Structure

### Core Components

1. **`synthetic_data_evaluator.py`** - Main evaluation class
   - `SyntheticDataEvaluator`: Primary evaluation interface
   - Individual evaluation methods for each aspect
   - Legacy function compatibility

2. **`evaluation_utils.py`** - Advanced utilities
   - `SimilarityCalculator`: Multiple similarity metrics
   - `DataQualityMetrics`: Additional quality assessments
   - `EvaluationReport`: Report generation utilities

3. **`evaluation_visualizer.py`** - Visualization capabilities
   - `EvaluationVisualizer`: Comprehensive plotting
   - Dashboard creation (static and interactive)
   - Comparison matrices and detailed plots

4. **`example_usage.py`** - Usage examples and testing

## Evaluation Aspects

### 1. Statistical Evaluation
- **Meta-features**: 21 statistical measures including correlation, covariance, skewness, kurtosis
- **Distribution Tests**: Kolmogorov-Smirnov tests for each feature
- **Similarity Scoring**: Multiple similarity calculation methods

### 2. Clustering Analysis
- **Cluster Quality**: Silhouette score, Calinski-Harabasz index
- **Structure Comparison**: Optimal cluster number analysis
- **Meta-features**: 8 clustering-based measures

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

### Advanced Visualization
```python
from evaluation_visualizer import EvaluationVisualizer

visualizer = EvaluationVisualizer()

# Create comprehensive dashboard
visualizer.create_comprehensive_dashboard(
    results, 
    save_path="dashboard.png",
    dataset_name="MyDataset"
)

# Create interactive dashboard
visualizer.create_interactive_dashboard(
    results,
    save_path="interactive_dashboard.html"
)
```

### Method Comparison
```python
# Compare multiple synthetic data generation methods
methods_results = {
    'SMOTE': evaluator1.evaluate_all(...),
    'ADASYN': evaluator2.evaluate_all(...),
    'GAN': evaluator3.evaluate_all(...)
}

visualizer.create_comparison_matrix(methods_results, "comparison.png")
```

## Output Files

The evaluation generates several output files:

- **`evaluation_summary_{dataset}.txt`**: Human-readable summary report
- **`detailed_results_{dataset}.csv`**: Detailed metrics in CSV format
- **`dashboard_{dataset}.png`**: Comprehensive visualization dashboard
- **`complexity_comparison_{dataset}.png`**: Complexity analysis plots
- **`method_comparison.csv`**: Multi-method comparison (when applicable)

## Interpretation Guide

### Quality Scores
- **0.8-1.0**: Excellent synthetic data quality
- **0.7-0.8**: Good synthetic data quality
- **0.6-0.7**: Moderate synthetic data quality
- **0.0-0.6**: Poor synthetic data quality

### Component Interpretation
- **Statistical**: How well distributions match
- **Clustering**: Similarity in cluster structure
- **Complexity**: Data complexity pattern preservation
- **Utility**: Usefulness for model training
- **Hardness**: Instance difficulty pattern similarity

## Advanced Features

### Custom Similarity Metrics
```python
from evaluation_utils import SimilarityCalculator

calculator = SimilarityCalculator()
similarity = calculator.calculate_similarity(
    real_values, synth_values, 
    method='correlation_based'
)
```

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

## Performance Considerations

- **Large Datasets**: Automatic cluster-based sampling for datasets >10,000 samples
- **Memory Management**: Efficient handling of high-dimensional data
- **Computation Time**: Parallel processing where possible

## Troubleshooting

### Common Issues

1. **Single Class Error**: Ensure minority data contains multiple classes or samples
2. **Memory Issues**: Reduce dataset size or increase sampling ratio
3. **Missing Dependencies**: Install optional packages for full functionality

### Error Handling
The module includes comprehensive error handling with graceful degradation when components fail.

## Contributing

The module is designed to be extensible. To add new evaluation metrics:

1. Implement the metric in the appropriate evaluation method
2. Add visualization support in `evaluation_visualizer.py`
3. Update the summary generation logic

## License

This module is provided as-is for research and educational purposes.

## Citation

If you use this evaluation module in your research, please cite:
```
Synthetic Data Evaluation Module for Class Imbalance
Enhanced comprehensive evaluation framework
2024
```

