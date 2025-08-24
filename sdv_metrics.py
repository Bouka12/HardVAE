# import sdmetrics
from sdmetrics.reports.single_table  import QualityReport, DiagnosticReport
from sdmetrics.single_column import RangeCoverage, KSComplement, BoundaryAdherence, StatisticSimilarity
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
import numpy as np

#from config import Pima_metadata
#Q_report = QualityReport()
# Q_report.generate(real_data, syntetic_data, metadata)
# Q_report.get_score()
# Q_report.get_properties()
#D_report = DiagnosticReport()
# D_report.generate(real_data, synthetic_data, metadata)
# D_report.get_score()
# D_report.get_properties()


# RangeCoverage.compute(real_data=real_table['column_name'], synthetic_data=synthetic_table['column_name'])
# KSComplement.compute(real_data=real_table['column_name'], synthetic_data=synthetic_table['column_name'])
# from sdmetrics.column_pairs import CorrelationSimilarity
# CorrelationSimilarity.compute( real_data=real_table[['column_1', 'column_2']], synthetic_data=synthetic_table[['column_1', 'column_2']], coefficient='Pearson')
# ContingencySimilarity.compute(real_data=real_table[['column_1', 'column_2']], synthetic_data=synthetic_table[['column_1', 'column_2']])

# sdmetrics: similarity metrics
def avgRangeCoverage(real_data, synthetic_data):
    avgRCov = {}
    for col in real_data.columns:
        avgRCov[col] = RangeCoverage.compute(real_data[col], synthetic_data[col])
    return np.mean(list(avgRCov.values()))

def avgKSComplement(real_data, synthetic_data):
    avgKSC = {}
    for col in real_data.columns:
        avgKSC[col] = KSComplement.compute(real_data[col], synthetic_data[col])
    return np.mean(list(avgKSC.values()))

def avgBoundaryAdherence(real_data, synthetic_data):
    avgBAdher = {}
    for col in real_data.columns:
        avgBAdher[col] = BoundaryAdherence.compute(real_data[col], synthetic_data[col])
    return np.mean(list(avgBAdher.values()))
# added similarity metric for numerical data
def avgStatisticalSimilarity(real_data, synthetic_data):
    avgStatSim = {}
    for col in real_data.columns:
        avgStatSim[col] = StatisticSimilarity.compute(real_data[col], synthetic_data[col], statistic='mean')
    return np.mean(list(avgStatSim.values()))

def similarity_metrics(real_data, synthetic_data):
    avgRangeCoverage_ = avgRangeCoverage(real_data, synthetic_data)
    avgKSComplement_ = avgKSComplement(real_data, synthetic_data)
    avgBoundaryAdherence_ = avgBoundaryAdherence(real_data, synthetic_data)
    avgStatisticalSimilarity_ = avgStatisticalSimilarity(real_data, synthetic_data)
    return ['avgRangeCoverage', 'avgKSComplement', 'avgBoundaryAdherence', 'avgStatisticalSimilarity'], [avgRangeCoverage_, avgKSComplement_, avgBoundaryAdherence_, avgStatisticalSimilarity_]
    

from scipy.stats import ks_2samp

def evaluate_synthetic_data(X_real, y_real, X_synth, y_synth, verbose=True):
    """
    Comprehensive evaluation of synthetic data quality.
    
    Args:
        X_real: Real feature data
        y_real: Real labels
        X_synth: Synthetic feature data
        y_synth: Synthetic labels
        verbose: Whether to print results
    
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # 1. Statistical Similarity (Kolmogorov-Smirnov test)
    ks_scores = []
    for i in range(X_real.shape[1]):
        ks_stat, p_value = ks_2samp(X_real[:, i], X_synth[:, i])
        ks_scores.append(ks_stat)
    
    results['ks_mean'] = np.mean(ks_scores)
    results['ks_std'] = np.std(ks_scores)
    
    # 2. Correlation Preservation
    # correlation_real = np.corrcoef(X_real.T)
    # correlation_synth = np.corrcoef(X_synth.T)
    # correlation_diff = np.mean(np.abs(correlation_real - correlation_synth))
    # results['correlation_preservation'] = 1 - correlation_diff
    
    # # 3. Classification Utility (Train on synthetic, test on real)
    # clf = RandomForestClassifier(random_state=42, n_estimators=100)
    # clf.fit(X_synth, y_synth)
    # y_pred = clf.predict(X_real)
    
    # results['utility_accuracy'] = accuracy_score(y_real, y_pred)
    # results['utility_f1'] = f1_score(y_real, y_pred, average='weighted')
    
    # # 4. Privacy: Distance to nearest real sample
    # from sklearn.metrics.pairwise import euclidean_distances
    # distances = euclidean_distances(X_synth, X_real)
    # min_distances = np.min(distances, axis=1)
    # results['privacy_score'] = np.mean(min_distances)
    
    # # 5. Class Distribution Comparison
    # real_class_dist = np.bincount(y_real.astype(int)) / len(y_real)
    # synth_class_dist = np.bincount(y_synth.astype(int)) / len(y_synth)
    # results['class_distribution_diff'] = np.mean(np.abs(real_class_dist - synth_class_dist))
    
    if verbose:
        print("=== Synthetic Data Evaluation ===")
        print(f"KS Test (lower is better): {results['ks_mean']:.4f} ± {results['ks_std']:.4f}")
        # print(f"Correlation Preservation: {results['correlation_preservation']:.4f}")
        # print(f"Utility Accuracy: {results['utility_accuracy']:.4f}")
        # print(f"Utility F1-Score: {results['utility_f1']:.4f}")
        # print(f"Privacy Score: {results['privacy_score']:.4f}")
        # print(f"Class Distribution Difference: {results['class_distribution_diff']:.4f}")
    
    return results