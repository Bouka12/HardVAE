""" 
This module contains the related class(es) used to evalaute the quality of the synthetic minority data. We will base our classes on a couple of packages as resources.
However, we will also try to make a package that is specific to evaluating synthetic data used for purposes of handling class imbalance (balancing originally imbalanced data)

It has the following components:
    ================================== 50% DONE =======================================
    1 -  Statistical evaluation: eg. `sdmetrics` and the meta features in `pymfe`
        - Correlation similarity >
        - KS Complement > DONE
        - Range Coverage >
        - Statistic Similarity (statistical measures of central, shape, and others) >>>  DONE HERE
    
    2 - Clustering-based evaluation: eg.  `pymfe` Clustering metrics
        - CHECK `pymfe`
        - think of how clustering could be used to compare the synthetic vs real: Number of clusters, inter-variance/intra-variance of clusters, inter-variance.intra-variance of latent variables (PCA or others)
        - WORK HERE ~.~ !!!!
        - One option:
            - check if the minority: synth >= real
                - do for k times the following:
                - sample (# real size) from synthetic data as much as real minority concatenate it with the majority class data in the training
                - calculate the Clustering features and store them {clus_1, clus_2, ..., clus_k} using pymfe
                - Average across the k dict/df of complexity scores
            - if real >= synth:
                - repeat the procedures of the previous loop but with:
                - sample (# synthetic size) from synthetic and sample from the majority class data to maintain the same class_imbalance_ratio to get a imbalanced dataset like the original
                - calculate the Clustering features and store them {clus_1, clus_2, ..., clus_k} using pymfe
                - Average across the k dict/df of complexity scores
    
    3 - Shape/Structure/homology/Topological/Geometric evaluation: topological data analysis (TDA)
        - TO CREATE FROM SCRATCH THE PIPELINE AND TOOLS OF TDA TO USE
        - WORK HERE!!!
    4 - Model-based (UTILITY): eg. Model-based/Landmarking from pymfe
        - CHECK Landmarking FROM `pymfe`
        - sample from majority and minority (real vs synthetic) to build models: clustering vs classification NEW IN IMBALANCE CONTEXT WE NEED TO CREATE FROM SCRATCH IN OUR PACKAGE
    
    =====================================Designed=======================================
    5 - Instance Hardness Analysis:
        - CHECK how to design the similarity measurement.
        - refer to `pyhard`
        - One option:
            - check if the minority: synth >= real
                - do for k times the following:
                - sample (# real size) from synthetic data as much as real minority concatenate it with the majority class data in the training
                - calculate the Hardness scores and store them {comp_1, comp_2, ..., comp_k} using problexity
                - Average across the k dict/df of complexity scores
            - if real >= synth:
                - repeat the procedures of the previous loop but with:
                - sample (# synthetic size) from synthetic and sample from the majority class data to maintain the same class_imbalance_ratio to get a imbalanced dataset like the original
                - calculate the Hardness scores and store them {comp_1, comp_2, ..., comp_k} using problexity
                - Average across the k dict/df of complexity scores
        
    6 - Complexity-based Analysis: from `problexity` and `pymfe`
        - CHECK HOW to do it minority (real vs. synthetic) and majority
        - One option:
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


Proposed similairty measurement:
    - The question is to design sort of metric to calculate or score the similarity between values of a metric in a group calculated on synthetic and the same calcualted on real.
    - if we design this metric to compare individually each metric we get the similarity in that metric that will easier the statistical test of it across datasets from one method to another.

        
            


"""
# Meta-features based evalaution: From the pymfe meta features calculation package, we are going to use: 
    ## 1 - Statistical MF
    ## 2 - Clustering MF
    ## 3 - Model-based MF (a ver)
    ## 4 - Landmarking - Subsampling-landmarking

#

# problexity: complexity metrics
# Wilcoxon test: test the significance of the difference between scores on `ft_orig` and `ft_synth`
# MiniBatchKMeans: clustering trick implementation
# px: problexity


from pymfe.mfe import MFE
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from Problexity import ComplexityCalculator 
from scipy.stats import wilcoxon
import problexity as problx
import problexity as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from collections import Counter


#print(MFE.valid_metafeatures(groups=['statistical']))
#print(MFE.metafeature_description())


# In our evaluation >> we are more interested in statistical metrics
# first, we compute the metrics for the original minority data 
# >> Statitical : mfe = MFE(features=['t_mean', 'var', 'sparsity', 'skeweness','sd', 'range', 'nr_outliers', 'nr_norm', 'nr_cor_attr', 'min', 'median', 'mean', 'max', 'mad', 'kurtosis', 'iq_range','h_mean', 'g_mean', 'eigenvalues','cov', 'cor' ])
# >> Complexity : mfe = MFE(groups=['complexity'],features=['t2', 't3', 't4'])

# >> Calculation example:
#mfe = MFE(groups=['complexity'],features=['t2', 't3', 't4'])
#mfe = MFE(features=['t_mean', 'var', 'sparsity', 'skeweness','sd', 'range', 'nr_outliers', 'nr_norm', 'nr_cor_attr', 'min', 'median', 'mean', 'max',                  'mad', 'kurtosis', 'iq_range','h_mean', 'g_mean', 'eigenvalues','cov', 'cor' ])
#mfe.fit(X)
#ft_orig = mfe.extract()
#print(ft_orig)

# and the metrics for the synthetic minoirty data
#mfe.fit(Xs)
#ft_synth = mfe.extract()

# second, we calcualte the difference respectively between the metrics
# third, we test if the difference is significantly different than 0 for each dataset
# >> Perform a Wilcoxon test to test the significance of the difference between scores on `ft_orig` and `ft_synth`

def statistical_features(real_data, real_y, augmented_data, augmented_y, rand_seed):
    """
    Statistical meta features:

    - selected statistical features: ['cor', 'cov', 'eigenvalues', 'gravity', 'iq_range', 'kurtosis', 'mad', 'max', 'mean', 'median', 'min', 'nr_cor_attr', 'nr_norm', 'nr_outliers', 'range', 'sd', 'sd_ratio', 'skewness', 'sparsity', 't_mean', 'var']
    
    - All Valid statistical features: ['can_cor', 'cor', 'cov', 'eigenvalues', 'g_mean', 'gravity', 'h_mean', 'iq_range', 'kurtosis', 'lh_trace', 'mad', 'max', 'mean', 'median', 'min', 'nr_cor_attr', 'nr_disc', 'nr_norm', 'nr_outliers', 'p_trace', 'range', 'roy_root', 'sd', 'sd_ratio', 'skewness', 'sparsity', 't_mean', 'var', 'w_lambda']

    Inputs:
    - real_data: the real data
    - real_y: the real labels
    - augmented_data: the augmented data
    - augmented_y: the augmented labels
    - rand_seed: the random state for reproducibility

    Output:
    - statistical_features: the statistical features names
    - ft_diff[1]: the difference between the statistical features of the real and augmented data
    - ft_real[1]: the statistical features of the real data
    - ft_synthetic[1]: the statistical features of the augmented data

    """
    print(f"Running statistical_features function...")  
    
    # Get Sampled data if dataset is >12000
    #if X_tr.shape[0]>=12000:
    real_data, real_y = ClusterSampling(real_data, real_y, random_state=rand_seed)
    augmented_data, augmented_y = ClusterSampling(augmented_data, augmented_y, random_state=rand_seed)
    mfe = MFE(groups='statistical', features =[ 'cor', 'cov', 'eigenvalues', 'gravity', 'iq_range', 'kurtosis', 'mad', 'max', 'mean', 'median', 'min', 'nr_cor_attr', 'nr_norm', 'nr_outliers', 'range', 'sd', 'sd_ratio', 'skewness', 'sparsity', 't_mean', 'var'], random_state=rand_seed)
    mfe.fit(real_data, real_y)
    ft_real = mfe.extract()
    mfe.fit(augmented_data, augmented_y)
    ft_synthetic = mfe.extract()
    ft_diff = tuple([ft_real[0], list(np.array(ft_real[1])- np.array(ft_synthetic[1]))])
    statistical_features = ft_real[0]
    return statistical_features, ft_diff[1], ft_real[1], ft_synthetic[1]


def complexity_multi(real_data, real_y, augmented_data, augmented_y, rand_seed):
    
    """
    Complexity metrics
    -  features = ['c1', 'c2', 'cls_coef', 'hubs', 'density','s4',..., 'lsc','n1', 'n2', 'n3', 'n4','t1', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 't2', 't3', 't4']
    - ['density', 'hubs', 'lsc','cls_coef', c2] -> (generating memmory issues)>> clustering trick implementation

    >> Network Measures -> *clustering*
    - `cls_coef`: The clusterign coefficient metric
    - `hubs`: The hub metric
    - `density`:  Average density of the network. 

    >> Class Imbalance measures
    - `c1`: the Entropy of Class Proportions
    - `c2`: the Imbalance Ratio

    >> Neighborhood measures -> *clustering*
    - `lsc`: The Local set average cardinality (LSC) metric
    - `n1`: the fraction f points on the class boundary
    - `n2`: The ratio of average intra/inter class NN distance
    - `n3`: Leave-one-out error rate f the 1NN classifier
    - `n4`: Nonlinearity of a 1-NN classifier
    - `t1`: Fraction of maximum covering spheres

    >> Feature-based Measures (aka Feature overlapping measures)
    - `f1`: Maximum Fisher's discriminant ratio.
    - `f1v`: Directional-vector maximum Fisher's discriminant.
    - `f2`: Volume of the overlapping region.
    - `f3`: Compute feature maximum individual efficiency.
    - `f4`: Compute the collective feature effciency.

    >> Linearity measures (aka Linear Separability measures)
    - `l1`: Minimized sum of error distance of a linear classifier.
    - `l2`: Compute the OVO subsets eror rate of linear classifer.
    - `l3`: Non linearity of a linear classifier.

    >> Dimensionality measures
    - `t2`: Compute the average number of features per dimension.
    - `t3`: Compute the average number of PCA dimensions per points.
    - `t4`: Compute the ratio of the PCA dimension to the original dimension.

    >> Other complexity measures for imbalanced datasets -> *clustering*
    - `CM`: Complexity measure for imbalanced datasets
    - `wCM`: Weighted complexity metric
    - `dwCM`: Dual weighted complexity metric
    - `BI^3`: Bayes imbalanced impact index
    

    Output: 
    """
    print(f"Running complexity_features function...")
    
    # Get Sampled data if dataset is >12000
    #sampling_rat = 0.05 if real_data.shape[0]>=100000 else 0.1 if real_data.shape[0]>=12000 else 0 # sample 5% of the data if the dataset is >100000, 10% if >12000, otherwise 0
    #if X_tr.shape[0]>=12000:
    real_data, real_y = ClusterSampling(real_data, real_y, random_state=rand_seed)
    augmented_data, augmented_y = ClusterSampling(augmented_data, augmented_y, random_state=rand_seed)
    mfe = MFE(groups=['complexity'], features=['c1', 'density', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 't2', 't3', 't4'],summary=["mean"],suppress_warnings=True ,random_state=rand_seed,)
    mfe.fit(real_data, real_y)
    ft_real = mfe.extract()
    #print(ft_real)
    mfe.fit(augmented_data, augmented_y)
    ft_synthetic = mfe.extract()
    #ft_diff = tuple([ft_real[0], list(np.array(ft_real[1])- np.array(ft_synthetic[1]))])
    complexity_features = ft_real[0]
    ft_real_score = np.mean(ft_real[1])
    ft_synth_score = np.mean(ft_synthetic[1])
    return complexity_features, ft_real_score, ft_synth_score, ft_real[1], ft_synthetic[1]


# Wilcoxon test: test if the difference between the metrics calculated on the real data and augmneted data are statistically significant
# >> if the p-value is less than 0.05, then the difference is statistically significant
# >> if the p-value is greater than 0.05, then the difference is not statistically significant
# This step is not necessary for the evaluation of the model, but it is useful to know if the difference is significant or not
# It is optional to include this step in the evaluation process >> we didn't use it in our evaluation process




# PROBLEXITY: check this package for complexity metrics calculation!
# Implement clustering on train data and augmented train data with k clusters ->> sample proportionally from each cluster
# ->> obtaining representative sample of train data and augmented train data ->> calculate the complexity metrics



#from sklearn.datasets import load_breast_cancer
#X,y = load_breast_cancer(return_X_y=True)

# Initialize ComplexityCalculator with default parametrization
#cc = ComplexityCalculator()
# Fit model with data
#cc.fit(X, y)

# Get the cc measures and metrics
#print(f"cc measures: {cc.complexity}\n")

# Complexity metrics names
#print(f"cc metrics of length {len(cc._metrics())}: {cc._metrics()}\n")

#labels = cc._metrics()


# Create Radar plot with the values of the complexity metrics

#categories = labels
#values = cc.complexity

# Convert categories to angles
#N = len(categories)
#angles = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Define angles
#m_neighborhood = ["n1", "n2", "n3", "n4"]
#m_network = ["hubs", "density", "discCoef", "lsc"]
#m_class_imbalance = ["c1", "c2"]
#m_feature_based = ["f1", "f1v", "f2", "f3", "f4"]
#m_linearity = ["l1", "l2", "l3"]
#m_dimensionality = ["t1", "t2", "t3", "t4"]
# Define bar colors based on values
#colors = ['red' if c in m_neighborhood else 'orange' if c in m_network else 'yellow' if c in m_class_imbalance else 'green' if c in m_feature_based else 'blue' if c in m_linearity else 'purple'  for c in categories]

# Create polar plot
#fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Plot bars
#bars = ax.bar(angles, values, width=2*np.pi/N, color=colors, alpha=0.8, linewidth=0)

# Set labels at angles
#ax.set_xticks(angles)
#ax.set_xticklabels(categories, fontsize=10)

# Adjust radial limits
#ax.set_ylim(0, 1)

# Remove y-ticks and labels for cleaner look
#ax.set_yticklabels([])
#ax.set_yticks([])

# Title
#plt.title("Radial Bar Chart", fontsize=14,)

# Show plot
#plt.show()








def px_complexity_multi(X_tr, y_tr, X_aug, y_aug, path, dataset_name, random_state=None):
    
    """
    Calculate the complexity metrics for the train data and augmented train data using the `Problexity` package

    Inputs:
    - X_tr: the train data
    - y_tr: the train labels
    - X_aug: the augmented train data
    - y_aug: the augmented train labels
    - path (str): the path to save the complexity plots
    - dataset_name (str): the name of the dataset
    - random_state (int): the random state for reproducibility used for clustering and metrics calculation

    Output:
    - cc_tr._metrics(): the complexity metrics names for the train data
    - cc_tr.complexity: the complexity values for the train data
    - cc_tr.score(): the complexity score for the train data
    - cc_aug.complexity: the complexity values for the augmented train data
    - cc_aug.score(): the complexity score for the augmented train data
    - cc_aug_wc._metrics(): the complexity metrics names for the augmented train data without *class_imbalance* metrics
    - cc_tr_wc.complexity: the complexity values for the train data without *class_imbalance* metrics
    - cc_tr_wc.score(): the complexity score for the train data without *class_imbalance* metrics
    - cc_aug_wc.complexity: the complexity values for the augmented train without *class_imbalance* metrics data
    - cc_aug_wc.score(): the complexity score for the augmented train data without *class_imbalance* metrics

    **Problexity package**
    Complexity metrics
    -  features = ['c1', 'c2', 'cls_coef', 'hubs', 'density','s4',..., 'lsc','n1', 'n2', 'n3', 'n4','t1', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 't2', 't3', 't4']
    - ['density', 'hubs', 'lsc','cls_coef', c2] -> (generating memmory issues)>> clustering trick implementation

    >> Network Measures -> *clustering*
    - `cls_coef`: The clusterign coefficient metric
    - `hubs`: The hub metric
    - `density`:  Average density of the network. 

    >> Class Imbalance measures
    - `c1`: the Entropy of Class Proportions
    - `c2`: the Imbalance Ratio

    >> Neighborhood measures -> *clustering*
    - `lsc`: The Local set average cardinality (LSC) metric
    - `n1`: the fraction f points on the class boundary
    - `n2`: The ratio of average intra/inter class NN distance
    - `n3`: Leave-one-out error rate f the 1NN classifier
    - `n4`: Nonlinearity of a 1-NN classifier
    - `t1`: Fraction of maximum covering spheres

    >> Feature-based Measures (aka Feature overlapping measures)
    - `f1`: Maximum Fisher's discriminant ratio.
    - `f1v`: Directional-vector maximum Fisher's discriminant.
    - `f2`: Volume of the overlapping region.
    - `f3`: Compute feature maximum individual efficiency.
    - `f4`: Compute the collective feature effciency.

    >> Linearity measures (aka Linear Separability measures)
    - `l1`: Minimized sum of error distance of a linear classifier.
    - `l2`: Compute the OVO subsets eror rate of linear classifer.
    - `l3`: Non linearity of a linear classifier.

    >> Dimensionality measures
    - `t2`: Compute the average number of features per dimension.
    - `t3`: Compute the average number of PCA dimensions per points.
    - `t4`: Compute the ratio of the PCA dimension to the original dimension.

    Output: 
    - `cc.complexity`: list of all the complexity values (e.g. [0.22, ...])
    - `cc._metrics(): list of the complexity metrics names (e.g. ['f1', ...])
    - `cc.score()`: value in range [0,1] returns the complexity score, the arithmetic meanof all measures (e.g. 0.23)
    """
    print(f"Running px_complexity_multi function...")
    ## Print the unique values of the target arrays
    #print(f"y_tr unique values: {np.unique(y_tr)}")
    #print(f"y_aug unique values: {np.unique(y_aug)}")



    #print(f"y_tr unique values: {np.unique(y_tr)}")
    #print(f"y_aug unique values: {np.unique(y_aug)}")
    ## Print shapes of input arrays
    #print(f"X_tr shape: {X_tr.shape}, y_tr shape: {y_tr.shape}")
    #print(f"X_aug shape: {X_aug.shape}, y_aug shape: {y_aug.shape}")

    # Get Sampled data if dataset is >12000
    #sampling_rat = 0.05 if X_tr.shape[0]>=100000 else 0.1 if X_tr.shape[0]>=12000 else 0 # sample 5% of the data if the dataset is >100000, 10% if >12000, otherwise 0
    #if X_tr.shape[0]>=12000:
    X_tr, y_tr = ClusterSampling(X_tr, y_tr, random_state=random_state)
    X_aug, y_aug = ClusterSampling(X_aug, y_aug, random_state=random_state)


    # Initialize ComplexityCalculator with default parametrization
    cc_tr = px.ComplexityCalculator().fit(X_tr, y_tr)
    cc_aug = px.ComplexityCalculator().fit(X_aug, y_aug)
    # metrics_wc = ['f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'n1', 'n2', 'n3', 'n4', 't1', 'lsc', 'density', 'clsCoef', 'hubs', 't2', 't3', 't4']
    # cc_aug_wc = ComplexityCalculator().fit(X_aug, y_aug)
    # cc_tr_wc = ComplexityCalculator().fit(X_tr, y_tr)    
    


    # Get the cc plots and save them in path/{cc_tr}_{dataset_name)_complexity.png
    fig = plt.figure(figsize=(10, 10))
    cc_tr.plot(fig, (1, 1, 1))
    fig_path = os.path.join(path, f"real_complexity_{dataset_name}_{random_state}.png")
    plt.savefig(fig_path, dpi=100)

    cc_aug.plot(fig, (1, 1, 1))
    fig_path = os.path.join(path, f"synth_complexity_{dataset_name}_{random_state}.png")
    plt.savefig(fig_path, dpi=100)

    # cc_aug_wc.plot(fig, (1, 1, 1))
    # plt.savefig(f"{path}_augmented_wc_complexity.png")

    # cc_tr_wc.plot(fig, (1, 1, 1))
    # plt.savefig(f"{path}_train_wc_complexity.png")

    return cc_tr._metrics(), cc_tr.score(),  cc_aug.score(),  cc_tr.complexity, cc_aug.complexity #, cc_aug_wc._metrics(), cc_tr_wc.complexity, cc_tr_wc.score(), cc_aug_wc.complexity, cc_aug_wc.score()



def px_complexity_one_class(X_tr, y_tr, X_aug, y_aug, path, dataset_name, random_state=None):
    
    """
    Calculate the complexity metrics for the train data and augmented train data using the `Problexity` package

    Inputs:
    - X_tr: the train data
    - y_tr: the train labels
    - X_aug: the augmented train data
    - y_aug: the augmented train labels
    - path (str): the path to save the complexity plots
    - dataset_name (str): the name of the dataset
    - random_state (int): the random state for reproducibility used for clustering and metrics calculation

    Output:
    - cc_tr._metrics(): the complexity metrics names for the train data
    - cc_tr.complexity: the complexity values for the train data
    - cc_tr.score(): the complexity score for the train data
    - cc_aug.complexity: the complexity values for the augmented train data
    - cc_aug.score(): the complexity score for the augmented train data
    - cc_aug_wc._metrics(): the complexity metrics names for the augmented train data without *class_imbalance* metrics
    - cc_tr_wc.complexity: the complexity values for the train data without *class_imbalance* metrics
    - cc_tr_wc.score(): the complexity score for the train data without *class_imbalance* metrics
    - cc_aug_wc.complexity: the complexity values for the augmented train without *class_imbalance* metrics data
    - cc_aug_wc.score(): the complexity score for the augmented train data without *class_imbalance* metrics

    **Problexity package**
    Complexity metrics
    -  features = ['c1', 'c2', 'cls_coef', 'hubs', 'density','s4',..., 'lsc','n1', 'n2', 'n3', 'n4','t1', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 't2', 't3', 't4']
    - ['density', 'hubs', 'lsc','cls_coef', c2] -> (generating memmory issues)>> clustering trick implementation

    >> Network Measures -> *clustering*
    - `cls_coef`: The clusterign coefficient metric
    - `hubs`: The hub metric
    - `density`:  Average density of the network. 

    >> Class Imbalance measures
    - `c1`: the Entropy of Class Proportions
    - `c2`: the Imbalance Ratio

    >> Neighborhood measures -> *clustering*
    - `lsc`: The Local set average cardinality (LSC) metric
    - `n1`: the fraction f points on the class boundary
    - `n2`: The ratio of average intra/inter class NN distance
    - `n3`: Leave-one-out error rate f the 1NN classifier
    - `n4`: Nonlinearity of a 1-NN classifier
    - `t1`: Fraction of maximum covering spheres

    >> Feature-based Measures (aka Feature overlapping measures)
    - `f1`: Maximum Fisher's discriminant ratio.
    - `f1v`: Directional-vector maximum Fisher's discriminant.
    - `f2`: Volume of the overlapping region.
    - `f3`: Compute feature maximum individual efficiency.
    - `f4`: Compute the collective feature effciency.

    >> Linearity measures (aka Linear Separability measures)
    - `l1`: Minimized sum of error distance of a linear classifier.
    - `l2`: Compute the OVO subsets eror rate of linear classifer.
    - `l3`: Non linearity of a linear classifier.

    >> Dimensionality measures
    - `t2`: Compute the average number of features per dimension.
    - `t3`: Compute the average number of PCA dimensions per points.
    - `t4`: Compute the ratio of the PCA dimension to the original dimension.

    Output: 
    - `cc.complexity`: list of all the complexity values (e.g. [0.22, ...])
    - `cc._metrics(): list of the complexity metrics names (e.g. ['f1', ...])
    - `cc.score()`: value in range [0,1] returns the complexity score, the arithmetic meanof all measures (e.g. 0.23)
    """
    print(f"Running px_complexity_one function...")
    ## Print the unique values of the target arrays
    #print(f"y_tr unique values: {np.unique(y_tr)}")
    #print(f"y_aug unique values: {np.unique(y_aug)}")
    
    # Ensure all values are integers and non-negative
        # If y is a PyTorch tensor, convert to NumPy array



    #print(f"y_tr unique values: {np.unique(y_tr)}")
    #print(f"y_aug unique values: {np.unique(y_aug)}")
    ## Print shapes of input arrays
    #print(f"X_tr shape: {X_tr.shape}, y_tr shape: {y_tr.shape}")
    #print(f"X_aug shape: {X_aug.shape}, y_aug shape: {y_aug.shape}")

    # Get Sampled data if dataset is >12000
    #sampling_rat = 0.05 if X_tr.shape[0]>=100000 else 0.1 if X_tr.shape[0]>=12000 else 0 # sample 5% of the data if the dataset is >100000, 10% if >12000, otherwise 0
    #if X_tr.shape[0]>=12000:
    X_tr, y_tr = ClusterSampling(X_tr, y_tr, k=2, random_state=random_state)
    X_aug, y_aug = ClusterSampling(X_aug, y_aug,k=2, random_state=random_state)


    # Initialize ComplexityCalculator with default parametrization
    cc_tr = px.ComplexityCalculator(metrics=['t2', 't3', 't4']).fit(X_tr, y_tr)
    cc_aug = px.ComplexityCalculator(metrics=['t2', 't3', 't4']).fit(X_aug, y_aug)
    # metrics_wc = ['f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'n1', 'n2', 'n3', 'n4', 't1', 'lsc', 'density', 'clsCoef', 'hubs', 't2', 't3', 't4']
    # cc_aug_wc = ComplexityCalculator().fit(X_aug, y_aug)
    # cc_tr_wc = ComplexityCalculator().fit(X_tr, y_tr)    
    


    # Get the cc plots and save them in path/{cc_tr}_{dataset_name)_complexity.png
    fig = plt.figure(figsize=(10, 10))
    cc_tr.plot(fig, (1, 1, 1))
    plt.savefig(f"{path}_real_complexity_{dataset_name}_{random_state}.png")

    cc_aug.plot(fig, (1, 1, 1))
    plt.savefig(f"{path}_synth_complexity_{dataset_name}_{random_state}.png")

    # cc_aug_wc.plot(fig, (1, 1, 1))
    # plt.savefig(f"{path}_augmented_wc_complexity.png")

    # cc_tr_wc.plot(fig, (1, 1, 1))
    # plt.savefig(f"{path}_train_wc_complexity.png")

    return cc_tr._metrics(), cc_tr.score(),  cc_aug.score(),  cc_tr.complexity, cc_aug.complexity #, cc_aug_wc._metrics(), cc_tr_wc.complexity, cc_tr_wc.score(), cc_aug_wc.complexity, cc_aug_wc.score()


# **CLUSTERING the train data before calculating the Neighborhood/Netwok measures**
# for datasets with n_instances >12000 ->> sample 10% of the data 
# 1- cluster the train data with (k=4) clusters using mini-batch k-means ->> return clusters of the data
# 2- sample 10% of the data from each cluster ->> get the representative sample of the train data and the corresponding labels
# 3- calculate the complexity metrics for the representative sample of the train data



def ClusterSampling(X, y, sampling_ratio=None, k=4, random_state=None):
    """
    Cluster Sampling 10% of the data using MiniBatchKmeans with k (k=4) clusters
    
    Inputs:
        * X (ndarray)
        * y (ndarray)
        * k (int): number of clusters and its default is 4
        * sampling_ratio (float): by default 0.1 becasue it is applied to a specific group of datasets (#instance>12000)
    Output:
        * X_sampled (ndarray): 10% of the input X
        * y_sampled (ndarray)
    """
    # Print message to know the function running now
    print(f"Running ClusterSampling function...")
    # Case when sampling_ratio is 0, return the original data >> because the data is small enough and clustering is not needed
    sampling_ratio = 0.01 if X.shape[0]>=100000 else 0.1 if X.shape[0]>=10000 else 0 # sample 5% of the data if the dataset is >100000, 10% if >12000, otherwise 0

    if sampling_ratio==0:
        print(f"Sampling ratio is 0, returning the original data")
        return X, y
    print(f"Sampling ratio is {sampling_ratio}")
    # Case when sampling ratio is not 0 >> apply clustering and sampling
    # Shuffle the data
    X, y = shuffle(X, y, random_state=random_state)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state).fit(X, y)
    X_clustered = kmeans.predict(X)
    X_sampled = []
    y_sampled = []
    cluster_counts = Counter(X_clustered)
    for i in range(k):
        cluster_count = cluster_counts[i]
        X_sampled.append(X[X_clustered==i][:int(sampling_ratio*cluster_count)]) # get 10% of each cluster by default
        y_sampled.append(y[X_clustered==i][:int(sampling_ratio*cluster_count)])
    return np.array(np.concatenate(X_sampled)), np.array(np.concatenate(y_sampled))

# Using our built function `ClusterSampling`

#X_c, y_c = ClusterSampling(X, y)
#print(f"*Using ClusterSampling*\nX original shape: {X.shape} -- X_c shape: {X_c.shape} -- Sample size of 10% shape: {int(0.9*X.shape[0])}")


#######################
# px_complexity_min : Get complexity measures on original minority class  vs synthetic minority class
# some complexity metrics only works for this task: 
#######################


def complexity_one_class(real_data, augmented_data, rand_seed):
    
    """
    Complexity metrics
    -  features = ['c1', 'density', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 't2', 't3', 't4']
    - ['hubs', 'lsc','cls_coef', c2] -> deleted (generating memmory issues)
    >> Class Imbalance Measures  (definitely different)
    - `c1`: Compute the entropy of class proportions.
    >> Neighborhood -> cluster and then proportianny sample to reduce
    >> Network measures solve it by clustering
    - `density`:  Average density of the network. -> to be deleted (graph-based metric uses 'gower' distance)
    >> Feature-based Measures
    - `f1`: Maximum Fisher's discriminant ratio.
    - `f1v`: Directional-vector maximum Fisher's discriminant.
    - `f2`: Volume of the overlapping region.
    - `f3`: Compute feature maximum individual efficiency.
    - `f4`: Compute the collective feature effciency.
    >> Linearity-based Measures
    - `l1`: Sum of error distance by linear programming.
    - `l2`: Compute the OVO subsets eror rate of linear classifer.
    - `l3`: Non linearity of a linear classifier.
    >> Dimensionality Measures (data sparsity)
    - `t2`: Compute the average number of features per dimension.
    - `t3`: Compute the average number of PCA dimensions per points.
    - `t4`: Compute the ratio of the PCA dimension to the original dimension.

    Output: 
    - tuple of the list of the features and a list of the scores
    Example:
    (['t2', 't3', 't4'], [0.02666666666666667, 0.013333333333333334, 0.5])
    """
    real_data=np.array(real_data)
    augmented_data = np.array(augmented_data)
    mfe = MFE(groups=['complexity'], features=['t2', 't3', 't4'],summary=["mean"],suppress_warnings=True ,random_state=rand_seed)
    mfe.fit(real_data)
    ft_real = mfe.extract()
    #print(ft_real)
    mfe.fit(augmented_data)
    ft_synthetic = mfe.extract()
    #ft_diff = tuple([ft_real[0], list(np.array(ft_real[1])- np.array(ft_synthetic[1]))])
    complexity_features = ft_real[0]
    ft_real_score = np.mean(np.array(ft_real[1]))
    ft_synth_score = np.mean(np.array(ft_synthetic[1]))

    return complexity_features, ft_real_score, ft_synth_score,  ft_real[1], ft_synthetic[1]
