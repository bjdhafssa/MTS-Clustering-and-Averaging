# evaluator.py

import numpy as np
from tslearn.clustering import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter
from tslearn.metrics import cdist_dtw, cdist_soft_dtw
from tslearn.utils import to_time_series_dataset
import numpy as np

# Adapt davies_bouldin score for multivariate time series 

def davies_bouldin(X, labels, metric="dtw", **metric_params):
    X = to_time_series_dataset(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if metric == "dtw" or metric is None:
        centroids = [dtw_barycenter_averaging(X[labels == label], **metric_params) 
                     for label in unique_labels]
        distances = cdist_dtw(X, centroids, **metric_params)
        centroid_distances = cdist_dtw(centroids, centroids, **metric_params)
    elif metric == "softdtw":
        centroids = [softdtw_barycenter(X[labels == label], **metric_params) 
                     for label in unique_labels]
        distances = cdist_soft_dtw(X, centroids, **metric_params)
        centroid_distances = cdist_soft_dtw(centroids, centroids, **metric_params)
    else:
        raise ValueError("Metric not supported. Use 'dtw' or 'softdtw'.")

    intra_dists = np.array([np.mean(distances[labels == label, i]) 
                            for i, label in enumerate(unique_labels)])

    db_ratios = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            db_ratios[i, j] = db_ratios[j, i] = (intra_dists[i] + intra_dists[j]) / centroid_distances[i, j]

    db_score = np.mean([np.max(db_ratios[i][db_ratios[i] != 0]) for i in range(n_clusters)])

    return db_score

#################################################################################################################


def evaluate_clustering(X, labels, true_labels=None, metric="dtw", **metric_params):
    """
    Evaluate clustering results using various metrics.
    
    Parameters:
    - X: The input data (array-like of shape (n_samples, n_timestamps, n_features)).
    - labels: The predicted cluster labels (array-like of shape (n_samples,)).
    - true_labels: The true labels (array-like of shape (n_samples,)), optional.
    
    Returns:
    A dictionary with evaluation metrics.
    """
    metrics = {}
    
    # Intrinsic metric (doesn't require true labels)
    metrics["Silhouette"] = silhouette_score(X, labels)
    metrics["Davies-Bouldin"] = davies_bouldin(X, labels, metric=metric, **metric_params)


    if true_labels is not None:
        # Extrinsic metrics (require true labels)
        metrics["ARI"] = adjusted_rand_score(true_labels, labels)
        metrics["AMI"] = adjusted_mutual_info_score(true_labels, labels)
        metrics["NMI"] = normalized_mutual_info_score(true_labels, labels)
        metrics["Homogeneity"] = homogeneity_score(true_labels, labels)
        metrics["Completeness"] = completeness_score(true_labels, labels)
        metrics["V-measure"] = v_measure_score(true_labels, labels)
        metrics["Fowlkes-Mallows"] = fowlkes_mallows_score(true_labels, labels)
    
    return metrics
