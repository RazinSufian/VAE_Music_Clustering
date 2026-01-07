"""
Evaluation Metrics for Music Clustering

This module contains functions to compute and compare
clustering evaluation metrics.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score, 
    normalized_mutual_info_score
)


def compute_purity(y_true, y_pred):
    """
    Compute cluster purity.
    
    Purity measures the fraction of the dominant class in each cluster.
    Higher purity indicates better clustering with respect to ground truth.
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
    
    Returns:
        Purity score between 0 and 1
    """
    contingency = {}
    for true, pred in zip(y_true, y_pred):
        if pred not in contingency:
            contingency[pred] = Counter()
        contingency[pred][true] += 1
    
    total_correct = sum(max(counts.values()) for counts in contingency.values())
    return total_correct / len(y_true)


def evaluate_clustering(features, labels, y_true=None, method_name="Unknown"):
    """
    Evaluate clustering using multiple metrics.
    
    Args:
        features: Feature matrix used for clustering
        labels: Cluster labels
        y_true: True labels (optional, for external metrics)
        method_name: Name of the clustering method
    
    Returns:
        Dictionary with all metric scores
    """
    # Filter out noise points (label = -1 for DBSCAN)
    valid_mask = labels >= 0
    if valid_mask.sum() < 10:
        print(f"Warning: {method_name} has too few valid points")
        return None
    
    features_v = features[valid_mask]
    labels_v = labels[valid_mask]
    
    n_clusters = len(set(labels_v))
    if n_clusters < 2:
        print(f"Warning: {method_name} has only {n_clusters} cluster(s)")
        return None
    
    results = {
        'Method': method_name,
        'Clusters': n_clusters,
        'Silhouette': silhouette_score(features_v, labels_v),
        'Calinski-Harabasz': calinski_harabasz_score(features_v, labels_v),
        'Davies-Bouldin': davies_bouldin_score(features_v, labels_v),
    }
    
    # External metrics (if ground truth available)
    if y_true is not None:
        y_true_v = np.array(y_true)[valid_mask] if hasattr(y_true, '__len__') else y_true[valid_mask]
        results['ARI'] = adjusted_rand_score(y_true_v, labels_v)
        results['NMI'] = normalized_mutual_info_score(y_true_v, labels_v)
        results['Purity'] = compute_purity(y_true_v, labels_v)
    
    return results


def compare_methods(results_list):
    """
    Create comparison DataFrame from list of results.
    
    Args:
        results_list: List of result dictionaries from evaluate_clustering
    
    Returns:
        DataFrame with all methods and their scores
    """
    results_list = [r for r in results_list if r is not None]
    return pd.DataFrame(results_list)


def print_results_table(results_df):
    """
    Print formatted results table.
    
    Args:
        results_df: DataFrame from compare_methods
    """
    print("\n" + "="*100)
    print("CLUSTERING EVALUATION RESULTS")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)


def analyze_cluster_composition(cluster_labels, genre_labels, genre_names):
    """
    Analyze the composition of each cluster by genre.
    
    Args:
        cluster_labels: Cluster assignments
        genre_labels: True genre labels (strings)
        genre_names: List of unique genre names
    
    Returns:
        Dictionary mapping cluster IDs to genre distributions
    """
    composition = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id < 0:  # Skip noise
            continue
        
        mask = cluster_labels == cluster_id
        cluster_genres = np.array(genre_labels)[mask]
        genre_counts = Counter(cluster_genres)
        total = len(cluster_genres)
        
        composition[cluster_id] = {
            'total': total,
            'distribution': {g: (c, c/total*100) for g, c in genre_counts.most_common()}
        }
    
    return composition


def print_cluster_composition(composition):
    """
    Print cluster composition analysis.
    
    Args:
        composition: Dictionary from analyze_cluster_composition
    """
    print("\nCluster Composition Analysis:")
    print("-" * 50)
    
    for cluster_id, data in composition.items():
        print(f"\nCluster {cluster_id} ({data['total']} songs):")
        for genre, (count, pct) in data['distribution'].items():
            print(f"  {genre}: {count} ({pct:.1f}%)")


def get_best_method(results_df, metric='Silhouette'):
    """
    Get the best performing method based on a metric.
    
    Args:
        results_df: DataFrame from compare_methods
        metric: Metric to use for comparison
    
    Returns:
        Name of the best method
    """
    if metric in ['Davies-Bouldin']:
        # Lower is better
        idx = results_df[metric].idxmin()
    else:
        # Higher is better
        idx = results_df[metric].idxmax()
    
    return results_df.loc[idx, 'Method']
