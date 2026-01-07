"""
Clustering Utilities for Music Clustering

This module contains clustering algorithms and
visualization functions for the VAE latent space.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def find_optimal_k(features, k_range=range(3, 12), random_state=42):
    """
    Find optimal number of clusters using silhouette analysis.
    
    Args:
        features: Latent features (N, latent_dim)
        k_range: Range of K values to try
        random_state: Random seed
    
    Returns:
        optimal_k: Best K value
        scores: Dictionary with K values and their scores
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    scores = {'k': [], 'silhouette': [], 'calinski': [], 'davies': []}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features)
        
        scores['k'].append(k)
        scores['silhouette'].append(silhouette_score(features, labels))
        scores['calinski'].append(calinski_harabasz_score(features, labels))
        scores['davies'].append(davies_bouldin_score(features, labels))
    
    optimal_k = scores['k'][np.argmax(scores['silhouette'])]
    return optimal_k, scores


def perform_clustering(features, n_clusters, random_state=42):
    """
    Perform multiple clustering algorithms.
    
    Args:
        features: Latent features (N, latent_dim)
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Dictionary with labels from each algorithm
    """
    results = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    results['kmeans'] = kmeans.fit_predict(features)
    
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    results['agglomerative'] = agg.fit_predict(features)
    
    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    results['dbscan'] = dbscan.fit_predict(features)
    
    return results


def compute_tsne(features, n_components=2, perplexity=30, random_state=42):
    """
    Compute t-SNE dimensionality reduction.
    
    Args:
        features: High-dimensional features
        n_components: Target dimensions
        perplexity: t-SNE perplexity
        random_state: Random seed
    
    Returns:
        2D projection of features
    """
    tsne = TSNE(n_components=n_components, random_state=random_state, 
                perplexity=perplexity, n_iter=1000)
    return tsne.fit_transform(features)


def compute_umap(features, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Compute UMAP dimensionality reduction.
    
    Args:
        features: High-dimensional features
        n_components: Target dimensions
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed
    
    Returns:
        2D projection of features (or None if UMAP not available)
    """
    if not HAS_UMAP:
        print("UMAP not available")
        return None
    
    reducer = umap.UMAP(n_components=n_components, random_state=random_state,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(features)


def plot_cluster_selection(scores, save_path=None):
    """
    Plot cluster selection metrics.
    
    Args:
        scores: Dictionary from find_optimal_k
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    optimal_k = scores['k'][np.argmax(scores['silhouette'])]
    
    axes[0].plot(scores['k'], scores['silhouette'], 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Analysis', fontweight='bold')
    axes[0].legend()
    
    axes[1].plot(scores['k'], scores['calinski'], 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Calinski-Harabasz Index')
    axes[1].set_title('Calinski-Harabasz Analysis', fontweight='bold')
    
    axes[2].plot(scores['k'], scores['davies'], 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin (Lower=Better)', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_latent_space(features_2d, labels_cluster, labels_genre, genre_names, save_path=None):
    """
    Plot 2D projection of latent space.
    
    Args:
        features_2d: 2D features from t-SNE or UMAP
        labels_cluster: Cluster labels
        labels_genre: True genre labels
        genre_names: List of genre names
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # By cluster
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=labels_cluster, cmap='viridis', alpha=0.6, s=15)
    axes[0].set_title('Latent Space (Clusters)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # By genre
    scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=labels_genre, cmap='Spectral', alpha=0.6, s=15)
    axes[1].set_title('Latent Space (True Genres)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    cbar = plt.colorbar(scatter2, ax=axes[1], label='Genre')
    cbar.set_ticks(range(len(genre_names)))
    cbar.set_ticklabels(genre_names)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, genre_names, n_clusters, save_path=None):
    """
    Plot confusion matrix between genres and clusters.
    
    Args:
        y_true: True genre labels
        y_pred: Predicted cluster labels
        genre_names: List of genre names
        n_clusters: Number of clusters
        save_path: Optional path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=[f'C{i}' for i in range(n_clusters)],
                yticklabels=genre_names)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Cluster')
    axes[0].set_ylabel('True Genre')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=[f'C{i}' for i in range(n_clusters)],
                yticklabels=genre_names)
    axes[1].set_title('Confusion Matrix (Row-Normalized)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Cluster')
    axes[1].set_ylabel('True Genre')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
