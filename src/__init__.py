"""
VAE Music Clustering - Source Package

This package provides modules for:
- vae.py: Beta-VAE model architecture
- dataset.py: Data loading and feature extraction
- clustering.py: Clustering algorithms and visualization
- evaluation.py: Metrics and evaluation utilities
"""

from .vae import BetaHybridVAE, beta_vae_loss
from .dataset import (
    load_and_match_data,
    process_text_features,
    prepare_data_for_training,
    extract_audio_features
)
from .clustering import (
    find_optimal_k,
    perform_clustering,
    compute_tsne,
    compute_umap
)
from .evaluation import (
    evaluate_clustering,
    compare_methods,
    compute_purity
)

__version__ = "1.0.0"
__author__ = "Razin Sufian"
