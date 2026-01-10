"""
VAE Music Clustering - Source Package

This package provides modules for:
- vae.py: Beta-VAE model architecture
- dataset.py: Data loading and feature extraction
- clustering.py: Clustering algorithms and visualization
- evaluation.py: Metrics and evaluation utilities
- train.py: Training loop and utilities
- config.py: Configuration and hyperparameters
- visualization.py: Plotting and visualization
- utils.py: Helper functions for saving/loading
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
from .train import (
    train_vae,
    create_dataloader,
    EarlyStopping,
    extract_latent_features
)
from .config import Config, config, get_device, set_seed
from .visualization import (
    plot_training_curves,
    plot_reconstruction_examples,
    plot_distributions
)
from .utils import (
    save_model,
    load_model,
    save_features,
    load_features,
    print_model_summary,
    print_final_summary
)

__version__ = "1.0.0"
__author__ = "Razin Sufian"

