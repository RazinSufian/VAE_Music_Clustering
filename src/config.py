"""
Configuration and Hyperparameters for VAE Music Clustering

This module contains all configurable parameters for the model,
training, and evaluation pipeline.
"""

import os


class Config:
    """
    Configuration class containing all hyperparameters.
    
    Usage:
        config = Config()
        config.LATENT_DIM = 64  # Override defaults
    """
    
    # ========== Data Configuration ==========
    N_MFCC = 20           # Number of MFCC coefficients
    N_CHROMA = 12         # Number of chroma features
    N_CONTRAST = 7        # Number of spectral contrast bands
    N_FEATURES = N_MFCC + N_CHROMA + N_CONTRAST  # Total = 39
    MAX_LEN = 130         # Maximum time frames
    SAMPLE_RATE = 22050   # Audio sample rate
    AUDIO_DURATION = 30   # Audio clip duration in seconds
    
    # Text features
    TFIDF_MAX_FEATURES = 500  # Max TF-IDF vocabulary size
    TEXT_DIM = 64             # PCA reduced text dimension
    
    # ========== Model Configuration ==========
    LATENT_DIM = 32       # Latent space dimension
    
    # Audio encoder
    AUDIO_ENC_CHANNELS = [32, 64, 128]  # Conv channels
    AUDIO_ENC_KERNEL = 3
    AUDIO_ENC_STRIDE = 2
    
    # Text encoder
    TEXT_ENC_HIDDEN = [64, 32]  # Hidden layer sizes
    
    # Dropout
    DROPOUT = 0.1
    
    # ========== Training Configuration ==========
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    EPOCHS = 100
    
    # Beta-VAE parameters
    MAX_BETA = 1.0
    WARMUP_EPOCHS = 50
    
    # Loss weights
    AUDIO_WEIGHT = 1.0
    TEXT_WEIGHT = 0.1
    
    # Regularization
    CLIP_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.0
    
    # Learning rate scheduling
    LR_PATIENCE = 10
    LR_FACTOR = 0.5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # ========== Clustering Configuration ==========
    K_RANGE = range(3, 12)  # Range of K values to try
    OPTIMAL_K = 4           # Found optimal K (can be overridden)
    
    # DBSCAN parameters
    DBSCAN_EPS = 1.5
    DBSCAN_MIN_SAMPLES = 10
    
    # ========== Visualization Configuration ==========
    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    
    FIGURE_DPI = 150
    
    # ========== Paths Configuration ==========
    # These should be set based on your environment
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'
    MODELS_DIR = 'results/models'
    FEATURES_DIR = 'results/features'
    FIGURES_DIR = 'results'
    
    # Random seed
    SEED = 42
    
    def __init__(self):
        """Initialize config with default values."""
        pass
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def __repr__(self):
        """String representation of config."""
        items = self.to_dict()
        return '\n'.join(f'{k}: {v}' for k, v in items.items())


def get_device():
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device object
    """
    import torch
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import torch
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Default config instance
config = Config()
