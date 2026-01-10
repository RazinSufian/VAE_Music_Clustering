"""
Utility Functions for VAE Music Clustering

This module contains helper functions for saving/loading models,
managing results, and other common operations.
"""

import os
import json
import pickle
import numpy as np
import torch
import pandas as pd


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def save_model(model, save_path, include_config=True, **config_kwargs):
    """
    Save model weights and optional configuration.
    
    Args:
        model: PyTorch model
        save_path: Path to save model
        include_config: Whether to save config alongside
        **config_kwargs: Additional config to save
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    
    if include_config and config_kwargs:
        config_path = save_path.replace('.pth', '_config.json')
        # Convert non-serializable values
        config = {}
        for k, v in config_kwargs.items():
            if isinstance(v, np.ndarray):
                config[k] = v.tolist()
            elif isinstance(v, (np.int64, np.float64)):
                config[k] = int(v) if isinstance(v, np.int64) else float(v)
            else:
                config[k] = v
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {config_path}")


def load_model(model, load_path, device='cpu'):
    """
    Load model weights from file.
    
    Args:
        model: PyTorch model (architecture must match)
        load_path: Path to load model from
        device: Device to load to
    
    Returns:
        Model with loaded weights
    """
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from: {load_path}")
    return model


def save_features(features, labels, save_dir, prefix=''):
    """
    Save extracted features and labels as numpy files.
    
    Args:
        features: Feature array
        labels: Dictionary of label arrays
        save_dir: Directory to save to
        prefix: Optional prefix for filenames
    """
    ensure_dir(save_dir)
    
    # Save features
    if features is not None:
        np.save(os.path.join(save_dir, f'{prefix}latent_features.npy'), features)
    
    # Save labels
    if isinstance(labels, dict):
        for name, arr in labels.items():
            np.save(os.path.join(save_dir, f'{prefix}{name}.npy'), arr)
    
    print(f"Features saved to: {save_dir}")


def load_features(load_dir, prefix=''):
    """
    Load features and labels from numpy files.
    
    Args:
        load_dir: Directory to load from
        prefix: Optional prefix for filenames
    
    Returns:
        features, labels dictionary
    """
    features = np.load(os.path.join(load_dir, f'{prefix}latent_features.npy'))
    
    labels = {}
    # Try to load common label files
    label_files = ['cluster_labels_kmeans', 'genre_labels', 'tsne_2d']
    for name in label_files:
        path = os.path.join(load_dir, f'{prefix}{name}.npy')
        if os.path.exists(path):
            labels[name] = np.load(path)
    
    return features, labels


def save_results(results_df, save_path):
    """
    Save clustering results to CSV.
    
    Args:
        results_df: DataFrame with results
        save_path: Path to save CSV
    """
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")


def load_results(load_path):
    """
    Load clustering results from CSV.
    
    Args:
        load_path: Path to CSV file
    
    Returns:
        DataFrame with results
    """
    return pd.read_csv(load_path)


def print_model_summary(model):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")


def print_final_summary(dataset_info, model_info, results_df, best_method='VAE + K-Means'):
    """
    Print comprehensive final project summary.
    
    Args:
        dataset_info: Dictionary with dataset information
        model_info: Dictionary with model information
        results_df: DataFrame with clustering results
        best_method: Name of the best method
    """
    best_row = results_df[results_df['Method'] == best_method].iloc[0]
    
    print("\n" + "="*80)
    print("                        📊 FINAL PROJECT SUMMARY")
    print("="*80)
    
    print(f"\n Dataset:")
    print(f"   Total songs processed: {dataset_info.get('n_songs', 'N/A')}")
    print(f"   Audio features: {dataset_info.get('audio_features', 'N/A')}")
    print(f"   Text features: {dataset_info.get('text_features', 'N/A')}")
    print(f"   Genres: {dataset_info.get('genres', 'N/A')}")
    
    print(f"\n Model:")
    print(f"   Architecture: {model_info.get('architecture', 'Hybrid CNN-VAE (Beta-VAE)')}")
    print(f"   Latent dimension: {model_info.get('latent_dim', 32)}")
    print(f"   Total parameters: {model_info.get('total_params', 'N/A'):,}")
    print(f"   Best training loss: {model_info.get('best_loss', 'N/A'):.4f}")
    
    print(f"\n Clustering:")
    print(f"   Optimal K: {model_info.get('optimal_k', 4)}")
    print(f"   Best method: {best_method}")
    
    print(f"\n Best Metrics ({best_method}):")
    print(f"   Silhouette Score: {best_row.get('Silhouette', 'N/A'):.4f}")
    print(f"   Calinski-Harabasz: {best_row.get('Calinski-Harabasz', 'N/A'):.2f}")
    print(f"   Davies-Bouldin: {best_row.get('Davies-Bouldin', 'N/A'):.4f}")
    if 'ARI' in best_row:
        print(f"   ARI: {best_row['ARI']:.4f}")
    if 'NMI' in best_row:
        print(f"   NMI: {best_row['NMI']:.4f}")
    if 'Purity' in best_row:
        print(f"   Purity: {best_row['Purity']:.4f}")
    
    print("\n" + "="*80)
    print("                        🎉 PROJECT COMPLETE!")
    print("="*80 + "\n")
