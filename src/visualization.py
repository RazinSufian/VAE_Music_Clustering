"""
Visualization Utilities for VAE Music Clustering

This module contains functions for creating training curves,
reconstruction visualizations, and other plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(history, save_path=None):
    """
    Plot training curves from history dictionary.
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history['epoch']
    
    # Total loss
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction losses
    axes[0, 1].plot(epochs, history['audio_loss'], 'g-', linewidth=2, label='Audio MSE')
    axes[0, 1].plot(epochs, history['text_loss'], 'r-', linewidth=2, label='Text MSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Reconstruction Losses', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence and beta
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    
    line1, = ax1.plot(epochs, history['kl_loss'], 'purple', linewidth=2, label='KL Divergence')
    line2, = ax2.plot(epochs, history['beta'], 'orange', linewidth=2, linestyle='--', label='β')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('KL Divergence', color='purple')
    ax2.set_ylabel('Beta (β)', color='orange')
    ax1.set_title('KL Divergence and Beta Annealing', fontweight='bold')
    ax1.legend(handles=[line1, line2], loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_reconstruction_examples(model, data_loader, device, n_samples=6, save_path=None):
    """
    Plot original vs reconstructed spectrograms.
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader with data
        device: Device to use
        n_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    import torch
    
    model.eval()
    
    # Get one batch
    batch_audio, batch_text = next(iter(data_loader))
    batch_audio = batch_audio[:n_samples].to(device)
    batch_text = batch_text[:n_samples].to(device)
    
    with torch.no_grad():
        recon_audio, recon_text, mu, logvar, z = model(batch_audio, batch_text)
    
    # Convert to numpy
    original = batch_audio.cpu().numpy()[:, 0, :, :]
    reconstructed = recon_audio.cpu().numpy()[:, 0, :, :]
    
    # Plot
    fig, axes = plt.subplots(3, n_samples, figsize=(3*n_samples, 9))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original[i], aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i], aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}', fontsize=10)
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(original[i] - reconstructed[i])
        axes[2, i].imshow(diff, aspect='auto', origin='lower', cmap='hot')
        axes[2, i].set_title(f'Difference {i+1}', fontsize=10)
        axes[2, i].axis('off')
    
    # Row labels
    axes[0, 0].text(-0.1, 0.5, 'Original', transform=axes[0, 0].transAxes, 
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    axes[1, 0].text(-0.1, 0.5, 'Reconstructed', transform=axes[1, 0].transAxes, 
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    axes[2, 0].text(-0.1, 0.5, 'Difference', transform=axes[2, 0].transAxes, 
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    
    plt.suptitle('Reconstruction Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_distributions(cluster_labels, genre_labels, genre_names, 
                       results_df=None, save_path=None):
    """
    Plot cluster and genre distributions.
    
    Args:
        cluster_labels: Cluster assignments
        genre_labels: Genre labels
        genre_names: List of genre names
        results_df: Optional DataFrame with clustering results
        save_path: Optional path to save figure
    """
    from collections import Counter
    
    n_plots = 3 if results_df is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # Cluster distribution
    cluster_counts = Counter(cluster_labels)
    clusters = sorted(cluster_counts.keys())
    counts = [cluster_counts[c] for c in clusters]
    
    bars1 = axes[0].bar([f'C{c}' for c in clusters], counts, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Number of Songs')
    axes[0].set_title('Cluster Distribution', fontweight='bold')
    
    # Add count labels
    for bar, count in zip(bars1, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                     str(count), ha='center', va='bottom', fontsize=10)
    
    # Genre distribution
    genre_counts = Counter(genre_labels)
    genres = list(genre_names) if isinstance(genre_names, (list, np.ndarray)) else sorted(genre_counts.keys())
    g_counts = [genre_counts.get(g, 0) for g in genres]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))
    bars2 = axes[1].bar(genres, g_counts, color=colors, edgecolor='black')
    axes[1].set_xlabel('Genre')
    axes[1].set_ylabel('Number of Songs')
    axes[1].set_title('Genre Distribution', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Metrics comparison (if available)
    if results_df is not None and n_plots == 3:
        methods = results_df['Method'].tolist()
        silhouette = results_df['Silhouette'].tolist()
        
        x = np.arange(len(methods))
        bars3 = axes[2].bar(x, silhouette, color='coral', edgecolor='black')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(methods, rotation=45, ha='right')
        axes[2].set_ylabel('Silhouette Score')
        axes[2].set_title('Clustering Methods Comparison', fontweight='bold')
        axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_history_to_csv(history, save_path):
    """
    Save training history to CSV file.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save CSV
    """
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    print(f"Training history saved to: {save_path}")
