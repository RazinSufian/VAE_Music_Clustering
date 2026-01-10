"""
Training Utilities for VAE Music Clustering

This module contains the training loop, learning rate scheduling,
and model checkpointing functionality.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def create_dataloader(X_audio, X_text, batch_size=64, shuffle=True):
    """
    Create a PyTorch DataLoader from numpy arrays.
    
    Args:
        X_audio: Audio features (N, 1, n_features, max_len)
        X_text: Text features (N, text_dim)
        batch_size: Batch size
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader object
    """
    audio_tensor = torch.FloatTensor(X_audio)
    text_tensor = torch.FloatTensor(X_text)
    
    dataset = TensorDataset(audio_tensor, text_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_beta(epoch, warmup_epochs=50, max_beta=1.0):
    """
    Compute beta value for KL annealing.
    
    Args:
        epoch: Current epoch
        warmup_epochs: Number of epochs to reach max_beta
        max_beta: Maximum beta value
    
    Returns:
        Current beta value
    """
    return min(max_beta, (epoch / warmup_epochs) * max_beta)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore best model weights
    """
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        
        return self.should_stop


def train_vae(model, train_loader, optimizer, scheduler, device, 
              epochs=100, warmup_epochs=50, max_beta=1.0,
              audio_weight=1.0, text_weight=0.1,
              clip_grad_norm=1.0, early_stopping=None,
              save_dir=None, verbose=True):
    """
    Train the VAE model.
    
    Args:
        model: BetaHybridVAE model
        train_loader: Training DataLoader
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        epochs: Number of training epochs
        warmup_epochs: Epochs for beta warmup
        max_beta: Maximum beta value
        audio_weight: Weight for audio reconstruction loss
        text_weight: Weight for text reconstruction loss
        clip_grad_norm: Maximum gradient norm for clipping
        early_stopping: EarlyStopping instance (optional)
        save_dir: Directory to save checkpoints (optional)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history
    """
    from .vae import beta_vae_loss
    
    model.to(device)
    history = {
        'epoch': [], 'total_loss': [], 'audio_loss': [], 
        'text_loss': [], 'kl_loss': [], 'beta': [], 'lr': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'audio': 0, 'text': 0, 'kl': 0}
        
        beta = get_beta(epoch, warmup_epochs, max_beta)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', 
                           disable=not verbose)
        
        for batch_audio, batch_text in progress_bar:
            batch_audio = batch_audio.to(device)
            batch_text = batch_text.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_audio, recon_text, mu, logvar, z = model(batch_audio, batch_text)
            
            # Compute loss
            total, audio_loss, text_loss, kl = beta_vae_loss(
                recon_audio, batch_audio, recon_text, batch_text,
                mu, logvar, beta, audio_weight, text_weight
            )
            
            # Check for NaN
            if torch.isnan(total):
                print(f"NaN detected at epoch {epoch+1}, stopping training")
                return history
            
            # Backward pass
            total.backward()
            
            # Gradient clipping
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total.item()
            epoch_losses['audio'] += audio_loss.item()
            epoch_losses['text'] += text_loss.item()
            epoch_losses['kl'] += kl.item()
            
            progress_bar.set_postfix({
                'loss': f"{total.item():.4f}",
                'β': f"{beta:.3f}"
            })
        
        # Average losses
        n_batches = len(train_loader)
        avg_loss = epoch_losses['total'] / n_batches
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg_loss)
        history['audio_loss'].append(epoch_losses['audio'] / n_batches)
        history['text_loss'].append(epoch_losses['text'] / n_batches)
        history['kl_loss'].append(epoch_losses['kl'] / n_batches)
        history['beta'].append(beta)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
        
        # Save best model
        if save_dir and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(avg_loss, model):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, β={beta:.3f}")
    
    # Save final model
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    
    return history


def extract_latent_features(model, data_loader, device):
    """
    Extract latent features from trained model.
    
    Args:
        model: Trained BetaHybridVAE model
        data_loader: DataLoader with data
        device: Device to use
    
    Returns:
        Numpy array of latent features
    """
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch_audio, batch_text in data_loader:
            batch_audio = batch_audio.to(device)
            batch_text = batch_text.to(device)
            
            z = model.get_latent(batch_audio, batch_text)
            latent_features.append(z.cpu().numpy())
    
    return np.vstack(latent_features)
