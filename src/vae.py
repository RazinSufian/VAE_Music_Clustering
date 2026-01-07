"""
VAE Model Architecture for Music Clustering

This module contains the Beta-VAE implementation with hybrid
audio (CNN) and text (MLP) encoders for learning disentangled
latent representations of music.
"""

import torch
import torch.nn as nn


class BetaHybridVAE(nn.Module):
    """
    Hybrid Variational Autoencoder with Beta-VAE formulation
    for disentangled latent representations.
    
    Architecture:
    - Audio Encoder: CNN (Conv2d layers)
    - Text Encoder: MLP
    - Latent Space: Gaussian with reparameterization
    - Audio Decoder: Transposed CNN
    - Text Decoder: MLP
    
    Args:
        n_feat: Number of audio features (default: 39 = 20 MFCC + 12 Chroma + 7 Spectral)
        max_len: Maximum time steps (default: 130)
        text_dim: Dimension of text features (default: 64)
        latent_dim: Dimension of latent space (default: 32)
    """
    
    def __init__(self, n_feat=39, max_len=130, text_dim=64, latent_dim=32):
        super().__init__()
        self.n_feat = n_feat
        self.max_len = max_len
        self.latent_dim = latent_dim
        
        # ========== AUDIO ENCODER ==========
        self.audio_enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        # Calculate dimensions dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_feat, max_len)
            enc_out = self.audio_enc(dummy)
            self.audio_flat_dim = enc_out.shape[1]
            
            # Get spatial dimensions before flatten
            conv_layers = nn.Sequential(*list(self.audio_enc.children())[:-1])
            conv_out = conv_layers(dummy)
            self.enc_c = conv_out.shape[1]
            self.enc_h = conv_out.shape[2]
            self.enc_w = conv_out.shape[3]
        
        # ========== TEXT ENCODER ==========
        self.text_enc = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # ========== LATENT SPACE ==========
        self.fusion_dim = self.audio_flat_dim + 32
        self.fc_mu = nn.Linear(self.fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.fusion_dim, latent_dim)
        
        # Initialize logvar to small values to prevent explosion
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)
        
        # ========== AUDIO DECODER ==========
        self.audio_dec_input = nn.Linear(latent_dim, self.audio_flat_dim)
        self.audio_dec = nn.Sequential(
            nn.Unflatten(1, (self.enc_c, self.enc_h, self.enc_w)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        )
        
        # ========== TEXT DECODER ==========
        self.text_dec = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, text_dim)
        )
        
        self.target_shape = (n_feat, max_len)
    
    def encode(self, x_audio, x_text):
        """Encode inputs to latent distribution parameters."""
        h_audio = self.audio_enc(x_audio)
        h_text = self.text_enc(x_text)
        h = torch.cat([h_audio, h_text], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstructions."""
        # Audio reconstruction
        h_audio = self.audio_dec_input(z)
        recon_audio = self.audio_dec(h_audio)
        
        # Text reconstruction
        recon_text = self.text_dec(z)
        
        return recon_audio, recon_text
    
    def forward(self, x_audio, x_text):
        """Full forward pass through the VAE."""
        mu, logvar = self.encode(x_audio, x_text)
        z = self.reparameterize(mu, logvar)
        recon_audio, recon_text = self.decode(z)
        
        # Resize audio if needed
        if recon_audio.shape[2:] != x_audio.shape[2:]:
            recon_audio = nn.functional.interpolate(
                recon_audio, size=self.target_shape, 
                mode='bilinear', align_corners=False
            )
        
        return recon_audio, recon_text, mu, logvar, z
    
    def get_latent(self, x_audio, x_text):
        """Get latent representation (mean only, no sampling)."""
        mu, _ = self.encode(x_audio, x_text)
        return mu


def beta_vae_loss(recon_audio, audio, recon_text, text, mu, logvar, beta, 
                  audio_weight=1.0, text_weight=0.1):
    """
    Beta-VAE Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        recon_audio: Reconstructed audio spectrogram
        audio: Original audio spectrogram
        recon_text: Reconstructed text features
        text: Original text features
        mu: Latent mean
        logvar: Latent log variance
        beta: KL divergence weight (higher = more disentanglement)
        audio_weight: Weight for audio reconstruction loss
        text_weight: Weight for text reconstruction loss
    
    Returns:
        total_loss, audio_loss, text_loss, kl_loss
    """
    # Reconstruction losses (mean reduction for stability)
    mse_audio = nn.functional.mse_loss(recon_audio, audio, reduction='mean')
    mse_text = nn.functional.mse_loss(recon_text, text, reduction='mean')
    
    # KL Divergence: D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total = audio_weight * mse_audio + text_weight * mse_text + beta * kl
    
    return total, mse_audio, mse_text, kl
