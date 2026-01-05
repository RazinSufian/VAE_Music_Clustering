# 🎵 VAE Music Clustering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A hybrid **Variational Autoencoder (VAE)** for music clustering using audio features (MFCC, Chroma, Spectral Contrast) and lyrics embeddings (TF-IDF). This project implements a **Beta-VAE** architecture with convolutional audio encoding for learning disentangled latent representations of music.

## 📋 Project Overview

This project explores unsupervised music clustering using deep generative models. The goal is to learn meaningful latent representations of songs that capture both audio characteristics and lyrical content, then cluster songs based on these representations.

### Key Features

- 🎧 **Hybrid Architecture**: Combines CNN-based audio encoder with MLP text encoder
- 🔄 **Beta-VAE**: Implements β-annealing for disentangled representations
- 📊 **Multi-Modal**: Fuses MFCC/Chroma/Spectral audio features with TF-IDF lyrics
- 🎯 **Multiple Clustering**: K-Means, Agglomerative, DBSCAN comparison
- 📈 **Comprehensive Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, Purity

## 📊 Results

| Method | Silhouette | CH Index | DB Index | ARI | NMI | Purity |
|--------|------------|----------|----------|-----|-----|--------|
| **VAE + K-Means** | 0.935 | 12067.75 | 0.225 | 0.004 | 0.018 | 22.5% |
| VAE + Agglomerative | 0.889 | 9856.32 | 0.287 | 0.003 | 0.015 | 21.8% |
| PCA + K-Means | 0.174 | 210.44 | 2.497 | 0.010 | 0.022 | 25.8% |

### Visualizations

<p align="center">
  <img src="results/tsne_visualization.png" width="80%" alt="t-SNE Visualization"/>
</p>

<p align="center">
  <img src="results/confusion_matrix.png" width="80%" alt="Confusion Matrix"/>
</p>

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Beta-VAE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio Input (1, 39, 130)    Text Input (64)               │
│         │                          │                        │
│         ▼                          ▼                        │
│  ┌─────────────┐            ┌─────────────┐                │
│  │ Conv2d(32)  │            │ Linear(64)  │                │
│  │ Conv2d(64)  │            │ Linear(32)  │                │
│  │ Conv2d(128) │            └──────┬──────┘                │
│  └──────┬──────┘                   │                        │
│         │                          │                        │
│         └──────────┬───────────────┘                        │
│                    ▼                                        │
│            ┌──────────────┐                                 │
│            │   Fusion     │                                 │
│            │  (Concat)    │                                 │
│            └──────┬───────┘                                 │
│                   ▼                                         │
│         ┌─────────────────┐                                 │
│         │ μ (mean)        │                                 │
│         │ σ (logvar)      │  → Latent Space (32-dim)       │
│         └────────┬────────┘                                 │
│                  │                                          │
│         ┌────────┴────────┐                                 │
│         ▼                 ▼                                 │
│  ┌─────────────┐   ┌─────────────┐                         │
│  │Audio Decoder│   │Text Decoder │                         │
│  │(TransConv2d)│   │  (Linear)   │                         │
│  └─────────────┘   └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
VAE_Music_Clustering/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── notebooks/
│   └── VAE_Music_Clustering_FINAL.ipynb  # Main notebook (run on Colab)
│
├── data/
│   └── README.md                # Instructions for obtaining dataset
│
├── results/                     # Output visualizations (after running)
│   ├── training_curves.png
│   ├── reconstruction_examples.png
│   ├── tsne_visualization.png
│   ├── umap_visualization.png
│   ├── confusion_matrix.png
│   ├── cluster_selection.png
│   └── clustering_metrics.csv
│
└── docs/
    └── report.pdf               # NeurIPS-style report (if available)
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/RazinSufian/VAE_Music_Clustering.git
cd VAE_Music_Clustering
```

### 2. Set Up Dataset

See [data/README.md](data/README.md) for instructions on obtaining and preparing the dataset.

### 3. Run on Google Colab

1. Upload `notebooks/VAE_Music_Clustering_FINAL.ipynb` to Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → GPU`
3. Mount your Google Drive with the dataset
4. Update the `DRIVE_PATH` variable to point to your data
5. Run all cells

### 4. Local Installation (Optional)

```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- Python 3.8+
- PyTorch 2.0+
- librosa
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- umap-learn (optional)

## 🎯 Dataset

The project uses a music dataset with:
- **2,890 songs** across 6 genres (pop, rock, rap, r&b, edm, latin)
- **Audio files**: 30-second WAV clips
- **Metadata**: Track names, genres, lyrics

> ⚠️ The audio files (~2GB) are not included in this repository due to size constraints. See [data/README.md](data/README.md) for download instructions.

## 📈 Metrics Explained

| Metric | Description | Optimal |
|--------|-------------|---------|
| **Silhouette Score** | Cluster cohesion vs separation | Higher (max 1) |
| **Calinski-Harabasz** | Ratio of between/within cluster variance | Higher |
| **Davies-Bouldin** | Average cluster similarity | Lower |
| **ARI** | Agreement with ground truth | Higher (max 1) |
| **NMI** | Mutual information with labels | Higher (max 1) |
| **Purity** | Dominant class fraction per cluster | Higher |

## 🔬 Key Findings

1. **Beta-VAE learns smooth latent representations** - The β-annealing strategy prevents posterior collapse
2. **Hybrid features outperform audio-only** - Combining audio + lyrics improves clustering quality
3. **Genre boundaries are fuzzy** - Music genres have significant overlap, explaining modest ARI/NMI scores
4. **Optimal clusters ≠ Number of genres** - The model found K=4 optimal despite having 6 genre labels

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@misc{vae_music_clustering_2026,
  author = {Razin Sufian},
  title = {VAE Music Clustering: Hybrid Audio-Lyrics Representation Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/RazinSufian/VAE_Music_Clustering}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course: CSE425 - Neural Networks
- Dataset: Music Dataset with lyrics and audio features
- Frameworks: PyTorch, librosa, scikit-learn
