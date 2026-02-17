# Results

This folder contains the output files from the VAE Music Clustering experiment.

## Directory Structure

```
results/
├── README.md
├── clustering_metrics.csv          # Evaluation metrics table
├── training_history.csv            # Loss per epoch
│
├── latent_visualization/           # Latent space plots
│   ├── tsne_visualization.png
│   └── umap_visualization.png
│
├── models/                         # Trained model weights
│   ├── model_final.pth
│   └── best_model.pth
│
├── features/                       # Extracted features
│   ├── latent_features.npy         # 32-dim VAE features
│   ├── cluster_labels_kmeans.npy   # Cluster assignments
│   ├── genre_labels.npy            # Encoded genres
│   └── tsne_2d.npy                 # 2D coordinates
│
├── training_curves.png             # Training loss plots
├── reconstruction_examples.png     # Original vs reconstructed
├── cluster_selection.png           # Silhouette analysis
├── confusion_matrix.png            # Genre-cluster matrix
└── distributions.png               # Distribution plots
```

## Clustering Results

```
================================================================================
                        FINAL CLUSTERING RESULTS
================================================================================
    Method          Clusters  Silhouette  Calinski-Harabasz  Davies-Bouldin    ARI     NMI   Purity
VAE + K-Means            4       0.935         12067.75           0.225      0.004   0.018   0.225
VAE + Agglomerative      4       0.889          9856.32           0.287      0.003   0.015   0.218
PCA + K-Means            4       0.174           210.44           2.497      0.010   0.022   0.258
================================================================================
```

## Training Summary

- **Dataset**: 2,890 songs across 6 genres
- **Features**: 39 audio (MFCC+Chroma+Spectral) + 64 text (TF-IDF→PCA)
- **Latent Dim**: 32
- **Optimal K**: 4 clusters
- **Best Loss**: 1.0371
- **Model Parameters**: 1,256,897

## File Descriptions

### Metrics
| File | Description |
|------|-------------|
| `clustering_metrics.csv` | All evaluation metrics for each method |
| `training_history.csv` | Loss values per epoch |

### Visualizations
| File | Description |
|------|-------------|
| `training_curves.png` | Training loss curves (total, audio, text, KL) |
| `reconstruction_examples.png` | Original vs reconstructed spectrograms |
| `cluster_selection.png` | Silhouette analysis for optimal K |
| `confusion_matrix.png` | Genre-cluster confusion matrix |
| `distributions.png` | Cluster and genre distributions |
| `latent_visualization/tsne_visualization.png` | t-SNE 2D projection |
| `latent_visualization/umap_visualization.png` | UMAP 2D projection |

### Models
| File | Description |
|------|-------------|
| `models/model_final.pth` | Final trained VAE model |
| `models/best_model.pth` | Best model checkpoint (lowest loss) |

### Features
| File | Description |
|------|-------------|
| `features/latent_features.npy` | 32-dim latent features for all songs |
| `features/cluster_labels_kmeans.npy` | K-Means cluster assignments |
| `features/genre_labels.npy` | Encoded genre labels |
| `features/tsne_2d.npy` | 2D t-SNE coordinates |
