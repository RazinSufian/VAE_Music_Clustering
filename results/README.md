# 📊 Results

This folder contains the output files from the VAE Music Clustering experiment.

## 📁 Files

### Visualizations

| File | Description |
|------|-------------|
| `training_curves.png` | Training loss curves (total, audio, text, KL) |
| `reconstruction_examples.png` | Original vs reconstructed spectrograms |
| `cluster_selection.png` | Silhouette analysis for optimal K |
| `confusion_matrix.png` | Genre-cluster confusion matrix |
| `distributions.png` | Cluster and genre distributions |

### Latent Space Visualizations (`latent_visualization/`)

| File | Description |
|------|-------------|
| `tsne_visualization.png` | t-SNE 2D projection with clusters and genres |
| `umap_visualization.png` | UMAP 2D projection with clusters and genres |

### Metrics

| File | Description |
|------|-------------|
| `clustering_metrics.csv` | All evaluation metrics for each method |
| `training_history.csv` | Loss values per epoch |

### Model & Data

| File | Description |
|------|-------------|
| `model_final.pth` | Trained VAE model weights |
| `best_model.pth` | Best model checkpoint (lowest loss) |
| `latent_features.npy` | 32-dim latent features for all songs |
| `cluster_labels_kmeans.npy` | K-Means cluster assignments |
| `genre_labels.npy` | Encoded genre labels |
| `tsne_2d.npy` | 2D t-SNE coordinates |

## 📈 Clustering Results

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

## 📊 Summary

- **Dataset**: 2,890 songs across 6 genres
- **Features**: 39 audio (MFCC+Chroma+Spectral) + 64 text (TF-IDF→PCA)
- **Latent Dim**: 32
- **Optimal K**: 4 clusters
- **Best Loss**: 1.0371
- **Model Parameters**: 1,256,897
