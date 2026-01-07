# 📁 Dataset

This folder contains the dataset for the VAE Music Clustering project.

## 📥 Download Links

### Audio Files (WAV)
> **Google Drive**: [Download wav_files (~2GB)](https://drive.google.com/drive/folders/1Vkr92gfxhmQvyf0wFdiIuj0popHXyqUI?usp=sharing)

Download the `wav_files` folder and place it in the `data/` directory:
```
data/
├── audio/
│   └── wav_files/
│       ├── song1_30s.wav
│       ├── song2_30s.wav
│       └── ... (2891 files)
└── lyrics/
    └── Music Dataset - Sheet1.csv
```

### CSV Metadata
The `Music Dataset - Sheet1.csv` file is included in this repository.

## 📊 Dataset Description

| Component | Description | Size |
|-----------|-------------|------|
| `Music Dataset - Sheet1.csv` | Track name, lyrics, genre | ~6.3 MB |
| `wav_files/` | 30-second audio clips | ~2 GB |

### Statistics
- **Total Songs**: 3,103 in CSV, 2,891 audio files
- **Matched Songs**: 2,890 (99.97% match rate)
- **Genres**: 6 (pop, rock, rap, r&b, edm, latin)

### Genre Distribution
| Genre | Count |
|-------|-------|
| pop | 673 |
| rock | 606 |
| rap | 586 |
| r&b | 543 |
| edm | 350 |
| latin | 345 |

### CSV Columns
| Column | Type | Description |
|--------|------|-------------|
| `track_name` | string | Name of the song |
| `lyrics` | string | Full lyrics text |
| `playlist_genre` | string | Genre category |

### Audio Files
- **Format**: WAV (mono)
- **Sample Rate**: 22,050 Hz
- **Duration**: 30 seconds each
- **Naming**: `{track_name}_30s.wav`

## 🔗 File Matching

The code matches audio files to CSV entries using cleaned track names:

```python
def clean_track_name(name):
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

# Example:
# CSV:   "24K Magic" → "24k_magic"
# Audio: "24k_magic_30s.wav" → "24k_magic"
# Result: ✅ Match!
```

## 🚀 Setup for Google Colab

1. Download wav_files from the Drive link above
2. Upload to your Google Drive:
   ```
   Google Drive/
   └── VAE_Music_Clustering_Project/
       ├── Music Dataset - Sheet1.csv
       └── wav_files/
           └── *.wav
   ```

3. Update path in notebook:
   ```python
   DRIVE_PATH = '/content/drive/MyDrive/VAE_Music_Clustering_Project'
   ```
