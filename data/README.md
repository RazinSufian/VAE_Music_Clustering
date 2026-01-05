# 📁 Dataset Instructions

This folder should contain the dataset for the VAE Music Clustering project.

## Required Files

```
data/
├── Music Dataset - Sheet1.csv    # Song metadata with lyrics
└── wav_files/                    # Audio files (30-second clips)
    ├── song1_30s.wav
    ├── song2_30s.wav
    └── ... (2891 files)
```

## Dataset Description

| File | Description | Size |
|------|-------------|------|
| `Music Dataset - Sheet1.csv` | Contains track_name, lyrics, playlist_genre | ~6.3 MB |
| `wav_files/` | 30-second audio clips in WAV format | ~2 GB |

### CSV Columns

- `track_name`: Name of the song
- `lyrics`: Song lyrics (text)
- `playlist_genre`: Genre label (edm, latin, pop, r&b, rap, rock)

### Audio Files

- Format: WAV (mono, 22050 Hz)
- Duration: 30 seconds each
- Naming: `{track_name}_30s.wav`

## 🔽 How to Obtain the Dataset

### Option 1: Use Your Own Dataset

If you have your own music dataset:
1. Prepare audio files as 30-second WAV clips
2. Create a CSV with columns: `track_name`, `lyrics`, `playlist_genre`
3. Name audio files to match the `track_name` column

### Option 2: Download from Course Materials

If this is for CSE425:
1. Download the dataset from the course portal
2. Extract to this `data/` folder

### Option 3: Use Public Datasets

You can adapt public music datasets:

- **GTZAN Genre Collection**: http://marsyas.info/downloads/datasets.html
- **Million Song Dataset**: http://millionsongdataset.com/
- **Jamendo Dataset**: https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset

## 📂 Setting Up for Google Colab

1. Upload the dataset to your Google Drive:
   ```
   Google Drive/
   └── VAE_Music_Clustering_Project/
       ├── Music Dataset - Sheet1.csv
       └── wav_files/
           └── *.wav
   ```

2. Update the path in the notebook:
   ```python
   DRIVE_PATH = '/content/drive/MyDrive/VAE_Music_Clustering_Project'
   ```

## ⚠️ Important Notes

- Audio files are **NOT included** in this repository due to size (~2 GB)
- The CSV file may or may not be included depending on licensing
- Make sure audio filenames match the `track_name` column in the CSV

## 🔗 Matching Logic

The code matches audio files to CSV entries using:

```python
def clean_track_name(name):
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

# Audio: "24k_magic_30s.wav" → "24k_magic"
# CSV:   "24K Magic" → "24k_magic"
# Result: ✅ Match!
```
