"""
Dataset Utilities for Music Clustering

This module handles data loading, audio feature extraction,
text processing, and data preparation for the VAE model.
"""

import os
import re
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_track_name(name):
    """
    Clean track name for matching with audio files.
    
    Args:
        name: Original track name
    
    Returns:
        Cleaned lowercase name with only alphanumeric and underscores
    """
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def extract_audio_features(file_path, n_mfcc=20, max_len=130):
    """
    Extract MFCC + Chroma + Spectral Contrast features from audio.
    
    Args:
        file_path: Path to the audio file
        n_mfcc: Number of MFCC coefficients (default: 20)
        max_len: Maximum time steps to pad/truncate to (default: 130)
    
    Returns:
        Feature matrix of shape (39, max_len) or None if extraction fails
        39 = 20 MFCC + 12 Chroma + 7 Spectral Contrast
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        # MFCC (20 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Chroma (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Spectral Contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Stack: 20 + 12 + 7 = 39 features
        combined = np.vstack([mfcc, chroma, contrast])
        
        # Pad or truncate time dimension
        if combined.shape[1] < max_len:
            combined = np.pad(combined, ((0, 0), (0, max_len - combined.shape[1])), mode='constant')
        else:
            combined = combined[:, :max_len]
        
        return combined
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def create_track_lookup(df):
    """
    Create a lookup dictionary from CSV dataframe.
    
    Args:
        df: DataFrame with columns ['track_name', 'lyrics', 'playlist_genre']
    
    Returns:
        Dictionary mapping cleaned track names to their metadata
    """
    df = df.copy()
    df['clean_name'] = df['track_name'].apply(clean_track_name)
    df['lyrics'] = df['lyrics'].fillna("")
    
    track_lookup = {}
    for _, row in df.iterrows():
        track_lookup[row['clean_name']] = {
            'lyrics': row['lyrics'],
            'genre': row['playlist_genre'],
            'original_name': row['track_name']
        }
    
    return track_lookup


def load_and_match_data(csv_path, audio_path, n_mfcc=20, max_len=130):
    """
    Load CSV data, extract audio features, and match audio files to metadata.
    
    Args:
        csv_path: Path to the CSV file with track metadata
        audio_path: Path to directory containing audio files
        n_mfcc: Number of MFCC coefficients
        max_len: Maximum time steps
    
    Returns:
        X_audio: Audio feature array (N, n_features, max_len)
        X_lyrics: List of lyrics strings
        valid_genres: List of genre strings
        valid_files: List of matched audio filenames
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} songs from CSV")
    
    # Create lookup
    track_lookup = create_track_lookup(df)
    print(f"Created lookup for {len(track_lookup)} tracks")
    
    # Get audio files
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files")
    
    # Extract and match
    X_audio_list, X_lyrics_list, valid_genres, valid_files = [], [], [], []
    matched, unmatched = 0, 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        clean = audio_file.replace('.wav', '').replace('_30s', '')
        
        if clean in track_lookup:
            file_path = os.path.join(audio_path, audio_file)
            features = extract_audio_features(file_path, n_mfcc, max_len)
            
            if features is not None:
                X_audio_list.append(features)
                X_lyrics_list.append(track_lookup[clean]['lyrics'])
                valid_genres.append(track_lookup[clean]['genre'])
                valid_files.append(audio_file)
                matched += 1
        else:
            unmatched += 1
    
    print(f"Matched: {matched}, Unmatched: {unmatched}")
    
    X_audio = np.array(X_audio_list)
    return X_audio, X_lyrics_list, valid_genres, valid_files


def process_text_features(lyrics_list, max_features=500, text_dim=64):
    """
    Process lyrics with TF-IDF and reduce dimensionality with PCA.
    
    Args:
        lyrics_list: List of lyrics strings
        max_features: Maximum TF-IDF features
        text_dim: Target dimension after PCA
    
    Returns:
        X_text: Processed text features (N, text_dim)
        vectorizer: Fitted TfidfVectorizer
        pca: Fitted PCA transformer
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', min_df=2)
    tfidf_matrix = vectorizer.fit_transform(lyrics_list).toarray()
    
    pca = PCA(n_components=min(text_dim, tfidf_matrix.shape[1]))
    X_text = pca.fit_transform(tfidf_matrix)
    
    print(f"Text features shape: {X_text.shape}")
    return X_text, vectorizer, pca


def prepare_data_for_training(X_audio, X_text, genres):
    """
    Normalize features and encode labels for training.
    
    Args:
        X_audio: Audio features (N, n_features, max_len)
        X_text: Text features (N, text_dim)
        genres: List of genre labels
    
    Returns:
        X_audio_cnn: Normalized audio for CNN (N, 1, n_features, max_len)
        X_text_scaled: Normalized text features
        y_genres: Encoded genre labels
        label_encoder: Fitted LabelEncoder
        scaler_audio: Fitted StandardScaler for audio
        scaler_text: Fitted StandardScaler for text
    """
    N, n_feat, max_len = X_audio.shape
    
    # Flatten, normalize, reshape for CNN
    X_audio_flat = X_audio.reshape(N, -1)
    scaler_audio = StandardScaler()
    X_audio_scaled = scaler_audio.fit_transform(X_audio_flat)
    X_audio_cnn = X_audio_scaled.reshape(N, 1, n_feat, max_len)
    
    # Normalize text
    scaler_text = StandardScaler()
    X_text_scaled = scaler_text.fit_transform(X_text)
    
    # Encode genres
    label_encoder = LabelEncoder()
    y_genres = label_encoder.fit_transform(genres)
    
    print(f"Audio shape: {X_audio_cnn.shape}")
    print(f"Text shape: {X_text_scaled.shape}")
    print(f"Genres: {list(label_encoder.classes_)}")
    
    return X_audio_cnn, X_text_scaled, y_genres, label_encoder, scaler_audio, scaler_text
