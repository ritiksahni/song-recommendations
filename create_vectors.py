import librosa
import numpy as np
import faiss
import os

def extract_audio_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    
    # Extract MFCCs (Timbre)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)  # Averaging across time

    # Extract Chroma (Pitch Content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Extract Spectral Contrast (High vs. Low Energy Bands)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)

    # Extract Tempo (Rhythm)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Extract RMS Energy (Loudness)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    feature_vector = np.hstack([
        mfccs_mean,
        chroma_mean,
        spec_contrast_mean,
        tempo,
        rms_mean
    ])  

    return feature_vector



