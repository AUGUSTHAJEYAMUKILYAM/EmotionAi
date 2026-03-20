
# utils.py — Audio feature extraction utilities
# EmotionSense: Speech Emotion Recognition for Emergency Calls
#
# How to run the app:
#   streamlit run app.py

import numpy as np
import librosa


def extract_features(audio_path: str) -> np.ndarray:
    """
    Extract audio features from a .wav file using librosa.

    Features extracted:
    - MFCCs (40 coefficients): capture spectral shape of speech
    - Pitch (fundamental frequency): related to emotional tone
    - Energy (RMS): relates to loudness/intensity of speech
    - Spectral centroid: brightness of sound
    - Zero crossing rate: roughness / noisiness of signal

    Returns:
        np.ndarray: 1D feature vector of shape (52,)
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # MFCCs — 40 coefficients, take mean over time frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)  # shape (40,)

    # Pitch (fundamental frequency via piptrack)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0.0

    # Energy (RMS root mean square)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(spectral_centroid)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Combine all into a single 1D feature vector (52 features total)
    features = np.concatenate([
        mfcc_mean,          # 40
        [pitch_mean],       # 1
        [pitch_std],        # 1
        [rms_mean],         # 1
        [rms_std],          # 1
        [centroid_mean],    # 1
        [zcr_mean],         # 1
        [zcr_std],          # 1
    ])

    return features
