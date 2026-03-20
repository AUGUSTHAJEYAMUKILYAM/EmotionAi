
# model.py — Emotion prediction logic
# EmotionSense: Speech Emotion Recognition for Emergency Calls
#
# How to run the app:
#   streamlit run app.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

EMOTIONS = ["Angry", "Fear", "Sad", "Happy", "Neutral"]
N_FEATURES = 47  # must match utils.py extract_features output size
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")


def _generate_synthetic_training_data(n_samples: int = 1500):
    """
    Generate synthetic but plausible training data for each emotion.

    Each emotion is given distinct statistical properties in feature space
    to simulate real-world acoustic differences between emotional states:
    - Angry:   high energy, high pitch variance, high zero crossing rate
    - Fear:    high pitch, trembling (high std), moderate energy
    - Sad:     low energy, low pitch, slow speech
    - Happy:   high energy, bright spectrum (high centroid), moderate pitch
    - Neutral: moderate all features, low variance

    Returns:
        X (np.ndarray): Feature matrix (n_samples, N_FEATURES)
        y (np.ndarray): Emotion labels (n_samples,) as integers
    """
    rng = np.random.default_rng(seed=42)
    X_parts = []
    y_parts = []

    samples_per_class = n_samples // len(EMOTIONS)

    # Feature index reference (see utils.py):
    # 0–39: MFCCs, 40: pitch_mean, 41: pitch_std, 42: rms_mean,
    # 43: rms_std, 44: centroid_mean, 45: zcr_mean, 46: zcr_std

    emotion_params = {
        "Angry": {
            "mfcc_mean": [2.0] + [-0.5] * 39,
            "mfcc_std": 3.5,
            "pitch_mean": 220.0, "pitch_std_val": 80.0,
            "rms_mean": 0.18, "rms_std_val": 0.05,
            "centroid_mean": 3000.0, "zcr_mean": 0.12, "zcr_std_val": 0.04,
        },
        "Fear": {
            "mfcc_mean": [-1.5] + [0.3] * 39,
            "mfcc_std": 3.0,
            "pitch_mean": 260.0, "pitch_std_val": 110.0,
            "rms_mean": 0.10, "rms_std_val": 0.04,
            "centroid_mean": 2400.0, "zcr_mean": 0.10, "zcr_std_val": 0.035,
        },
        "Sad": {
            "mfcc_mean": [-3.0] + [-1.0] * 39,
            "mfcc_std": 2.0,
            "pitch_mean": 140.0, "pitch_std_val": 30.0,
            "rms_mean": 0.05, "rms_std_val": 0.02,
            "centroid_mean": 1500.0, "zcr_mean": 0.05, "zcr_std_val": 0.02,
        },
        "Happy": {
            "mfcc_mean": [1.0] + [0.8] * 39,
            "mfcc_std": 3.0,
            "pitch_mean": 210.0, "pitch_std_val": 60.0,
            "rms_mean": 0.14, "rms_std_val": 0.04,
            "centroid_mean": 3500.0, "zcr_mean": 0.09, "zcr_std_val": 0.03,
        },
        "Neutral": {
            "mfcc_mean": [0.0] * 40,
            "mfcc_std": 1.5,
            "pitch_mean": 175.0, "pitch_std_val": 25.0,
            "rms_mean": 0.08, "rms_std_val": 0.02,
            "centroid_mean": 2000.0, "zcr_mean": 0.07, "zcr_std_val": 0.02,
        },
    }

    for label_idx, emotion in enumerate(EMOTIONS):
        p = emotion_params[emotion]
        mfcc_base = np.array(p["mfcc_mean"])
        noise = rng.normal(0, p["mfcc_std"], size=(samples_per_class, 40))
        mfccs = mfcc_base[None, :] + noise

        pitch_mean_col = rng.normal(p["pitch_mean"], p["pitch_std_val"] * 0.3, size=(samples_per_class, 1))
        pitch_std_col = np.abs(rng.normal(p["pitch_std_val"], 15.0, size=(samples_per_class, 1)))
        rms_mean_col = np.abs(rng.normal(p["rms_mean"], 0.01, size=(samples_per_class, 1)))
        rms_std_col = np.abs(rng.normal(p["rms_std_val"], 0.005, size=(samples_per_class, 1)))
        centroid_col = rng.normal(p["centroid_mean"], 200.0, size=(samples_per_class, 1))
        zcr_mean_col = np.abs(rng.normal(p["zcr_mean"], 0.01, size=(samples_per_class, 1)))
        zcr_std_col = np.abs(rng.normal(p["zcr_std_val"], 0.005, size=(samples_per_class, 1)))

        X_class = np.hstack([
            mfccs, pitch_mean_col, pitch_std_col,
            rms_mean_col, rms_std_col,
            centroid_col, zcr_mean_col, zcr_std_col
        ])

        X_parts.append(X_class)
        y_parts.append(np.full(samples_per_class, label_idx, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def _train_and_save_model():
    """Train a RandomForest classifier on synthetic data and persist to disk."""
    X, y = _generate_synthetic_training_data(n_samples=1500)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return clf, scaler


def load_model():
    """Load (or train + cache) the RandomForest classifier and scaler."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        clf, scaler = _train_and_save_model()
    return clf, scaler


def predict_emotion(features: np.ndarray, clf, scaler):
    """
    Predict the emotion label and confidence scores for the given feature vector.

    Args:
        features: 1D numpy array of shape (N_FEATURES,)
        clf: Trained RandomForest classifier
        scaler: Fitted StandardScaler

    Returns:
        predicted_label (str): The most likely emotion name
        confidence_scores (dict): {emotion: probability} for all 5 emotions
    """
    features_scaled = scaler.transform(features.reshape(1, -1))
    proba = clf.predict_proba(features_scaled)[0]

    confidence_scores = {EMOTIONS[i]: float(proba[i]) for i in range(len(EMOTIONS))}
    predicted_label = EMOTIONS[int(np.argmax(proba))]

    return predicted_label, confidence_scores
