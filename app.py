# app.py — Streamlit application entry point
# EmotionSense: AI Emotion Detection for Emergency Calls

import sys
import traceback
import os
import tempfile
import concurrent.futures

print("Starting EmotionSense app...", flush=True)

try:
    import matplotlib
    matplotlib.use('Agg')  # Fix for headless server
    import matplotlib.pyplot as plt
    print("matplotlib loaded", flush=True)
except Exception as e:
    print(f"matplotlib error: {e}", flush=True)

try:
    import numpy as np
    print("numpy loaded", flush=True)
except Exception as e:
    print(f"numpy error: {e}", flush=True)

try:
    import streamlit as st
    print("streamlit loaded", flush=True)
except Exception as e:
    print(f"streamlit error: {e}", flush=True)
    sys.exit(1)

try:
    from model import EMOTIONS, load_model, predict_emotion
    print("model loaded", flush=True)
except Exception as e:
    print(f"model error: {e}", flush=True)
    traceback.print_exc()
    st.error(f"Failed to load model: {e}")
    sys.exit(1)

try:
    from utils import extract_features
    print("utils loaded", flush=True)
except Exception as e:
    print(f"utils error: {e}", flush=True)
    traceback.print_exc()
    st.error(f"Failed to load utils: {e}")
    sys.exit(1)

try:
    from transcribe import transcribe_audio
    print("transcribe loaded", flush=True)
except Exception as e:
    print(f"transcribe error: {e}", flush=True)
    traceback.print_exc()
    st.error(f"Failed to load transcribe: {e}")
    sys.exit(1)

print("All imports successful!", flush=True)

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotionSense",
    page_icon="🎙️",
    layout="centered",
)

# ── Title & introduction ────────────────────────────────────────────────────
st.title("🎙️ EmotionSense – AI Emotion Detection for Emergency Calls")
st.markdown(
    """
    Upload a voice recording (**.wav**) to analyze the emotional state of the speaker.
    Powered by **Whisper AI** for multilingual transcription and a **RandomForest** classifier
    for real-time emotion recognition — suitable for any language.
    """
)

st.divider()

# ── Model loading (cached so it only trains once per session) ───────────────
@st.cache_resource(show_spinner="Initializing emotion recognition model…")
def get_model():
    return load_model()

try:
    clf, scaler = get_model()
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    traceback.print_exc()
    st.stop()

# ── File uploader ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
    help="Supports WAV, MP3, M4A, OGG, FLAC. Any language is supported.",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    st.caption(f"Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):

        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        transcription_result = None
        features = None
        predicted_emotion = None
        confidence_scores = None
        transcription_error = None
        feature_error = None

        with st.spinner("Transcribing audio and analyzing emotion…"):

            def run_transcription():
                return transcribe_audio(tmp_path)

            def run_feature_extraction():
                feats = extract_features(tmp_path)
                lbl, scores = predict_emotion(feats, clf, scaler)
                return feats, lbl, scores

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fut_transcription = executor.submit(run_transcription)
                fut_features = executor.submit(run_feature_extraction)

                try:
                    transcription_result = fut_transcription.result(timeout=60)
                except Exception as e:
                    transcription_error = str(e)
                    print(f"Transcription error: {e}", flush=True)

                try:
                    features, predicted_emotion, confidence_scores = fut_features.result(timeout=30)
                except Exception as e:
                    feature_error = str(e)
                    print(f"Feature extraction error: {e}", flush=True)

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        if feature_error:
            st.error(f"Could not analyze audio features: {feature_error}")
            st.stop()

        st.divider()

        # ── Prediction result ───────────────────────────────────────────────
        emotion_emoji = {
            "Angry": "😡",
            "Fear":  "😨",
            "Sad":   "😢",
            "Happy": "😄",
            "Neutral": "😐",
        }
        emoji = emotion_emoji.get(predicted_emotion, "🎙️")

        st.subheader("Predicted Emotion")
        st.markdown(
            f"<h2 style='text-align:center; margin-top:0'>{emoji} {predicted_emotion}</h2>",
            unsafe_allow_html=True,
        )

        if predicted_emotion in ("Angry", "Fear"):
            st.error(
                "⚠️ High distress detected. Operator attention required.",
                icon="🚨",
            )

        st.divider()

        # ── Transcription ───────────────────────────────────────────────────
        st.subheader("🌐 Multilingual Transcription (Whisper AI)")

        if transcription_error:
            st.warning(f"Transcription unavailable: {transcription_error}", icon="⚠️")
        elif transcription_result:
            lang_name = transcription_result.get("language_name", "Unknown")
            lang_code = transcription_result.get("language", "")
            text = transcription_result.get("text", "")

            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Detected Language", lang_name)
            with col2:
                if text:
                    st.text_area(
                        "Transcript",
                        value=text,
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                else:
                    st.info("No speech detected in the audio.", icon="ℹ️")
        else:
            st.info("Transcription not available.", icon="ℹ️")

        st.divider()

        # ── Confidence chart ────────────────────────────────────────────────
        st.subheader("Emotion Confidence Scores")

        emotions_list = list(confidence_scores.keys())
        scores = [confidence_scores[e] * 100 for e in emotions_list]

        colors = []
        for e in emotions_list:
            if e == predicted_emotion:
                colors.append("#E74C3C" if predicted_emotion in ("Angry", "Fear") else "#2ECC71")
            else:
                colors.append("#AEB6BF")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(emotions_list, scores, color=colors, edgecolor="none", height=0.55)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%",
                va="center",
                fontsize=10,
                color="#333333",
            )

        ax.set_xlim(0, 115)
        ax.set_xlabel("Confidence (%)", fontsize=10)
        ax.tick_params(axis="y", labelsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("#F8F9FA")

        st.pyplot(fig)
        plt.close(fig)

        # ── Feature summary ─────────────────────────────────────────────────
        with st.expander("📊 View extracted audio features"):
            import pandas as pd

            summary = {
                "Pitch Mean (Hz)":         round(float(features[40]), 2),
                "Pitch Std Dev (Hz)":      round(float(features[41]), 2),
                "Energy (RMS Mean)":       round(float(features[42]), 5),
                "Energy Std Dev":          round(float(features[43]), 5),
                "Spectral Centroid (Hz)":  round(float(features[44]), 2),
                "Zero Crossing Rate":      round(float(features[45]), 5),
            }
            df = pd.DataFrame(summary.items(), columns=["Feature", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info(
        "👆 Upload an audio file to get started. Any language is supported — "
        "Whisper AI will automatically detect and transcribe the spoken language.",
        icon="ℹ️",
    )

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "EmotionSense uses Groq Whisper for multilingual transcription and a RandomForest "
    "classifier on MFCC/pitch/energy features for emotion detection. "
    "For demonstration purposes — not a substitute for professional emergency services."
)
