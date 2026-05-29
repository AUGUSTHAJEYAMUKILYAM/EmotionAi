# transcribe.py — Multilingual speech-to-text using Groq Whisper API
# EmotionSense: AI Emotion Detection for Emergency Calls
#
# Uses Groq's free Whisper API for fast, accurate multilingual transcription.
# Supports 50+ languages automatically via Whisper's language detection.
#
# How to run the app:
#   streamlit run app.py

import os
import streamlit as st
from groq import Groq

# Language code → human-readable name mapping for display purposes
LANGUAGE_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "hy": "Armenian", "az": "Azerbaijani",
    "be": "Belarusian", "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan",
    "zh": "Chinese", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "gl": "Galician", "de": "German", "el": "Greek",
    "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada",
    "kk": "Kazakh", "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian",
    "mk": "Macedonian", "ms": "Malay", "mr": "Marathi", "mi": "Maori",
    "ne": "Nepali", "no": "Norwegian", "fa": "Persian", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "es": "Spanish", "sw": "Swahili",
    "sv": "Swedish", "tl": "Filipino", "ta": "Tamil", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "cy": "Welsh",
}

def get_groq_client() -> Groq:
    """
    Create a Groq client using the GROQ_API_KEY from Streamlit secrets
    or environment variables.
    """
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your Streamlit secrets or environment variables."
        )
    return Groq(api_key=api_key)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using Groq's Whisper API.
    Automatically detects the spoken language — supports 50+ languages.

    Args:
        audio_path: Path to a local audio file (.wav, .mp3, .m4a, etc.)

    Returns:
        dict with keys:
            - "text" (str): The transcribed text
            - "language" (str): Detected language code (e.g. "en", "fr", "ta")
            - "language_name" (str): Human-readable language name
    """
    client = get_groq_client()

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="verbose_json",
        )

    language_code = getattr(response, "language", None) or "unknown"
    language_name = LANGUAGE_NAMES.get(language_code, language_code.capitalize())

    return {
        "text": response.text.strip(),
        "language": language_code,
        "language_name": language_name,
    }
