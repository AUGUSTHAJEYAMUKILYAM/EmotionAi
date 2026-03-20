
# transcribe.py — Multilingual speech-to-text using OpenAI Whisper via Replit AI Integrations
# EmotionSense: AI Emotion Detection for Emergency Calls
#
# Uses gpt-4o-mini-transcribe for fast, accurate multilingual transcription.
# Supports 50+ languages automatically via Whisper's language detection.
#
# How to run the app:
#   streamlit run app.py

import os
from openai import OpenAI

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


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client configured for Replit AI Integrations.
    Uses AI_INTEGRATIONS_OPENAI_BASE_URL and AI_INTEGRATIONS_OPENAI_API_KEY
    environment variables (automatically set by Replit — no manual setup needed).
    """
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY", "replit")

    if not base_url:
        raise RuntimeError(
            "AI_INTEGRATIONS_OPENAI_BASE_URL is not set. "
            "Ensure the OpenAI AI Integration is configured in your Replit project."
        )

    return OpenAI(base_url=base_url, api_key=api_key)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using OpenAI's Whisper (gpt-4o-mini-transcribe).
    Automatically detects the spoken language — supports 50+ languages.

    Args:
        audio_path: Path to a local audio file (.wav, .mp3, .m4a, etc.)

    Returns:
        dict with keys:
            - "text" (str): The transcribed text
            - "language" (str): Detected language code (e.g. "en", "fr", "ar")
            - "language_name" (str): Human-readable language name
    """
    client = get_openai_client()

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            response_format="json",
        )

    language_code = getattr(response, "language", None) or "unknown"
    language_name = LANGUAGE_NAMES.get(language_code, language_code.capitalize())

    return {
        "text": response.text.strip(),
        "language": language_code,
        "language_name": language_name,
    }
