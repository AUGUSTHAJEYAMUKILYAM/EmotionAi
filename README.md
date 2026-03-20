# EmotionSense – AI Emotion Detection for Emergency Calls

A Streamlit web application that analyzes speech emotion from audio files using OpenAI Whisper for multilingual transcription and a RandomForest classifier for emotion detection.

## Features

- **Multilingual Support**: Powered by OpenAI Whisper (`gpt-4o-mini-transcribe`), supporting 50+ languages with automatic language detection
- **Emotion Detection**: Classifies speech into 5 emotions — Angry, Fear, Sad, Happy, Neutral
- **High-Distress Alert**: Automatically flags calls with Angry or Fear emotion
- **Confidence Visualization**: Bar chart showing prediction confidence for each emotion
- **Audio Feature Analysis**: Displays extracted MFCC, pitch, energy, and spectral features

## Project Structure

```
├── app.py           # Main Streamlit application
├── model.py         # RandomForest emotion classifier
├── utils.py         # Audio feature extraction (librosa)
├── transcribe.py    # Whisper AI multilingual transcription
├── requirements.txt # Python dependencies
└── .streamlit/
    └── config.toml  # Streamlit server configuration
```

## Requirements

```
streamlit
librosa
numpy
scikit-learn
pandas
matplotlib
soundfile
openai
```

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

For Whisper transcription, set the following environment variables (provided automatically on Replit via AI Integrations):

```
AI_INTEGRATIONS_OPENAI_BASE_URL=<your_openai_base_url>
AI_INTEGRATIONS_OPENAI_API_KEY=<your_openai_api_key>
```

If running outside Replit, set `OPENAI_API_KEY` and update `transcribe.py` to use the standard OpenAI client initialization.

## Emotions Detected

| Emotion | Description |
|---------|-------------|
| Angry | High energy, aggressive speech |
| Fear | High pitch, trembling voice |
| Sad | Low energy, slow speech |
| Happy | Bright spectrum, moderate-high energy |
| Neutral | Balanced, calm speech |

## Alert System

If **Fear** or **Angry** is detected, a red alert is displayed:

> ⚠️ High distress detected. Operator attention required.

---

*For demonstration purposes — not a substitute for professional emergency services.*
