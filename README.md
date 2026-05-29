# 🎙️ EmotionSense — AI Speech Emotion Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> An AI-powered speech emotion recognition system that analyzes voice recordings to detect human emotions in real time — built for emergency call monitoring and beyond.

---

## 🌟 What is EmotionSense?

EmotionSense listens to a voice recording and identifies **what emotion the speaker is feeling** — instantly. It's designed with emergency call centers in mind, where detecting distress early can make a real difference.

When **Fear** or **Anger** is detected, the system triggers an automatic **high-distress alert** so operators can respond faster.

---

## ✨ Features

- 🌍 **Multilingual Support** — powered by OpenAI Whisper, supporting 50+ languages with automatic detection
- 🧠 **5-Emotion Classification** — detects Angry, Fear, Sad, Happy, and Neutral
- 🚨 **High-Distress Alert** — automatically flags calls with Fear or Angry emotion
- 📊 **Confidence Visualization** — bar chart showing prediction confidence for each emotion
- 🎵 **Audio Feature Analysis** — displays extracted MFCC, pitch, energy, and spectral features
- ⚡ **Real-time Processing** — analyze uploaded audio files instantly

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web application framework |
| **OpenAI Whisper** | Multilingual speech-to-text transcription |
| **RandomForest Classifier** | Emotion prediction model |
| **Librosa** | Audio feature extraction (MFCC, pitch, energy) |
| **scikit-learn** | Machine learning pipeline |
| **Matplotlib** | Confidence visualization charts |

---

## 📁 Project Structure

```
EmotionSense/
├── app.py              # Main Streamlit application & UI
├── model.py            # RandomForest emotion classifier
├── utils.py            # Audio feature extraction (MFCC, pitch, energy)
├── transcribe.py       # OpenAI Whisper multilingual transcription
├── requirements.txt    # Python dependencies
└── .streamlit/
    └── config.toml     # Streamlit server configuration
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API Key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AUGUSTHAJEYAMUKILYAM/EmotionAi.git
cd EmotionAi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# 4. Run the app
streamlit run app.py
```

Then open your browser at `http://localhost:8501` 🎉

---

## 🎭 Emotions Detected

| Emotion | Voice Characteristics |
|---|---|
| 😠 **Angry** | High energy, aggressive speech patterns |
| 😨 **Fear** | High pitch, trembling or shaky voice |
| 😢 **Sad** | Low energy, slow and heavy speech |
| 😊 **Happy** | Bright spectrum, moderate-high energy |
| 😐 **Neutral** | Balanced, calm speech |

---

## 🚨 Alert System

When **Fear** or **Angry** is detected, the system displays:

> ⚠️ **High distress detected. Operator attention required.**

This is especially useful in emergency call monitoring scenarios where quick human response matters.

---

## 🔬 How It Works

```
Audio Input
    ↓
Feature Extraction (Librosa)
→ MFCC, Pitch, Energy, Spectral features
    ↓
Speech Transcription (OpenAI Whisper)
→ Multilingual text output
    ↓
Emotion Classification (RandomForest)
→ Predicts emotion with confidence score
    ↓
Result Display (Streamlit)
→ Emotion label + confidence chart + alert if needed
```

---

## 🔮 Future Improvements

- [ ] Live microphone input for real-time emotion detection
- [ ] Emotion history log with timestamps
- [ ] Support for batch audio file processing
- [ ] Export reports as PDF
- [ ] Integration with emergency call center APIs
- [ ] Model accuracy improvement with larger datasets

---

## 👩‍💻 About the Developer

**Augustha Jeya Mukilya M**
- 🎓 Student | Data Analytics Enthusiast
- 💼 Data Analytics Intern @ AAA Techno Park Pvt. Ltd.
- 🌍 GitHub: [@AUGUSTHAJEYAMUKILYAM](https://github.com/AUGUSTHAJEYAMUKILYAM)

---

## ⚠️ Disclaimer

*This project is for demonstration and educational purposes only. It is not a substitute for professional emergency services.*

---

## 📄 License

This project is licensed under the MIT License — feel free to use and build on it!

---

⭐ **If you found this project interesting, please star the repo!** ⭐
