import os
import sys
import librosa
import numpy as np
import pyaudio
import wave
import speech_recognition as sr
import joblib
import pandas as pd
from utils import extract_features
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
# -------------------- Load Trained Models --------------------
MODEL_PATH = "speech_emotion_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found! Train the model first using train_model.py")

# Load models, scaler, and label encoder
stacking_model, mlp_model, lgbm_model, xgb_model, scaler, label_encoder = joblib.load(MODEL_PATH)

# -------------------- Real-Time Audio Recording --------------------
def record_audio(filename="live_audio.wav", duration=5, rate=22050, channels=1):
    """Records audio for 5 seconds and saves it to a file."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=1024)
    frames = []
    print("\U0001F3A4 Recording... (5 sec)")

    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("✅ Recording complete!")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # ✅ Save as a WAV file
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    # ✅ Check if recorded audio is valid
    audio, sr = librosa.load(filename, sr=rate, mono=True)
    if len(audio) == 0:
        raise ValueError("Recorded audio is empty! Check your microphone and recording settings.")

    return filename  # ✅ Return filename instead of raw audio data

# -------------------- Predict Emotion --------------------
import os

def predict_emotion(filename="live_audio.wav"):
    """Predicts emotion using the trained StackingClassifier."""
    try:
        filename = os.path.abspath(filename)  # ✅ Ensure absolute path
        print(f"🔍 Processing file: {filename}")  # ✅ Debugging

        if not os.path.exists(filename):
            print(f"❌ File does not exist: {filename}")
            return "Unknown"

        if os.path.getsize(filename) < 1024:  # Less than 1 KB
            print(f"❌ File too small or empty: {filename}")
            return "Unknown"

        # ✅ Load and preprocess the audio file
        audio, sr = librosa.load(filename, sr=22050, mono=True)
        if len(audio) == 0:
            print(f"❌ Audio file is empty: {filename}")
            return "Unknown"

        # ✅ Extract features correctly
        features = extract_features(filename).reshape(1, -1)

        # ✅ Ensure correct feature shape before transformation
        if features.shape[1] != scaler.n_features_in_:
            print(f"❌ Feature mismatch: Expected {scaler.n_features_in_}, got {features.shape[1]}")
            return "Unknown"

        # ✅ Scale the features
        features_scaled = scaler.transform(features)

        # ✅ Use trained Stacking Model
        stacking_pred = stacking_model.predict(features_scaled)[0]

        # ✅ Convert label to emotion
        emotion = label_encoder.inverse_transform([stacking_pred])[0]
        print(f"🎭 Predicted Emotion: {emotion}")  # ✅ Debugging
        return emotion

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return "Unknown"



# -------------------- Speech-to-Text Transcription --------------------
def speech_to_text(filename="live_audio.wav"):
    """Converts speech to text using SpeechRecognition with better handling."""
    recognizer = sr.Recognizer()

    try:
        filename = os.path.abspath(filename)  # ✅ Ensure absolute path
        print(f"🔍 Transcribing file: {filename}")  # ✅ Debugging

        with sr.AudioFile(filename) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)  # ✅ Handle background noise
            audio_data = recognizer.record(source)  # ✅ Capture audio for processing

        if len(audio_data.frame_data) == 0:
            print("❌ No speech detected.")
            return "No speech detected"

        text = recognizer.recognize_google(audio_data)
        print(f"📝 Transcribed Text: {text}")  # ✅ Debugging
        return text

    except sr.UnknownValueError:
        print("❌ Speech not recognized.")
        return "Speech not recognized"
    except sr.RequestError:
        print("❌ Speech recognition service unavailable.")
        return "Speech recognition service unavailable"
    except Exception as e:
        print(f"❌ Transcription Error: {e}")
        return f"Error in transcription: {e}"  # ✅ Debugging



# -------------------- Run Real-Time Recognition --------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]  # Use provided file
    else:
        filename = record_audio()  # 🔹 This will record new audio

    print(f"🔍 Predicting emotion for: {filename}", file=sys.stderr)
    
    emotion = predict_emotion(filename)
    transcript = speech_to_text(filename)

    sys.stderr.flush()
    sys.stdout.flush()

    print(json.dumps({"emotion": emotion, "transcription": transcript}))