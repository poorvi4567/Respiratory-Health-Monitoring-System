# streamlit_app.py

import streamlit as st
import numpy as np
import librosa
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Load your model and preprocessors (assumed to be saved previously) ---
# These can be saved using joblib or pickle after training
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# Load trained model, scaler, and encoder
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# --- Feature extraction function (same as in training) ---
def extract_features(file):
    y, sr = librosa.load(file, duration=5, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# --- Streamlit UI ---
st.title("🫁 Respiratory Sound Disease Classifier")
st.markdown("Upload a breathing sound (.wav) to detect respiratory disease.")

uploaded_file = st.file_uploader("Upload audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Extract features and preprocess
    try:
        with st.spinner('Analyzing audio...'):
            features = extract_features(uploaded_file)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🩺 Predicted Disease: **{predicted_label}**")
    except Exception as e:
        st.error(f"Error processing file: {e}")
