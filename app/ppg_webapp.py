
import streamlit as st
import numpy as np
import joblib
import scipy.signal as signal
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
import pandas as pd

model_path = "models/ppg_quality_model_xgboost.joblib"
clf = joblib.load(model_path)

def extract_time_features(ppg):
    peaks, _ = signal.find_peaks(ppg, distance=30)
    intervals = np.diff(peaks)
    if len(intervals) < 2:
        return [0] * 4
    return [np.mean(intervals), np.std(intervals), np.min(intervals), np.max(intervals)]

def extract_frequency_features(ppg, fs=100):
    freqs = rfftfreq(len(ppg), 1/fs)
    fft_vals = np.abs(rfft(ppg))
    dominant_freq = freqs[np.argmax(fft_vals)]
    spectral_entropy = -np.sum((fft_vals / np.sum(fft_vals)) * np.log2((fft_vals / np.sum(fft_vals)) + 1e-12))
    return [dominant_freq, spectral_entropy]

def extract_statistical_features(ppg):
    return [np.mean(ppg), np.std(ppg), stats.skew(ppg), stats.kurtosis(ppg)]

def extract_features(ppg):
    features = (
        extract_time_features(ppg) +
        extract_frequency_features(ppg) +
        extract_statistical_features(ppg)
    )
    column_names = [
        "mean_interval", "std_interval", "min_interval", "max_interval",
        "dominant_freq", "spectral_entropy",
        "mean_ppg", "std_ppg", "skew_ppg", "kurtosis_ppg"
    ]
    return pd.DataFrame([features], columns=column_names)

st.set_page_config(page_title="PPG Quality Classifier", layout="centered")
st.title("🩺 PPG Signal Quality Classifier")
st.write("Upload a `.npy` PPG signal to classify its quality using your trained XGBoost model.")

uploaded_file = st.file_uploader("📂 Choose a PPG .npy file", type=["npy"])

if uploaded_file:
    try:
        ppg = np.load(uploaded_file)
        st.subheader("📈 Signal Preview (First 500 Samples)")
        st.line_chart(ppg[:500])

        features_df = extract_features(ppg)
        prediction = clf.predict(features_df)[0]
        proba = clf.predict_proba(features_df)[0]
        confidence = proba[prediction]

        label = "✅ Good Quality" if prediction == 1 else "❌ Poor Quality"
        st.subheader("🧠 Prediction Result")
        st.markdown(f"**{label}**  \nConfidence: `{confidence:.2%}`")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
