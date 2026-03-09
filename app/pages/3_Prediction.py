import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.title("🔮 Voice Gender Prediction")

# -------------------------------
# LOAD MODELS
# -------------------------------

classifier = joblib.load("../models/classifier_model.pkl")
cluster = joblib.load("../models/cluster_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("../data/vocal_gender_features.csv")
features = df.drop("label", axis=1).columns

# -------------------------------
# AUDIO FEATURE EXTRACTION
# -------------------------------

def extract_features(audio_file):

    y, sr = librosa.load(audio_file)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]

    mean_pitch = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
    max_pitch = np.max(pitch_vals) if len(pitch_vals) > 0 else 0
    min_pitch = np.min(pitch_vals) if len(pitch_vals) > 0 else 0

    energy = np.sum(y**2)
    log_energy = np.log(energy + 1e-6)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

    feature_dict = {
        "mean_pitch": mean_pitch,
        "max_pitch": max_pitch,
        "min_pitch": min_pitch,
        "log_energy": log_energy,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "mean_spectral_contrast": contrast
    }

    return feature_dict, y, sr


# -------------------------------
# TABS
# -------------------------------

tab1, tab2, tab3 = st.tabs([
    "Manual Prediction",
    "Dataset Testing",
    "Audio Prediction"
])

# =========================================================
# TAB 1 MANUAL INPUT
# =========================================================

with tab1:

    st.subheader("Manual Feature Prediction")

    inputs = []

    for feature in features:
        val = st.number_input(feature, value=0.0)
        inputs.append(val)

    if st.button("Predict Manual"):

        input_df = pd.DataFrame([inputs], columns=features)

        scaled = scaler.transform(input_df)

        gender = classifier.predict(scaled)[0]
        cluster_id = cluster.predict(scaled)[0]

        st.write("Cluster:", cluster_id)

        if gender == 1:
            st.success("Male Voice")
        else:
            st.success("Female Voice")


# =========================================================
# TAB 2 DATASET TESTING
# =========================================================

with tab2:

    st.subheader("Dataset Row Testing")

    row_index = st.number_input(
        "Select Dataset Row",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    sample_row = df.iloc[row_index]

    st.dataframe(sample_row)

    actual = sample_row["label"]

    X = sample_row.drop("label").values.reshape(1, -1)

    if st.button("Predict Dataset Row"):

        scaled = scaler.transform(X)

        pred = classifier.predict(scaled)[0]

        st.write("Actual:", "Male" if actual == 1 else "Female")
        st.write("Predicted:", "Male" if pred == 1 else "Female")


# =========================================================
# TAB 3 AUDIO PREDICTION
# =========================================================

with tab3:

    st.subheader("Upload Voice Audio (.wav)")

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file is not None:

        features_dict, y, sr = extract_features(audio_file)

        st.audio(audio_file)

        # -------------------------------
        # Waveform Visualization
        # -------------------------------

        st.subheader("Audio Waveform")

        fig, ax = plt.subplots()

        librosa.display.waveshow(y, sr=sr, ax=ax)

        ax.set_title("Waveform")

        st.pyplot(fig)

        # -------------------------------
        # Feature Table
        # -------------------------------

        input_df = pd.DataFrame([features_dict])

        input_df = input_df.reindex(columns=features, fill_value=0)

        st.subheader("Extracted Features")

        st.dataframe(input_df)

        if st.button("Predict Audio"):

            scaled = scaler.transform(input_df)

            pred = classifier.predict(scaled)[0]
            cluster_id = cluster.predict(scaled)[0]

            st.subheader("Prediction Result")

            st.write("Cluster:", cluster_id)

            st.divider()

            st.subheader("Prediction Result")

            if pred == 1:
                st.success("🎤 Predicted Gender: Male")
            else:
                st.success("🎤 Predicted Gender: Female")