import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

st.title("🎧 Audio Visualization")

st.markdown("Upload a **.wav audio file** to visualize waveform, spectrogram and MFCC features.")

audio_file = st.file_uploader("Upload Audio File", type=["wav"])

if audio_file is not None:

    # Load audio
    y, sr = librosa.load(audio_file)

    st.subheader("Audio Player")
    st.audio(audio_file)

    # -------------------------------
    # Waveform
    # -------------------------------

    st.subheader("Waveform")

    fig, ax = plt.subplots()

    librosa.display.waveshow(y, sr=sr, ax=ax)

    ax.set_title("Audio Waveform")

    st.pyplot(fig)

    # -------------------------------
    # Spectrogram
    # -------------------------------

    st.subheader("Spectrogram")

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))

    fig2, ax2 = plt.subplots()

    img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax2)

    ax2.set_title("Spectrogram")

    fig2.colorbar(img, ax=ax2)

    st.pyplot(fig2)

    # -------------------------------
    # MFCC
    # -------------------------------

    st.subheader("MFCC Features")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    fig3, ax3 = plt.subplots()

    img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax3)

    ax3.set_title("MFCC")

    fig3.colorbar(img2, ax=ax3)

    st.pyplot(fig3)