import librosa
import numpy as np

def extract_features(file):

    y, sr = librosa.load(file, sr=None)

    features = []

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    rms = np.mean(librosa.feature.rms(y=y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features.extend([
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate,
        rms
    ])

    features.extend(mfcc_mean)
    features.extend(mfcc_std)

    return features