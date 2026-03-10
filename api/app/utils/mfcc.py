import librosa
import numpy as np

def normalize_rms(y, target_dB=-20):
    """
    Normalise le signal audio à un niveau RMS cible (en dB).
    """
    rms = np.sqrt(np.mean(y**2))
    scalar = 10 ** (target_dB / 20) / (rms + 1e-6)
    return y * scalar

def apply_pre_emphasis(y, coef=0.97):
    """
    Accentue les hautes fréquences du signal audio.
    """
    return np.append(y[0], y[1:] - coef * y[:-1])

def extract_mfcc_from_audio(y, sr, n_mfcc=13):

    # 🔊 Étape 1 : normalisation RMS
    y = normalize_rms(y)

    # 🔊 Étape 2 : pre-emphasis
    y = apply_pre_emphasis(y)

    # 🔊 Étape 3 : extraction des MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 🔢 Étape 4 : moyenne + écart-type
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    mfccs_combined = np.concatenate([mfccs_mean, mfccs_std])

    return mfccs_combined.tolist()
