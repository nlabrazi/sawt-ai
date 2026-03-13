# ROLE
# ----
# Prédit les 3 imams les plus probables à partir de l'audio.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import librosa

from app.core.model_loader import get_imam_model, get_label_encoder
from app.utils.mfcc import extract_mfcc_from_audio

TARGET_SAMPLE_RATE = 16000


def predict_imam(audio_path: str):
    try:
        y_audio, sr = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )

        mfcc = extract_mfcc_from_audio(y=y_audio, sr=sr)

        model = get_imam_model()
        label_encoder = get_label_encoder()

        predictions = model.predict(np.array([mfcc]), verbose=0)[0]
        top_indices = predictions.argsort()[-3:][::-1]

        return [
            {
                "name": label_encoder.inverse_transform([idx])[0],
                "score": float(predictions[idx]),
            }
            for idx in top_indices
        ]

    except Exception:
        return []
