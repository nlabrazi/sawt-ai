# ROLE
# ----
# Prédit les 3 imams les plus probables à partir de l'audio.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from whisper.audio import load_audio, SAMPLE_RATE

from app.core.model_loader import get_imam_model, get_label_encoder
from app.utils.mfcc import extract_mfcc_from_audio


def predict_imam(audio_path: str):
    try:
        y_audio = load_audio(audio_path)
        sr = SAMPLE_RATE

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
