# ROLE
# ----
# Prédit l'imam à partir d'un fichier audio
# en utilisant le modèle CNN déjà entraîné.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from whisper.audio import load_audio, SAMPLE_RATE

from app.utils.mfcc import extract_mfcc_from_audio


MODEL_PATH = "models/model_cnn_imam_v4.keras"
LABEL_PATH = "models/label_encoder_imam.pkl"


def predict_imam(audio_path: str):
    """
    Retourne les 3 imams les plus probables.
    """
    try:
        y_audio = load_audio(audio_path)
        sr = SAMPLE_RATE

        mfcc = extract_mfcc_from_audio(y=y_audio, sr=sr)

        model = load_model(MODEL_PATH)

        with open(LABEL_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        predictions = model.predict(np.array([mfcc]), verbose=0)[0]
        top_indices = predictions.argsort()[-3:][::-1]

        return [
            {
                "name": label_encoder.inverse_transform([idx])[0],
                "score": float(predictions[idx]),
            }
            for idx in top_indices
        ]

    except Exception as e:
        return {
            "error": f"Erreur prédiction imam: {str(e)}"
        }
