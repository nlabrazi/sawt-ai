# ROLE
# ----
# Charge une seule fois les ressources lourdes au démarrage de l'API :
# - modèle Whisper via faster-whisper
# - modèle imam CNN
# - label encoder imam
# - versets du Coran

import json
import pickle

from faster_whisper import WhisperModel
from tensorflow.keras.models import load_model


WHISPER_MODEL_NAME = "base"
QURAN_VERSETS_PATH = "assets/quran_versets.json"

# À adapter si besoin selon vos vrais fichiers
IMAM_MODEL_PATH = "models/model_cnn_imam_v4.keras"
IMAM_LABEL_ENCODER_PATH = "models/label_encoder_imam.pkl"

whisper_model = None
quran_versets = None
imam_model = None
label_encoder = None


def load_all_models():
    global whisper_model, quran_versets, imam_model, label_encoder

    if whisper_model is None:
        whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )

    if quran_versets is None:
        with open(QURAN_VERSETS_PATH, "r", encoding="utf-8") as f:
            quran_versets = json.load(f)

    if imam_model is None:
        imam_model = load_model(IMAM_MODEL_PATH)

    if label_encoder is None:
        with open(IMAM_LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)


def get_whisper_model():
    return whisper_model


def get_quran_versets():
    return quran_versets


def get_imam_model():
    return imam_model


def get_label_encoder():
    return label_encoder
