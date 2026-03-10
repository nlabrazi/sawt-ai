# ROLE
# ----
# Charge une seule fois les ressources lourdes au démarrage de l'API :
# - modèle Whisper
# - modèle imam
# - label encoder
# - versets du Coran

import json
import pickle

import whisper
from tensorflow.keras.models import load_model


WHISPER_MODEL_NAME = "base"
QURAN_VERSETS_PATH = "assets/quran_versets.json"
IMAM_MODEL_PATH = "models/model_cnn_imam_v4.keras"
LABEL_ENCODER_PATH = "models/label_encoder_imam.pkl"


whisper_model = None
imam_model = None
label_encoder = None
quran_versets = None


def load_all_models():
    global whisper_model, imam_model, label_encoder, quran_versets

    if whisper_model is None:
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

    if imam_model is None:
        imam_model = load_model(IMAM_MODEL_PATH)

    if label_encoder is None:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

    if quran_versets is None:
        with open(QURAN_VERSETS_PATH, "r", encoding="utf-8") as f:
            quran_versets = json.load(f)


def get_whisper_model():
    return whisper_model


def get_imam_model():
    return imam_model


def get_label_encoder():
    return label_encoder


def get_quran_versets():
    return quran_versets
