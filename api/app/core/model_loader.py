# ROLE
# ----
# Charge une seule fois les ressources lourdes au démarrage de l'API :
# - modèle Whisper via faster-whisper
# - modèle imam CNN
# - label encoder imam
# - versets du Coran

import json
from faster_whisper import WhisperModel

WHISPER_MODEL_NAME = "base"
QURAN_VERSETS_PATH = "assets/quran_versets.json"

whisper_model = None
quran_versets = None


def load_all_models():
    global whisper_model, quran_versets

    if whisper_model is None:
        whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )

    if quran_versets is None:
        with open(QURAN_VERSETS_PATH, "r", encoding="utf-8") as f:
            quran_versets = json.load(f)


def get_whisper_model():
    return whisper_model


def get_quran_versets():
    return quran_versets
