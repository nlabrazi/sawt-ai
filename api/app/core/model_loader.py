# ROLE
# ----
# Charge une seule fois les ressources lourdes au démarrage de l'API :
# - modèle Whisper via faster-whisper
# - versets du Coran

import os
import json
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parents[2]
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")
QURAN_VERSETS_PATH = Path(
    os.getenv("QURAN_VERSETS_PATH", str(BASE_DIR / "assets" / "quran_versets.json"))
)

whisper_model: WhisperModel | None = None
quran_versets: list[dict[str, Any]] | None = None


def load_all_models() -> None:
    global whisper_model, quran_versets

    if whisper_model is None:
        whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )

    if quran_versets is None:
        with QURAN_VERSETS_PATH.open("r", encoding="utf-8") as file:
            quran_versets = json.load(file)


def get_whisper_model() -> WhisperModel:
    if whisper_model is None:
        raise RuntimeError("Whisper model is not loaded.")

    return whisper_model


def get_quran_versets() -> list[dict[str, Any]]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    return quran_versets
