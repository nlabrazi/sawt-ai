# ROLE
# ----
# Charge une seule fois les ressources lourdes au démarrage de l'API :
# - modèle Whisper via faster-whisper
# - versets du Coran

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.utils.normalize_arabic import normalize_arabic

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parents[2]
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")
QURAN_VERSETS_PATH = Path(
    os.getenv("QURAN_VERSETS_PATH", str(BASE_DIR / "assets" / "quran_versets.json"))
)
MAX_VERSE_DETECTION_WINDOW_SIZE = 5

whisper_model: WhisperModel | None = None
quran_versets: list[dict[str, Any]] | None = None
quran_verse_candidates: tuple["QuranVerseCandidate", ...] | None = None


@dataclass(frozen=True, slots=True)
class QuranVerseCandidate:
    sourate_id: int
    sourate_name: str
    transliteration: str
    start_verse: int
    end_verse: int
    normalized_text: str


def _build_quran_verse_candidates(
    versets_data: list[dict[str, Any]],
) -> tuple[QuranVerseCandidate, ...]:
    candidates: list[QuranVerseCandidate] = []

    for sourate in versets_data:
        verses = sourate["verses"]
        max_window_size = min(MAX_VERSE_DETECTION_WINDOW_SIZE, len(verses))
        normalized_verses = [
            {
                "id": verse["id"],
                "text": normalize_arabic(verse["text"]),
            }
            for verse in verses
        ]

        for window_size in range(1, max_window_size + 1):
            for start_index in range(len(normalized_verses) - window_size + 1):
                chunk = normalized_verses[start_index:start_index + window_size]
                candidates.append(
                    QuranVerseCandidate(
                        sourate_id=sourate["id"],
                        sourate_name=sourate["name"],
                        transliteration=sourate.get("transliteration", ""),
                        start_verse=chunk[0]["id"],
                        end_verse=chunk[-1]["id"],
                        normalized_text=" ".join(verse["text"] for verse in chunk),
                    )
                )

    return tuple(candidates)


def _ensure_quran_verse_candidates_loaded() -> tuple[QuranVerseCandidate, ...]:
    global quran_verse_candidates

    if quran_verse_candidates is None:
        quran_verse_candidates = _build_quran_verse_candidates(get_quran_versets())

    return quran_verse_candidates


def load_all_models() -> None:
    global whisper_model

    if whisper_model is None:
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )

    load_quran_catalog()


def load_quran_catalog() -> None:
    global quran_versets

    if quran_versets is None:
        with QURAN_VERSETS_PATH.open("r", encoding="utf-8") as file:
            quran_versets = json.load(file)

    _ensure_quran_verse_candidates_loaded()


def get_whisper_model() -> WhisperModel:
    if whisper_model is None:
        raise RuntimeError("Whisper model is not loaded.")

    return whisper_model


def get_quran_versets() -> list[dict[str, Any]]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    return quran_versets


def get_quran_verse_candidates() -> tuple[QuranVerseCandidate, ...]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    return _ensure_quran_verse_candidates_loaded()
