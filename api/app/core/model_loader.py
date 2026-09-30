# ROLE
# ----
# Charge une seule fois les ressources lourdes au démarrage de l'API :
# - modèle Whisper via faster-whisper
# - versets du Coran

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.utils.normalize_arabic import normalize_arabic

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parents[2]
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "turbo")
QURAN_VERSETS_PATH = Path(
    os.getenv("QURAN_VERSETS_PATH", str(BASE_DIR / "assets" / "quran_versets.json"))
)
MAX_VERSE_DETECTION_WINDOW_SIZE = 10
MAX_VERSE_DETECTION_WORD_COUNT = 64

whisper_model: WhisperModel | None = None
quran_versets: list[dict[str, Any]] | None = None
quran_verse_candidates: tuple["QuranVerseCandidate", ...] | None = None
quran_candidate_texts: tuple[str, ...] | None = None
quran_text_occurrences: Counter[str] | None = None
quran_single_verse_candidates_by_sourate: (
    dict[int, tuple[tuple[int, "QuranVerseCandidate"], ...]] | None
) = None
quran_candidates_by_range: (
    dict[tuple[int, int, int], tuple[int, "QuranVerseCandidate"]] | None
) = None


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
                normalized_text = " ".join(verse["text"] for verse in chunk)

                # Un verset long doit rester détectable, mais les passages
                # multi-versets sont bornés pour contenir le coût mémoire et CPU.
                if (
                    window_size > 1
                    and len(normalized_text.split()) > MAX_VERSE_DETECTION_WORD_COUNT
                ):
                    continue

                candidates.append(
                    QuranVerseCandidate(
                        sourate_id=sourate["id"],
                        sourate_name=sourate["name"],
                        transliteration=sourate.get("transliteration", ""),
                        start_verse=chunk[0]["id"],
                        end_verse=chunk[-1]["id"],
                        normalized_text=normalized_text,
                    )
                )

    return tuple(candidates)


def _ensure_quran_verse_candidates_loaded() -> tuple[QuranVerseCandidate, ...]:
    global quran_verse_candidates, quran_candidate_texts, quran_text_occurrences
    global quran_single_verse_candidates_by_sourate, quran_candidates_by_range

    if quran_verse_candidates is None:
        candidates = _build_quran_verse_candidates(get_quran_versets())
        quran_verse_candidates = candidates
        quran_candidate_texts = tuple(candidate.normalized_text for candidate in candidates)

        single_by_sourate: dict[int, list[tuple[int, QuranVerseCandidate]]] = {}
        all_single_verse_texts: list[str] = []
        candidates_by_range: dict[tuple[int, int, int], tuple[int, QuranVerseCandidate]] = {}

        for candidate_index, candidate in enumerate(candidates):
            candidates_by_range[
                (candidate.sourate_id, candidate.start_verse, candidate.end_verse)
            ] = (candidate_index, candidate)
            if candidate.start_verse == candidate.end_verse:
                all_single_verse_texts.append(candidate.normalized_text)
                single_by_sourate.setdefault(candidate.sourate_id, []).append(
                    (candidate_index, candidate)
                )

        quran_text_occurrences = Counter(all_single_verse_texts)
        quran_single_verse_candidates_by_sourate = {
            sourate_id: tuple(items) for sourate_id, items in single_by_sourate.items()
        }
        quran_candidates_by_range = candidates_by_range

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


def is_catalog_candidates(candidates: Any) -> bool:
    return (
        quran_versets is not None
        and quran_verse_candidates is not None
        and candidates is quran_verse_candidates
    )


def get_quran_verse_candidates() -> tuple[QuranVerseCandidate, ...]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    return _ensure_quran_verse_candidates_loaded()


def get_quran_candidate_texts() -> tuple[str, ...]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    _ensure_quran_verse_candidates_loaded()
    assert quran_candidate_texts is not None
    return quran_candidate_texts


def get_quran_text_occurrences() -> Counter[str]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    _ensure_quran_verse_candidates_loaded()
    assert quran_text_occurrences is not None
    return quran_text_occurrences


def get_quran_single_verse_candidates_by_sourate(
    sourate_id: int,
) -> tuple[tuple[int, QuranVerseCandidate], ...]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    _ensure_quran_verse_candidates_loaded()
    assert quran_single_verse_candidates_by_sourate is not None
    return quran_single_verse_candidates_by_sourate.get(sourate_id, ())


def get_quran_candidates_by_range() -> dict[
    tuple[int, int, int], tuple[int, QuranVerseCandidate]
]:
    if quran_versets is None:
        raise RuntimeError("Quran verses are not loaded.")

    _ensure_quran_verse_candidates_loaded()
    assert quran_candidates_by_range is not None
    return quran_candidates_by_range
