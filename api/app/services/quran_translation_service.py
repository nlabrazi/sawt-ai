"""Read the validated French translation snapshot without network access."""

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import ValidationError

from app.schemas.quran import QuranTranslation
from app.services.quran_catalog_service import get_surah_metadata

DEFAULT_TRANSLATION_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "quran_translation_fr.json"
)
_translation_index: dict[tuple[int, int], QuranTranslation] | None = None
_translation_lock = Lock()


class QuranTranslationServiceError(Exception):
    pass


def validate_verse_range(surah_id: int, start_verse: int, end_verse: int) -> None:
    if any(type(value) is not int for value in (surah_id, start_verse, end_verse)):
        raise QuranTranslationServiceError("Les références de versets doivent être des entiers.")
    if start_verse < 1 or end_verse < start_verse:
        raise QuranTranslationServiceError("Plage de versets invalide.")
    surah = get_surah_metadata(surah_id)
    if surah is None or end_verse > surah["total_verses"]:
        raise QuranTranslationServiceError("Référence absente du catalogue coranique.")


def build_translation_index(payload: Any) -> dict[tuple[int, int], QuranTranslation]:
    """Validate the whole snapshot before publishing any of it in memory."""
    if not isinstance(payload, dict):
        raise QuranTranslationServiceError("Snapshot de traduction invalide.")
    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("schema_version") != 1:
        raise QuranTranslationServiceError("Version du format de traduction invalide.")
    entries = payload.get("translations")
    if not isinstance(entries, list) or not entries:
        raise QuranTranslationServiceError("Le snapshot ne contient aucune traduction.")

    index: dict[tuple[int, int], QuranTranslation] = {}
    for item in entries:
        try:
            entry = QuranTranslation.model_validate(item)
        except ValidationError as exc:
            raise QuranTranslationServiceError("Entrée de traduction invalide.") from exc
        validate_verse_range(entry.surah_id, entry.ayah, entry.ayah)
        reference = (entry.surah_id, entry.ayah)
        if reference in index:
            raise QuranTranslationServiceError(f"Traduction en doublon pour {reference}.")
        if any(
            getattr(entry, field) != meta.get(field)
            for field in ("source", "translator", "version")
        ):
            raise QuranTranslationServiceError("Provenance incohérente dans le snapshot.")
        index[reference] = entry
    return index


def load_translation_index(data_path: Path) -> dict[tuple[int, int], QuranTranslation]:
    try:
        with data_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError) as exc:
        raise QuranTranslationServiceError(
            "Impossible de lire le snapshot local de traduction."
        ) from exc
    return build_translation_index(payload)


def clear_quran_translation_cache() -> None:
    global _translation_index
    with _translation_lock:
        _translation_index = None


def _load_translation_index() -> dict[tuple[int, int], QuranTranslation]:
    global _translation_index
    with _translation_lock:
        if _translation_index is None:
            configured_path = os.getenv("QURAN_TRANSLATION_PATH", "").strip()
            data_path = Path(configured_path) if configured_path else DEFAULT_TRANSLATION_PATH
            _translation_index = load_translation_index(data_path)
        return _translation_index


def fetch_quran_translations(
    surah_id: int, start_verse: int, end_verse: int,
) -> list[QuranTranslation]:
    """Return only existing entries, in verse order, for the exact surah."""
    validate_verse_range(surah_id, start_verse, end_verse)
    index = _load_translation_index()
    return [
        index[(surah_id, ayah)]
        for ayah in range(start_verse, end_verse + 1)
        if (surah_id, ayah) in index
    ]
