from __future__ import annotations

from typing import Any

from app.core.model_loader import get_quran_versets


def _serialize_surah_metadata(surah: dict[str, Any]) -> dict[str, Any]:
    verses = surah.get("verses", [])

    return {
        "id": surah["id"],
        "name": surah["name"],
        "transliteration": surah.get("transliteration", ""),
        "total_verses": surah.get("total_verses") or len(verses),
    }


def list_surah_metadata() -> list[dict[str, Any]]:
    return sorted([
        _serialize_surah_metadata(surah)
        for surah in get_quran_versets()
    ], key=lambda surah: surah["id"])


def get_surah_metadata(surah_id: int) -> dict[str, Any] | None:
    for surah in get_quran_versets():
        if surah["id"] == surah_id:
            return _serialize_surah_metadata(surah)

    return None
