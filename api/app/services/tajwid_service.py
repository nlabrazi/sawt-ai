# ROLE
# ----
# Récupère le texte tajwid depuis un snapshot local si disponible,
# sinon via l'API AlQuran Cloud. Les données sont ensuite gardées
# en mémoire pour éviter de recharger tout le corpus à chaque requête.

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TAJWID_DATA_PATH = BASE_DIR / "assets" / "quran_tajwid.json"
TAJWID_API_BASE = "https://api.alquran.cloud/v1/quran/quran-tajweed"
TAJWID_TIMEOUT_SECONDS = 15

_tajwid_index: dict[int, dict[int, str]] | None = None
_tajwid_index_lock = Lock()


class TajwidServiceError(Exception):
    pass


def clear_tajwid_cache() -> None:
    global _tajwid_index

    with _tajwid_index_lock:
        _tajwid_index = None


def _get_configured_tajwid_data_path() -> Path | None:
    raw_path = os.getenv("TAJWID_DATA_PATH", "").strip()

    if not raw_path:
        return None

    return Path(raw_path)


def _load_tajwid_payload_from_file(data_path: Path) -> dict[str, Any]:
    try:
        with data_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        logger.exception("Local tajwid snapshot not found: path=%s", data_path)
        raise TajwidServiceError("Snapshot tajwid local introuvable.") from exc
    except json.JSONDecodeError as exc:
        logger.exception("Local tajwid snapshot is invalid JSON: path=%s", data_path)
        raise TajwidServiceError("Snapshot tajwid local invalide.") from exc
    except OSError as exc:
        logger.exception("Local tajwid snapshot could not be read: path=%s", data_path)
        raise TajwidServiceError("Impossible de lire le snapshot tajwid local.") from exc


def _download_tajwid_payload() -> dict[str, Any]:
    try:
        with urlopen(TAJWID_API_BASE, timeout=TAJWID_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise TajwidServiceError(f"Erreur API tajwid ({exc.code}).") from exc
    except URLError as exc:
        raise TajwidServiceError("Impossible de joindre l'API tajwid.") from exc
    except json.JSONDecodeError as exc:
        logger.exception("Remote tajwid payload is invalid JSON")
        raise TajwidServiceError("Réponse tajwid invalide.") from exc
    except Exception as exc:
        raise TajwidServiceError("Erreur inattendue pendant le chargement du tajwid.") from exc


def _build_tajwid_index(payload: dict[str, Any]) -> dict[int, dict[int, str]]:
    if not isinstance(payload, dict):
        raise TajwidServiceError("Données tajwid invalides.")

    surahs = payload.get("data", {}).get("surahs")

    if not isinstance(surahs, list):
        raise TajwidServiceError("Données tajwid invalides.")

    tajwid_index: dict[int, dict[int, str]] = {}

    for surah in surahs:
        surah_id = surah.get("number")
        ayahs = surah.get("ayahs")

        if not isinstance(surah_id, int) or not isinstance(ayahs, list):
            continue

        verses_by_number: dict[int, str] = {}

        for ayah in ayahs:
            verse_number = ayah.get("numberInSurah")
            text = ayah.get("text", "")

            if isinstance(verse_number, int) and isinstance(text, str):
                verses_by_number[verse_number] = text.strip()

        if verses_by_number:
            tajwid_index[surah_id] = verses_by_number

    if not tajwid_index:
        raise TajwidServiceError("Données tajwid invalides.")

    return tajwid_index


def _load_tajwid_index() -> dict[int, dict[int, str]]:
    global _tajwid_index

    if _tajwid_index is not None:
        return _tajwid_index

    with _tajwid_index_lock:
        if _tajwid_index is not None:
            return _tajwid_index

        configured_data_path = _get_configured_tajwid_data_path()
        data_path = configured_data_path or DEFAULT_TAJWID_DATA_PATH

        if data_path.is_file():
            payload = _load_tajwid_payload_from_file(data_path)
            source = str(data_path)
        elif configured_data_path is not None:
            raise TajwidServiceError("Snapshot tajwid local introuvable.")
        else:
            payload = _download_tajwid_payload()
            source = TAJWID_API_BASE

        _tajwid_index = _build_tajwid_index(payload)
        logger.info(
            "Tajwid corpus loaded: source=%s surahs=%s",
            source,
            len(_tajwid_index),
        )
        return _tajwid_index


def fetch_tajwid_text(surah_id: int, start_verse: int, end_verse: int) -> dict[str, Any]:
    tajwid_index = _load_tajwid_index()
    surah_verses = tajwid_index.get(surah_id)

    if not surah_verses:
        raise TajwidServiceError("Sourate introuvable dans les données tajwid.")

    selected_ayahs: list[str] = []

    for verse_number in range(start_verse, end_verse + 1):
        verse_text = surah_verses.get(verse_number)

        if not verse_text:
            raise TajwidServiceError("Aucun verset tajwid trouvé pour cette plage.")

        selected_ayahs.append(verse_text)

    text = " ".join(selected_ayahs).strip()

    return {
        "surah_id": surah_id,
        "start_verse": start_verse,
        "end_verse": end_verse,
        "text": text,
    }
