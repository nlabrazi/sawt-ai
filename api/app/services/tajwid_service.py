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
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TAJWID_DATA_PATH = BASE_DIR / "assets" / "quran_tajwid.json"
TAJWID_API_BASE = "https://api.alquran.cloud/v1/quran/quran-tajweed"
TAJWID_TIMEOUT_SECONDS = 15

_tajwid_index: dict[int, dict[int, str]] | None = None
_tajwid_index_lock = Lock()


class TajwidServiceError(Exception):
    pass


def warm_tajwid_cache() -> None:
    _load_tajwid_index()


def clear_tajwid_cache() -> None:
    global _tajwid_index

    with _tajwid_index_lock:
        _tajwid_index = None


def _get_configured_tajwid_data_path() -> Path | None:
    raw_path = os.getenv("TAJWID_DATA_PATH", "").strip()

    if not raw_path:
        return None

    return Path(raw_path)


def _get_tajwid_backup_url() -> str | None:
    value = os.getenv("TAJWID_BACKUP_URL", "").strip()
    return value or None


def _load_tajwid_payload_from_file(data_path: Path) -> dict[str, Any]:
    try:
        with data_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise TajwidServiceError("Snapshot tajwid local introuvable.") from exc
    except json.JSONDecodeError as exc:
        raise TajwidServiceError("Snapshot tajwid local invalide.") from exc
    except OSError as exc:
        raise TajwidServiceError("Impossible de lire le snapshot tajwid local.") from exc


def _download_json_payload(url: str, *, source_name: str) -> dict[str, Any]:
    request = Request(url)

    try:
        with urlopen(request, timeout=TAJWID_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise TajwidServiceError(f"Erreur {source_name} ({exc.code}).") from exc
    except URLError as exc:
        raise TajwidServiceError(f"Impossible de joindre {source_name}.") from exc
    except json.JSONDecodeError as exc:
        raise TajwidServiceError(f"Réponse {source_name} invalide.") from exc
    except Exception as exc:
        raise TajwidServiceError(f"Erreur inattendue pendant le chargement depuis {source_name}.") from exc


def _download_tajwid_payload() -> dict[str, Any]:
    return _download_json_payload(TAJWID_API_BASE, source_name="l'API tajwid")


def _download_tajwid_backup_payload(backup_url: str) -> dict[str, Any]:
    return _download_json_payload(backup_url, source_name="la sauvegarde tajwid")


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
        backup_url = _get_tajwid_backup_url()
        sources: list[tuple[str, Callable[[], dict[str, Any]]]] = []

        if configured_data_path is not None:
            sources.append((str(configured_data_path), lambda: _load_tajwid_payload_from_file(configured_data_path)))
        elif DEFAULT_TAJWID_DATA_PATH.is_file():
            sources.append((str(DEFAULT_TAJWID_DATA_PATH), lambda: _load_tajwid_payload_from_file(DEFAULT_TAJWID_DATA_PATH)))

        if backup_url:
            sources.append((backup_url, lambda: _download_tajwid_backup_payload(backup_url)))

        sources.append((TAJWID_API_BASE, _download_tajwid_payload))

        last_error: TajwidServiceError | None = None

        for source, loader in sources:
            try:
                payload = loader()
                _tajwid_index = _build_tajwid_index(payload)
                logger.info(
                    "Tajwid corpus loaded: source=%s surahs=%s",
                    source,
                    len(_tajwid_index),
                )
                return _tajwid_index
            except TajwidServiceError as exc:
                last_error = exc
                logger.warning("Tajwid source failed: source=%s error=%s", source, exc)

        raise last_error or TajwidServiceError("Aucune source tajwid disponible.")


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
