# ROLE
# ----
# Récupère le texte tajwid depuis l'API AlQuran Cloud
# pour une sourate et une plage de versets.

from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import json


TAJWID_API_BASE = "https://api.alquran.cloud/v1/quran/quran-tajweed"


class TajwidServiceError(Exception):
    pass


def fetch_tajwid_text(surah_id: int, start_verse: int, end_verse: int) -> dict:
    try:
        with urlopen(TAJWID_API_BASE, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise TajwidServiceError(f"Erreur API tajwid ({exc.code}).") from exc
    except URLError as exc:
        raise TajwidServiceError("Impossible de joindre l'API tajwid.") from exc
    except Exception as exc:
        raise TajwidServiceError("Erreur inattendue pendant le chargement du tajwid.") from exc

    surahs = payload.get("data", {}).get("surahs", [])
    surah = next((item for item in surahs if item.get("number") == surah_id), None)

    if not surah:
        raise TajwidServiceError("Sourate introuvable dans la réponse tajwid.")

    ayahs = surah.get("ayahs", [])
    selected_ayahs = [
        ayah for ayah in ayahs
        if start_verse <= ayah.get("numberInSurah", 0) <= end_verse
    ]

    if not selected_ayahs:
        raise TajwidServiceError("Aucun verset tajwid trouvé pour cette plage.")

    text = " ".join(ayah.get("text", "").strip() for ayah in selected_ayahs).strip()

    return {
        "surah_id": surah_id,
        "start_verse": start_verse,
        "end_verse": end_verse,
        "text": text,
    }
