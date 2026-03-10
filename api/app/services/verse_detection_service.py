# ROLE
# ----
# Trouve le verset ou groupe de versets le plus proche
# à partir des segments retournés par Whisper.

import json
from difflib import SequenceMatcher

from app.utils.normalize_arabic import normalize_arabic


VERSes_PATH = "assets/quran_versets.json"


def load_versets(path: str = VERSes_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_versets(segments):
    """
    Reçoit les segments Whisper et retourne le meilleur match coranique.
    """
    versets = load_versets()

    transcription = normalize_arabic(
        " ".join(segment["text"] for segment in segments).strip()
    )

    matches = []

    for sourate in versets:
        verses = sourate["verses"]

        for window_size in [1, 2, 3, 4, 5]:
            for i in range(len(verses) - window_size + 1):
                chunk = verses[i:i + window_size]
                combined_text = normalize_arabic(" ".join(v["text"] for v in chunk))
                score = SequenceMatcher(None, transcription, combined_text).ratio()

                matches.append({
                    "sourate_id": sourate["id"],
                    "sourate_name": sourate["name"],
                    "start_verse": chunk[0]["id"],
                    "end_verse": chunk[-1]["id"],
                    "text": combined_text,
                    "similarity": score,
                })

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches[0] if matches else None
