# ROLE
# ----
# Trouve le meilleur match de versets à partir des segments de transcription.

from difflib import SequenceMatcher

from app.core.model_loader import get_quran_versets
from app.utils.normalize_arabic import normalize_arabic


def detect_versets(segments):
    versets_data = get_quran_versets()

    transcription = normalize_arabic(
        " ".join(segment["text"] for segment in segments).strip()
    )

    matches = []

    for sourate in versets_data:
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
