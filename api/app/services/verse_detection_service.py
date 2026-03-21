# ROLE
# ----
# Trouve le meilleur match de versets à partir des segments de transcription.

import logging
from difflib import SequenceMatcher

from app.core.model_loader import get_quran_versets
from app.utils.normalize_arabic import normalize_arabic

logger = logging.getLogger(__name__)


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
                    "transliteration": sourate.get("transliteration", ""),
                    "start_verse": chunk[0]["id"],
                    "end_verse": chunk[-1]["id"],
                    "text": combined_text,
                    "similarity": score,
                })

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    best_match = matches[0] if matches else None

    logger.info(
        "Verse detection complete: transcription_chars=%s best_sourate=%s best_similarity=%s",
        len(transcription),
        best_match["sourate_name"] if best_match else None,
        best_match["similarity"] if best_match else None,
    )

    return best_match
