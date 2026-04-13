# ROLE
# ----
# Trouve le meilleur match de versets à partir des segments de transcription.

import logging
from difflib import SequenceMatcher

from app.core.model_loader import get_quran_verse_candidates
from app.utils.normalize_arabic import normalize_arabic

logger = logging.getLogger(__name__)


def detect_versets(segments):
    transcription = normalize_arabic(
        " ".join(segment.get("text", "") for segment in segments).strip()
    )

    if not transcription:
        logger.info(
            "Verse detection skipped: empty transcription after normalization."
        )
        return None

    best_match = None
    best_similarity = -1.0
    matcher = SequenceMatcher(None, "", transcription)

    for candidate in get_quran_verse_candidates():
        matcher.set_seq1(candidate.normalized_text)
        score = matcher.ratio()

        if score <= best_similarity:
            continue

        best_similarity = score
        best_match = {
            "sourate_id": candidate.sourate_id,
            "sourate_name": candidate.sourate_name,
            "transliteration": candidate.transliteration,
            "start_verse": candidate.start_verse,
            "end_verse": candidate.end_verse,
            "text": candidate.normalized_text,
            "similarity": score,
        }

    logger.info(
        "Verse detection complete: transcription_chars=%s best_sourate=%s best_similarity=%s",
        len(transcription),
        best_match["sourate_name"] if best_match else None,
        best_match["similarity"] if best_match else None,
    )

    return best_match
