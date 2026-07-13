# ROLE
# ----
# Trouve le meilleur match de versets à partir des segments de transcription.

import logging

from rapidfuzz import fuzz, process

from app.core.model_loader import get_quran_verse_candidates
from app.utils.normalize_arabic import normalize_arabic

logger = logging.getLogger(__name__)

LOCAL_SIMILARITY_WEIGHT = 0.7
TOKEN_SIMILARITY_WEIGHT = 0.3


def compute_similarity_score(query: str, candidate: str, **_kwargs) -> float:
    """Combine la correspondance locale et la similarité des mots sur 100."""
    local_score = fuzz.partial_ratio(query, candidate)
    token_score = fuzz.token_sort_ratio(query, candidate)

    return (
        LOCAL_SIMILARITY_WEIGHT * local_score
        + TOKEN_SIMILARITY_WEIGHT * token_score
    )


def detect_versets(segments):
    transcription = normalize_arabic(
        " ".join(segment.get("text", "") for segment in segments).strip()
    )

    if not transcription:
        logger.info(
            "Verse detection skipped: empty transcription after normalization."
        )
        return None

    candidates = get_quran_verse_candidates()
    match = process.extractOne(
        transcription,
        (candidate.normalized_text for candidate in candidates),
        scorer=compute_similarity_score,
    )

    if match is None:
        logger.info(
            "Verse detection skipped: no Quran verse candidates available."
        )
        return None

    _, score_percent, candidate_index = match
    candidate = candidates[candidate_index]
    best_match = {
        "sourate_id": candidate.sourate_id,
        "sourate_name": candidate.sourate_name,
        "transliteration": candidate.transliteration,
        "start_verse": candidate.start_verse,
        "end_verse": candidate.end_verse,
        "text": candidate.normalized_text,
        "similarity": score_percent / 100,
    }

    logger.info(
        "Verse detection complete: transcription_chars=%s best_sourate=%s best_similarity=%s",
        len(transcription),
        best_match["sourate_name"] if best_match else None,
        best_match["similarity"] if best_match else None,
    )

    return best_match
