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
TRANSCRIPTION_WINDOW_WORD_SIZES = (4, 8, 12, 16, 24, 32)
MAX_WINDOWS_PER_SIZE = 6
MATCH_CANDIDATE_LIMIT = 20


def compute_similarity_score(query: str, candidate: str, **_kwargs) -> float:
    """Combine la correspondance locale et la similarité des mots sur 100."""
    local_score = fuzz.partial_ratio(query, candidate)
    token_score = fuzz.token_sort_ratio(query, candidate)

    return (
        LOCAL_SIMILARITY_WEIGHT * local_score
        + TOKEN_SIMILARITY_WEIGHT * token_score
    )


def build_transcription_windows(transcription: str) -> tuple[str, ...]:
    """Construit des fenêtres chevauchantes tout en conservant le texte complet."""
    words = transcription.split()
    windows = [transcription]
    seen_windows = {transcription}

    for window_size in TRANSCRIPTION_WINDOW_WORD_SIZES:
        if window_size >= len(words):
            continue

        stride = max(1, window_size // 2)
        last_start = len(words) - window_size
        starts = list(range(0, last_start + 1, stride))

        if starts[-1] != last_start:
            starts.append(last_start)

        if len(starts) > MAX_WINDOWS_PER_SIZE:
            last_index = len(starts) - 1
            starts = [
                starts[round(index * last_index / (MAX_WINDOWS_PER_SIZE - 1))]
                for index in range(MAX_WINDOWS_PER_SIZE)
            ]

        for start in starts:
            window = " ".join(words[start:start + window_size])

            if window not in seen_windows:
                windows.append(window)
                seen_windows.add(window)

    return tuple(windows)


def extract_best_match(query: str, candidate_texts: tuple[str, ...]):
    """Présélectionne rapidement les candidats avant le score combiné."""
    shortlisted_matches = process.extract(
        query,
        candidate_texts,
        scorer=fuzz.ratio,
        limit=MATCH_CANDIDATE_LIMIT,
    )

    if not shortlisted_matches:
        return None

    return max(
        (
            (candidate_text, compute_similarity_score(query, candidate_text), candidate_index)
            for candidate_text, _score, candidate_index in shortlisted_matches
        ),
        key=lambda match: match[1],
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
    candidate_texts = tuple(candidate.normalized_text for candidate in candidates)
    match = None
    matched_window = transcription

    for window in build_transcription_windows(transcription):
        window_match = extract_best_match(window, candidate_texts)

        if window_match is not None and (match is None or window_match[1] > match[1]):
            match = window_match
            matched_window = window

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
        "Verse detection complete: transcription_chars=%s matched_window_words=%s best_sourate=%s best_similarity=%s",
        len(transcription),
        len(matched_window.split()),
        best_match["sourate_name"] if best_match else None,
        best_match["similarity"] if best_match else None,
    )

    return best_match
