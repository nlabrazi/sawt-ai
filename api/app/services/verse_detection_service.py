# ROLE
# ----
# Trouve le meilleur match de versets à partir des segments de transcription.

import logging
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process

from app.core.detection_policy import (
    MIN_ACCEPTED_SIMILARITY,
    MIN_MATCHED_WORD_COUNT,
    MIN_PROBABLE_SIMILARITY,
    MIN_SCORE_MARGIN,
)
from app.core.model_loader import QuranVerseCandidate, get_quran_verse_candidates
from app.utils.normalize_arabic import normalize_arabic

logger = logging.getLogger(__name__)

LOCAL_SIMILARITY_WEIGHT = 0.7
TOKEN_SIMILARITY_WEIGHT = 0.3
TRANSCRIPTION_WINDOW_WORD_SIZES = (4, 8, 12, 16, 24, 32)
MAX_WINDOWS_PER_SIZE = 6
MATCH_CANDIDATE_LIMIT = 20
RANKED_CANDIDATE_LIMIT = 2
AMBIGUITY_CANDIDATE_LIMIT = 10

DetectionStatus = Literal["confident", "probable", "ambiguous", "insufficient"]
RejectionReason = Literal[
    "no_match",
    "score_too_low",
    "transcription_too_short",
    "ambiguous_match",
]


@dataclass(frozen=True, slots=True)
class RankedVerseCandidate:
    candidate_index: int
    candidate: QuranVerseCandidate
    score_percent: float
    matched_window: str


@dataclass(frozen=True, slots=True)
class MatchAcceptance:
    accepted: bool
    reason: RejectionReason | None
    matched_word_count: int
    score_margin_percent: float | None
    competing_match: RankedVerseCandidate | None


@dataclass(frozen=True, slots=True)
class VerseDetectionOutcome:
    verse: dict | None
    status: DetectionStatus
    score: float | None
    score_margin: float | None
    matched_word_count: int
    rejection_reason: RejectionReason | None

    def metadata(self) -> dict:
        return {
            "status": self.status,
            "score": self.score,
            "score_margin": self.score_margin,
            "matched_word_count": self.matched_word_count,
            "rejection_reason": self.rejection_reason,
        }


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


def extract_ranked_matches(
    query: str,
    candidate_texts: tuple[str, ...],
    limit: int = RANKED_CANDIDATE_LIMIT,
):
    """Présélectionne rapidement les candidats avant le score combiné."""
    shortlisted_matches = process.extract(
        query,
        candidate_texts,
        scorer=fuzz.ratio,
        limit=MATCH_CANDIDATE_LIMIT,
    )

    if not shortlisted_matches:
        return []

    rescored_matches = sorted(
        (
            (candidate_text, compute_similarity_score(query, candidate_text), candidate_index)
            for candidate_text, _score, candidate_index in shortlisted_matches
        ),
        key=lambda match: (-match[1], match[2]),
    )

    return rescored_matches[:limit]


def rank_verse_candidates(
    transcription: str,
    candidates: tuple[QuranVerseCandidate, ...],
    limit: int = RANKED_CANDIDATE_LIMIT,
) -> list[RankedVerseCandidate]:
    """Classe les candidats uniques selon leur meilleur score toutes fenêtres confondues."""
    candidate_texts = tuple(candidate.normalized_text for candidate in candidates)
    best_matches_by_index: dict[int, RankedVerseCandidate] = {}

    for window in build_transcription_windows(transcription):
        for _text, score_percent, candidate_index in extract_ranked_matches(
            window,
            candidate_texts,
            limit=limit,
        ):
            previous_match = best_matches_by_index.get(candidate_index)

            if previous_match is None or (
                score_percent,
                len(window.split()),
            ) > (
                previous_match.score_percent,
                len(previous_match.matched_window.split()),
            ):
                best_matches_by_index[candidate_index] = RankedVerseCandidate(
                    candidate_index=candidate_index,
                    candidate=candidates[candidate_index],
                    score_percent=score_percent,
                    matched_window=window,
                )

    return sorted(
        best_matches_by_index.values(),
        key=lambda match: (
            -match.score_percent,
            -len(match.matched_window.split()),
            -len(match.candidate.normalized_text.split()),
            match.candidate_index,
        ),
    )[:limit]


def candidates_overlap(
    first: QuranVerseCandidate,
    second: QuranVerseCandidate,
) -> bool:
    if first.sourate_id != second.sourate_id:
        return False

    return not (
        first.end_verse < second.start_verse
        or second.end_verse < first.start_verse
    )


def assess_match_acceptance(
    ranked_matches: list[RankedVerseCandidate],
) -> MatchAcceptance:
    if not ranked_matches:
        return MatchAcceptance(False, "no_match", 0, None, None)

    top_match = ranked_matches[0]
    matched_word_count = len(top_match.matched_window.split())
    competing_match = next(
        (
            match
            for match in ranked_matches[1:]
            if not candidates_overlap(top_match.candidate, match.candidate)
        ),
        None,
    )
    score_margin_percent = (
        top_match.score_percent - competing_match.score_percent
        if competing_match is not None
        else None
    )

    if top_match.score_percent < MIN_ACCEPTED_SIMILARITY * 100:
        reason = "score_too_low"
    elif matched_word_count < MIN_MATCHED_WORD_COUNT:
        reason = "transcription_too_short"
    elif (
        score_margin_percent is not None
        and score_margin_percent < MIN_SCORE_MARGIN * 100
    ):
        reason = "ambiguous_match"
    else:
        reason = None

    return MatchAcceptance(
        accepted=reason is None,
        reason=reason,
        matched_word_count=matched_word_count,
        score_margin_percent=score_margin_percent,
        competing_match=competing_match,
    )


def build_detection_outcome(
    ranked_matches: list[RankedVerseCandidate],
    acceptance: MatchAcceptance,
    include_ambiguous_verse: bool = False,
) -> VerseDetectionOutcome:
    if not ranked_matches:
        return VerseDetectionOutcome(
            verse=None,
            status="insufficient",
            score=None,
            score_margin=None,
            matched_word_count=0,
            rejection_reason="no_match",
        )

    top_match = ranked_matches[0]
    score = top_match.score_percent / 100
    score_margin = (
        acceptance.score_margin_percent / 100
        if acceptance.score_margin_percent is not None
        else None
    )

    if acceptance.accepted:
        status: DetectionStatus = "confident"
    elif acceptance.reason == "ambiguous_match":
        status = "ambiguous"
    elif acceptance.reason == "score_too_low" and score >= MIN_PROBABLE_SIMILARITY:
        status = "probable"
    else:
        status = "insufficient"

    should_include_verse = acceptance.accepted or (
        include_ambiguous_verse and acceptance.reason == "ambiguous_match"
    )

    if should_include_verse:
        candidate = top_match.candidate
        verse = {
            "sourate_id": candidate.sourate_id,
            "sourate_name": candidate.sourate_name,
            "transliteration": candidate.transliteration,
            "start_verse": candidate.start_verse,
            "end_verse": candidate.end_verse,
            "text": candidate.normalized_text,
            "similarity": score,
        }
    else:
        verse = None

    return VerseDetectionOutcome(
        verse=verse,
        status=status,
        score=score,
        score_margin=score_margin,
        matched_word_count=acceptance.matched_word_count,
        rejection_reason=acceptance.reason,
    )


def detect_verse_with_metadata(
    segments,
    include_ambiguous_verse: bool = False,
) -> VerseDetectionOutcome:
    transcription = normalize_arabic(
        " ".join(segment.get("text", "") for segment in segments).strip()
    )

    if not transcription:
        logger.info(
            "Verse detection skipped: empty transcription after normalization."
        )
        return build_detection_outcome([], assess_match_acceptance([]))

    candidates = get_quran_verse_candidates()
    ranked_matches = rank_verse_candidates(
        transcription,
        candidates,
        limit=AMBIGUITY_CANDIDATE_LIMIT,
    )

    if not ranked_matches:
        logger.info(
            "Verse detection skipped: no Quran verse candidates available."
        )
        return build_detection_outcome([], assess_match_acceptance([]))

    acceptance = assess_match_acceptance(ranked_matches)
    outcome = build_detection_outcome(
        ranked_matches,
        acceptance,
        include_ambiguous_verse=include_ambiguous_verse,
    )

    if not acceptance.accepted:
        logger.info(
            "Verse detection rejected: reason=%s best_similarity=%s matched_window_words=%s score_margin=%s",
            acceptance.reason,
            ranked_matches[0].score_percent / 100,
            acceptance.matched_word_count,
            (
                acceptance.score_margin_percent / 100
                if acceptance.score_margin_percent is not None
                else None
            ),
        )
        return outcome

    top_match = ranked_matches[0]

    logger.info(
        "Verse detection complete: transcription_chars=%s matched_window_words=%s best_sourate=%s best_similarity=%s",
        len(transcription),
        len(top_match.matched_window.split()),
        outcome.verse["sourate_name"] if outcome.verse else None,
        outcome.score,
    )

    return outcome


def detect_versets(segments):
    return detect_verse_with_metadata(segments).verse
