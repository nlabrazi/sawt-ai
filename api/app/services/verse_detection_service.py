# ROLE
# ----
# Trouve le meilleur match de versets à partir des segments de transcription.

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process

from app.core.detection_policy import (
    MIN_ACCEPTED_SIMILARITY,
    MIN_MATCHED_WORD_COUNT,
    MIN_PROBABLE_SIMILARITY,
    MIN_PROPOSAL_MATCHED_WORD_COUNT,
    MIN_PROPOSAL_SCORE_MARGIN,
    MIN_PROPOSAL_SIMILARITY,
    MIN_SCORE_MARGIN,
)
from app.core.model_loader import QuranVerseCandidate, get_quran_verse_candidates
from app.utils.normalize_arabic import normalize_arabic

logger = logging.getLogger(__name__)

LOCAL_SIMILARITY_WEIGHT = 0.7
TOKEN_SIMILARITY_WEIGHT = 0.3
MIN_EVIDENCE_FACTOR = 0.8
LENGTH_COMPATIBILITY_WEIGHT = 0.2
FULL_EVIDENCE_WORD_COUNT = 16
TRANSCRIPTION_WINDOW_WORD_SIZES = (4, 8, 12, 16, 24, 32)
MAX_WINDOWS_PER_SIZE = 6
MATCH_CANDIDATE_LIMIT = 20
RANKED_CANDIDATE_LIMIT = 2
AMBIGUITY_CANDIDATE_LIMIT = 10
DIAGNOSTIC_CANDIDATE_LIMIT = 3
PASSAGE_ANCHOR_MIN_WORD_COUNT = 4
PASSAGE_ANCHOR_MIN_SIMILARITY = 90.0
PASSAGE_ANCHOR_MIN_TEXT_COVERAGE = 0.85
MIN_CONFIDENT_INFERRED_PASSAGE_SIMILARITY = 97.0
MIN_STRONG_EVIDENCE_SIMILARITY = 97.0
MIN_STRONG_EVIDENCE_WORD_COUNT = 12
PASSAGE_ANCHOR_BOUNDARY_TOLERANCE = 1
MAX_UNSUPPORTED_VERSES_BETWEEN_ANCHORS = 2

DetectionStatus = Literal["confident", "probable", "ambiguous", "insufficient"]
RejectionReason = Literal[
    "no_match",
    "score_too_low",
    "transcription_too_short",
    "ambiguous_match",
    "insufficient_speech",
    "non_arabic_speech",
    "low_transcription_confidence",
]


@dataclass(frozen=True, slots=True)
class RankedVerseCandidate:
    candidate_index: int
    candidate: QuranVerseCandidate
    score_percent: float
    matched_window: str
    ranking_score_percent: float | None = None


@dataclass(frozen=True, slots=True)
class MatchAcceptance:
    accepted: bool
    reason: RejectionReason | None
    matched_word_count: int
    score_margin_percent: float | None
    competing_match: RankedVerseCandidate | None


@dataclass(frozen=True, slots=True)
class PassageVerseEvidence:
    candidate_index: int
    candidate: QuranVerseCandidate
    score_percent: float
    transcription_start: int
    transcription_end: int


@dataclass(frozen=True, slots=True)
class VerseDetectionOutcome:
    verse: dict | None
    status: DetectionStatus
    score: float | None
    score_margin: float | None
    matched_word_count: int
    rejection_reason: RejectionReason | None
    candidates: tuple[dict[str, float | int], ...] = ()

    def metadata(self) -> dict:
        return {
            "status": self.status,
            "score": self.score,
            "score_margin": self.score_margin,
            "matched_word_count": self.matched_word_count,
            "rejection_reason": self.rejection_reason,
        }


def build_candidate_evidence(
    ranked_matches: list[RankedVerseCandidate],
    *,
    limit: int = DIAGNOSTIC_CANDIDATE_LIMIT,
) -> tuple[dict[str, float | int], ...]:
    """Expose les preuves de classement sans inclure le texte coranique."""
    evidence = []
    seen_candidates = set()

    for match in ranked_matches:
        candidate = match.candidate
        candidate_key = (
            candidate.sourate_id,
            candidate.start_verse,
            candidate.end_verse,
        )
        if candidate_key in seen_candidates:
            continue

        seen_candidates.add(candidate_key)
        matched_word_count = len(match.matched_window.split())
        candidate_word_count = len(candidate.normalized_text.split())
        covered_word_count = min(matched_word_count, candidate_word_count)
        coverage = (
            covered_word_count / candidate_word_count
            if candidate_word_count
            else 0.0
        )
        continuity = fuzz.ratio(
            match.matched_window,
            candidate.normalized_text,
        ) / 100
        evidence.append(
            {
                "rank": len(evidence) + 1,
                "sourate_id": candidate.sourate_id,
                "start_verse": candidate.start_verse,
                "end_verse": candidate.end_verse,
                "similarity": match.score_percent / 100,
                "ranking_score": effective_ranking_score(match) / 100,
                "matched_word_count": matched_word_count,
                "coverage": coverage,
                "continuity": continuity,
            }
        )
        if len(evidence) >= limit:
            break

    return tuple(evidence)


def compute_similarity_score(query: str, candidate: str, **_kwargs) -> float:
    """Combine la correspondance locale et la similarité des mots sur 100."""
    local_score = fuzz.partial_ratio(query, candidate)
    token_score = fuzz.token_sort_ratio(query, candidate)

    return (
        LOCAL_SIMILARITY_WEIGHT * local_score
        + TOKEN_SIMILARITY_WEIGHT * token_score
    )


def compute_ranking_score(
    similarity_score: float,
    matched_window: str,
    candidate: str,
) -> float:
    """Pondère la similarité par la quantité et la cohérence des preuves.

    Une courte sous-fenêtre exacte reste un bon signal, mais ne doit pas dominer
    un passage long presque exact. La pondération est volontairement bornée :
    elle sert uniquement au classement et ne transforme pas une correspondance
    faible en résultat acceptable.
    """
    window_word_count = len(matched_window.split())
    candidate_word_count = len(candidate.split())

    if window_word_count == 0 or candidate_word_count == 0:
        return 0.0

    evidence_word_count = min(window_word_count, candidate_word_count)
    evidence_factor = MIN_EVIDENCE_FACTOR + (
        (1 - MIN_EVIDENCE_FACTOR)
        * min(evidence_word_count / FULL_EVIDENCE_WORD_COUNT, 1)
    )
    length_compatibility = evidence_word_count / max(
        window_word_count,
        candidate_word_count,
    )
    length_factor = (
        1 - LENGTH_COMPATIBILITY_WEIGHT
        + LENGTH_COMPATIBILITY_WEIGHT * length_compatibility
    )

    return similarity_score * evidence_factor * length_factor


def effective_ranking_score(match: RankedVerseCandidate) -> float:
    """Garde les candidats construits manuellement compatibles avec l'API interne."""
    if match.ranking_score_percent is None:
        return match.score_percent

    return match.ranking_score_percent


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
            candidate = candidates[candidate_index]
            ranking_score_percent = compute_ranking_score(
                score_percent,
                window,
                candidate.normalized_text,
            )
            previous_match = best_matches_by_index.get(candidate_index)

            if previous_match is None or (
                ranking_score_percent,
                score_percent,
                len(window.split()),
            ) > (
                effective_ranking_score(previous_match),
                previous_match.score_percent,
                len(previous_match.matched_window.split()),
            ):
                best_matches_by_index[candidate_index] = RankedVerseCandidate(
                    candidate_index=candidate_index,
                    candidate=candidate,
                    score_percent=score_percent,
                    matched_window=window,
                    ranking_score_percent=ranking_score_percent,
                )

    return sorted(
        best_matches_by_index.values(),
        key=lambda match: (
            -effective_ranking_score(match),
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


def _alignment_is_near_word_boundaries(
    transcription: str,
    start: int,
    end: int,
) -> bool:
    """Tolère une lettre tronquée par Whisper, pas une sous-chaîne arbitraire."""
    word_start = transcription.rfind(" ", 0, start) + 1
    next_separator = transcription.find(" ", end)
    word_end = len(transcription) if next_separator == -1 else next_separator

    return (
        start - word_start <= PASSAGE_ANCHOR_BOUNDARY_TOLERANCE
        and word_end - end <= PASSAGE_ANCHOR_BOUNDARY_TOLERANCE
    )


def extract_passage_verse_evidence(
    transcription: str,
    candidates: tuple[QuranVerseCandidate, ...],
    sourate_id: int,
) -> list[PassageVerseEvidence]:
    """Extrait des ancres fortes et non ambiguës pour une seule sourate.

    Les versets courts et les refrains identiques au sein d'une sourate sont
    exclus : ils sont utiles au classement classique, mais trop faibles pour
    justifier l'extension d'une plage.
    """
    all_single_verse_candidates = [
        (candidate_index, candidate)
        for candidate_index, candidate in enumerate(candidates)
        if candidate.start_verse == candidate.end_verse
    ]
    text_occurrences = Counter(
        candidate.normalized_text for _, candidate in all_single_verse_candidates
    )
    single_verse_candidates = (
        (candidate_index, candidate)
        for candidate_index, candidate in all_single_verse_candidates
        if candidate.sourate_id == sourate_id
    )
    evidence = []

    for candidate_index, candidate in single_verse_candidates:
        candidate_text = candidate.normalized_text

        if (
            len(candidate_text.split()) < PASSAGE_ANCHOR_MIN_WORD_COUNT
            or text_occurrences[candidate_text] != 1
        ):
            continue

        alignment = fuzz.partial_ratio_alignment(candidate_text, transcription)

        if alignment is None or alignment.score < PASSAGE_ANCHOR_MIN_SIMILARITY:
            continue

        covered_character_count = alignment.src_end - alignment.src_start
        text_coverage = covered_character_count / len(candidate_text)

        if (
            text_coverage < PASSAGE_ANCHOR_MIN_TEXT_COVERAGE
            or not _alignment_is_near_word_boundaries(
                transcription,
                alignment.dest_start,
                alignment.dest_end,
            )
        ):
            continue

        evidence.append(
            PassageVerseEvidence(
                candidate_index=candidate_index,
                candidate=candidate,
                score_percent=alignment.score,
                transcription_start=alignment.dest_start,
                transcription_end=alignment.dest_end,
            )
        )

    return sorted(
        evidence,
        key=lambda item: (
            item.transcription_start,
            item.transcription_end,
            item.candidate.start_verse,
        ),
    )


def _passage_evidence_follows(
    previous: PassageVerseEvidence,
    current: PassageVerseEvidence,
    top_match: RankedVerseCandidate,
) -> bool:
    previous_end_verse = previous.candidate.end_verse
    current_start_verse = current.candidate.start_verse

    # Le meilleur candidat soutient déjà toute sa propre plage. Lorsqu'une
    # ancre se trouve dedans, seuls les versets entre cette plage et l'ancre
    # extérieure comptent comme non étayés.
    if (
        previous_end_verse < top_match.candidate.start_verse
        <= current_start_verse
        <= top_match.candidate.end_verse
    ):
        current_start_verse = top_match.candidate.start_verse
    elif (
        top_match.candidate.start_verse
        <= previous_end_verse
        <= top_match.candidate.end_verse
        < current_start_verse
    ):
        previous_end_verse = top_match.candidate.end_verse

    missing_verse_count = current_start_verse - previous_end_verse - 1

    return (
        current.transcription_start >= previous.transcription_end
        and 0 <= missing_verse_count <= MAX_UNSUPPORTED_VERSES_BETWEEN_ANCHORS
    )


def _passage_evidence_chain_quality(
    chain: tuple[PassageVerseEvidence, ...],
) -> tuple[int, int, float, int]:
    unsupported_verse_count = sum(
        current.candidate.start_verse - previous.candidate.end_verse - 1
        for previous, current in zip(chain, chain[1:])
    )

    return (
        len(chain),
        sum(len(item.candidate.normalized_text.split()) for item in chain),
        sum(item.score_percent for item in chain),
        -unsupported_verse_count,
    )


def _chain_relation_to_match(
    evidence: PassageVerseEvidence,
    match: RankedVerseCandidate,
) -> tuple[bool, bool]:
    verse_id = evidence.candidate.start_verse
    is_inside = match.candidate.start_verse <= verse_id <= match.candidate.end_verse
    return is_inside, not is_inside


def _find_supported_passage_chains(
    evidence: list[PassageVerseEvidence],
    top_match: RankedVerseCandidate,
) -> list[tuple[PassageVerseEvidence, ...]]:
    """Conserve les meilleures chaînes ordonnées avec preuve interne/externe."""
    states_by_end: list[
        dict[tuple[bool, bool], tuple[PassageVerseEvidence, ...]]
    ] = []

    for current_index, current in enumerate(evidence):
        relation = _chain_relation_to_match(current, top_match)
        states = {relation: (current,)}

        for previous_index in range(current_index):
            previous = evidence[previous_index]

            if not _passage_evidence_follows(previous, current, top_match):
                continue

            for previous_relation, previous_chain in states_by_end[
                previous_index
            ].items():
                combined_relation = (
                    previous_relation[0] or relation[0],
                    previous_relation[1] or relation[1],
                )
                combined_chain = (*previous_chain, current)
                existing_chain = states.get(combined_relation)

                if existing_chain is None or _passage_evidence_chain_quality(
                    combined_chain
                ) > _passage_evidence_chain_quality(existing_chain):
                    states[combined_relation] = combined_chain

        states_by_end.append(states)

    return [
        states[(True, True)]
        for states in states_by_end
        if (True, True) in states and len(states[(True, True)]) >= 2
    ]


def infer_enclosing_passage_match(
    transcription: str,
    ranked_matches: list[RankedVerseCandidate],
    candidates: tuple[QuranVerseCandidate, ...],
    acceptance: MatchAcceptance,
) -> RankedVerseCandidate | None:
    """Infère prudemment une plage englobante sans modifier l'acceptation.

    Une ambiguïté entre sourates reste intacte. Une ambiguïté interne à
    une sourate peut seulement produire un meilleur candidat manuel ; elle
    n'est jamais promue en résultat accepté.
    """
    if not ranked_matches or (
        not acceptance.accepted and acceptance.reason != "ambiguous_match"
    ):
        return None

    top_match = ranked_matches[0]
    competing_match = acceptance.competing_match

    if acceptance.reason == "ambiguous_match" and (
        competing_match is None
        or competing_match.candidate.sourate_id != top_match.candidate.sourate_id
    ):
        return None

    evidence = extract_passage_verse_evidence(
        transcription,
        candidates,
        top_match.candidate.sourate_id,
    )
    supported_chains = _find_supported_passage_chains(evidence, top_match)
    candidate_by_range = {
        (
            candidate.sourate_id,
            candidate.start_verse,
            candidate.end_verse,
        ): (candidate_index, candidate)
        for candidate_index, candidate in enumerate(candidates)
        if candidate.sourate_id == top_match.candidate.sourate_id
    }
    proposals = []

    for chain in supported_chains:
        start_verse = min(
            top_match.candidate.start_verse,
            chain[0].candidate.start_verse,
        )
        end_verse = max(
            top_match.candidate.end_verse,
            chain[-1].candidate.end_verse,
        )

        if (
            start_verse == top_match.candidate.start_verse
            and end_verse == top_match.candidate.end_verse
        ):
            continue

        passage_entry = candidate_by_range.get(
            (top_match.candidate.sourate_id, start_verse, end_verse)
        )

        if passage_entry is None:
            continue

        if acceptance.reason == "ambiguous_match" and competing_match is not None:
            competing_candidate = competing_match.candidate

            if not (
                start_verse <= competing_candidate.start_verse
                and competing_candidate.end_verse <= end_verse
            ):
                continue

        candidate_index, passage_candidate = passage_entry
        passage_similarity = compute_similarity_score(
            transcription,
            passage_candidate.normalized_text,
        )

        if passage_similarity < MIN_ACCEPTED_SIMILARITY * 100:
            continue

        proposals.append(
            (
                _passage_evidence_chain_quality(chain),
                passage_similarity,
                candidate_index,
                passage_candidate,
            )
        )

    if not proposals:
        return None

    proposals.sort(key=lambda proposal: (proposal[0], proposal[1]), reverse=True)
    best_proposal = proposals[0]

    if len(proposals) > 1:
        next_proposal = proposals[1]
        same_primary_evidence = best_proposal[0][:2] == next_proposal[0][:2]
        different_range = (
            best_proposal[3].start_verse,
            best_proposal[3].end_verse,
        ) != (
            next_proposal[3].start_verse,
            next_proposal[3].end_verse,
        )

        if same_primary_evidence and different_range:
            return None

    _, passage_similarity, candidate_index, passage_candidate = best_proposal

    return RankedVerseCandidate(
        candidate_index=candidate_index,
        candidate=passage_candidate,
        score_percent=passage_similarity,
        matched_window=transcription,
        ranking_score_percent=compute_ranking_score(
            passage_similarity,
            transcription,
            passage_candidate.normalized_text,
        ),
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
    score_margin_percent = None

    if competing_match is not None:
        raw_score_margin_percent = (
            top_match.score_percent - competing_match.score_percent
        )
        ranking_score_margin_percent = (
            effective_ranking_score(top_match)
            - effective_ranking_score(competing_match)
        )
        top_evidence_word_count = min(
            len(top_match.matched_window.split()),
            len(top_match.candidate.normalized_text.split()),
        )
        has_strong_length_evidence = (
            top_match.score_percent >= MIN_STRONG_EVIDENCE_SIMILARITY
            and top_evidence_word_count >= MIN_STRONG_EVIDENCE_WORD_COUNT
            and ranking_score_margin_percent >= MIN_SCORE_MARGIN * 100
        )
        score_margin_percent = (
            ranking_score_margin_percent
            if has_strong_length_evidence
            else raw_score_margin_percent
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
            candidates=(),
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

    is_probable_match = (
        acceptance.reason == "score_too_low"
        and score >= MIN_PROBABLE_SIMILARITY
    )
    is_supported_probable_proposal = (
        is_probable_match
        and score >= MIN_PROPOSAL_SIMILARITY
        and acceptance.matched_word_count >= MIN_PROPOSAL_MATCHED_WORD_COUNT
        and (
            score_margin is None
            or score_margin >= MIN_PROPOSAL_SCORE_MARGIN
        )
    )
    should_include_verse = acceptance.accepted or (
        include_ambiguous_verse
        and (
            acceptance.reason == "ambiguous_match"
            or is_supported_probable_proposal
        )
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
        candidates=build_candidate_evidence(ranked_matches),
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
    inferred_match = None
    outcome_acceptance = acceptance
    matches_for_outcome = ranked_matches

    if acceptance.accepted or include_ambiguous_verse:
        inferred_match = infer_enclosing_passage_match(
            transcription,
            ranked_matches,
            candidates,
            acceptance,
        )

    if inferred_match is not None:
        inferred_matches = [inferred_match, *ranked_matches]
        inferred_acceptance = assess_match_acceptance(inferred_matches)

        if acceptance.accepted and inferred_acceptance.accepted:
            matches_for_outcome = inferred_matches
            outcome_acceptance = inferred_acceptance
        elif (
            acceptance.reason == "ambiguous_match"
            and inferred_acceptance.accepted
            and inferred_match.score_percent
            >= MIN_CONFIDENT_INFERRED_PASSAGE_SIMILARITY
        ):
            # Une ambiguïté interne peut être levée lorsque plusieurs ancres
            # uniques reconstruisent un passage long, très similaire, dont le
            # classement pondéré domine désormais tous les concurrents.
            matches_for_outcome = inferred_matches
            outcome_acceptance = inferred_acceptance
        elif acceptance.reason == "ambiguous_match":
            matches_for_outcome = inferred_matches
            # Une inférence plus faible améliore seulement les bornes proposées
            # en revue manuelle, sans fabriquer une décision confiante.
            outcome_acceptance = MatchAcceptance(
                accepted=False,
                reason="ambiguous_match",
                matched_word_count=inferred_acceptance.matched_word_count,
                score_margin_percent=inferred_acceptance.score_margin_percent,
                competing_match=inferred_acceptance.competing_match,
            )
        else:
            # Le passage élargi n'est pas aussi solide que le meilleur passage
            # initial : conserver la prédiction courte plutôt que de lui prêter
            # une confiance qui ne lui appartient pas.
            inferred_match = None

    outcome = build_detection_outcome(
        matches_for_outcome,
        outcome_acceptance,
        include_ambiguous_verse=include_ambiguous_verse,
    )

    if inferred_match is not None:
        logger.info(
            "Verse passage inferred: sourate=%s start_verse=%s end_verse=%s anchor_decision=%s",
            inferred_match.candidate.sourate_id,
            inferred_match.candidate.start_verse,
            inferred_match.candidate.end_verse,
            "accepted" if acceptance.accepted else acceptance.reason,
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
