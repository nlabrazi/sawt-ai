import pytest
from rapidfuzz import fuzz

import app.services.verse_detection_service as verse_detection_service
from app.core.model_loader import QuranVerseCandidate


FATIHA_VERSES = (
    "بسم الله الرحمن الرحيم",
    "الحمد لله رب العلمين",
    "الرحمن الرحيم",
    "ملك يوم الدين",
    "اياك نعبد واياك نستعين",
    "اهدنا الصرط المستقيم",
    "صرط الذين انعمت عليهم غير المغضوب عليهم ولا الضا لين",
)


def make_fatiha_candidate(start_verse: int, end_verse: int):
    return QuranVerseCandidate(
        sourate_id=1,
        sourate_name="الفاتحة",
        transliteration="Al-Fatihah",
        start_verse=start_verse,
        end_verse=end_verse,
        normalized_text=" ".join(FATIHA_VERSES[start_verse - 1:end_verse]),
    )


def make_fatiha_inference_candidates():
    return (
        *(make_fatiha_candidate(verse_id, verse_id) for verse_id in range(1, 8)),
        make_fatiha_candidate(1, 2),
        make_fatiha_candidate(5, 7),
        make_fatiha_candidate(4, 7),
        make_fatiha_candidate(2, 7),
        make_fatiha_candidate(1, 7),
        QuranVerseCandidate(
            112,
            "الإخلاص",
            "Al-Ikhlas",
            1,
            4,
            "قل هو الله احد الله الصمد لم يلد ولم يولد",
        ),
    )


@pytest.fixture
def representative_candidates():
    return (
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=2,
            normalized_text="قل هو الله احد الله الصمد",
        ),
        QuranVerseCandidate(
            sourate_id=113,
            sourate_name="الفلق",
            transliteration="Al-Falaq",
            start_verse=1,
            end_verse=2,
            normalized_text="قل اعوذ برب الفلق من شر ما خلق",
        ),
        QuranVerseCandidate(
            sourate_id=114,
            sourate_name="الناس",
            transliteration="An-Nas",
            start_verse=1,
            end_verse=2,
            normalized_text="قل اعوذ برب الناس ملك الناس",
        ),
    )


@pytest.mark.parametrize(
    ("case_name", "transcription", "expected_surah_id"),
    [
        (
            "diacritics",
            "قُلْ هُوَ اللَّهُ أَحَدٌ اللَّهُ الصَّمَدُ",
            112,
        ),
        (
            "punctuation",
            "قل، هو الله أحد! الله الصمد.",
            112,
        ),
        (
            "partial_verse",
            "قل اعوذ برب الفلق",
            113,
        ),
        (
            "extra_transcribed_words",
            "اعوذ بالله قل اعوذ برب الناس ملك الناس من الجنه والناس",
            114,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and value.isascii() else None,
)
def test_detect_versets_ranks_representative_recitations(
    monkeypatch,
    representative_candidates,
    case_name,
    transcription,
    expected_surah_id,
):
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: representative_candidates,
    )

    match = verse_detection_service.detect_versets([{"text": transcription}])

    assert match is not None, case_name
    assert match["sourate_id"] == expected_surah_id, case_name
    assert 0 <= match["similarity"] <= 1, case_name


def test_detect_versets_rejects_short_transcription(monkeypatch):
    candidates = (
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=1,
            normalized_text="قل هو الله احد",
        ),
        QuranVerseCandidate(
            sourate_id=113,
            sourate_name="الفلق",
            transliteration="Al-Falaq",
            start_verse=1,
            end_verse=1,
            normalized_text="قل اعوذ برب الفلق",
        ),
    )
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    match = verse_detection_service.detect_versets([{"text": "قل"}])

    assert match is None


def test_detect_versets_returns_best_precomputed_match(monkeypatch):
    candidates = (
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=1,
            normalized_text="قل",
        ),
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=2,
            normalized_text="قل هو الله",
        ),
    )

    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    match = verse_detection_service.detect_versets([
        {"text": "قُلْ هُوَ اللَّهُ"},
    ])

    assert match == {
        "sourate_id": 112,
        "sourate_name": "الإخلاص",
        "transliteration": "Al-Ikhlas",
        "start_verse": 1,
        "end_verse": 2,
        "text": "قل هو الله",
        "similarity": 1.0,
    }


def test_detect_versets_skips_candidate_scan_for_empty_transcription(monkeypatch):
    candidate_calls = []

    def fail_if_called():
        candidate_calls.append(True)
        raise AssertionError("Candidates should not be loaded for an empty transcription")

    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        fail_if_called,
    )

    match = verse_detection_service.detect_versets([
        {"text": "   "},
        {"text": ""},
    ])

    assert match is None
    assert candidate_calls == []


def test_compute_similarity_score_combines_local_and_token_scores():
    transcription = "قل هو الله احد"
    candidate = "قل هو الله"

    score = verse_detection_service.compute_similarity_score(transcription, candidate)
    expected_score = (
        0.7 * fuzz.partial_ratio(transcription, candidate)
        + 0.3 * fuzz.token_sort_ratio(transcription, candidate)
    )

    assert score == pytest.approx(expected_score)
    assert fuzz.ratio(transcription, candidate) < score < 100


def test_compute_ranking_score_rewards_longer_consistent_evidence():
    short_score = verse_detection_service.compute_ranking_score(
        100,
        "بسم الله الرحمن الرحيم",
        "بسم الله الرحمن الرحيم",
    )
    long_score = verse_detection_service.compute_ranking_score(
        98,
        " ".join(["كلمه"] * 24),
        " ".join(["كلمه"] * 24),
    )

    assert long_score > short_score
    assert short_score < 100
    assert long_score == pytest.approx(98)


def test_build_transcription_windows_keeps_full_text_and_overlaps_chunks():
    transcription = " ".join(f"word{index}" for index in range(10))

    windows = verse_detection_service.build_transcription_windows(transcription)

    assert windows == (
        transcription,
        "word0 word1 word2 word3",
        "word2 word3 word4 word5",
        "word4 word5 word6 word7",
        "word6 word7 word8 word9",
        "word0 word1 word2 word3 word4 word5 word6 word7",
        "word2 word3 word4 word5 word6 word7 word8 word9",
    )


def test_build_transcription_windows_limits_each_size_across_long_transcription():
    transcription = " ".join(f"word{index}" for index in range(100))

    windows = verse_detection_service.build_transcription_windows(transcription)
    four_word_windows = [window for window in windows if len(window.split()) == 4]

    assert len(four_word_windows) == verse_detection_service.MAX_WINDOWS_PER_SIZE
    assert four_word_windows[0] == "word0 word1 word2 word3"
    assert four_word_windows[-1] == "word96 word97 word98 word99"


def test_detect_versets_uses_best_sliding_window(monkeypatch):
    candidates = (
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=1,
            normalized_text="قل هو الله احد",
        ),
    )
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    match = verse_detection_service.detect_versets([
        {"text": "كلمات زائده في البدايه"},
        {"text": "قل هو الله احد"},
        {"text": "كلمات زائده في النهايه"},
    ])

    assert match is not None
    assert match["sourate_id"] == 112
    assert match["similarity"] == 1


def test_rank_verse_candidates_returns_two_unique_candidates_in_score_order(
    representative_candidates,
):
    ranked_matches = verse_detection_service.rank_verse_candidates(
        "قل اعوذ برب الناس ملك الناس",
        representative_candidates,
    )

    assert [match.candidate.sourate_id for match in ranked_matches] == [114, 113]
    assert ranked_matches[0].score_percent > ranked_matches[1].score_percent
    assert ranked_matches[0].matched_window == "قل اعوذ برب الناس ملك الناس"


def test_rank_verse_candidates_deduplicates_candidate_across_windows():
    candidates = (
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=1,
            normalized_text="قل هو الله احد",
        ),
    )

    ranked_matches = verse_detection_service.rank_verse_candidates(
        "كلمات زائده قل هو الله احد كلمات اخري",
        candidates,
    )

    assert len(ranked_matches) == 1
    assert ranked_matches[0].candidate.sourate_id == 112
    assert ranked_matches[0].score_percent == 100


def test_assess_match_acceptance_rejects_close_competing_location():
    candidates = (
        QuranVerseCandidate(1, "First", "First", 1, 1, "نص اول واضح"),
        QuranVerseCandidate(2, "Second", "Second", 1, 1, "نص ثان واضح"),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(0, candidates[0], 92, "نص كامل واضح"),
        verse_detection_service.RankedVerseCandidate(1, candidates[1], 88, "نص كامل واضح"),
    ]

    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    assert acceptance.accepted is False
    assert acceptance.reason == "ambiguous_match"
    assert acceptance.score_margin_percent == 4


def test_assess_match_acceptance_ignores_overlapping_candidate_window():
    candidates = (
        QuranVerseCandidate(112, "Al-Ikhlas", "Al-Ikhlas", 1, 2, "نص اول واضح"),
        QuranVerseCandidate(112, "Al-Ikhlas", "Al-Ikhlas", 1, 1, "نص اول"),
        QuranVerseCandidate(113, "Al-Falaq", "Al-Falaq", 1, 1, "نص مختلف"),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(0, candidates[0], 96, "نص كامل واضح"),
        verse_detection_service.RankedVerseCandidate(1, candidates[1], 94, "نص كامل واضح"),
        verse_detection_service.RankedVerseCandidate(2, candidates[2], 80, "نص كامل واضح"),
    ]

    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    assert acceptance.accepted is True
    assert acceptance.competing_match == ranked_matches[2]
    assert acceptance.score_margin_percent == 16


def test_assess_match_acceptance_uses_evidence_weighted_margin():
    long_passage = "un deux trois quatre cinq six sept huit neuf dix onze douze"
    candidates = (
        QuranVerseCandidate(113, "Al-Falaq", "Al-Falaq", 1, 5, long_passage),
        QuranVerseCandidate(1, "Al-Fatihah", "Al-Fatihah", 1, 1, "passage court"),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            0,
            candidates[0],
            97.5,
            long_passage,
            ranking_score_percent=96,
        ),
        verse_detection_service.RankedVerseCandidate(
            1,
            candidates[1],
            100,
            "preuve courte",
            ranking_score_percent=85,
        ),
    ]

    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    assert acceptance.accepted is True
    assert acceptance.reason is None
    assert acceptance.score_margin_percent == 11


def test_evidence_weighted_margin_cannot_promote_a_short_exact_phrase():
    candidates = (
        QuranVerseCandidate(1, "Al-Fatihah", "Al-Fatihah", 1, 1, "mot un deux trois"),
        QuranVerseCandidate(113, "Al-Falaq", "Al-Falaq", 1, 1, "mot un deux autre"),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            0,
            candidates[0],
            100,
            candidates[0].normalized_text,
            ranking_score_percent=95,
        ),
        verse_detection_service.RankedVerseCandidate(
            1,
            candidates[1],
            96,
            candidates[1].normalized_text,
            ranking_score_percent=80,
        ),
    ]

    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    assert acceptance.accepted is False
    assert acceptance.reason == "ambiguous_match"
    assert acceptance.score_margin_percent == 4


def test_rank_and_acceptance_prefer_candidate_covering_overlapping_windows():
    candidates = (
        QuranVerseCandidate(113, "Al-Falaq", "Al-Falaq", 1, 1, "قل اعوذ برب الفلق"),
        QuranVerseCandidate(113, "Al-Falaq", "Al-Falaq", 2, 2, "من شر ما خلق"),
        QuranVerseCandidate(
            113,
            "Al-Falaq",
            "Al-Falaq",
            1,
            2,
            "قل اعوذ برب الفلق من شر ما خلق",
        ),
        QuranVerseCandidate(114, "An-Nas", "An-Nas", 1, 1, "قل اعوذ برب الناس"),
    )
    transcription = "قل اعوذ برب الفلق من شر ما خلق"

    ranked_matches = verse_detection_service.rank_verse_candidates(
        transcription,
        candidates,
        limit=4,
    )
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    assert ranked_matches[0].candidate.start_verse == 1
    assert ranked_matches[0].candidate.end_verse == 2
    assert acceptance.accepted is True
    assert acceptance.competing_match is not None
    assert acceptance.competing_match.candidate.sourate_id == 114


def test_rank_verse_candidates_prefers_complete_fatiha_over_exact_short_window():
    full_fatiha = (
        "بسم الله الرحمن الرحيم الحمد لله رب العلمين الرحمن الرحيم ملك يوم الدين "
        "اياك نعبد واياك نستعين اهدنا الصرط المستقيم صرط الذين انعمت عليهم "
        "غير المغضوب عليهم ولا الضالين"
    )
    candidates = (
        QuranVerseCandidate(
            1,
            "Al-Fatihah",
            "Al-Fatihah",
            1,
            1,
            "بسم الله الرحمن الرحيم",
        ),
        QuranVerseCandidate(
            1,
            "Al-Fatihah",
            "Al-Fatihah",
            1,
            7,
            full_fatiha,
        ),
        QuranVerseCandidate(
            112,
            "Al-Ikhlas",
            "Al-Ikhlas",
            1,
            4,
            "قل هو الله احد الله الصمد لم يلد ولم يولد",
        ),
    )

    ranked_matches = verse_detection_service.rank_verse_candidates(
        full_fatiha.replace("الدين", "الديت"),
        candidates,
        limit=3,
    )
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    assert ranked_matches[0].candidate.sourate_id == 1
    assert ranked_matches[0].candidate.start_verse == 1
    assert ranked_matches[0].candidate.end_verse == 7
    assert len(ranked_matches[0].matched_window.split()) > 20
    assert acceptance.accepted is True


@pytest.mark.parametrize(
    "transcription",
    [
        "ceci est un texte français sans rapport avec le coran",
        "هذا حديث يومي عن العمل والطقس وليس تلاوه",
    ],
)
def test_detect_versets_rejects_text_outside_quran(
    monkeypatch,
    representative_candidates,
    transcription,
):
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: representative_candidates,
    )

    outcome = verse_detection_service.detect_verse_with_metadata([
        {"text": transcription},
    ])

    assert outcome.verse is None
    assert outcome.status in {"insufficient", "probable"}


def test_assess_match_acceptance_rejects_low_score():
    candidate = QuranVerseCandidate(1, "First", "First", 1, 1, "نص اول واضح")
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(0, candidate, 79, "نص كامل واضح"),
    ]

    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    assert acceptance.accepted is False
    assert acceptance.reason == "score_too_low"


def test_detect_versets_returns_normalized_combined_score(monkeypatch):
    candidates = (
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=2,
            normalized_text="قل هو الله",
        ),
    )

    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    match = verse_detection_service.detect_versets([
        {"text": "قل هو الله احد"},
    ])

    expected_score = (
        verse_detection_service.compute_similarity_score(
            "قل هو الله احد",
            "قل هو الله",
        )
        / 100
    )

    assert match is not None
    assert match["similarity"] == pytest.approx(expected_score)
    assert 0 < match["similarity"] < 1


def test_rank_verse_candidates_keeps_catalog_order_when_scores_are_tied():
    candidates = (
        QuranVerseCandidate(
            sourate_id=1,
            sourate_name="الأولى",
            transliteration="First",
            start_verse=1,
            end_verse=1,
            normalized_text="نص كامل واضح",
        ),
        QuranVerseCandidate(
            sourate_id=2,
            sourate_name="الثانية",
            transliteration="Second",
            start_verse=1,
            end_verse=1,
            normalized_text="نص كامل واضح",
        ),
    )
    ranked_matches = verse_detection_service.rank_verse_candidates(
        "نص كامل واضح",
        candidates,
    )

    assert [match.candidate.sourate_id for match in ranked_matches] == [1, 2]


def test_detect_versets_returns_none_without_candidates(monkeypatch):
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: (),
    )

    assert verse_detection_service.detect_versets([{"text": "نص"}]) is None


def test_detect_verse_with_metadata_exposes_confident_decision(monkeypatch):
    candidates = (
        QuranVerseCandidate(
            sourate_id=112,
            sourate_name="الإخلاص",
            transliteration="Al-Ikhlas",
            start_verse=1,
            end_verse=1,
            normalized_text="قل هو الله احد",
        ),
    )
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    outcome = verse_detection_service.detect_verse_with_metadata([
        {"text": "قل هو الله احد"},
    ])

    assert outcome.verse is not None
    assert outcome.metadata() == {
        "status": "confident",
        "score": 1,
        "score_margin": None,
        "matched_word_count": 4,
        "rejection_reason": None,
    }


def test_build_detection_outcome_exposes_ambiguous_decision():
    candidates = (
        QuranVerseCandidate(1, "First", "First", 1, 1, "نص اول واضح"),
        QuranVerseCandidate(2, "Second", "Second", 1, 1, "نص ثان واضح"),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(0, candidates[0], 92, "نص كامل واضح"),
        verse_detection_service.RankedVerseCandidate(1, candidates[1], 88, "نص كامل واضح"),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    outcome = verse_detection_service.build_detection_outcome(ranked_matches, acceptance)

    assert outcome.verse is None
    assert outcome.metadata() == {
        "status": "ambiguous",
        "score": 0.92,
        "score_margin": 0.04,
        "matched_word_count": 3,
        "rejection_reason": "ambiguous_match",
    }


def test_build_detection_outcome_can_expose_ambiguous_candidate_for_manual_result():
    candidates = (
        QuranVerseCandidate(1, "First", "First", 1, 1, "نص اول واضح"),
        QuranVerseCandidate(2, "Second", "Second", 1, 1, "نص ثان واضح"),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(0, candidates[0], 92, "نص كامل واضح"),
        verse_detection_service.RankedVerseCandidate(1, candidates[1], 88, "نص كامل واضح"),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    outcome = verse_detection_service.build_detection_outcome(
        ranked_matches,
        acceptance,
        include_ambiguous_verse=True,
    )

    assert outcome.verse is not None
    assert outcome.verse["sourate_id"] == 1
    assert outcome.status == "ambiguous"
    assert outcome.rejection_reason == "ambiguous_match"


def test_build_detection_outcome_classifies_low_score_as_probable():
    candidate = QuranVerseCandidate(1, "First", "First", 1, 1, "نص اول واضح")
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            0,
            candidate,
            74,
            "نص كامل واضح",
        ),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    outcome = verse_detection_service.build_detection_outcome(ranked_matches, acceptance)

    assert outcome.verse is None
    assert outcome.status == "probable"
    assert outcome.score == 0.74
    assert outcome.rejection_reason == "score_too_low"


def test_detect_verse_infers_enclosing_fatiha_when_middle_verses_are_missing(
    monkeypatch,
):
    candidates = make_fatiha_inference_candidates()
    transcription = (
        "بسم الله الرحمن الرحيم الحمد لله رب العالمين "
        "وإياك نعبد وإياك نستعين إهدنا الصراط المستقيم "
        "صراط الذين أن أنت عليهم غير المضب عليهم ولا الضالين"
    )
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    hidden_outcome = verse_detection_service.detect_verse_with_metadata([
        {"text": transcription},
    ])
    manual_outcome = verse_detection_service.detect_verse_with_metadata(
        [{"text": transcription}],
        include_ambiguous_verse=True,
    )

    assert hidden_outcome.verse is None
    assert hidden_outcome.status == "ambiguous"
    assert hidden_outcome.rejection_reason == "ambiguous_match"
    assert manual_outcome.verse is not None
    assert manual_outcome.verse["sourate_id"] == 1
    assert manual_outcome.verse["start_verse"] == 1
    assert manual_outcome.verse["end_verse"] == 7
    assert manual_outcome.status == "ambiguous"
    assert manual_outcome.rejection_reason == "ambiguous_match"
    assert 0.8 <= manual_outcome.score < hidden_outcome.score
    assert manual_outcome.verse["similarity"] == manual_outcome.score
    assert manual_outcome.matched_word_count == len(transcription.split())


def test_detect_verse_uses_ranked_passage_to_bridge_supported_boundaries(
    monkeypatch,
):
    candidates = make_fatiha_inference_candidates()
    transcription = (
        "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم "
        "الحمد لله رب العالمين الرحمن الرحيم الرحمن الرحيم ملك يومين "
        "اياك نعبد واياك نسعين اهدنا الصرط المستقيم "
        "صرط الذين انعمت عليهم غير المرضوب عليهم ولن ضالين"
    )
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    outcome = verse_detection_service.detect_verse_with_metadata(
        [{"text": transcription}],
        include_ambiguous_verse=True,
    )

    assert outcome.status == "ambiguous"
    assert outcome.rejection_reason == "ambiguous_match"
    assert outcome.verse is not None
    assert outcome.verse["start_verse"] == 1
    assert outcome.verse["end_verse"] == 7


def test_passage_inference_does_not_expand_from_a_single_outside_anchor():
    outside_anchor = make_fatiha_candidate(1, 1)
    top_candidate = make_fatiha_candidate(5, 7)
    candidates = (
        outside_anchor,
        top_candidate,
        make_fatiha_candidate(1, 7),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            1,
            top_candidate,
            94,
            top_candidate.normalized_text,
        ),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    inferred_match = verse_detection_service.infer_enclosing_passage_match(
        outside_anchor.normalized_text,
        ranked_matches,
        candidates,
        acceptance,
    )

    assert acceptance.accepted is True
    assert inferred_match is None


def test_passage_inference_rejects_verse_evidence_recited_out_of_order():
    first_verse = make_fatiha_candidate(1, 1)
    fifth_verse = make_fatiha_candidate(5, 5)
    top_candidate = make_fatiha_candidate(5, 7)
    candidates = (
        first_verse,
        fifth_verse,
        top_candidate,
        make_fatiha_candidate(1, 7),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            2,
            top_candidate,
            94,
            top_candidate.normalized_text,
        ),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)
    reversed_transcription = (
        f"{fifth_verse.normalized_text} {first_verse.normalized_text}"
    )

    inferred_match = verse_detection_service.infer_enclosing_passage_match(
        reversed_transcription,
        ranked_matches,
        candidates,
        acceptance,
    )

    assert inferred_match is None


def test_passage_inference_does_not_count_repeated_verse_as_multiple_evidence():
    repeated_verse = make_fatiha_candidate(1, 1)
    top_candidate = make_fatiha_candidate(5, 7)
    candidates = (
        repeated_verse,
        top_candidate,
        make_fatiha_candidate(1, 7),
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            1,
            top_candidate,
            94,
            top_candidate.normalized_text,
        ),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)
    repeated_transcription = " ".join([repeated_verse.normalized_text] * 3)

    inferred_match = verse_detection_service.infer_enclosing_passage_match(
        repeated_transcription,
        ranked_matches,
        candidates,
        acceptance,
    )

    assert inferred_match is None


def test_passage_inference_ignores_refrain_shared_by_multiple_verses():
    refrain = "فباي الاء ربكما تكذبان"
    first_refrain = QuranVerseCandidate(55, "الرحمن", "Ar-Rahman", 13, 13, refrain)
    second_refrain = QuranVerseCandidate(55, "الرحمن", "Ar-Rahman", 16, 16, refrain)
    top_candidate = QuranVerseCandidate(
        55,
        "الرحمن",
        "Ar-Rahman",
        16,
        17,
        f"{refrain} خلق الانسان من صلصال كالفخار",
    )
    enclosing_candidate = QuranVerseCandidate(
        55,
        "الرحمن",
        "Ar-Rahman",
        13,
        17,
        f"{refrain} وله الجوار المنشات في البحر كالاعلم {refrain} "
        "خلق الانسان من صلصال كالفخار",
    )
    candidates = (
        first_refrain,
        second_refrain,
        top_candidate,
        enclosing_candidate,
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            2,
            top_candidate,
            96,
            top_candidate.normalized_text,
        ),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    inferred_match = verse_detection_service.infer_enclosing_passage_match(
        f"{refrain} {refrain}",
        ranked_matches,
        candidates,
        acceptance,
    )

    assert inferred_match is None


def test_passage_inference_requires_anchor_unique_across_the_quran():
    duplicated_text = FATIHA_VERSES[0]
    candidates = (
        make_fatiha_candidate(1, 1),
        QuranVerseCandidate(
            2,
            "البقرة",
            "Al-Baqarah",
            1,
            1,
            duplicated_text,
        ),
        make_fatiha_candidate(5, 5),
    )

    evidence = verse_detection_service.extract_passage_verse_evidence(
        f"{duplicated_text} {FATIHA_VERSES[4]}",
        candidates,
        sourate_id=1,
    )

    assert [item.candidate.start_verse for item in evidence] == [5]


def test_inferred_passage_does_not_inherit_confidence_from_shorter_match(
    monkeypatch,
):
    top_candidate = make_fatiha_candidate(5, 7)
    inferred_candidate = make_fatiha_candidate(1, 7)
    competing_candidate = QuranVerseCandidate(
        2,
        "البقرة",
        "Al-Baqarah",
        255,
        255,
        "الله لا اله الا هو الحي القيوم",
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            0,
            top_candidate,
            94,
            top_candidate.normalized_text,
        ),
        verse_detection_service.RankedVerseCandidate(
            2,
            competing_candidate,
            80,
            competing_candidate.normalized_text,
        ),
    ]
    inferred_match = verse_detection_service.RankedVerseCandidate(
        1,
        inferred_candidate,
        75,
        inferred_candidate.normalized_text,
    )
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: (top_candidate, inferred_candidate, competing_candidate),
    )
    monkeypatch.setattr(
        verse_detection_service,
        "rank_verse_candidates",
        lambda *_args, **_kwargs: ranked_matches,
    )
    monkeypatch.setattr(
        verse_detection_service,
        "infer_enclosing_passage_match",
        lambda *_args, **_kwargs: inferred_match,
    )

    outcome = verse_detection_service.detect_verse_with_metadata(
        [{"text": top_candidate.normalized_text}],
        include_ambiguous_verse=True,
    )

    assert outcome.status == "confident"
    assert outcome.verse is not None
    assert outcome.verse["start_verse"] == 5
    assert outcome.verse["end_verse"] == 7
    assert outcome.score == 0.94
    assert outcome.score_margin == 0.14


def test_passage_inference_preserves_cross_surah_ambiguity():
    top_candidate = make_fatiha_candidate(5, 7)
    competing_candidate = QuranVerseCandidate(
        2,
        "البقرة",
        "Al-Baqarah",
        1,
        1,
        "الم ذلك الكتب لا ريب فيه",
    )
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            0,
            top_candidate,
            94,
            top_candidate.normalized_text,
        ),
        verse_detection_service.RankedVerseCandidate(
            1,
            competing_candidate,
            90,
            competing_candidate.normalized_text,
        ),
    ]
    acceptance = verse_detection_service.assess_match_acceptance(ranked_matches)

    inferred_match = verse_detection_service.infer_enclosing_passage_match(
        " ".join(FATIHA_VERSES),
        ranked_matches,
        make_fatiha_inference_candidates(),
        acceptance,
    )

    assert acceptance.reason == "ambiguous_match"
    assert inferred_match is None


def test_strong_inferred_passage_can_resolve_internal_ambiguity(monkeypatch):
    short_candidate = make_fatiha_candidate(5, 7)
    competing_candidate = make_fatiha_candidate(1, 2)
    inferred_candidate = make_fatiha_candidate(1, 7)
    ranked_matches = [
        verse_detection_service.RankedVerseCandidate(
            0,
            short_candidate,
            96,
            short_candidate.normalized_text,
            ranking_score_percent=92,
        ),
        verse_detection_service.RankedVerseCandidate(
            1,
            competing_candidate,
            95,
            competing_candidate.normalized_text,
            ranking_score_percent=89,
        ),
    ]
    inferred_match = verse_detection_service.RankedVerseCandidate(
        2,
        inferred_candidate,
        98,
        inferred_candidate.normalized_text,
        ranking_score_percent=98,
    )
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: (short_candidate, competing_candidate, inferred_candidate),
    )
    monkeypatch.setattr(
        verse_detection_service,
        "rank_verse_candidates",
        lambda *_args, **_kwargs: ranked_matches,
    )
    monkeypatch.setattr(
        verse_detection_service,
        "infer_enclosing_passage_match",
        lambda *_args, **_kwargs: inferred_match,
    )

    outcome = verse_detection_service.detect_verse_with_metadata(
        [{"text": inferred_candidate.normalized_text}],
        include_ambiguous_verse=True,
    )

    assert outcome.status == "confident"
    assert outcome.rejection_reason is None
    assert outcome.verse is not None
    assert outcome.verse["start_verse"] == 1
    assert outcome.verse["end_verse"] == 7


def test_passage_inference_does_not_rescue_unrelated_arabic_text(monkeypatch):
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        make_fatiha_inference_candidates,
    )

    outcome = verse_detection_service.detect_verse_with_metadata(
        [{"text": "هذا حديث يومي عن العمل والطقس وليس تلاوه من القران"}],
        include_ambiguous_verse=True,
    )

    assert outcome.verse is None
    assert outcome.status in {"insufficient", "probable"}
