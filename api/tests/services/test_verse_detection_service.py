import pytest
from rapidfuzz import fuzz

import app.services.verse_detection_service as verse_detection_service
from app.core.model_loader import QuranVerseCandidate


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


def test_detect_versets_documents_short_transcription_ambiguity(monkeypatch):
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

    assert match is not None
    assert match["similarity"] < 0.8


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


def test_detect_versets_keeps_first_candidate_when_scores_are_tied(monkeypatch):
    candidates = (
        QuranVerseCandidate(
            sourate_id=1,
            sourate_name="الأولى",
            transliteration="First",
            start_verse=1,
            end_verse=1,
            normalized_text="نص",
        ),
        QuranVerseCandidate(
            sourate_id=2,
            sourate_name="الثانية",
            transliteration="Second",
            start_verse=1,
            end_verse=1,
            normalized_text="نص",
        ),
    )

    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: candidates,
    )

    match = verse_detection_service.detect_versets([{"text": "نص"}])

    assert match is not None
    assert match["sourate_id"] == 1


def test_detect_versets_returns_none_without_candidates(monkeypatch):
    monkeypatch.setattr(
        verse_detection_service,
        "get_quran_verse_candidates",
        lambda: (),
    )

    assert verse_detection_service.detect_versets([{"text": "نص"}]) is None
