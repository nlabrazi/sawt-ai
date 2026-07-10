import pytest
from rapidfuzz import fuzz

import app.services.verse_detection_service as verse_detection_service
from app.core.model_loader import QuranVerseCandidate


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


def test_detect_versets_returns_normalized_rapidfuzz_score(monkeypatch):
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

    expected_score = fuzz.ratio("قل هو الله احد", "قل هو الله") / 100

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
