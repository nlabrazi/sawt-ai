from pathlib import Path
from dataclasses import replace

import pytest

from evaluation.audio_benchmark.corpus import (
    BuiltAudioCase,
    BuiltAudioCorpus,
    ExpectedVerse,
)
from evaluation.audio_benchmark.evaluator import (
    evaluate_audio_corpus,
    evaluate_quality_gates,
)


def make_case(case_id, expected, category="quran_recitation"):
    return BuiltAudioCase(
        case_id=case_id,
        label="positive" if expected else "negative",
        category=category,
        audio_path=Path(f"/private/{case_id}.wav"),
        duration_seconds=2.0,
        expected_verse=expected,
        variant={"id": "clean"},
        tags=("test",),
    )


def pipeline_result(verse, transcription="texte privé", status="confident"):
    return {
        "verse": verse,
        "transcription_text": transcription,
        "detection": {
            "status": status,
            "score": 0.9 if verse else None,
            "rejection_reason": None if verse else "no_match",
        },
    }


def test_evaluate_audio_corpus_separates_positive_and_negative_failures():
    expected = ExpectedVerse(1, 1, 7)
    cases = (
        make_case("exact_case", expected),
        make_case("wrong_range", expected),
        make_case("wrong_surah", expected),
        make_case("false_negative", expected),
        make_case("true_negative", None, "french_speech"),
        make_case("false_positive", None, "music"),
    )
    outcomes = {
        "exact_case": pipeline_result({"sourate_id": 1, "start_verse": 1, "end_verse": 7}),
        "wrong_range": pipeline_result({"sourate_id": 1, "start_verse": 1, "end_verse": 5}),
        "wrong_surah": pipeline_result({"sourate_id": 2, "start_verse": 1, "end_verse": 2}),
        "false_negative": pipeline_result(None, status="insufficient"),
        "true_negative": pipeline_result(None, status="insufficient"),
        "false_positive": pipeline_result({"sourate_id": 97, "start_verse": 1, "end_verse": 1}),
    }
    clock_values = iter(value / 1000 for value in range(0, 12))
    corpus = BuiltAudioCorpus(Path("manifest.json"), 16_000, cases, ())

    report = evaluate_audio_corpus(
        corpus,
        lambda case: outcomes[case.case_id],
        clock=lambda: next(clock_values),
    )

    assert report["summary"] == {
        "total_cases": 6,
        "skipped_cases": 0,
        "errors": 0,
        "evaluated_cases": 6,
        "positive_cases": 4,
        "negative_cases": 2,
        "exact_match": 1,
        "correct_surah_wrong_range": 1,
        "wrong_surah": 1,
        "false_negative": 1,
        "true_negative": 1,
        "false_positive": 1,
        "overall_exact_accuracy": 2 / 6,
        "positive_exact_accuracy": 0.25,
        "positive_surah_accuracy": 0.5,
        "negative_rejection_rate": 0.5,
        "false_positive_rate": 0.5,
        "exact_precision": 0.25,
        "exact_recall": 0.25,
        "unique_source_cases": 6,
        "positive_source_cases": 4,
        "negative_source_cases": 2,
        "vocal_negative_cases": 0,
        "vocal_negative_source_cases": 0,
        "noisy_positive_cases": 0,
        "noisy_positive_source_cases": 0,
        "macro_positive_exact_accuracy": 0.25,
        "macro_negative_rejection_rate": 0.5,
        "macro_false_positive_rate": 0.5,
        "average_latency_ms": pytest.approx(1),
        "p50_latency_ms": pytest.approx(1),
        "p95_latency_ms": pytest.approx(1),
        "average_realtime_factor": pytest.approx(0.0005),
        "p95_realtime_factor": pytest.approx(0.0005),
        "status_counts": {"confident": 4, "insufficient": 2},
        "rejection_reason_counts": {"none": 4, "no_match": 2},
    }
    assert [case["classification"] for case in report["cases"]] == [
        "exact_match",
        "correct_surah_wrong_range",
        "wrong_surah",
        "false_negative",
        "true_negative",
        "false_positive",
    ]
    assert all("transcription_text" not in case for case in report["cases"])
    assert all("transcription_sha256" not in case for case in report["cases"])
    assert all("/private/" not in str(case) for case in report["cases"])
    assert report["categories"]["french_speech"]["negative_source_cases"] == 1


def test_evaluation_can_explicitly_include_transcriptions():
    case = make_case("negative_case", None, "french_speech")
    corpus = BuiltAudioCorpus(Path("manifest.json"), 16_000, (case,), ())

    report = evaluate_audio_corpus(
        corpus,
        lambda _case: pipeline_result(None, transcription="contenu français"),
        include_transcriptions=True,
    )

    assert report["cases"][0]["transcription_text"] == "contenu français"


def test_pipeline_error_does_not_count_as_a_quality_result():
    case = make_case("broken_case", None, "noise")
    corpus = BuiltAudioCorpus(Path("manifest.json"), 16_000, (case,), ())

    report = evaluate_audio_corpus(
        corpus,
        lambda _case: (_ for _ in ()).throw(RuntimeError("private path")),
    )

    assert report["summary"]["errors"] == 1
    assert report["summary"]["evaluated_cases"] == 0
    assert report["cases"][0]["error_type"] == "RuntimeError"
    assert "private path" not in str(report)


def test_variants_from_one_recording_count_as_one_source():
    expected = ExpectedVerse(112, 1, 4)
    clean = replace(
        make_case("ikhlas_clean", expected),
        source_case_id="ikhlas_reciter_01",
    )
    noisy = replace(
        make_case("ikhlas_noisy", expected),
        source_case_id="ikhlas_reciter_01",
        variant={"id": "pink_snr10", "noise": {"type": "pink", "snr_db": 10}},
    )
    corpus = BuiltAudioCorpus(Path("manifest.json"), 16_000, (clean, noisy), ())

    report = evaluate_audio_corpus(
        corpus,
        lambda case: (
            pipeline_result({"sourate_id": 112, "start_verse": 1, "end_verse": 4})
            if case.case_id == "ikhlas_clean"
            else pipeline_result(None, status="insufficient")
        ),
    )

    assert report["summary"]["positive_cases"] == 2
    assert report["summary"]["positive_source_cases"] == 1
    assert report["summary"]["noisy_positive_source_cases"] == 1
    assert report["summary"]["macro_positive_exact_accuracy"] == 0.5


def test_quality_gates_report_missing_sets_and_threshold_failures():
    failures = evaluate_quality_gates(
        {
            "errors": 1,
            "positive_cases": 0,
            "negative_cases": 2,
            "positive_source_cases": 0,
            "negative_source_cases": 2,
            "noisy_positive_source_cases": 0,
            "vocal_negative_source_cases": 0,
            "macro_positive_exact_accuracy": 0.0,
            "macro_negative_rejection_rate": 0.5,
            "macro_false_positive_rate": 0.5,
        },
        categories={"french_speech": {"negative_source_cases": 1}},
        min_macro_positive_exact_accuracy=0.9,
        min_macro_negative_rejection_rate=0.95,
        max_macro_false_positive_rate=0.01,
        min_positive_sources=3,
        min_noisy_positive_sources=3,
        min_vocal_negative_sources=2,
        required_negative_categories=(
            "french_speech",
            "french_conversation",
            "vocal_music",
            "arabic_non_quran",
        ),
    )

    assert failures == [
        "errors=1 dépasse la limite 0",
        "positive_source_cases=0 < 3",
        "noisy_positive_source_cases=0 < 3",
        "vocal_negative_source_cases=0 < 2",
        "negative_category_source_cases[arabic_non_quran]=0 < 1",
        "negative_category_source_cases[french_conversation]=0 < 1",
        "negative_category_source_cases[vocal_music]=0 < 1",
        "aucune source positive évaluée",
        "macro_negative_rejection_rate=0.500 < 0.950",
        "macro_false_positive_rate=0.500 > 0.010",
    ]
