import json
from pathlib import Path


BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "audio_quality_baseline.json"
)


def test_versioned_audio_quality_snapshot_is_internally_consistent():
    """Ce test valide le JSON versionné ; il ne rejoue ni audio ni Whisper."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    corpus = baseline["corpus"]
    metrics = baseline["metrics"]
    gates = baseline["release_gates"]

    assert baseline["verification"] == {
        "kind": "manual_measured_snapshot",
        "automated_test_scope": "json_integrity_only",
        "whisper_replayed_in_ci": False,
    }
    assert corpus["positive_cases"] + corpus["negative_cases"] == corpus["total_cases"]
    assert len(set(corpus["positive_surahs"])) == corpus["distinct_positive_surahs"]
    assert corpus["positive_source_cases"] >= gates["min_positive_source_cases"]
    assert corpus["distinct_positive_surahs"] >= gates["min_distinct_positive_surahs"]
    assert (
        corpus["noisy_positive_source_cases"]
        >= gates["min_noisy_positive_source_cases"]
    )
    assert (
        corpus["vocal_negative_source_cases"]
        >= gates["min_vocal_negative_source_cases"]
    )
    assert corpus["errors"] <= gates["max_errors"]
    assert sum(metrics["detection_status_counts"].values()) == corpus["total_cases"]
    assert (
        metrics["exact_match_count"]
        + metrics["correct_surah_wrong_range_count"]
        + metrics["wrong_surah_count"]
        + metrics["false_negative_count"]
        == corpus["positive_cases"]
    )
    assert (
        metrics["true_negative_count"] + metrics["false_positive_count"]
        == corpus["negative_cases"]
    )
    assert (
        metrics["macro_positive_exact_accuracy"]
        >= gates["min_macro_positive_exact_accuracy"]
    )
    assert (
        metrics["macro_positive_surah_accuracy"]
        >= gates["min_macro_positive_surah_accuracy"]
    )
    assert (
        metrics["macro_positive_confident_exact_accuracy"]
        >= gates["min_macro_positive_confident_exact_accuracy"]
    )
    assert (
        metrics["macro_positive_confident_surah_accuracy"]
        >= gates["min_macro_positive_confident_surah_accuracy"]
    )
    assert metrics["confident_exact_match"] <= metrics["confident_correct_surah"]
    assert (
        metrics["confident_correct_surah"]
        <= metrics["confident_positive_predictions"]
        <= corpus["positive_cases"]
    )
    assert (
        metrics["macro_negative_rejection_rate"]
        >= gates["min_macro_negative_rejection_rate"]
    )
    assert (
        metrics["macro_false_positive_rate"]
        <= gates["max_macro_false_positive_rate"]
    )
