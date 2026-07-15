import json
from pathlib import Path


BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "audio_quality_baseline.json"
)


def test_versioned_audio_quality_baseline_meets_its_release_gates():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    corpus = baseline["corpus"]
    metrics = baseline["metrics"]
    gates = baseline["release_gates"]

    assert corpus["positive_cases"] + corpus["negative_cases"] == corpus["total_cases"]
    assert corpus["errors"] <= gates["max_errors"]
    assert (
        metrics["macro_positive_exact_accuracy"]
        >= gates["min_macro_positive_exact_accuracy"]
    )
    assert (
        metrics["macro_positive_surah_accuracy"]
        >= gates["min_macro_positive_surah_accuracy"]
    )
    assert (
        metrics["macro_negative_rejection_rate"]
        >= gates["min_macro_negative_rejection_rate"]
    )
    assert (
        metrics["macro_false_positive_rate"]
        <= gates["max_macro_false_positive_rate"]
    )
