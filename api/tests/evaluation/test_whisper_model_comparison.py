import runpy
from pathlib import Path


def load_comparison_namespace():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "compare_whisper_models.py"
    )
    return runpy.run_path(str(script_path), run_name="comparison_import_test")


def test_model_comparison_extracts_quality_and_performance_metrics():
    namespace = load_comparison_namespace()
    reports = {
        "turbo": {
            "summary": {
                "macro_positive_exact_accuracy": 0.8,
                "macro_negative_rejection_rate": 1.0,
                "average_latency_ms": 100,
            }
        },
        "large-v3": {
            "summary": {
                "macro_positive_exact_accuracy": 0.9,
                "macro_negative_rejection_rate": 0.99,
                "average_latency_ms": 220,
            }
        },
    }

    comparison = namespace["build_comparison_summary"](reports)

    assert comparison["turbo"]["macro_positive_exact_accuracy"] == 0.8
    assert comparison["large-v3"]["macro_negative_rejection_rate"] == 0.99
    assert comparison["large-v3"]["average_latency_ms"] == 220
    assert comparison["turbo"]["p95_latency_ms"] is None


def test_model_comparison_builds_strict_command_when_requested(tmp_path):
    namespace = load_comparison_namespace()

    command = namespace["build_benchmark_command"](
        corpus_path=tmp_path / "corpus.json",
        output_path=tmp_path / "turbo.json",
        allow_ambiguous_result=False,
    )

    assert "--no-allow-ambiguous-result" in command
    assert "--mode" in command
    assert "smoke" in command
