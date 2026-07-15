import json
import os
import runpy
import sys
from pathlib import Path

import pytest

from evaluation.audio_benchmark.corpus import BuiltAudioCorpus


def load_runner_namespace():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_audio_recognition.py"
    return runpy.run_path(str(script_path), run_name="audio_runner_import_test")


def test_audio_runner_forces_huggingface_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")

    load_runner_namespace()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_audio_runner_supports_commit_without_transcription_policy():
    namespace = load_runner_namespace()

    def missing_policy(module_name):
        raise ModuleNotFoundError(name=module_name)

    configuration = namespace["_build_transcription_policy_configuration"](
        missing_policy
    )

    assert configuration == {"available": False, "thresholds": {}}


def quality_ready_report():
    categories = {
        category: {
            "negative_source_cases": 1,
            "positive_source_cases": 0,
        }
        for category in (
            "french_speech",
            "french_conversation",
            "vocal_music",
            "arabic_non_quran",
        )
    }
    return {
        "summary": {
            "errors": 0,
            "positive_source_cases": 3,
            "negative_source_cases": 4,
            "noisy_positive_source_cases": 3,
            "vocal_negative_source_cases": 4,
            "macro_positive_exact_accuracy": 0.95,
            "macro_negative_rejection_rate": 1.0,
            "macro_false_positive_rate": 0.0,
        },
        "categories": categories,
        "variants": {},
        "cases": [{"audio_sha256": "private-audio-hash"}],
        "skipped": [],
        "corpus_manifest_sha256": "private-manifest-hash",
        "corpus_privacy": "private",
    }


def configure_private_runner(namespace, report):
    corpus = BuiltAudioCorpus(
        Path("private-manifest.json"),
        16_000,
        (),
        (),
        privacy="private",
    )
    runner_globals = namespace["main"].__globals__
    runner_globals["load_built_corpus"] = lambda _path: corpus
    runner_globals["load_all_models"] = lambda: None
    runner_globals["evaluate_audio_corpus"] = lambda *_args, **_kwargs: report


def test_private_runner_requires_output(monkeypatch):
    namespace = load_runner_namespace()
    configure_private_runner(namespace, quality_ready_report())
    monkeypatch.setattr(sys, "argv", ["evaluate_audio_recognition.py"])

    with pytest.raises(SystemExit) as exc:
        namespace["main"]()

    assert exc.value.code == 2


def test_private_quality_report_is_0600_and_stdout_is_redacted(
    monkeypatch,
    tmp_path,
    capsys,
):
    namespace = load_runner_namespace()
    report = quality_ready_report()
    configure_private_runner(namespace, report)
    output_path = tmp_path / "reports" / "private.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_audio_recognition.py",
            "--mode",
            "quality",
            "--output",
            str(output_path),
        ],
    )

    namespace["main"]()

    console_report = json.loads(capsys.readouterr().out)
    private_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert output_path.parent.stat().st_mode & 0o777 == 0o700
    assert console_report["quality_gate"]["passed"] is True
    assert console_report["coverage"]["negative_category_source_cases"] == {
        "arabic_non_quran": 1,
        "french_conversation": 1,
        "french_speech": 1,
        "vocal_music": 1,
    }
    assert "cases" not in console_report
    assert "corpus_manifest_sha256" not in console_report
    assert "private-audio-hash" not in json.dumps(console_report)
    assert private_report["cases"][0]["audio_sha256"] == "private-audio-hash"
    assert private_report["quality_gate"]["metrics"] == {
        "macro_positive_exact_accuracy": 0.95,
        "macro_negative_rejection_rate": 1.0,
        "macro_false_positive_rate": 0.0,
    }
