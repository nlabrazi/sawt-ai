#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

API_DIR = Path(__file__).resolve().parents[1]

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from evaluation.audio_benchmark.corpus import load_built_corpus
from evaluation.audio_benchmark.secure_io import atomic_write_text, prepare_directory

DEFAULT_CORPUS = API_DIR / "evaluation" / "generated" / "audio" / "manifest.json"
EVALUATOR = API_DIR / "scripts" / "evaluate_audio_recognition.py"
DEFAULT_MODELS = ("turbo", "large-v3")
COMPARISON_METRICS = (
    "macro_positive_exact_accuracy",
    "macro_positive_surah_accuracy",
    "macro_positive_confident_exact_accuracy",
    "macro_positive_confident_surah_accuracy",
    "macro_negative_rejection_rate",
    "macro_false_positive_rate",
    "average_latency_ms",
    "p95_latency_ms",
    "average_realtime_factor",
    "p95_realtime_factor",
)


def build_benchmark_command(
    *,
    corpus_path: Path,
    output_path: Path,
    allow_ambiguous_result: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--corpus",
        str(corpus_path),
        "--output",
        str(output_path),
        "--mode",
        "smoke",
    ]
    if not allow_ambiguous_result:
        command.append("--no-allow-ambiguous-result")
    return command


def build_comparison_summary(
    reports: Mapping[str, Mapping],
) -> dict[str, dict[str, object]]:
    return {
        model_name: {
            metric: report.get("summary", {}).get(metric)
            for metric in COMPARISON_METRICS
        }
        for model_name, report in reports.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Whisper models on the same offline audio corpus."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        choices=DEFAULT_MODELS,
        help="Model to evaluate (repeatable, defaults to turbo and large-v3).",
    )
    parser.add_argument(
        "--allow-ambiguous-result",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    corpus = load_built_corpus(corpus_path)
    output_dir = args.output_dir.resolve()
    private = corpus.privacy == "private"
    prepare_directory(output_dir, private=private, tighten_existing=private)
    reports = {}

    for model_name in args.model or DEFAULT_MODELS:
        report_path = output_dir / f"{model_name}.json"
        environment = os.environ.copy()
        environment["WHISPER_MODEL_NAME"] = model_name
        environment["HF_HUB_OFFLINE"] = "1"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        completed = subprocess.run(
            build_benchmark_command(
                corpus_path=corpus_path,
                output_path=report_path,
                allow_ambiguous_result=args.allow_ambiguous_result,
            ),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"Benchmark failed for {model_name} with code {completed.returncode}."
            )
        reports[model_name] = json.loads(report_path.read_text(encoding="utf-8"))

    comparison = {
        "corpus_privacy": corpus.privacy,
        "allow_ambiguous_result": args.allow_ambiguous_result,
        "models": build_comparison_summary(reports),
    }
    comparison_path = output_dir / "comparison.json"
    atomic_write_text(
        comparison_path,
        json.dumps(comparison, ensure_ascii=False, indent=2),
        private=private,
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
