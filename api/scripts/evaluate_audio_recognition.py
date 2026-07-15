#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Le benchmark ne doit jamais télécharger implicitement un modèle ou un corpus.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

API_DIR = Path(__file__).resolve().parents[1]

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.core.detection_policy import build_detection_policy
from app.core.model_loader import WHISPER_MODEL_NAME, load_all_models
from app.services import transcription_service
from app.services.inference_pipeline import run_inference_pipeline
from evaluation.audio_benchmark.corpus import BuiltAudioCase, load_built_corpus
from evaluation.audio_benchmark.evaluator import (
    evaluate_audio_corpus,
    evaluate_quality_gates,
)
from evaluation.audio_benchmark.secure_io import atomic_write_text, prepare_directory

DEFAULT_CORPUS = API_DIR / "evaluation" / "generated" / "audio" / "manifest.json"
DEFAULT_REQUIRED_NEGATIVE_CATEGORIES = (
    "french_speech",
    "french_conversation",
    "vocal_music",
    "arabic_non_quran",
)


def _ratio(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("La valeur doit être comprise entre 0 et 1.")
    return number


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _non_negative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("La valeur doit être positive ou nulle.")
    return number


def _build_transcription_policy_configuration(importer=import_module) -> dict:
    module_name = "app.core.transcription_policy"
    try:
        policy_module = importer(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return {"available": False, "thresholds": {}}

    return {
        "available": True,
        "thresholds": {
            name.lower(): value
            for name, value in vars(policy_module).items()
            if name.isupper() and isinstance(value, (int, float, str, bool))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the complete local audio -> Whisper -> Quran passage pipeline."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("smoke", "quality"), default="smoke")
    parser.add_argument(
        "--include-transcriptions",
        action="store_true",
        help="Include transcription content in a private output file (requires --output).",
    )
    parser.add_argument(
        "--allow-ambiguous-result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the current API behavior by exposing ambiguous verse hypotheses.",
    )
    parser.add_argument("--min-macro-positive-exact-accuracy", type=_ratio)
    parser.add_argument("--min-macro-negative-rejection-rate", type=_ratio)
    parser.add_argument("--max-macro-false-positive-rate", type=_ratio)
    parser.add_argument("--max-errors", type=_non_negative_integer, default=0)
    parser.add_argument("--min-positive-sources", type=_non_negative_integer)
    parser.add_argument("--min-noisy-positive-sources", type=_non_negative_integer)
    parser.add_argument("--min-vocal-negative-sources", type=_non_negative_integer)
    parser.add_argument(
        "--required-negative-category",
        action="append",
        default=[],
        help="Require an additional negative category in quality mode (repeatable).",
    )
    args = parser.parse_args()

    if args.include_transcriptions and args.output is None:
        parser.error("--include-transcriptions requires --output to avoid leaking speech to stdout")

    corpus = load_built_corpus(args.corpus)
    if corpus.privacy == "private" and args.output is None:
        parser.error("a private corpus requires --output for its protected full report")
    private_report = args.include_transcriptions or corpus.privacy == "private"
    if args.output is not None:
        managed_reports_dir = (API_DIR / "evaluation" / "reports").resolve()
        output_path = args.output.resolve()
        output_parent = output_path.parent
        prepare_directory(
            output_parent,
            private=private_report,
            tighten_existing=(
                private_report and output_parent.is_relative_to(managed_reports_dir)
            ),
        )
        if private_report and output_path.exists() and not output_path.is_symlink():
            output_path.chmod(0o600)
    # Le benchmark neutralise ce logger afin qu'une configuration de logs externe
    # ne divulgue jamais de contenu dérivé des voix.
    logging.getLogger("app.services.transcription_service").setLevel(logging.WARNING)
    load_all_models()

    def infer(case: BuiltAudioCase):
        return run_inference_pipeline(
            str(case.audio_path),
            detect_imam=False,
            audio_duration_seconds=case.duration_seconds,
            allow_ambiguous_result=args.allow_ambiguous_result,
        )

    report = evaluate_audio_corpus(
        corpus,
        infer,
        include_transcriptions=args.include_transcriptions,
    )
    report["configuration"] = {
        "whisper_model": Path(WHISPER_MODEL_NAME).name,
        "faster_whisper_version": _package_version("faster-whisper"),
        "offline_model_loading": True,
        "detect_imam": False,
        "allow_ambiguous_result": args.allow_ambiguous_result,
        "transcription": {
            "language": getattr(transcription_service, "WHISPER_LANGUAGE", None),
            "beam_size": transcription_service.WHISPER_BEAM_SIZE,
            "log_prob_threshold": transcription_service.WHISPER_LOG_PROB_THRESHOLD,
            "no_speech_threshold": transcription_service.WHISPER_NO_SPEECH_THRESHOLD,
            "vad_min_silence_duration_ms": (
                transcription_service.VAD_MIN_SILENCE_DURATION_MS
            ),
            "vad_speech_pad_ms": transcription_service.VAD_SPEECH_PAD_MS,
        },
        "detection_policy": build_detection_policy(),
        "transcription_policy": _build_transcription_policy_configuration(),
    }
    explicit_quality_gate = any(
        value is not None
        for value in (
            args.min_macro_positive_exact_accuracy,
            args.min_macro_negative_rejection_rate,
            args.max_macro_false_positive_rate,
            args.min_positive_sources,
            args.min_noisy_positive_sources,
            args.min_vocal_negative_sources,
            *args.required_negative_category,
        )
    )
    run_mode = "quality" if args.mode == "quality" or explicit_quality_gate else "smoke"
    required_positive_sources = (
        3 if args.min_positive_sources is None else args.min_positive_sources
    )
    required_noisy_positive_sources = (
        3
        if args.min_noisy_positive_sources is None
        else args.min_noisy_positive_sources
    )
    required_vocal_negative_sources = (
        3
        if args.min_vocal_negative_sources is None
        else args.min_vocal_negative_sources
    )
    required_negative_categories = tuple(
        sorted(
            set(DEFAULT_REQUIRED_NEGATIVE_CATEGORIES)
            | set(args.required_negative_category)
        )
    )
    negative_category_source_cases = {
        category: int(
            report["categories"].get(category, {}).get("negative_source_cases", 0)
        )
        for category in required_negative_categories
    }

    missing_coverage = []
    if report["summary"]["positive_source_cases"] < required_positive_sources:
        missing_coverage.append("positive_sources")
    if report["summary"]["noisy_positive_source_cases"] < required_noisy_positive_sources:
        missing_coverage.append("noisy_positive_sources")
    if report["summary"]["vocal_negative_source_cases"] < required_vocal_negative_sources:
        missing_coverage.append("vocal_negative_sources")
    missing_coverage.extend(
        f"negative_category:{category}"
        for category, source_count in negative_category_source_cases.items()
        if source_count < 1
    )
    report["coverage"] = {
        "status": "quality_ready" if not missing_coverage else "smoke_incomplete",
        "missing": missing_coverage,
        "positive_source_cases": report["summary"]["positive_source_cases"],
        "noisy_positive_source_cases": report["summary"][
            "noisy_positive_source_cases"
        ],
        "vocal_negative_source_cases": report["summary"][
            "vocal_negative_source_cases"
        ],
        "negative_category_source_cases": negative_category_source_cases,
        "requirements": {
            "min_positive_sources": required_positive_sources,
            "min_noisy_positive_sources": required_noisy_positive_sources,
            "min_vocal_negative_sources": required_vocal_negative_sources,
            "required_negative_categories": list(required_negative_categories),
        },
    }

    execution_failures = evaluate_quality_gates(
        report["summary"],
        max_errors=args.max_errors,
    )
    report["execution_gate"] = {
        "passed": not execution_failures,
        "failures": execution_failures,
    }

    quality_failures: list[str] = []
    if run_mode == "quality":
        quality_failures = evaluate_quality_gates(
            report["summary"],
            categories=report["categories"],
            min_macro_positive_exact_accuracy=(
                args.min_macro_positive_exact_accuracy
                if args.min_macro_positive_exact_accuracy is not None
                else 0.90
            ),
            min_macro_negative_rejection_rate=(
                args.min_macro_negative_rejection_rate
                if args.min_macro_negative_rejection_rate is not None
                else 1.0
            ),
            max_macro_false_positive_rate=(
                args.max_macro_false_positive_rate
                if args.max_macro_false_positive_rate is not None
                else 0.0
            ),
            max_errors=args.max_errors,
            min_positive_sources=required_positive_sources,
            min_noisy_positive_sources=required_noisy_positive_sources,
            min_vocal_negative_sources=required_vocal_negative_sources,
            required_negative_categories=required_negative_categories,
        )
        report["quality_gate"] = {
            "evaluated": True,
            "passed": not quality_failures,
            "failures": quality_failures,
            "metrics": {
                "macro_positive_exact_accuracy": report["summary"][
                    "macro_positive_exact_accuracy"
                ],
                "macro_negative_rejection_rate": report["summary"][
                    "macro_negative_rejection_rate"
                ],
                "macro_false_positive_rate": report["summary"][
                    "macro_false_positive_rate"
                ],
            },
            "coverage": report["coverage"],
        }
    else:
        report["quality_gate"] = {
            "evaluated": False,
            "passed": None,
            "failures": ["run in --mode quality after adding real vocal cases"],
        }
    serialized_report = json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    if args.output is not None:
        atomic_write_text(
            args.output,
            serialized_report,
            private=private_report,
        )
    if args.include_transcriptions or corpus.privacy == "private":
        safe_console_report = {
            "summary": report["summary"],
            "categories": report["categories"],
            "coverage": report["coverage"],
            "skipped": report["skipped"],
            "configuration": report["configuration"],
            "execution_gate": report["execution_gate"],
            "quality_gate": report["quality_gate"],
            "private_report_file": args.output.name,
        }
        print(json.dumps(safe_console_report, ensure_ascii=False, indent=2))
    else:
        print(serialized_report, end="")

    if execution_failures or quality_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
