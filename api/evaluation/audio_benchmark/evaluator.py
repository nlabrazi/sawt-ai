from __future__ import annotations

import math
from collections import Counter, defaultdict
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from evaluation.audio_benchmark.corpus import (
    BuiltAudioCase,
    BuiltAudioCorpus,
    ExpectedVerse,
)


class AudioEvaluationError(ValueError):
    """Le résultat du pipeline ne respecte pas le contrat attendu."""


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _serialize_expected(expected: ExpectedVerse | None) -> dict[str, int] | None:
    if expected is None:
        return None
    return {
        "sourate_id": expected.sourate_id,
        "start_verse": expected.start_verse,
        "end_verse": expected.end_verse,
    }


def _parse_predicted_verse(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AudioEvaluationError("verse doit être un objet ou null.")

    prediction = {}
    for field_name in ("sourate_id", "start_verse", "end_verse"):
        field_value = value.get(field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise AudioEvaluationError(f"verse.{field_name} doit être un entier positif.")
        prediction[field_name] = field_value
    return prediction


def _classify(
    expected: ExpectedVerse | None,
    predicted: Mapping[str, int] | None,
) -> str:
    if expected is None:
        return "true_negative" if predicted is None else "false_positive"
    if predicted is None:
        return "false_negative"

    expected_tuple = (
        expected.sourate_id,
        expected.start_verse,
        expected.end_verse,
    )
    predicted_tuple = (
        predicted["sourate_id"],
        predicted["start_verse"],
        predicted["end_verse"],
    )
    if predicted_tuple == expected_tuple:
        return "exact_match"
    if predicted["sourate_id"] == expected.sourate_id:
        return "correct_surah_wrong_range"
    return "wrong_surah"


def _distribution_summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    sorted_values = sorted(values)
    p50_index = max(0, math.ceil(len(sorted_values) * 0.50) - 1)
    p95_index = max(0, math.ceil(len(sorted_values) * 0.95) - 1)
    return (
        sum(values) / len(values),
        sorted_values[p50_index],
        sorted_values[p95_index],
    )


def _optional_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _quality_summary(counters: Counter) -> dict[str, int | float]:
    evaluated_cases = counters["positive_cases"] + counters["negative_cases"]
    accepted_predictions = (
        counters["exact_match"]
        + counters["correct_surah_wrong_range"]
        + counters["wrong_surah"]
        + counters["false_positive"]
    )
    correct_surah = counters["exact_match"] + counters["correct_surah_wrong_range"]

    return {
        "evaluated_cases": evaluated_cases,
        "positive_cases": counters["positive_cases"],
        "negative_cases": counters["negative_cases"],
        "exact_match": counters["exact_match"],
        "correct_surah_wrong_range": counters["correct_surah_wrong_range"],
        "wrong_surah": counters["wrong_surah"],
        "false_negative": counters["false_negative"],
        "true_negative": counters["true_negative"],
        "false_positive": counters["false_positive"],
        "confident_positive_predictions": counters[
            "confident_positive_predictions"
        ],
        "confident_exact_match": counters["confident_exact_match"],
        "confident_correct_surah": counters["confident_correct_surah"],
        "overall_exact_accuracy": _safe_ratio(
            counters["exact_match"] + counters["true_negative"],
            evaluated_cases,
        ),
        "positive_exact_accuracy": _safe_ratio(
            counters["exact_match"],
            counters["positive_cases"],
        ),
        "positive_surah_accuracy": _safe_ratio(
            correct_surah,
            counters["positive_cases"],
        ),
        # Le dénominateur reste l'ensemble des récitations attendues : une
        # hypothèse ambiguë ne peut donc pas gonfler la qualité « confident ».
        "positive_confident_exact_accuracy": _safe_ratio(
            counters["confident_exact_match"],
            counters["positive_cases"],
        ),
        "positive_confident_surah_accuracy": _safe_ratio(
            counters["confident_correct_surah"],
            counters["positive_cases"],
        ),
        "negative_rejection_rate": _safe_ratio(
            counters["true_negative"],
            counters["negative_cases"],
        ),
        "false_positive_rate": _safe_ratio(
            counters["false_positive"],
            counters["negative_cases"],
        ),
        "exact_precision": _safe_ratio(
            counters["exact_match"],
            accepted_predictions,
        ),
        "exact_recall": _safe_ratio(
            counters["exact_match"],
            counters["positive_cases"],
        ),
    }


def _case_result(
    case: BuiltAudioCase,
    pipeline_result: Mapping[str, Any],
    latency_ms: float,
    *,
    include_transcriptions: bool,
) -> tuple[str, dict[str, Any]]:
    predicted = _parse_predicted_verse(pipeline_result.get("verse"))
    classification = _classify(case.expected_verse, predicted)
    detection = pipeline_result.get("detection", {})
    if not isinstance(detection, Mapping):
        raise AudioEvaluationError("detection doit être un objet.")

    transcription = pipeline_result.get("transcription_text", "")
    if not isinstance(transcription, str):
        raise AudioEvaluationError("transcription_text doit être une chaîne.")

    diagnostics = pipeline_result.get("recognition_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    candidate_evidence = diagnostics.get("topCandidates", [])
    if not isinstance(candidate_evidence, (list, tuple)):
        candidate_evidence = []

    result = {
        "id": case.case_id,
        "source_case_id": case.source_case_id or case.case_id,
        "audio_sha256": case.audio_sha256 or None,
        "label": case.label,
        "category": case.category,
        "classification": classification,
        "expected_verse": _serialize_expected(case.expected_verse),
        "predicted_verse": predicted,
        "detection_status": detection.get("status"),
        "detection_score": detection.get("score"),
        "score_margin": detection.get("score_margin"),
        "matched_word_count": detection.get("matched_word_count"),
        "rejection_reason": detection.get("rejection_reason"),
        "analysis_attempts": detection.get("analysis_attempts"),
        "candidate_count": len(candidate_evidence),
        "candidate_evidence": [
            dict(candidate)
            for candidate in candidate_evidence
            if isinstance(candidate, Mapping)
        ],
        "audio_quality": {
            key: diagnostics.get(key)
            for key in (
                "language",
                "languageProbability",
                "arabicProbability",
                "averageLogProbability",
                "averageNoSpeechProbability",
                "maxCompressionRatio",
                "maxTemperature",
                "speechDurationSeconds",
                "audioQualityWarning",
            )
            if diagnostics.get(key) is not None
        },
        "duration_seconds": case.duration_seconds,
        "latency_ms": latency_ms,
        "realtime_factor": (latency_ms / 1000) / case.duration_seconds,
        "variant": dict(case.variant),
        "tags": list(case.tags),
        "transcription_chars": len(transcription),
    }
    if include_transcriptions:
        result["transcription_text"] = transcription
    return classification, result


def evaluate_audio_corpus(
    corpus: BuiltAudioCorpus,
    inferencer: Callable[[BuiltAudioCase], Mapping[str, Any]],
    *,
    include_transcriptions: bool = False,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Évalue audio → transcription → passage, sans exposer les sources privées."""
    counters = Counter()
    category_counters: dict[str, Counter] = defaultdict(Counter)
    source_counters: dict[str, Counter] = defaultdict(Counter)
    variant_counters: dict[str, Counter] = defaultdict(Counter)
    category_source_ids: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"positive": set(), "negative": set()}
    )
    status_counts = Counter()
    rejection_reason_counts = Counter()
    latencies_ms: list[float] = []
    realtime_factors: list[float] = []
    detection_scores: list[float] = []
    score_margins: list[float] = []
    matched_word_counts: list[float] = []
    case_results: list[dict[str, Any]] = []

    for case in corpus.cases:
        started_at = clock()
        try:
            pipeline_result = inferencer(case)
            latency_ms = (clock() - started_at) * 1000
            if not isinstance(pipeline_result, Mapping):
                raise AudioEvaluationError("Le pipeline doit retourner un objet.")

            classification, result = _case_result(
                case,
                pipeline_result,
                latency_ms,
                include_transcriptions=include_transcriptions,
            )
            source_case_id = case.source_case_id or case.case_id
            category_source_ids[case.category][case.label].add(source_case_id)
            variant_id = str(case.variant.get("id", "unknown"))
            quality_counters = (
                counters,
                category_counters[case.category],
                source_counters[source_case_id],
                variant_counters[variant_id],
            )
            for quality_counter in quality_counters:
                quality_counter[case.label + "_cases"] += 1
                quality_counter[classification] += 1

            is_confident_positive_prediction = (
                case.label == "positive"
                and result["predicted_verse"] is not None
                and result["detection_status"] == "confident"
            )
            if is_confident_positive_prediction:
                for quality_counter in quality_counters:
                    quality_counter["confident_positive_predictions"] += 1
                    if classification == "exact_match":
                        quality_counter["confident_exact_match"] += 1
                    if classification in {
                        "exact_match",
                        "correct_surah_wrong_range",
                    }:
                        quality_counter["confident_correct_surah"] += 1
            if case.label == "negative" and "vocal" in case.tags:
                counters["vocal_negative_cases"] += 1
                source_counters[source_case_id]["vocal_negative_cases"] += 1
            if case.label == "positive" and isinstance(case.variant.get("noise"), Mapping):
                counters["noisy_positive_cases"] += 1
                source_counters[source_case_id]["noisy_positive_cases"] += 1
            status_counts[result["detection_status"] or "missing"] += 1
            rejection_reason_counts[result["rejection_reason"] or "none"] += 1
            latencies_ms.append(latency_ms)
            realtime_factors.append(result["realtime_factor"])
            for value, destination in (
                (result["detection_score"], detection_scores),
                (result["score_margin"], score_margins),
                (result["matched_word_count"], matched_word_counts),
            ):
                parsed_value = _optional_number(value)
                if parsed_value is not None:
                    destination.append(float(parsed_value))
            case_results.append(result)
        except Exception as exc:  # Le rapport doit conserver les autres cas du corpus.
            latency_ms = (clock() - started_at) * 1000
            counters["errors"] += 1
            category_counters[case.category]["errors"] += 1
            case_results.append(
                {
                    "id": case.case_id,
                    "source_case_id": case.source_case_id or case.case_id,
                    "audio_sha256": case.audio_sha256 or None,
                    "label": case.label,
                    "category": case.category,
                    "classification": "error",
                    "error_type": type(exc).__name__,
                    "duration_seconds": case.duration_seconds,
                    "latency_ms": latency_ms,
                    "variant": dict(case.variant),
                    "tags": list(case.tags),
                }
            )

    average_latency_ms, p50_latency_ms, p95_latency_ms = _distribution_summary(latencies_ms)
    average_realtime_factor, _p50_realtime_factor, p95_realtime_factor = (
        _distribution_summary(realtime_factors)
    )
    average_detection_score, _p50_detection_score, _p95_detection_score = (
        _distribution_summary(detection_scores)
    )
    average_score_margin, _p50_score_margin, _p95_score_margin = (
        _distribution_summary(score_margins)
    )
    average_matched_word_count, _p50_words, _p95_words = _distribution_summary(
        matched_word_counts
    )
    positive_source_summaries = [
        _quality_summary(source_counter)
        for source_counter in source_counters.values()
        if source_counter["positive_cases"]
    ]
    negative_source_summaries = [
        _quality_summary(source_counter)
        for source_counter in source_counters.values()
        if source_counter["negative_cases"]
    ]
    vocal_negative_sources = sum(
        source_counter["vocal_negative_cases"] > 0
        for source_counter in source_counters.values()
    )
    noisy_positive_sources = sum(
        source_counter["noisy_positive_cases"] > 0
        for source_counter in source_counters.values()
    )
    distinct_positive_surahs = {
        case.expected_verse.sourate_id
        for case in corpus.cases
        if case.label == "positive" and case.expected_verse is not None
    }

    summary = {
        "total_cases": len(corpus.cases),
        "skipped_cases": len(corpus.skipped),
        "errors": counters["errors"],
        **_quality_summary(counters),
        "unique_source_cases": len(source_counters),
        "positive_source_cases": len(positive_source_summaries),
        "distinct_positive_surahs": len(distinct_positive_surahs),
        "negative_source_cases": len(negative_source_summaries),
        "vocal_negative_cases": counters["vocal_negative_cases"],
        "vocal_negative_source_cases": vocal_negative_sources,
        "noisy_positive_cases": counters["noisy_positive_cases"],
        "noisy_positive_source_cases": noisy_positive_sources,
        "macro_positive_exact_accuracy": (
            sum(item["positive_exact_accuracy"] for item in positive_source_summaries)
            / len(positive_source_summaries)
            if positive_source_summaries
            else 0.0
        ),
        "macro_positive_surah_accuracy": (
            sum(item["positive_surah_accuracy"] for item in positive_source_summaries)
            / len(positive_source_summaries)
            if positive_source_summaries
            else 0.0
        ),
        "macro_positive_confident_exact_accuracy": (
            sum(
                item["positive_confident_exact_accuracy"]
                for item in positive_source_summaries
            )
            / len(positive_source_summaries)
            if positive_source_summaries
            else 0.0
        ),
        "macro_positive_confident_surah_accuracy": (
            sum(
                item["positive_confident_surah_accuracy"]
                for item in positive_source_summaries
            )
            / len(positive_source_summaries)
            if positive_source_summaries
            else 0.0
        ),
        "macro_negative_rejection_rate": (
            sum(item["negative_rejection_rate"] for item in negative_source_summaries)
            / len(negative_source_summaries)
            if negative_source_summaries
            else 0.0
        ),
        "macro_false_positive_rate": (
            sum(item["false_positive_rate"] for item in negative_source_summaries)
            / len(negative_source_summaries)
            if negative_source_summaries
            else 0.0
        ),
        "average_latency_ms": average_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "average_realtime_factor": average_realtime_factor,
        "p95_realtime_factor": p95_realtime_factor,
        "average_detection_score": average_detection_score,
        "average_score_margin": average_score_margin,
        "average_matched_word_count": average_matched_word_count,
        "status_counts": dict(status_counts),
        "rejection_reason_counts": dict(rejection_reason_counts),
    }

    return {
        "summary": summary,
        "categories": {
            category: {
                "errors": category_counter["errors"],
                "positive_source_cases": len(
                    category_source_ids[category]["positive"]
                ),
                "negative_source_cases": len(
                    category_source_ids[category]["negative"]
                ),
                **_quality_summary(category_counter),
            }
            for category, category_counter in sorted(category_counters.items())
        },
        "variants": {
            variant_id: _quality_summary(variant_counter)
            for variant_id, variant_counter in sorted(variant_counters.items())
        },
        "cases": case_results,
        "skipped": [dict(item) for item in corpus.skipped],
        "corpus_manifest_sha256": corpus.manifest_sha256 or None,
        "corpus_privacy": corpus.privacy,
    }


def evaluate_quality_gates(
    summary: Mapping[str, Any],
    *,
    categories: Mapping[str, Mapping[str, Any]] | None = None,
    min_macro_positive_exact_accuracy: float | None = None,
    min_macro_positive_surah_accuracy: float | None = None,
    min_macro_positive_confident_exact_accuracy: float | None = None,
    min_macro_positive_confident_surah_accuracy: float | None = None,
    min_macro_negative_rejection_rate: float | None = None,
    max_macro_false_positive_rate: float | None = None,
    max_errors: int = 0,
    min_positive_sources: int = 0,
    min_distinct_positive_surahs: int = 0,
    min_noisy_positive_sources: int = 0,
    min_vocal_negative_sources: int = 0,
    required_negative_categories: Iterable[str] = (),
) -> list[str]:
    failures = []

    if int(summary.get("errors", 0)) > max_errors:
        failures.append(
            f"errors={summary.get('errors')} dépasse la limite {max_errors}"
        )

    if int(summary.get("positive_source_cases", 0)) < min_positive_sources:
        failures.append(
            "positive_source_cases="
            f"{summary.get('positive_source_cases', 0)} < {min_positive_sources}"
        )

    if int(summary.get("distinct_positive_surahs", 0)) < min_distinct_positive_surahs:
        failures.append(
            "distinct_positive_surahs="
            f"{summary.get('distinct_positive_surahs', 0)} < "
            f"{min_distinct_positive_surahs}"
        )

    if int(summary.get("noisy_positive_source_cases", 0)) < min_noisy_positive_sources:
        failures.append(
            "noisy_positive_source_cases="
            f"{summary.get('noisy_positive_source_cases', 0)} < {min_noisy_positive_sources}"
        )

    if int(summary.get("vocal_negative_source_cases", 0)) < min_vocal_negative_sources:
        failures.append(
            "vocal_negative_source_cases="
            f"{summary.get('vocal_negative_source_cases', 0)} < {min_vocal_negative_sources}"
        )

    category_summaries = categories or {}
    for category in sorted(set(required_negative_categories)):
        category_summary = category_summaries.get(category, {})
        negative_source_cases = int(category_summary.get("negative_source_cases", 0))
        if negative_source_cases < 1:
            failures.append(
                f"negative_category_source_cases[{category}]=0 < 1"
            )

    if min_macro_positive_exact_accuracy is not None:
        if int(summary.get("positive_source_cases", 0)) == 0:
            failures.append("aucune source positive évaluée")
        elif (
            float(summary.get("macro_positive_exact_accuracy", 0))
            < min_macro_positive_exact_accuracy
        ):
            failures.append(
                "macro_positive_exact_accuracy="
                f"{summary.get('macro_positive_exact_accuracy'):.3f} < "
                f"{min_macro_positive_exact_accuracy:.3f}"
            )

    if min_macro_positive_surah_accuracy is not None:
        if int(summary.get("positive_source_cases", 0)) == 0:
            failures.append("aucune source positive évaluée pour la sourate")
        elif (
            float(summary.get("macro_positive_surah_accuracy", 0))
            < min_macro_positive_surah_accuracy
        ):
            failures.append(
                "macro_positive_surah_accuracy="
                f"{summary.get('macro_positive_surah_accuracy'):.3f} < "
                f"{min_macro_positive_surah_accuracy:.3f}"
            )

    if min_macro_positive_confident_exact_accuracy is not None:
        if int(summary.get("positive_source_cases", 0)) == 0:
            failures.append("aucune source positive évaluée avec statut confident")
        elif (
            float(summary.get("macro_positive_confident_exact_accuracy", 0))
            < min_macro_positive_confident_exact_accuracy
        ):
            failures.append(
                "macro_positive_confident_exact_accuracy="
                f"{summary.get('macro_positive_confident_exact_accuracy', 0):.3f} < "
                f"{min_macro_positive_confident_exact_accuracy:.3f}"
            )

    if min_macro_positive_confident_surah_accuracy is not None:
        if int(summary.get("positive_source_cases", 0)) == 0:
            failures.append(
                "aucune source positive évaluée avec statut confident pour la sourate"
            )
        elif (
            float(summary.get("macro_positive_confident_surah_accuracy", 0))
            < min_macro_positive_confident_surah_accuracy
        ):
            failures.append(
                "macro_positive_confident_surah_accuracy="
                f"{summary.get('macro_positive_confident_surah_accuracy', 0):.3f} < "
                f"{min_macro_positive_confident_surah_accuracy:.3f}"
            )

    if min_macro_negative_rejection_rate is not None:
        if int(summary.get("negative_source_cases", 0)) == 0:
            failures.append("aucune source négative évaluée")
        elif (
            float(summary.get("macro_negative_rejection_rate", 0))
            < min_macro_negative_rejection_rate
        ):
            failures.append(
                "macro_negative_rejection_rate="
                f"{summary.get('macro_negative_rejection_rate'):.3f} < "
                f"{min_macro_negative_rejection_rate:.3f}"
            )

    if max_macro_false_positive_rate is not None:
        if int(summary.get("negative_source_cases", 0)) == 0:
            failures.append("aucune source négative évaluée")
        elif (
            float(summary.get("macro_false_positive_rate", 0))
            > max_macro_false_positive_rate
        ):
            failures.append(
                "macro_false_positive_rate="
                f"{summary.get('macro_false_positive_rate'):.3f} > "
                f"{max_macro_false_positive_rate:.3f}"
            )

    return failures
