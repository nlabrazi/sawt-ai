from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from app.services.verse_detection_service import (
    VerseDetectionOutcome,
    detect_verse_with_metadata,
)


@dataclass(frozen=True, slots=True)
class ExpectedVerse:
    sourate_id: int
    start_verse: int
    end_verse: int


@dataclass(frozen=True, slots=True)
class DetectionEvaluationCase:
    case_id: str
    transcription: str
    expected_verse: ExpectedVerse | None


def load_evaluation_cases(path: str | Path) -> list[DetectionEvaluationCase]:
    with Path(path).open("r", encoding="utf-8") as corpus_file:
        payload = json.load(corpus_file)

    cases = []

    for item in payload["cases"]:
        expected_payload = item.get("expected_verse")
        expected_verse = (
            ExpectedVerse(**expected_payload)
            if expected_payload is not None
            else None
        )
        cases.append(
            DetectionEvaluationCase(
                case_id=item["id"],
                transcription=item["transcription"],
                expected_verse=expected_verse,
            )
        )

    return cases


def matches_expected_verse(verse: dict | None, expected: ExpectedVerse) -> bool:
    return verse is not None and (
        verse["sourate_id"],
        verse["start_verse"],
        verse["end_verse"],
    ) == (
        expected.sourate_id,
        expected.start_verse,
        expected.end_verse,
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def serialize_expected_verse(expected: ExpectedVerse | None) -> dict | None:
    if expected is None:
        return None

    return {
        "sourate_id": expected.sourate_id,
        "start_verse": expected.start_verse,
        "end_verse": expected.end_verse,
    }


def serialize_predicted_verse(verse: dict | None) -> dict | None:
    if verse is None:
        return None

    return {
        "sourate_id": verse["sourate_id"],
        "start_verse": verse["start_verse"],
        "end_verse": verse["end_verse"],
    }


def evaluate_detection_cases(
    cases: list[DetectionEvaluationCase],
    detector: Callable[[list[dict[str, str]]], VerseDetectionOutcome] = detect_verse_with_metadata,
) -> dict:
    counters = Counter()
    status_counts = Counter()
    latencies_ms = []
    case_results = []

    for case in cases:
        started_at = perf_counter()
        outcome = detector([{"text": case.transcription}])
        latency_ms = (perf_counter() - started_at) * 1000
        latencies_ms.append(latency_ms)
        status_counts[outcome.status] += 1

        if case.expected_verse is None:
            counters["negative_cases"] += 1

            if outcome.verse is None:
                classification = "true_negative"
            else:
                classification = "false_positive"
        else:
            counters["positive_cases"] += 1

            if matches_expected_verse(outcome.verse, case.expected_verse):
                classification = "correct"
            elif outcome.verse is None:
                classification = "false_negative"
            else:
                classification = "wrong_match"

        counters[classification] += 1
        case_results.append(
            {
                "id": case.case_id,
                "classification": classification,
                "status": outcome.status,
                "score": outcome.score,
                "expected_verse": serialize_expected_verse(case.expected_verse),
                "predicted_verse": serialize_predicted_verse(outcome.verse),
                "latency_ms": latency_ms,
            }
        )

    total_cases = len(cases)
    accepted_predictions = (
        counters["correct"]
        + counters["wrong_match"]
        + counters["false_positive"]
    )
    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)

    return {
        "summary": {
            "total_cases": total_cases,
            "positive_cases": counters["positive_cases"],
            "negative_cases": counters["negative_cases"],
            "correct": counters["correct"],
            "true_negative": counters["true_negative"],
            "false_positive": counters["false_positive"],
            "false_negative": counters["false_negative"],
            "wrong_match": counters["wrong_match"],
            "accuracy": safe_ratio(
                counters["correct"] + counters["true_negative"],
                total_cases,
            ),
            "precision": safe_ratio(counters["correct"], accepted_predictions),
            "recall": safe_ratio(counters["correct"], counters["positive_cases"]),
            "false_positive_rate": safe_ratio(
                counters["false_positive"],
                counters["negative_cases"],
            ),
            "average_latency_ms": safe_ratio(sum(latencies_ms), total_cases),
            "p95_latency_ms": sorted_latencies[p95_index] if sorted_latencies else 0.0,
            "status_counts": dict(status_counts),
        },
        "cases": case_results,
    }
