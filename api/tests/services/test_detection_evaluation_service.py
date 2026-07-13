from app.services.detection_evaluation_service import (
    DetectionEvaluationCase,
    ExpectedVerse,
    evaluate_detection_cases,
    load_evaluation_cases,
)
from app.services.verse_detection_service import VerseDetectionOutcome


def build_outcome(verse, status="confident", score=0.9):
    return VerseDetectionOutcome(
        verse=verse,
        status=status,
        score=score,
        score_margin=None,
        matched_word_count=4,
        rejection_reason=None if verse else "no_match",
    )


def test_load_evaluation_cases_reads_positive_and_negative_cases(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        """
        {
          "cases": [
            {
              "id": "positive",
              "transcription": "نص اول",
              "expected_verse": {"sourate_id": 1, "start_verse": 1, "end_verse": 2}
            },
            {"id": "negative", "transcription": "نص عادي", "expected_verse": null}
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = load_evaluation_cases(corpus_path)

    assert cases == [
        DetectionEvaluationCase("positive", "نص اول", ExpectedVerse(1, 1, 2)),
        DetectionEvaluationCase("negative", "نص عادي", None),
    ]


def test_evaluate_detection_cases_computes_quality_metrics():
    expected = ExpectedVerse(112, 1, 1)
    cases = [
        DetectionEvaluationCase("correct", "correct", expected),
        DetectionEvaluationCase("missed", "missed", expected),
        DetectionEvaluationCase("false-positive", "false-positive", None),
        DetectionEvaluationCase("true-negative", "true-negative", None),
    ]
    outcomes = {
        "correct": build_outcome({"sourate_id": 112, "start_verse": 1, "end_verse": 1}),
        "missed": build_outcome(None, status="insufficient", score=None),
        "false-positive": build_outcome({"sourate_id": 2, "start_verse": 1, "end_verse": 1}),
        "true-negative": build_outcome(None, status="insufficient", score=None),
    }

    report = evaluate_detection_cases(
        cases,
        detector=lambda segments: outcomes[segments[0]["text"]],
    )

    assert report["summary"] | {"average_latency_ms": 0, "p95_latency_ms": 0} == {
        "total_cases": 4,
        "positive_cases": 2,
        "negative_cases": 2,
        "correct": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "wrong_match": 0,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "false_positive_rate": 0.5,
        "average_latency_ms": 0,
        "p95_latency_ms": 0,
        "status_counts": {"confident": 2, "insufficient": 2},
    }
    assert report["cases"][0]["expected_verse"] == {
        "sourate_id": 112,
        "start_verse": 1,
        "end_verse": 1,
    }
    assert report["cases"][0]["predicted_verse"] == {
        "sourate_id": 112,
        "start_verse": 1,
        "end_verse": 1,
    }
