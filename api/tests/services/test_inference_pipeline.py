import app.services.inference_pipeline as inference_pipeline
from app.services.imam_prediction_service import ImamPredictionError
from app.services.verse_detection_service import VerseDetectionOutcome

from app.services.inference_pipeline import compute_imam_status


def test_compute_imam_status_returns_disabled_when_feature_is_off():
    assert compute_imam_status([{"name": "X", "score": 0.99}], detect_imam=False) == "disabled"


def test_compute_imam_status_returns_unknown_without_predictions():
    assert compute_imam_status([], detect_imam=True) == "unknown"


def test_compute_imam_status_returns_unavailable_when_prediction_failed():
    assert compute_imam_status([], detect_imam=True, unavailable=True) == "unavailable"


def test_compute_imam_status_uses_score_thresholds():
    assert compute_imam_status([{"name": "A", "score": 0.9}], detect_imam=True) == "high"
    assert compute_imam_status([{"name": "A", "score": 0.7}], detect_imam=True) == "medium"
    assert compute_imam_status([{"name": "A", "score": 0.4}], detect_imam=True) == "low"


def test_run_inference_pipeline_returns_unavailable_when_imam_prediction_fails(monkeypatch):
    monkeypatch.setattr(
        inference_pipeline,
        "transcribe_audio",
        lambda _audio_path: [{"text": "قل هو الله احد"}],
    )
    monkeypatch.setattr(
        inference_pipeline,
        "detect_verse_with_metadata",
        lambda _segments: VerseDetectionOutcome(
            verse=None,
            status="insufficient",
            score=None,
            score_margin=None,
            matched_word_count=0,
            rejection_reason="no_match",
        ),
    )

    def fail_imam_prediction(_audio_path: str):
        raise ImamPredictionError("imam backend unavailable")

    monkeypatch.setattr(inference_pipeline, "predict_imam", fail_imam_prediction)

    result = inference_pipeline.run_inference_pipeline("/tmp/audio.wav", detect_imam=True)

    assert result["imam_predictions"] == []
    assert result["imam_status"] == "unavailable"
    assert result["imam_detection_enabled"] is True
    assert result["detection"] == {
        "status": "insufficient",
        "score": None,
        "score_margin": None,
        "matched_word_count": 0,
        "rejection_reason": "no_match",
    }
