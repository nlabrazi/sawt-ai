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


def test_build_progressive_analysis_endpoints_includes_full_duration():
    assert inference_pipeline.build_progressive_analysis_endpoints(12.5) == [5, 10, 12.5]
    assert inference_pipeline.build_progressive_analysis_endpoints(5) == [5]


def test_detect_verse_progressively_stops_after_confident_match(monkeypatch):
    transcription_calls = []
    outcomes = iter([
        VerseDetectionOutcome(None, "probable", 0.72, 0.04, 4, "score_too_low"),
        VerseDetectionOutcome(
            {"similarity": 0.91},
            "confident",
            0.91,
            0.12,
            7,
            None,
        ),
    ])

    def fake_transcribe(_audio_path, clip_end_seconds=None):
        transcription_calls.append(clip_end_seconds)
        return [{"text": "قل هو الله احد"}]

    monkeypatch.setattr(inference_pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        inference_pipeline,
        "detect_verse_with_metadata",
        lambda _segments: next(outcomes),
    )

    _segments, detection, analyzed_duration, attempts = (
        inference_pipeline.detect_verse_progressively("/tmp/audio.wav", 18)
    )

    assert transcription_calls == [5, 10]
    assert detection.status == "confident"
    assert analyzed_duration == 10
    assert attempts == 2


def test_detect_verse_progressively_falls_back_to_full_audio(monkeypatch):
    transcription_calls = []

    def fake_transcribe(_audio_path, clip_end_seconds=None):
        transcription_calls.append(clip_end_seconds)
        return [{"text": "نص غير كاف"}]

    monkeypatch.setattr(inference_pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        inference_pipeline,
        "detect_verse_with_metadata",
        lambda _segments: VerseDetectionOutcome(
            None,
            "insufficient",
            0.4,
            None,
            3,
            "score_too_low",
        ),
    )

    _segments, detection, analyzed_duration, attempts = (
        inference_pipeline.detect_verse_progressively("/tmp/audio.wav", 12)
    )

    assert transcription_calls == [5, 10, None]
    assert detection.status == "insufficient"
    assert analyzed_duration == 12
    assert attempts == 3


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
        "analyzed_duration_seconds": None,
        "analysis_attempts": 1,
    }
