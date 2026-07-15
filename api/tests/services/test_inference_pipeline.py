import app.services.inference_pipeline as inference_pipeline
from app.services.imam_prediction_service import ImamPredictionError
from app.services.verse_detection_service import VerseDetectionOutcome
from app.services.transcription_service import (
    TranscriptionMetadata,
    TranscriptionResult,
)

from app.services.inference_pipeline import compute_imam_status


def build_transcription(
    text="قل هو الله احد",
    *,
    language="ar",
    language_probability=0.95,
    arabic_probability=0.95,
    average_log_probability=-0.4,
    max_compression_ratio=1.8,
    max_temperature=0.0,
):
    metadata = TranscriptionMetadata(
        language=language,
        language_probability=language_probability,
        arabic_probability=arabic_probability,
        language_probabilities=(),
        duration_seconds=4.0,
        duration_after_vad_seconds=3.0,
        speech_duration_seconds=3.0,
        average_log_probability=average_log_probability,
        average_no_speech_probability=0.1,
        max_compression_ratio=max_compression_ratio,
        max_temperature=max_temperature,
        segment_metrics=(),
    )
    segments = [] if text is None else [{"text": text}]
    return TranscriptionResult(segments, metadata)


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


def test_audio_quality_rejects_empty_transcription():
    quality = inference_pipeline.assess_audio_quality(build_transcription(None))

    assert quality.accepted is False
    assert quality.rejection_reason == "insufficient_speech"


def test_audio_quality_rejects_only_confidently_non_arabic_speech():
    french = build_transcription(
        "Bonjour, ceci est un texte français.",
        language="fr",
        language_probability=0.91,
        arabic_probability=0.03,
    )
    uncertain = build_transcription(
        "قل هو الله احد",
        language="ur",
        language_probability=0.61,
        arabic_probability=0.2,
    )

    assert inference_pipeline.assess_audio_quality(french).rejection_reason == (
        "non_arabic_speech"
    )
    assert inference_pipeline.assess_audio_quality(uncertain).accepted is True


def test_audio_quality_rejects_low_decode_confidence():
    low_log_probability = build_transcription(average_log_probability=-1.01)
    unstable_temperature = build_transcription(
        average_log_probability=-0.81,
        max_temperature=0.8,
    )
    suspicious_compression = build_transcription(
        average_log_probability=-0.71,
        max_compression_ratio=2.41,
    )

    assert inference_pipeline.assess_audio_quality(
        low_log_probability
    ).rejection_reason == "low_transcription_confidence"
    assert inference_pipeline.assess_audio_quality(
        unstable_temperature
    ).rejection_reason == "low_transcription_confidence"
    assert inference_pipeline.assess_audio_quality(
        suspicious_compression
    ).rejection_reason == "low_transcription_confidence"


def test_audio_quality_does_not_reject_compression_without_weak_decode():
    compressed_but_stable = build_transcription(
        average_log_probability=-0.4,
        max_compression_ratio=2.8,
    )

    assert inference_pipeline.assess_audio_quality(compressed_but_stable).accepted is True


def test_audio_quality_rejection_happens_before_verse_matching(monkeypatch):
    def unexpected_matching(*_args, **_kwargs):
        raise AssertionError("Quran matching must not run for rejected audio")

    monkeypatch.setattr(
        inference_pipeline,
        "detect_verse_with_metadata",
        unexpected_matching,
    )

    detection = inference_pipeline.detect_verse_after_audio_quality_check(
        build_transcription(
            "Texte lu en français",
            language="fr",
            language_probability=0.94,
            arabic_probability=0.01,
        ),
        include_ambiguous_verse=True,
    )

    assert detection.verse is None
    assert detection.status == "insufficient"
    assert detection.rejection_reason == "non_arabic_speech"


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
        lambda _segments, include_ambiguous_verse=False: next(outcomes),
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
    ambiguous_result_flags = []

    def fake_transcribe(_audio_path, clip_end_seconds=None):
        transcription_calls.append(clip_end_seconds)
        return [{"text": "نص غير كاف"}]

    monkeypatch.setattr(inference_pipeline, "transcribe_audio", fake_transcribe)
    def fake_detect(_segments, include_ambiguous_verse=False):
        ambiguous_result_flags.append(include_ambiguous_verse)
        return VerseDetectionOutcome(
            None,
            "insufficient",
            0.4,
            None,
            3,
            "score_too_low",
        )

    monkeypatch.setattr(
        inference_pipeline,
        "detect_verse_with_metadata",
        fake_detect,
    )

    _segments, detection, analyzed_duration, attempts = (
        inference_pipeline.detect_verse_progressively("/tmp/audio.wav", 12)
    )

    assert transcription_calls == [5, 10, None]
    assert ambiguous_result_flags == [False, False, True]
    assert detection.status == "insufficient"
    assert analyzed_duration == 12
    assert attempts == 3


def test_detect_verse_progressively_retries_after_a_low_quality_prefix(monkeypatch):
    transcription_calls = []
    transcriptions = iter([
        build_transcription(average_log_probability=-1.2),
        build_transcription(),
    ])

    def fake_transcribe(_audio_path, clip_end_seconds=None):
        transcription_calls.append(clip_end_seconds)
        return next(transcriptions)

    monkeypatch.setattr(inference_pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        inference_pipeline,
        "detect_verse_with_metadata",
        lambda _segments, include_ambiguous_verse=False: VerseDetectionOutcome(
            {"similarity": 0.9},
            "confident",
            0.9,
            0.2,
            4,
            None,
        ),
    )

    _segments, detection, analyzed_duration, attempts = (
        inference_pipeline.detect_verse_progressively("/tmp/audio.wav", 8)
    )

    assert transcription_calls == [5, None]
    assert detection.status == "confident"
    assert analyzed_duration == 8
    assert attempts == 2


def test_run_inference_pipeline_returns_unavailable_when_imam_prediction_fails(monkeypatch):
    monkeypatch.setattr(
        inference_pipeline,
        "transcribe_audio",
        lambda _audio_path: [{"text": "قل هو الله احد"}],
    )
    monkeypatch.setattr(
        inference_pipeline,
        "detect_verse_with_metadata",
        lambda _segments, include_ambiguous_verse=False: VerseDetectionOutcome(
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


def test_run_inference_pipeline_skips_imam_for_rejected_audio(monkeypatch):
    monkeypatch.setattr(
        inference_pipeline,
        "transcribe_audio",
        lambda _audio_path: build_transcription(None),
    )

    def unexpected_imam_prediction(_audio_path):
        raise AssertionError("Imam prediction must not run for rejected audio")

    monkeypatch.setattr(
        inference_pipeline,
        "predict_imam",
        unexpected_imam_prediction,
    )

    result = inference_pipeline.run_inference_pipeline(
        "/tmp/silence.wav",
        detect_imam=True,
    )

    assert result["verse"] is None
    assert result["detection"]["rejection_reason"] == "insufficient_speech"
    assert result["imam_predictions"] == []
    assert result["imam_status"] == "unknown"
