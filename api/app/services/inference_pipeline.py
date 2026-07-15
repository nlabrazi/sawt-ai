# ROLE
# ----
# Orchestration principale :
# transcription -> détection verset -> détection imam

import logging
import math
from dataclasses import dataclass
from typing import Literal

from app.core.detection_policy import PROGRESSIVE_ANALYSIS_STEP_SECONDS
from app.core.transcription_policy import (
    HIGH_COMPRESSION_MAX_LOG_PROBABILITY,
    HIGH_COMPRESSION_RATIO,
    HIGH_TEMPERATURE,
    HIGH_TEMPERATURE_MAX_LOG_PROBABILITY,
    MIN_AVERAGE_LOG_PROBABILITY,
    NON_ARABIC_LANGUAGE_MIN_PROBABILITY,
    NON_ARABIC_MAX_ARABIC_PROBABILITY,
)
from app.services.transcription_service import (
    TranscriptionMetadata,
    TranscriptionResult,
    transcribe_audio,
)
from app.services.verse_detection_service import (
    VerseDetectionOutcome,
    detect_verse_with_metadata,
)
from app.services.imam_prediction_service import ImamPredictionError, predict_imam

logger = logging.getLogger(__name__)

ImamStatus = Literal["disabled", "unknown", "unavailable", "high", "medium", "low"]
AudioRejectionReason = Literal[
    "insufficient_speech",
    "non_arabic_speech",
    "low_transcription_confidence",
]
AUDIO_REJECTION_REASONS: frozenset[AudioRejectionReason] = frozenset(
    {
        "insufficient_speech",
        "non_arabic_speech",
        "low_transcription_confidence",
    }
)


@dataclass(frozen=True, slots=True)
class AudioQualityAssessment:
    accepted: bool
    rejection_reason: AudioRejectionReason | None


def assess_audio_quality(segments) -> AudioQualityAssessment:
    """Écarte uniquement les cas dont les signaux Whisper sont suffisamment nets."""
    has_transcription = any(
        bool(segment.get("text", "").strip())
        for segment in segments
    )

    if not has_transcription:
        return AudioQualityAssessment(False, "insufficient_speech")

    metadata: TranscriptionMetadata | None = getattr(segments, "metadata", None)
    if metadata is None:
        # Compatibilité avec les doubles de tests et les anciens appelants.
        return AudioQualityAssessment(True, None)

    is_confidently_non_arabic = (
        metadata.language not in (None, "ar")
        and metadata.language_probability is not None
        and metadata.language_probability >= NON_ARABIC_LANGUAGE_MIN_PROBABILITY
        and metadata.arabic_probability is not None
        and metadata.arabic_probability < NON_ARABIC_MAX_ARABIC_PROBABILITY
    )
    if is_confidently_non_arabic:
        return AudioQualityAssessment(False, "non_arabic_speech")

    average_log_probability = metadata.average_log_probability
    has_low_average_log_probability = (
        average_log_probability is not None
        and average_log_probability < MIN_AVERAGE_LOG_PROBABILITY
    )
    has_unstable_temperature = (
        average_log_probability is not None
        and metadata.max_temperature is not None
        and metadata.max_temperature >= HIGH_TEMPERATURE
        and average_log_probability < HIGH_TEMPERATURE_MAX_LOG_PROBABILITY
    )
    has_suspicious_compression = (
        average_log_probability is not None
        and metadata.max_compression_ratio is not None
        and metadata.max_compression_ratio > HIGH_COMPRESSION_RATIO
        and average_log_probability < HIGH_COMPRESSION_MAX_LOG_PROBABILITY
    )

    if (
        has_low_average_log_probability
        or has_unstable_temperature
        or has_suspicious_compression
    ):
        return AudioQualityAssessment(False, "low_transcription_confidence")

    return AudioQualityAssessment(True, None)


def build_audio_rejection_outcome(
    rejection_reason: AudioRejectionReason,
) -> VerseDetectionOutcome:
    return VerseDetectionOutcome(
        verse=None,
        status="insufficient",
        score=None,
        score_margin=None,
        matched_word_count=0,
        rejection_reason=rejection_reason,
    )


def detect_verse_after_audio_quality_check(
    segments,
    *,
    include_ambiguous_verse: bool,
) -> VerseDetectionOutcome:
    quality = assess_audio_quality(segments)

    if not quality.accepted:
        if quality.rejection_reason is None:
            raise RuntimeError("A rejected audio quality assessment requires a reason.")

        logger.info(
            "Verse detection skipped after audio quality check: reason=%s",
            quality.rejection_reason,
        )
        return build_audio_rejection_outcome(quality.rejection_reason)

    return detect_verse_with_metadata(
        segments,
        include_ambiguous_verse=include_ambiguous_verse,
    )


def build_progressive_analysis_endpoints(duration_seconds: float) -> list[float]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return []

    endpoints = []
    endpoint = float(PROGRESSIVE_ANALYSIS_STEP_SECONDS)

    while endpoint < duration_seconds:
        endpoints.append(endpoint)
        endpoint += PROGRESSIVE_ANALYSIS_STEP_SECONDS

    endpoints.append(duration_seconds)
    return endpoints


def detect_verse_progressively(
    audio_path: str,
    audio_duration_seconds: float | None,
    allow_ambiguous_result: bool = True,
) -> tuple[
    TranscriptionResult | list[dict],
    VerseDetectionOutcome,
    float | None,
    int,
]:
    if audio_duration_seconds is None:
        segments = transcribe_audio(audio_path)
        return (
            segments,
            detect_verse_after_audio_quality_check(
                segments,
                include_ambiguous_verse=allow_ambiguous_result,
            ),
            None,
            1,
        )

    endpoints = build_progressive_analysis_endpoints(audio_duration_seconds)

    for attempt, endpoint in enumerate(endpoints, start=1):
        is_full_audio = endpoint == audio_duration_seconds
        segments = transcribe_audio(
            audio_path,
            clip_end_seconds=None if is_full_audio else endpoint,
        )
        detection = detect_verse_after_audio_quality_check(
            segments,
            include_ambiguous_verse=allow_ambiguous_result and is_full_audio,
        )

        if detection.status == "confident" or is_full_audio:
            return segments, detection, endpoint, attempt

    segments = transcribe_audio(audio_path)
    return (
        segments,
        detect_verse_after_audio_quality_check(
            segments,
            include_ambiguous_verse=allow_ambiguous_result,
        ),
        audio_duration_seconds,
        1,
    )


def compute_imam_status(
    predictions,
    detect_imam: bool = True,
    unavailable: bool = False,
) -> ImamStatus:
    """
    Détermine le niveau de confiance.
    """
    if not detect_imam:
        return "disabled"

    if unavailable:
        return "unavailable"

    if not predictions:
        return "unknown"

    top_score = predictions[0]["score"]

    if top_score >= 0.85:
        return "high"

    if top_score >= 0.65:
        return "medium"

    return "low"


def run_inference_pipeline(
    audio_path: str,
    detect_imam: bool = True,
    audio_duration_seconds: float | None = None,
    allow_ambiguous_result: bool = True,
):
    """
    Pipeline principal
    """

    # 1️⃣ transcription whisper
    segments, verse_detection, analyzed_duration_seconds, analysis_attempts = (
        detect_verse_progressively(
            audio_path,
            audio_duration_seconds,
            allow_ambiguous_result=allow_ambiguous_result,
        )
    )

    transcription_text = " ".join(
        segment.get("text", "").strip()
        for segment in segments
    ).strip()

    # 2️⃣ détection verset
    verse = verse_detection.verse

    # 3️⃣ détection imam
    imam_predictions = []
    imam_unavailable = False

    audio_was_rejected = verse_detection.rejection_reason in AUDIO_REJECTION_REASONS

    if detect_imam and not audio_was_rejected:
        try:
            imam_predictions = predict_imam(audio_path)
        except ImamPredictionError:
            imam_unavailable = True

    # 4️⃣ calcul statut
    imam_status = compute_imam_status(
        imam_predictions,
        detect_imam=detect_imam,
        unavailable=imam_unavailable,
    )

    logger.info(
        "Inference complete: segments=%s transcription_chars=%s verse_found=%s verse_similarity=%s analyzed_duration_seconds=%s analysis_attempts=%s imam_predictions=%s imam_status=%s",
        len(segments),
        len(transcription_text),
        verse is not None,
        verse.get("similarity") if verse else None,
        analyzed_duration_seconds,
        analysis_attempts,
        len(imam_predictions),
        imam_status,
    )

    return {
        "transcription_text": transcription_text,
        "verse": verse,
        "detection": {
            **verse_detection.metadata(),
            "analyzed_duration_seconds": analyzed_duration_seconds,
            "analysis_attempts": analysis_attempts,
        },
        "imam_predictions": imam_predictions,
        "imam_status": imam_status,
        "imam_detection_enabled": detect_imam,
    }
