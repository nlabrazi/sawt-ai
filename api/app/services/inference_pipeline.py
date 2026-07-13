# ROLE
# ----
# Orchestration principale :
# transcription -> détection verset -> détection imam

import logging
import math
from typing import Literal

from app.core.detection_policy import PROGRESSIVE_ANALYSIS_STEP_SECONDS
from app.services.transcription_service import transcribe_audio
from app.services.verse_detection_service import (
    VerseDetectionOutcome,
    detect_verse_with_metadata,
)
from app.services.imam_prediction_service import ImamPredictionError, predict_imam

logger = logging.getLogger(__name__)

ImamStatus = Literal["disabled", "unknown", "unavailable", "high", "medium", "low"]


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
) -> tuple[list[dict], VerseDetectionOutcome, float | None, int]:
    if audio_duration_seconds is None:
        segments = transcribe_audio(audio_path)
        return segments, detect_verse_with_metadata(segments), None, 1

    endpoints = build_progressive_analysis_endpoints(audio_duration_seconds)

    for attempt, endpoint in enumerate(endpoints, start=1):
        is_full_audio = endpoint == audio_duration_seconds
        segments = transcribe_audio(
            audio_path,
            clip_end_seconds=None if is_full_audio else endpoint,
        )
        detection = detect_verse_with_metadata(segments)

        if detection.status == "confident" or is_full_audio:
            return segments, detection, endpoint, attempt

    segments = transcribe_audio(audio_path)
    return segments, detect_verse_with_metadata(segments), audio_duration_seconds, 1


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
):
    """
    Pipeline principal
    """

    # 1️⃣ transcription whisper
    segments, verse_detection, analyzed_duration_seconds, analysis_attempts = (
        detect_verse_progressively(audio_path, audio_duration_seconds)
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

    if detect_imam:
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
