# ROLE
# ----
# Orchestration principale :
# transcription -> détection verset -> détection imam

import logging
from typing import Literal

from app.services.transcription_service import transcribe_audio
from app.services.verse_detection_service import detect_verse_with_metadata
from app.services.imam_prediction_service import ImamPredictionError, predict_imam

logger = logging.getLogger(__name__)

ImamStatus = Literal["disabled", "unknown", "unavailable", "high", "medium", "low"]


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


def run_inference_pipeline(audio_path: str, detect_imam: bool = True):
    """
    Pipeline principal
    """

    # 1️⃣ transcription whisper
    segments = transcribe_audio(audio_path)

    transcription_text = " ".join(
        segment.get("text", "").strip()
        for segment in segments
    ).strip()

    # 2️⃣ détection verset
    verse_detection = detect_verse_with_metadata(segments)
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
        "Inference complete: segments=%s transcription_chars=%s verse_found=%s verse_similarity=%s imam_predictions=%s imam_status=%s",
        len(segments),
        len(transcription_text),
        verse is not None,
        verse.get("similarity") if verse else None,
        len(imam_predictions),
        imam_status,
    )

    return {
        "transcription_text": transcription_text,
        "verse": verse,
        "detection": verse_detection.metadata(),
        "imam_predictions": imam_predictions,
        "imam_status": imam_status,
        "imam_detection_enabled": detect_imam,
    }
