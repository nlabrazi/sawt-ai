# ROLE
# ----
# Transcrit un fichier audio avec le modèle faster-whisper déjà chargé.

import logging

from app.core.model_loader import get_whisper_model

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str):
    model = get_whisper_model()

    segments, _info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5,
    )

    result = [
        {
            "text": segment.text
        }
        for segment in segments
    ]

    logger.info(
        "Transcription complete: segments=%s preview=%s",
        len(result),
        " ".join(item["text"].strip() for item in result)[:160],
    )

    return result
