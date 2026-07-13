# ROLE
# ----
# Transcrit un fichier audio avec le modèle faster-whisper déjà chargé.

import logging

from app.core.model_loader import get_whisper_model

logger = logging.getLogger(__name__)

WHISPER_LANGUAGE = "ar"
WHISPER_BEAM_SIZE = 5
WHISPER_LOG_PROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6
VAD_MIN_SILENCE_DURATION_MS = 500
VAD_SPEECH_PAD_MS = 400


def transcribe_audio(audio_path: str, clip_end_seconds: float | None = None):
    model = get_whisper_model()
    clip_options = (
        {"clip_timestamps": [0, clip_end_seconds]}
        if clip_end_seconds is not None
        else {}
    )

    segments, _info = model.transcribe(
        audio_path,
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        },
        **clip_options,
    )

    result = [
        {"text": text}
        for segment in segments
        if (text := segment.text.strip())
    ]

    logger.info(
        "Transcription complete: segments=%s preview=%s",
        len(result),
        " ".join(item["text"].strip() for item in result)[:160],
    )

    return result
