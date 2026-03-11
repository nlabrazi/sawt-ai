# ROLE
# ----
# Transcrit un fichier audio avec le modèle faster-whisper déjà chargé.

from app.core.model_loader import get_whisper_model


def transcribe_audio(audio_path: str):
    model = get_whisper_model()

    segments, _info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5,
    )

    return [
        {
            "text": segment.text
        }
        for segment in segments
    ]
