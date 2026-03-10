# ROLE
# ----
# Transcrit un fichier audio avec le modèle Whisper déjà chargé.

from app.core.model_loader import get_whisper_model


def transcribe_audio(audio_path: str):
    model = get_whisper_model()
    result = model.transcribe(audio_path, language="ar")

    return result.get("segments", [])
