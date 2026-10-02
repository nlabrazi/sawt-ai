"""Transcribe a spoken French search using the existing Whisper model."""

from app.services.transcription_service import transcribe_audio

MAX_HADITH_AUDIO_DURATION_SECONDS = 30
# MediaRecorder stops asynchronously and may include a final codec frame.
HADITH_AUDIO_DURATION_TOLERANCE_SECONDS = 1


class HadithVoiceQueryError(Exception):
    """The audio did not produce a usable search query."""


def transcribe_hadith_query(audio_path: str) -> str:
    segments = transcribe_audio(audio_path, language="fr", vad_filter=True)
    query = " ".join(segment["text"] for segment in segments).strip()
    if len(query) < 3 or not any(character.isalpha() for character in query):
        raise HadithVoiceQueryError("Aucune demande comprise. Réessayez en parlant clairement en français.")
    if len(query) > 300:
        raise HadithVoiceQueryError("Votre demande doit contenir au maximum 300 caractères. Essayez une phrase plus courte.")
    return query
