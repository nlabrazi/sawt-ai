# ROLE
# ----
# Orchestration principale :
# transcription -> détection verset
# La détection imam est temporairement désactivée pour la V1.

from app.services.transcription_service import transcribe_audio
from app.services.verse_detection_service import detect_versets


def run_inference_pipeline(audio_path: str):
    segments = transcribe_audio(audio_path)

    transcription_text = " ".join(
        segment.get("text", "").strip()
        for segment in segments
    ).strip()

    verse = detect_versets(segments)

    return {
        "transcription_text": transcription_text,
        "verse": verse,
        "imam_predictions": [],
        "imam_status": "soon",
    }
