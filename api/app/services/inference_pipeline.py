# ROLE
# ----
# Orchestration principale :
# transcription -> détection verset -> prédiction imam

from app.services.transcription_service import transcribe_audio
from app.services.verse_detection_service import detect_versets
from app.services.imam_prediction_service import predict_imam


def run_inference_pipeline(audio_path: str):
    segments = transcribe_audio(audio_path)

    transcription_text = " ".join(
        segment.get("text", "").strip()
        for segment in segments
    ).strip()

    verse = detect_versets(segments)
    imam_predictions = predict_imam(audio_path)

    return {
        "transcription_text": transcription_text,
        "verse": verse,
        "imam_predictions": imam_predictions,
    }
