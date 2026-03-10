# ROLE
# ----
# Pipeline central de reconnaissance Sawt AI.
# Ce fichier orchestre les 3 étapes :
# 1) transcription audio
# 2) détection des versets
# 3) reconnaissance de l'imam


from app.services.transcription_service import transcribe_audio
from app.services.verse_detection_service import detect_versets
from app.services.imam_prediction_service import predict_imam


def run_inference_pipeline(audio_path: str):

    # 1️⃣ transcription
    transcription = transcribe_audio(audio_path)

    # 2️⃣ détection du verset
    verse = detect_versets(transcription)

    # 3️⃣ reconnaissance de l'imam
    imam = predict_imam(audio_path)

    return {
        "transcription": transcription,
        "verse": verse,
        "imam": imam
    }
