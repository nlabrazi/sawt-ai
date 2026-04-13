# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.
# Schémas de réponse pour l'endpoint /recognize.

from pydantic import BaseModel


class ImamPrediction(BaseModel):
    name: str
    score: float


class VerseMatch(BaseModel):
    sourate_id: int
    sourate_name: str
    transliteration: str
    start_verse: int
    end_verse: int
    text: str
    similarity: float


class RecognizeResponse(BaseModel):
    transcription_text: str
    verse: VerseMatch | None
    imam_predictions: list[ImamPrediction]
    imam_status: str
    imam_detection_enabled: bool
