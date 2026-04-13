# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.
# Schémas de réponse pour l'endpoint /recognize.

from typing import Literal

from pydantic import BaseModel

ImamStatus = Literal["disabled", "unknown", "unavailable", "high", "medium", "low"]


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
    imam_status: ImamStatus
    imam_detection_enabled: bool
