# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.
# Schémas de réponse pour l'endpoint /recognize.

from typing import Literal

from pydantic import BaseModel

ImamStatus = Literal["disabled", "unknown", "unavailable", "high", "medium", "low"]
DetectionStatus = Literal["confident", "probable", "ambiguous", "insufficient"]
RejectionReason = Literal[
    "no_match",
    "score_too_low",
    "transcription_too_short",
    "ambiguous_match",
]


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


class VerseDetectionMetadata(BaseModel):
    status: DetectionStatus
    score: float | None
    score_margin: float | None
    matched_word_count: int
    rejection_reason: RejectionReason | None


class RecognizeResponse(BaseModel):
    transcription_text: str
    verse: VerseMatch | None
    detection: VerseDetectionMetadata
    imam_predictions: list[ImamPrediction]
    imam_status: ImamStatus
    imam_detection_enabled: bool
