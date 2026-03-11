# ROLE
# ----
# Schémas Pydantic pour recevoir un feedback utilisateur
# sur un résultat de reconnaissance.

from typing import Optional
from pydantic import BaseModel


class VerseMatchPayload(BaseModel):
    sourate_id: int
    sourate_name: str
    start_verse: int
    end_verse: int
    text: str
    similarity: float


class VerseCorrectionPayload(BaseModel):
    sourate_name: str
    start_verse: int
    end_verse: int


class FeedbackPayload(BaseModel):
    is_correct: bool
    transcription_text: str
    detected_verse: Optional[VerseMatchPayload] = None
    correction: Optional[VerseCorrectionPayload] = None
    comment: Optional[str] = None
