from typing import Literal

from pydantic import BaseModel


class ImamDetectionHealth(BaseModel):
    available: bool
    status: Literal["available", "unavailable"]
    message: str | None


class UploadPolicy(BaseModel):
    max_file_size_bytes: int
    max_audio_duration_seconds: int
    accepted_mime_types: list[str]
    accepted_file_extensions: list[str]


class DetectionPolicy(BaseModel):
    min_accepted_similarity: float
    min_probable_similarity: float
    min_matched_word_count: int
    min_score_margin: float
    progressive_analysis_step_seconds: int


class HealthServices(BaseModel):
    imam_detection: ImamDetectionHealth
    upload_policy: UploadPolicy
    detection_policy: DetectionPolicy


class HealthResponse(BaseModel):
    status: Literal["ok"]
    services: HealthServices
