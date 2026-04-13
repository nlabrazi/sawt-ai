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


class HealthServices(BaseModel):
    imam_detection: ImamDetectionHealth
    upload_policy: UploadPolicy


class HealthResponse(BaseModel):
    status: Literal["ok"]
    services: HealthServices
