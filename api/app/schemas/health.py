from typing import Literal

from pydantic import BaseModel


class ImamDetectionHealth(BaseModel):
    available: bool
    status: Literal["available", "unavailable"]
    message: str | None


class HealthServices(BaseModel):
    imam_detection: ImamDetectionHealth


class HealthResponse(BaseModel):
    status: Literal["ok"]
    services: HealthServices
