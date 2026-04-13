import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.recognize import router as recognize_router
from app.routes.tajwid import router as tajwid_router
from app.routes.feedback import router as feedback_router
from app.core.model_loader import load_all_models
from app.schemas.health import HealthResponse
from app.services.imam_prediction_service import (
    get_imam_service_health,
    preflight_imam_resources,
)
from app.services.tajwid_service import TajwidServiceError, warm_tajwid_cache

LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
logging.getLogger().setLevel(LOG_LEVEL)
logging.getLogger("app").setLevel(LOG_LEVEL)
logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def parse_allowed_origins(raw_allowed_origins: str | None) -> list[str]:
    origins = raw_allowed_origins or ",".join(DEFAULT_ALLOWED_ORIGINS)
    normalized_origins: list[str] = []
    seen_origins: set[str] = set()

    for origin in origins.split(","):
        normalized_origin = origin.strip().rstrip("/")

        if not normalized_origin or normalized_origin in seen_origins:
            continue

        normalized_origins.append(normalized_origin)
        seen_origins.add(normalized_origin)

    return normalized_origins


def build_cors_options(raw_allowed_origins: str | None = None) -> dict[str, Any]:
    return {
        "allow_origins": parse_allowed_origins(raw_allowed_origins),
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
CORS_OPTIONS = build_cors_options(",".join(ALLOWED_ORIGINS))


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_all_models()
    preflight_imam_resources()

    try:
        warm_tajwid_cache()
    except TajwidServiceError:
        logger.warning(
            "Tajwid warmup failed during startup; the API will retry on demand.",
            exc_info=True,
        )

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    **CORS_OPTIONS,
)


app.include_router(recognize_router)
app.include_router(feedback_router)
app.include_router(tajwid_router)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "services": {
            "imam_detection": get_imam_service_health(),
        },
    }
