# ROLE
# ----
# Endpoint API pour recevoir et stocker le feedback utilisateur.

import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.schemas.feedback import FeedbackPayload
from app.services.feedback_store import (
    FeedbackStoreConfigError,
    FeedbackStoreError,
    save_feedback,
)

router = APIRouter()
logger = logging.getLogger(__name__)

FEEDBACK_CONFIG_ERROR_MESSAGE = "Service de feedback temporairement indisponible."
FEEDBACK_STORE_ERROR_MESSAGE = "Impossible d'enregistrer le feedback pour le moment."


@router.post("/feedback")
async def create_feedback(payload: FeedbackPayload):
    try:
        await run_in_threadpool(save_feedback, payload.model_dump(mode="json"))
    except FeedbackStoreConfigError as exc:
        logger.exception("Feedback store configuration error")
        raise HTTPException(status_code=503, detail=FEEDBACK_CONFIG_ERROR_MESSAGE) from exc
    except FeedbackStoreError as exc:
        raise HTTPException(status_code=502, detail=FEEDBACK_STORE_ERROR_MESSAGE) from exc

    return {
        "success": True,
        "message": "Feedback enregistré."
    }
