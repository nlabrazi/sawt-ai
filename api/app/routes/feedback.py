# ROLE
# ----
# Endpoint API pour recevoir et stocker le feedback utilisateur.

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.schemas.feedback import FeedbackPayload
from app.services.feedback_payload_service import (
    FeedbackPayloadValidationError,
    build_feedback_store_payload,
)
from app.services.feedback_store import (
    FeedbackStoreConfigError,
    FeedbackStoreError,
    save_feedback,
)

router = APIRouter()

FEEDBACK_CONFIG_ERROR_MESSAGE = "Service de feedback temporairement indisponible."
FEEDBACK_STORE_ERROR_MESSAGE = "Impossible d'enregistrer le feedback pour le moment."


@router.post("/feedback")
async def create_feedback(payload: FeedbackPayload):
    try:
        prepared_payload = build_feedback_store_payload(payload)
        await run_in_threadpool(save_feedback, prepared_payload)
    except FeedbackPayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FeedbackStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=FEEDBACK_CONFIG_ERROR_MESSAGE) from exc
    except FeedbackStoreError as exc:
        raise HTTPException(status_code=502, detail=FEEDBACK_STORE_ERROR_MESSAGE) from exc

    return {
        "success": True,
        "message": "Feedback enregistré."
    }
