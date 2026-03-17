# ROLE
# ----
# Endpoint API pour recevoir et stocker le feedback utilisateur.

from fastapi import APIRouter, HTTPException

from app.schemas.feedback import FeedbackPayload
from app.services.feedback_store import (
    FeedbackStoreConfigError,
    FeedbackStoreError,
    save_feedback,
)

router = APIRouter()


@router.post("/feedback")
async def create_feedback(payload: FeedbackPayload):
    try:
        save_feedback(payload.model_dump(mode="json"))
    except FeedbackStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FeedbackStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "success": True,
        "message": "Feedback enregistré."
    }
