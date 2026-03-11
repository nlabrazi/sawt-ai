# ROLE
# ----
# Endpoint API pour recevoir et stocker le feedback utilisateur.

from fastapi import APIRouter

from app.schemas.feedback import FeedbackPayload
from app.services.feedback_store import save_feedback

router = APIRouter()


@router.post("/feedback")
async def create_feedback(payload: FeedbackPayload):
    save_feedback(payload.model_dump())

    return {
        "success": True,
        "message": "Feedback enregistré."
    }
