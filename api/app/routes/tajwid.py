# ROLE
# ----
# Endpoint API pour récupérer le texte tajwid d'une plage de versets.

from fastapi import APIRouter, HTTPException, Query

from app.services.tajwid_service import fetch_tajwid_text, TajwidServiceError

router = APIRouter()


@router.get("/tajwid")
def get_tajwid(
    surah_id: int = Query(..., ge=1, le=114),
    start_verse: int = Query(..., ge=1),
    end_verse: int = Query(..., ge=1),
):
    if end_verse < start_verse:
        raise HTTPException(
            status_code=400,
            detail="end_verse doit être supérieur ou égal à start_verse.",
        )

    try:
        return fetch_tajwid_text(
            surah_id=surah_id,
            start_verse=start_verse,
            end_verse=end_verse,
        )
    except TajwidServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
