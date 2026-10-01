# ROLE
# ----
# Endpoint API pour rechercher des hadiths par mots-clés ou phrase française.

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.schemas.hadith import HadithSearchRequest, HadithSearchResponse
from app.services.hadith_search_service import HadithSearchError, get_hadith_search_service

router = APIRouter()


@router.post("/hadith/search", response_model=HadithSearchResponse)
async def search_hadith(request: HadithSearchRequest) -> HadithSearchResponse:
    try:
        return await run_in_threadpool(
            get_hadith_search_service().search,
            request.query,
            request.limit,
        )
    except HadithSearchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
