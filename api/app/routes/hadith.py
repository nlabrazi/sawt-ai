# ROLE
# ----
# Endpoint API pour rechercher des hadiths par mots-clés ou phrase française.

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.audio_upload import enforce_audio_duration_limit, persist_upload_to_temp_file
from app.core.inference_runtime import get_inference_semaphore
from app.schemas.hadith import HadithSearchRequest, HadithSearchResponse, HadithTranscriptionResponse
from app.services.hadith_transcription_service import (
    MAX_HADITH_AUDIO_DURATION_SECONDS, HadithVoiceQueryError, transcribe_hadith_query,
)
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


@router.post("/hadith/transcribe", response_model=HadithTranscriptionResponse)
async def transcribe_hadith(file: UploadFile) -> HadithTranscriptionResponse:
    temp_file: Path | None = None
    try:
        temp_file, _, _ = await persist_upload_to_temp_file(file)
        await run_in_threadpool(
            enforce_audio_duration_limit, temp_file, MAX_HADITH_AUDIO_DURATION_SECONDS,
        )
        try:
            async with get_inference_semaphore():
                query = await run_in_threadpool(transcribe_hadith_query, str(temp_file))
        except HadithVoiceQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="La recherche vocale est temporairement indisponible.",
            ) from exc
        return HadithTranscriptionResponse(query=query)
    finally:
        await file.close()
        if temp_file is not None:
            temp_file.unlink(missing_ok=True)
