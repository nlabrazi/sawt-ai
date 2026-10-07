"""Read-only French content, loaded separately from audio recognition."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.core.api_logger import log_api_event
from app.schemas.quran import QuranAyahContent, QuranContentResponse
from app.schemas.tafsir import VerifiedTafsirEntry
from app.services.quran_translation_service import (
    QuranTranslationServiceError,
    fetch_quran_translations,
    validate_verse_range,
)
from app.services.tafsir_store import TafsirStoreError, fetch_verified_tafsirs

router = APIRouter()


@router.get("/quran/content", response_model=QuranContentResponse)
async def get_quran_content(
    response: Response,
    surah_id: Annotated[int, Query(ge=1, le=114)],
    start_verse: Annotated[int, Query(ge=1, le=286)],
    end_verse: Annotated[int, Query(ge=1, le=286)],
):
    # A later correction can revoke verification: never reuse an HTTP snapshot.
    response.headers["Cache-Control"] = "no-store"
    try:
        validate_verse_range(surah_id, start_verse, end_verse)
    except QuranTranslationServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        translations = await run_in_threadpool(
            fetch_quran_translations, surah_id, start_verse, end_verse,
        )
    except QuranTranslationServiceError as exc:
        raise HTTPException(
            status_code=503, detail="La traduction française est temporairement indisponible.",
            headers={"Cache-Control": "no-store"},
        ) from exc

    tafsir_status = "available"
    try:
        entries = await run_in_threadpool(
            fetch_verified_tafsirs, surah_id, start_verse, end_verse,
        )
        # Require verified again in the public contract; never return internal rows.
        tafsirs = [VerifiedTafsirEntry.model_validate(entry.model_dump()) for entry in entries]
    except (TafsirStoreError, ValidationError) as exc:
        tafsirs = []
        tafsir_status = "unavailable"
        log_api_event(
            level="warning", message="Verified tafsir unavailable for Quran content.",
            route="/quran/content",
            extra={"surah_id": surah_id, "start_verse": start_verse,
                   "end_verse": end_verse, "errorType": type(exc).__name__},
        )

    translations_by_ayah = {entry.ayah: entry for entry in translations}
    tafsirs_by_ayah: dict[int, list[VerifiedTafsirEntry]] = {}
    for entry in tafsirs:
        tafsirs_by_ayah.setdefault(entry.ayah, []).append(entry)

    return QuranContentResponse(
        surah_id=surah_id, start_verse=start_verse, end_verse=end_verse,
        tafsir_status=tafsir_status,
        ayahs=[
            QuranAyahContent(
                ayah=ayah, translation=translations_by_ayah.get(ayah),
                tafsirs=tafsirs_by_ayah.get(ayah, []),
            )
            for ayah in range(start_verse, end_verse + 1)
        ],
    )
