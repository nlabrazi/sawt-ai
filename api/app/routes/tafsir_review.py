"""Private review endpoints; never used by the public recognition flow."""

import os
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from app.schemas.tafsir import TafsirReviewEntry, TafsirSource, TafsirStatus
from app.services import tafsir_store
from app.services.quran_catalog_service import get_surah_metadata

bearer = HTTPBearer(auto_error=False)


async def require_review_access(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    password = os.getenv("TAFSIR_REVIEW_PASSWORD", "")
    if not password.strip():
        raise HTTPException(503, "L’accès à la review n’est pas configuré.")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode("utf-8"), password.encode("utf-8"),
    ):
        raise HTTPException(401, "Mot de passe incorrect.", headers={"WWW-Authenticate": "Bearer"})


router = APIRouter(
    prefix="/internal/tafsir", dependencies=[Depends(require_review_access)],
    include_in_schema=False,
)


class ReviewRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_updated_at: AwareDatetime


class TafsirTextUpdate(ReviewRevision):
    text_fr: str = Field(min_length=1)

    @field_validator("text_fr")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Le texte français ne doit pas être vide.")
        return value


SurahId = Annotated[int, Path(ge=1, le=114)]
Ayah = Annotated[int, Path(ge=1, le=286)]


def check_verse(surah_id: int, ayah: int) -> None:
    surah = get_surah_metadata(surah_id)
    if surah is None or ayah > surah["total_verses"]:
        raise HTTPException(422, "Ce verset n’existe pas dans le catalogue.")


async def call_store(operation, *args, **kwargs):
    try:
        return await run_in_threadpool(operation, *args, **kwargs)
    except tafsir_store.TafsirStoreConflict as exc:
        raise HTTPException(409, "Le tafsir a changé. Rechargez-le et relisez-le avant de continuer.") from exc
    except tafsir_store.TafsirStoreConfigError as exc:
        raise HTTPException(503, "Le stockage des tafsirs n’est pas configuré.") from exc
    except tafsir_store.TafsirStoreError as exc:
        raise HTTPException(502, "Le stockage des tafsirs est temporairement indisponible.") from exc


@router.get("/access", status_code=204)
async def check_access():
    return Response(status_code=204)


@router.get("", response_model=list[TafsirReviewEntry])
async def list_reviews(
    surah_id: Annotated[int | None, Query(ge=1, le=114)] = None,
    source: TafsirSource | None = None,
    status: TafsirStatus | None = "need_review",
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return await call_store(
        tafsir_store.list_tafsirs_for_review, surah_id=surah_id, source=source,
        status=status, limit=limit, offset=offset,
    )


@router.patch("/{surah_id}/{ayah}/{source}", response_model=TafsirReviewEntry)
async def edit_review(surah_id: SurahId, ayah: Ayah, source: TafsirSource, payload: TafsirTextUpdate):
    check_verse(surah_id, ayah)
    return await call_store(
        tafsir_store.update_tafsir_text, surah_id, ayah, source, payload.text_fr,
        expected_updated_at=payload.expected_updated_at,
    )


@router.post("/{surah_id}/{ayah}/{source}/verify", response_model=TafsirReviewEntry)
async def validate_review(surah_id: SurahId, ayah: Ayah, source: TafsirSource, payload: ReviewRevision):
    check_verse(surah_id, ayah)
    return await call_store(
        tafsir_store.verify_tafsir, surah_id, ayah, source,
        expected_updated_at=payload.expected_updated_at,
    )
