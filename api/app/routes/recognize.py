# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.

import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.api_logger import log_api_event
from app.core.upload_policy import (
    MAX_AUDIO_DURATION_SECONDS,
    MAX_FILE_SIZE_BYTES,
    canonicalize_content_type,
    resolve_temp_extension,
)
from app.schemas.recognize import RecognizeResponse
from app.services.audio_metadata_service import (
    AudioMetadataError,
    get_audio_duration_seconds,
)
from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()

HEADER_SNIFF_BYTES = 4096
READ_CHUNK_SIZE_BYTES = 1024 * 1024


def sniff_audio_content_type(header_bytes: bytes) -> str | None:
    if (
        len(header_bytes) >= 12
        and header_bytes.startswith(b"RIFF")
        and header_bytes[8:12] == b"WAVE"
    ):
        return "audio/wav"

    if header_bytes.startswith(b"OggS"):
        return "audio/ogg"

    if len(header_bytes) >= 12 and header_bytes[4:8] == b"ftyp":
        return "audio/mp4"

    if (
        len(header_bytes) >= 4
        and header_bytes[:4] == b"\x1a\x45\xdf\xa3"
        and b"webm" in header_bytes.lower()
    ):
        return "audio/webm"

    if header_bytes.startswith(b"ID3"):
        return "audio/mpeg"

    if (
        len(header_bytes) >= 2
        and header_bytes[0] == 0xFF
        and (header_bytes[1] & 0xE0) == 0xE0
    ):
        return "audio/mpeg"

    return None


async def persist_upload_to_temp_file(file: UploadFile) -> tuple[Path, int, str]:
    header_bytes = await file.read(HEADER_SNIFF_BYTES)
    total_bytes = len(header_bytes)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")

    if total_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

    detected_content_type = sniff_audio_content_type(header_bytes)

    if detected_content_type is None:
        raise HTTPException(
            status_code=415,
            detail="Format audio invalide ou non pris en charge.",
        )

    with NamedTemporaryFile(
        mode="wb",
        suffix=resolve_temp_extension(detected_content_type),
        delete=False,
        dir="/tmp",
    ) as temp_buffer:
        temp_path = Path(temp_buffer.name)

        try:
            temp_buffer.write(header_bytes)

            while True:
                chunk = await file.read(READ_CHUNK_SIZE_BYTES)
                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

                temp_buffer.write(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    return temp_path, total_bytes, detected_content_type


def enforce_audio_duration_limit(audio_path: Path) -> float:
    try:
        duration_seconds = get_audio_duration_seconds(audio_path)
    except AudioMetadataError as exc:
        raise HTTPException(
            status_code=415,
            detail="Impossible de lire la durée du fichier audio.",
        ) from exc

    if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"Audio trop long. Maximum {MAX_AUDIO_DURATION_SECONDS} secondes.",
        )

    return duration_seconds


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile,
    detect_imam: bool = Form(True),
    allow_ambiguous_result: bool = Form(True),
):
    request_id = uuid.uuid4()
    temp_file: Path | None = None

    try:
        temp_file, file_size, detected_content_type = await persist_upload_to_temp_file(file)
        audio_duration_seconds = enforce_audio_duration_limit(temp_file)
        declared_content_type = canonicalize_content_type(file.content_type)

        log_api_event(
            message="Recognize request received",
            route="/recognize",
            extra={
                "requestId": str(request_id),
                "filename": file.filename,
                "declaredContentType": declared_content_type,
                "detectedContentType": detected_content_type,
                "size": file_size,
                "durationSeconds": round(audio_duration_seconds, 3),
                "detectImam": detect_imam,
                "allowAmbiguousResult": allow_ambiguous_result,
            },
        )

        if declared_content_type and declared_content_type != detected_content_type:
            log_api_event(
                level="warning",
                message="Recognize content type mismatch",
                route="/recognize",
                extra={
                    "requestId": str(request_id),
                    "filename": file.filename,
                    "declaredContentType": declared_content_type,
                    "detectedContentType": detected_content_type,
                },
            )

        return await run_in_threadpool(
            run_inference_pipeline,
            str(temp_file),
            detect_imam,
            audio_duration_seconds,
            allow_ambiguous_result,
            str(request_id),
        )
    finally:
        await file.close()

        if temp_file is not None:
            temp_file.unlink(missing_ok=True)
