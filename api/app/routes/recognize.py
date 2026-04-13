# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.

import logging
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas.recognize import RecognizeResponse
from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024
HEADER_SNIFF_BYTES = 4096
READ_CHUNK_SIZE_BYTES = 1024 * 1024
CONTENT_TYPE_ALIASES = {
    "audio/wav": "audio/wav",
    "audio/x-wav": "audio/wav",
    "audio/mpeg": "audio/mpeg",
    "audio/mp3": "audio/mpeg",
    "audio/mp4": "audio/mp4",
    "audio/x-m4a": "audio/mp4",
    "audio/ogg": "audio/ogg",
    "audio/webm": "audio/webm",
}
CONTENT_TYPE_TO_EXTENSION = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}


def canonicalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""

    normalized_content_type = content_type.strip().lower()
    return CONTENT_TYPE_ALIASES.get(normalized_content_type, normalized_content_type)


def resolve_temp_extension(content_type: str) -> str:
    return CONTENT_TYPE_TO_EXTENSION.get(content_type, ".bin")


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


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile,
    detect_imam: bool = Form(True),
):
    request_id = uuid.uuid4()
    temp_file: Path | None = None

    try:
        temp_file, file_size, detected_content_type = await persist_upload_to_temp_file(file)
        declared_content_type = canonicalize_content_type(file.content_type)

        logger.info(
            "Recognize request received: request_id=%s filename=%s declared_content_type=%s detected_content_type=%s size=%s detect_imam=%s temp_path=%s",
            request_id,
            file.filename,
            declared_content_type or None,
            detected_content_type,
            file_size,
            detect_imam,
            temp_file,
        )

        if declared_content_type and declared_content_type != detected_content_type:
            logger.warning(
                "Recognize content type mismatch: request_id=%s filename=%s declared=%s detected=%s",
                request_id,
                file.filename,
                declared_content_type,
                detected_content_type,
            )

        return await run_in_threadpool(
            run_inference_pipeline,
            str(temp_file),
            detect_imam,
        )
    finally:
        await file.close()

        if temp_file is not None:
            temp_file.unlink(missing_ok=True)
