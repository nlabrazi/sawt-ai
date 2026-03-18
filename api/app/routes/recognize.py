# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.

import logging
from pathlib import Path
from fastapi import APIRouter, Form, HTTPException, UploadFile
import uuid

from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024
CONTENT_TYPE_TO_EXTENSION = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}


def resolve_temp_extension(file: UploadFile) -> str:
    original_suffix = Path(file.filename or "").suffix.lower()

    if original_suffix:
        return original_suffix

    return CONTENT_TYPE_TO_EXTENSION.get(file.content_type or "", ".bin")


@router.post("/recognize")
async def recognize(
    file: UploadFile,
    detect_imam: bool = Form(True),
):
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

    temp_extension = resolve_temp_extension(file)
    temp_path = f"/tmp/{uuid.uuid4()}{temp_extension}"
    temp_file = Path(temp_path)

    logger.info(
        "Recognize request received: filename=%s content_type=%s size=%s detect_imam=%s temp_path=%s",
        file.filename,
        file.content_type,
        len(file_bytes),
        detect_imam,
        temp_path,
    )

    with temp_file.open("wb") as buffer:
        buffer.write(file_bytes)

    try:
        return run_inference_pipeline(temp_path, detect_imam=detect_imam)
    finally:
        temp_file.unlink(missing_ok=True)
