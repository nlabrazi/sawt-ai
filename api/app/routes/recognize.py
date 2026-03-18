# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.

from fastapi import APIRouter, Form, HTTPException, UploadFile
from pathlib import Path
import uuid

from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()

MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024


@router.post("/recognize")
async def recognize(
    file: UploadFile,
    detect_imam: bool = Form(True),
):
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

    temp_path = f"/tmp/{uuid.uuid4()}.wav"
    temp_file = Path(temp_path)

    with temp_file.open("wb") as buffer:
        buffer.write(file_bytes)

    try:
        return run_inference_pipeline(temp_path, detect_imam=detect_imam)
    finally:
        temp_file.unlink(missing_ok=True)
