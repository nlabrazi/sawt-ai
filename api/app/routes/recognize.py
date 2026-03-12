# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.

from fastapi import APIRouter, HTTPException, UploadFile
import shutil
import uuid

from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()

MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024


@router.post("/recognize")
async def recognize(file: UploadFile):
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

    temp_path = f"/tmp/{uuid.uuid4()}.wav"

    with open(temp_path, "wb") as buffer:
        buffer.write(file_bytes)

    result = run_inference_pipeline(temp_path)

    return result
