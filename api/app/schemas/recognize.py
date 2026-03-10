# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.

import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.recognize import RecognizeResponse
from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)):
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_inference_pipeline(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
