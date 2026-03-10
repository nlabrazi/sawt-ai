# ROLE
# ----
# Endpoint API qui reçoit un fichier audio
# et déclenche le pipeline Sawt AI.


from fastapi import APIRouter, UploadFile
import shutil
import uuid

from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()


@router.post("/recognize")
async def recognize(file: UploadFile):

    # sauvegarde temporaire du fichier
    temp_path = f"/tmp/{uuid.uuid4()}.wav"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_inference_pipeline(temp_path)

    return result
