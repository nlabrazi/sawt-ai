# Endpoint API de reconnaissance coranique.

import uuid
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.api_logger import log_api_event
from app.core.audio_upload import enforce_audio_duration_limit, persist_upload_to_temp_file
from app.core.inference_runtime import get_inference_semaphore
from app.core.upload_policy import canonicalize_content_type
from app.schemas.recognize import RecognizeResponse
from app.services.inference_pipeline import run_inference_pipeline

router = APIRouter()


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
        audio_duration_seconds = await run_in_threadpool(
            enforce_audio_duration_limit,
            temp_file,
        )
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

        async with get_inference_semaphore():
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
