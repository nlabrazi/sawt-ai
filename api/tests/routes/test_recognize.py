import asyncio
from pathlib import Path
from tempfile import SpooledTemporaryFile

from fastapi import HTTPException

import app.routes.recognize as recognize_route


def build_upload_file(filename: str, content: bytes, content_type: str):
    temp_file = SpooledTemporaryFile()
    temp_file.write(content)
    temp_file.seek(0)

    return recognize_route.UploadFile(
        filename=filename,
        file=temp_file,
        headers={"content-type": content_type},
    )


def test_resolve_temp_extension_prefers_filename_suffix():
    upload = build_upload_file("recitation.ogg", b"", "audio/webm")

    assert recognize_route.resolve_temp_extension(upload) == ".ogg"


def test_resolve_temp_extension_falls_back_to_content_type():
    upload = build_upload_file("", b"", "audio/webm")

    assert recognize_route.resolve_temp_extension(upload) == ".webm"


def test_recognize_runs_pipeline_and_cleans_temp_file(monkeypatch):
    captured = {}

    def fake_pipeline(audio_path: str, detect_imam: bool):
        path = Path(audio_path)
        captured["path"] = path
        captured["detect_imam"] = detect_imam
        captured["bytes"] = path.read_bytes()

        return {
            "transcription_text": "قل هو الله احد",
            "verse": None,
            "imam_predictions": [],
            "imam_status": "disabled",
            "imam_detection_enabled": detect_imam,
        }

    monkeypatch.setattr(recognize_route, "run_inference_pipeline", fake_pipeline)

    upload = build_upload_file("recitation.webm", b"test-audio", "audio/webm")

    response = asyncio.run(recognize_route.recognize(file=upload, detect_imam=False))

    assert response["imam_detection_enabled"] is False
    assert captured["detect_imam"] is False
    assert captured["bytes"] == b"test-audio"
    assert not captured["path"].exists()


def test_recognize_rejects_too_large_files():
    oversized_upload = build_upload_file(
        "too-big.webm",
        b"a" * (recognize_route.MAX_FILE_SIZE_BYTES + 1),
        "audio/webm",
    )

    try:
        asyncio.run(recognize_route.recognize(file=oversized_upload, detect_imam=True))
    except HTTPException as exc:
        assert exc.status_code == 413
        assert exc.detail == "Fichier trop volumineux."
    else:
        raise AssertionError("Expected HTTPException")
