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


def build_wav_like_audio(payload: bytes = b"test-audio") -> bytes:
    return b"RIFF\x00\x00\x00\x00WAVE" + payload


def build_mp3_like_audio(payload: bytes = b"test-audio") -> bytes:
    return b"ID3" + payload


def test_resolve_temp_extension_returns_extension_for_detected_content_type():
    assert recognize_route.resolve_temp_extension("audio/ogg") == ".ogg"


def test_sniff_audio_content_type_detects_wav_header():
    assert recognize_route.sniff_audio_content_type(build_wav_like_audio()) == "audio/wav"


def test_recognize_runs_pipeline_in_threadpool_and_cleans_temp_file(monkeypatch):
    captured = {
        "threadpool_called": False,
    }

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

    async def fake_run_in_threadpool(func, *args, **kwargs):
        captured["threadpool_called"] = True
        return func(*args, **kwargs)

    monkeypatch.setattr(recognize_route, "run_inference_pipeline", fake_pipeline)
    monkeypatch.setattr(recognize_route, "run_in_threadpool", fake_run_in_threadpool)

    upload = build_upload_file(
        "recitation.webm",
        build_wav_like_audio(),
        "audio/webm",
    )

    response = asyncio.run(recognize_route.recognize(file=upload, detect_imam=False))

    assert response["imam_detection_enabled"] is False
    assert captured["threadpool_called"] is True
    assert captured["detect_imam"] is False
    assert captured["bytes"] == build_wav_like_audio()
    assert captured["path"].suffix == ".wav"
    assert not captured["path"].exists()


def test_recognize_rejects_invalid_audio_signature():
    invalid_upload = build_upload_file(
        "invalid.webm",
        b"not-audio",
        "audio/webm",
    )

    try:
        asyncio.run(recognize_route.recognize(file=invalid_upload, detect_imam=True))
    except HTTPException as exc:
        assert exc.status_code == 415
        assert exc.detail == "Format audio invalide ou non pris en charge."
    else:
        raise AssertionError("Expected HTTPException")


def test_recognize_rejects_too_large_files_during_streaming(monkeypatch):
    monkeypatch.setattr(recognize_route, "MAX_FILE_SIZE_BYTES", 10)
    monkeypatch.setattr(recognize_route, "HEADER_SNIFF_BYTES", 3)
    monkeypatch.setattr(recognize_route, "READ_CHUNK_SIZE_BYTES", 4)

    oversized_upload = build_upload_file(
        "too-big.mp3",
        build_mp3_like_audio(b"abcdefgh"),
        "audio/mpeg",
    )

    try:
        asyncio.run(recognize_route.recognize(file=oversized_upload, detect_imam=True))
    except HTTPException as exc:
        assert exc.status_code == 413
        assert exc.detail == "Fichier trop volumineux."
    else:
        raise AssertionError("Expected HTTPException")
