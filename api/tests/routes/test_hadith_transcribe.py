from pathlib import Path
from threading import get_ident
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.audio_upload as audio_upload
import app.routes.hadith as hadith_route
from app.core.inference_runtime import reset_inference_semaphore
from app.services.audio_metadata_service import AudioMetadataError
from app.services.hadith_transcription_service import HadithVoiceQueryError

AUDIO = b"RIFF\x00\x00\x00\x00WAVEaudio"
QUERY = "Trouve-moi le ou les hadiths qui parlent du mariage"


@pytest.fixture
def transcription(monkeypatch):
    reset_inference_semaphore()
    captured = []

    def transcribe(path):
        captured.append(Path(path))
        assert Path(path).read_bytes() == AUDIO
        return QUERY

    mock = Mock(side_effect=transcribe)
    mock.paths = captured
    monkeypatch.setattr(hadith_route, "transcribe_hadith_query", mock)
    monkeypatch.setattr(audio_upload, "get_audio_duration_seconds", lambda _: 5.0)
    yield mock
    reset_inference_semaphore()


@pytest.fixture
def client(transcription):
    app = FastAPI()
    app.include_router(hadith_route.router)
    with TestClient(app) as client:
        yield client


def post_audio(client, audio=AUDIO):
    # The signature, rather than the browser's declared type, controls decoding.
    return client.post("/hadith/transcribe", files={"file": ("query.webm", audio, "audio/webm")})


def test_returns_editable_query_without_searching_and_cleans_upload(client, transcription, monkeypatch):
    search = Mock()
    monkeypatch.setattr(hadith_route, "get_hadith_search_service", search)
    response = post_audio(client)
    assert response.status_code == 200
    assert response.json() == {"query": QUERY}
    assert transcription.paths[0].suffix == ".wav"
    assert not transcription.paths[0].exists()
    search.assert_not_called()


@pytest.mark.parametrize("audio,status", [(b"", 400), (b"not audio", 415)])
def test_rejects_empty_or_invalid_audio_before_inference(client, transcription, audio, status):
    assert post_audio(client, audio).status_code == status
    transcription.assert_not_called()


def test_requires_an_audio_file(client, transcription):
    assert client.post("/hadith/transcribe").status_code == 422
    transcription.assert_not_called()


@pytest.mark.parametrize("duration,status", [(30.0, 200), (30.1, 413)])
def test_limits_voice_requests_to_thirty_seconds(client, transcription, monkeypatch, duration, status):
    monkeypatch.setattr(audio_upload, "get_audio_duration_seconds", lambda _: duration)
    paths = []
    persist = hadith_route.persist_upload_to_temp_file

    async def capture(file):
        result = await persist(file)
        paths.append(result[0])
        return result

    monkeypatch.setattr(hadith_route, "persist_upload_to_temp_file", capture)
    assert post_audio(client).status_code == status
    assert not paths[0].exists()
    if status == 413:
        transcription.assert_not_called()


def test_rejects_oversized_uploads(client, transcription, monkeypatch):
    monkeypatch.setattr(audio_upload, "MAX_FILE_SIZE_BYTES", 12)
    assert post_audio(client).status_code == 413
    transcription.assert_not_called()


def test_rejects_undecodable_audio(client, transcription, monkeypatch):
    def invalid(_):
        raise AudioMetadataError("invalid audio")

    monkeypatch.setattr(audio_upload, "get_audio_duration_seconds", invalid)
    assert post_audio(client).status_code == 415
    transcription.assert_not_called()


@pytest.mark.parametrize("error,status", [
    (HadithVoiceQueryError("Aucune demande comprise."), 422),
    (RuntimeError("private model failure"), 503),
])
def test_cleans_temp_files_after_transcription_failure(client, transcription, monkeypatch, error, status):
    paths = []

    def fail(path):
        paths.append(Path(path))
        raise error

    transcription.side_effect = fail
    response = post_audio(client)
    assert response.status_code == status
    assert not paths[0].exists()
    assert "private model failure" not in response.text


def test_offloads_transcription_and_uses_the_quran_concurrency_budget(client, transcription, monkeypatch):
    import app.routes.recognize as recognize_route

    assert hadith_route.get_inference_semaphore() is recognize_route.get_inference_semaphore()
    event_loop_threads, worker_threads = [], []
    original = hadith_route.run_in_threadpool

    async def record(func, *args):
        event_loop_threads.append(get_ident())
        return await original(func, *args)

    def transcribe(path):
        worker_threads.append(get_ident())
        return QUERY

    monkeypatch.setattr(hadith_route, "run_in_threadpool", record)
    transcription.side_effect = transcribe
    assert post_audio(client).status_code == 200
    assert worker_threads[0] not in event_loop_threads
