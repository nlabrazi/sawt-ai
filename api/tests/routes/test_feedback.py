import asyncio

from fastapi import HTTPException

import app.routes.feedback as feedback_route
from app.schemas.feedback import FeedbackPayload
from app.services.feedback_store import FeedbackStoreConfigError, FeedbackStoreError


def build_payload():
    return {
        "is_correct": True,
        "transcription_text": "قل هو الله احد",
        "detected_verse": None,
        "correction": None,
        "comment": None,
    }


def test_create_feedback_returns_success(monkeypatch):
    captured_payload = {}
    captured_call = {}

    def fake_save_feedback(payload):
        captured_payload.update(payload)

    async def fake_run_in_threadpool(func, payload):
        captured_call["func"] = func
        captured_call["payload"] = payload
        return func(payload)

    monkeypatch.setattr(feedback_route, "build_feedback_store_payload", lambda payload: payload.model_dump(mode="json"))
    monkeypatch.setattr(feedback_route, "save_feedback", fake_save_feedback)
    monkeypatch.setattr(feedback_route, "run_in_threadpool", fake_run_in_threadpool)

    response = asyncio.run(feedback_route.create_feedback(FeedbackPayload(**build_payload())))

    assert response == {
        "success": True,
        "message": "Feedback enregistré.",
    }
    assert captured_call["func"] is fake_save_feedback
    assert captured_call["payload"]["transcription_text"] == "قل هو الله احد"
    assert captured_payload["transcription_text"] == "قل هو الله احد"


def test_create_feedback_maps_config_error_to_503(monkeypatch):
    async def fake_run_in_threadpool(_func, _payload):
        raise FeedbackStoreConfigError("missing config")

    monkeypatch.setattr(feedback_route, "build_feedback_store_payload", lambda payload: payload.model_dump(mode="json"))
    monkeypatch.setattr(feedback_route, "run_in_threadpool", fake_run_in_threadpool)

    try:
        asyncio.run(feedback_route.create_feedback(FeedbackPayload(**build_payload())))
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail == feedback_route.FEEDBACK_CONFIG_ERROR_MESSAGE
    else:
        raise AssertionError("Expected HTTPException")


def test_create_feedback_maps_store_error_to_502(monkeypatch):
    async def fake_run_in_threadpool(_func, _payload):
        raise FeedbackStoreError("upstream failure with secret details")

    monkeypatch.setattr(feedback_route, "build_feedback_store_payload", lambda payload: payload.model_dump(mode="json"))
    monkeypatch.setattr(feedback_route, "run_in_threadpool", fake_run_in_threadpool)

    try:
        asyncio.run(feedback_route.create_feedback(FeedbackPayload(**build_payload())))
    except HTTPException as exc:
        assert exc.status_code == 502
        assert exc.detail == feedback_route.FEEDBACK_STORE_ERROR_MESSAGE
    else:
        raise AssertionError("Expected HTTPException")


def test_create_feedback_maps_payload_validation_error_to_422(monkeypatch):
    monkeypatch.setattr(
        feedback_route,
        "build_feedback_store_payload",
        lambda _payload: (_ for _ in ()).throw(
            feedback_route.FeedbackPayloadValidationError("correction invalide")
        ),
    )

    try:
        asyncio.run(feedback_route.create_feedback(FeedbackPayload(**build_payload())))
    except HTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail == "correction invalide"
    else:
        raise AssertionError("Expected HTTPException")
