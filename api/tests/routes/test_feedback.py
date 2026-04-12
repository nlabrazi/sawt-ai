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

    def fake_save_feedback(payload):
        captured_payload.update(payload)

    monkeypatch.setattr(feedback_route, "save_feedback", fake_save_feedback)

    response = asyncio.run(feedback_route.create_feedback(FeedbackPayload(**build_payload())))

    assert response == {
        "success": True,
        "message": "Feedback enregistré.",
    }
    assert captured_payload["transcription_text"] == "قل هو الله احد"


def test_create_feedback_maps_config_error_to_503(monkeypatch):
    def fake_save_feedback(_payload):
        raise FeedbackStoreConfigError("missing config")

    monkeypatch.setattr(feedback_route, "save_feedback", fake_save_feedback)

    try:
        asyncio.run(feedback_route.create_feedback(FeedbackPayload(**build_payload())))
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail == "missing config"
    else:
        raise AssertionError("Expected HTTPException")

def test_create_feedback_maps_store_error_to_502(monkeypatch):
    def fake_save_feedback(_payload):
        raise FeedbackStoreError("upstream failure")

    monkeypatch.setattr(feedback_route, "save_feedback", fake_save_feedback)

    try:
        asyncio.run(feedback_route.create_feedback(FeedbackPayload(**build_payload())))
    except HTTPException as exc:
        assert exc.status_code == 502
        assert exc.detail == "upstream failure"
    else:
        raise AssertionError("Expected HTTPException")
