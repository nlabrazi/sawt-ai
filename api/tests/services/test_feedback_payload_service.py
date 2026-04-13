import pytest

from app.schemas.feedback import FeedbackPayload
from app.services.feedback_payload_service import (
    FeedbackPayloadValidationError,
    build_feedback_store_payload,
)


def build_surah_metadata():
    return {
        "id": 114,
        "name": "الناس",
        "transliteration": "An-Nas",
        "total_verses": 6,
    }


def build_feedback_payload(**overrides):
    payload = {
        "is_correct": False,
        "transcription_text": "قل اعوذ برب الناس",
        "detected_verse": None,
        "correction": {
            "sourate_id": 114,
            "sourate_name": "الناس",
            "transliteration": "An-Nas",
            "start_verse": 1,
            "end_verse": 4,
        },
        "comment": "correction utile",
    }
    payload.update(overrides)
    return FeedbackPayload(**payload)


def test_build_feedback_store_payload_requires_correction_for_negative_feedback():
    payload = FeedbackPayload(
        is_correct=False,
        transcription_text="قل اعوذ برب الناس",
        detected_verse=None,
        correction=None,
        comment=None,
    )

    with pytest.raises(FeedbackPayloadValidationError, match="obligatoire"):
        build_feedback_store_payload(payload)


def test_build_feedback_store_payload_rejects_out_of_range_correction(monkeypatch):
    monkeypatch.setattr(
        "app.services.feedback_payload_service.get_surah_metadata",
        lambda surah_id: build_surah_metadata() if surah_id == 114 else None,
    )

    payload = build_feedback_payload(correction={
        "sourate_id": 114,
        "sourate_name": "الناس",
        "transliteration": "An-Nas",
        "start_verse": 2,
        "end_verse": 7,
    })

    with pytest.raises(FeedbackPayloadValidationError, match="plage de versets"):
        build_feedback_store_payload(payload)


def test_build_feedback_store_payload_canonicalizes_feedback_surah_metadata(monkeypatch):
    monkeypatch.setattr(
        "app.services.feedback_payload_service.get_surah_metadata",
        lambda surah_id: build_surah_metadata() if surah_id == 114 else None,
    )

    payload = build_feedback_payload()
    serialized_payload = build_feedback_store_payload(payload)

    assert serialized_payload["correction"] == {
        "sourate_id": 114,
        "sourate_name": "الناس",
        "transliteration": "An-Nas",
        "start_verse": 1,
        "end_verse": 4,
    }
