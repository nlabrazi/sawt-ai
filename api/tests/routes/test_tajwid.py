from fastapi import HTTPException

import app.routes.tajwid as tajwid_route
from app.services.tajwid_service import TajwidServiceError


def test_get_surahs_returns_metadata(monkeypatch):
    monkeypatch.setattr(
        tajwid_route,
        "list_surah_metadata",
        lambda: [
            {
                "id": 114,
                "name": "الناس",
                "transliteration": "An-Nas",
                "total_verses": 6,
            },
        ],
    )

    response = tajwid_route.get_surahs()

    assert response == [{
        "id": 114,
        "name": "الناس",
        "transliteration": "An-Nas",
        "total_verses": 6,
    }]


def test_get_tajwid_returns_payload(monkeypatch):
    def fake_fetch_tajwid_text(surah_id: int, start_verse: int, end_verse: int):
        return {
            "surah_id": surah_id,
            "start_verse": start_verse,
            "end_verse": end_verse,
            "text": "tajwid text",
        }

    monkeypatch.setattr(tajwid_route, "fetch_tajwid_text", fake_fetch_tajwid_text)

    response = tajwid_route.get_tajwid(surah_id=112, start_verse=1, end_verse=4)

    assert response["text"] == "tajwid text"


def test_get_tajwid_rejects_invalid_range():
    try:
        tajwid_route.get_tajwid(surah_id=112, start_verse=4, end_verse=1)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "end_verse doit être supérieur ou égal à start_verse."
    else:
        raise AssertionError("Expected HTTPException")


def test_get_tajwid_maps_service_error_to_502(monkeypatch):
    def fake_fetch_tajwid_text(surah_id: int, start_verse: int, end_verse: int):
        raise TajwidServiceError("tajwid unavailable")

    monkeypatch.setattr(tajwid_route, "fetch_tajwid_text", fake_fetch_tajwid_text)

    try:
        tajwid_route.get_tajwid(surah_id=112, start_verse=1, end_verse=4)
    except HTTPException as exc:
        assert exc.status_code == 502
        assert exc.detail == "tajwid unavailable"
    else:
        raise AssertionError("Expected HTTPException")
