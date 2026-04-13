import json

import pytest

import app.services.tajwid_service as tajwid_service
from app.services.tajwid_service import TajwidServiceError, fetch_tajwid_text


def build_tajwid_payload():
    return {
        "data": {
            "surahs": [
                {
                    "number": 112,
                    "ayahs": [
                        {"numberInSurah": 1, "text": "[rule[قل]]"},
                        {"numberInSurah": 2, "text": "[rule[هو]]"},
                        {"numberInSurah": 3, "text": "[rule[الله]]"},
                        {"numberInSurah": 4, "text": "[rule[أحد]]"},
                    ],
                },
            ],
        },
    }


class FakeUrlOpenResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


@pytest.fixture(autouse=True)
def reset_tajwid_cache(monkeypatch, tmp_path):
    tajwid_service.clear_tajwid_cache()
    monkeypatch.delenv("TAJWID_DATA_PATH", raising=False)
    monkeypatch.setattr(
        tajwid_service,
        "DEFAULT_TAJWID_DATA_PATH",
        tmp_path / "quran_tajwid.json",
    )
    yield
    tajwid_service.clear_tajwid_cache()


def test_fetch_tajwid_text_downloads_payload_once_and_reuses_cache(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return FakeUrlOpenResponse(build_tajwid_payload())

    monkeypatch.setattr(tajwid_service, "urlopen", fake_urlopen)

    first_response = fetch_tajwid_text(112, 1, 2)
    second_response = fetch_tajwid_text(112, 3, 4)

    assert first_response["text"] == "[rule[قل]] [rule[هو]]"
    assert second_response["text"] == "[rule[الله]] [rule[أحد]]"
    assert calls == [(tajwid_service.TAJWID_API_BASE, tajwid_service.TAJWID_TIMEOUT_SECONDS)]


def test_fetch_tajwid_text_prefers_local_snapshot_when_available(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "tajwid_snapshot.json"
    snapshot_path.write_text(json.dumps(build_tajwid_payload()), encoding="utf-8")

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setenv("TAJWID_DATA_PATH", str(snapshot_path))
    monkeypatch.setattr(tajwid_service, "urlopen", fail_urlopen)

    response = fetch_tajwid_text(112, 2, 3)

    assert response["text"] == "[rule[هو]] [rule[الله]]"


def test_fetch_tajwid_text_rejects_ranges_with_missing_verses(monkeypatch):
    payload = build_tajwid_payload()
    payload["data"]["surahs"][0]["ayahs"] = [
        {"numberInSurah": 1, "text": "[rule[قل]]"},
        {"numberInSurah": 3, "text": "[rule[الله]]"},
    ]

    monkeypatch.setattr(
        tajwid_service,
        "urlopen",
        lambda _url, timeout: FakeUrlOpenResponse(payload),
    )

    with pytest.raises(TajwidServiceError, match="plage"):
        fetch_tajwid_text(112, 1, 3)


def test_fetch_tajwid_text_raises_when_configured_snapshot_is_missing(monkeypatch, tmp_path):
    missing_snapshot_path = tmp_path / "missing_tajwid_snapshot.json"
    monkeypatch.setenv("TAJWID_DATA_PATH", str(missing_snapshot_path))

    with pytest.raises(TajwidServiceError, match="introuvable"):
        fetch_tajwid_text(112, 1, 1)
